"""flare/sensors/gps.py — Beitian BN-880 GPS + HMC5883L compass model.

When the agent has clear sky, the sensor reports::

    x_meas = x_true + N(0, σ_xy²)
    y_meas = y_true + N(0, σ_xy²)
    ψ_meas = ψ_true + N(0, σ_ψ²)

with ``σ_xy = CEP / √2 ≈ 1.4 m`` (BN-880 spec, CEP = 2 m) and
``σ_ψ = 1.5°`` (HMC5883L magnetometer noise).

In dense urban canyons, multipath and signal blockage trigger dropout.
Per-step dropout probability::

    p_drop = σ(k · (building_density − θ))

with ``σ`` the logistic sigmoid, ``k`` the steepness, ``θ`` the threshold,
and ``building_density ∈ [0, 1]`` the local density around the agent.
During dropout the sensor performs dead-reckoning from the last good
fix, accumulating Wiener-process drift::

    σ_drift(t) = σ_dr · √t,  σ_dr = 0.5 m / √s

(using the diffusion-style scaling rather than the literal ``m/s`` of
CLAUDE.md, which is the standard interpretation for IMU drift).

Determinism: all randomness via the caller-supplied
``np.random.Generator`` (Invariant I2). The dropout state itself is
internal to the sensor and should be reset at the start of every
episode via :meth:`GPSSensor.reset`.

Citations
---------
- Beitian BN-880 GPS module datasheet — CEP < 2.0 m at HDOP < 1.
- Honeywell HMC5883L 3-axis digital compass — heading noise σ ≈ 1.5°.
- IEEE 802.11 / GNSS multipath: Mendoza-Silva et al., *Multipath
  detection methods for GNSS-based vehicular localization* (2018).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np

# === BN-880 / HMC5883L constants ==========================================

GPS_CEP_M: float = 2.0
"""Circular Error Probable (m). BN-880 datasheet @ HDOP < 1."""

GPS_SIGMA_PER_AXIS_M: float = 1.4  # ≈ CEP / sqrt(2)
"""Per-axis Gaussian σ (m) — derived from CEP under bivariate Rayleigh
assumption. BN-880 spec."""

COMPASS_SIGMA_DEG: float = 1.5
"""Heading noise σ (degrees). HMC5883L digital compass spec."""

GPS_DROPOUT_THRESHOLD: float = 0.4
"""Building-density threshold at which p_drop = 0.5."""

GPS_DROPOUT_STEEPNESS: float = 10.0
"""Logistic steepness ``k``: large ⇒ sharp on/off transition."""

GPS_DR_DRIFT_SIGMA_M_PER_RTSEC: float = 0.5
"""Wiener-process drift coefficient (m / √s) during dropout."""


class GPSSensor:
    """BN-880 + HMC5883L sensor model with urban-canyon dropout.

    Stateful — owns ``last_good_*`` and a dropout counter. Call
    :meth:`reset` at the start of every episode.

    Parameters
    ----------
    rng:
        Caller-supplied ``np.random.Generator``.
    sigma_xy_m, sigma_heading_deg, dropout_threshold, dropout_steepness,
    dr_drift_sigma_m_per_rtsec:
        See module-level constants for defaults and citations.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        sigma_xy_m: float = GPS_SIGMA_PER_AXIS_M,
        sigma_heading_deg: float = COMPASS_SIGMA_DEG,
        dropout_threshold: float = GPS_DROPOUT_THRESHOLD,
        dropout_steepness: float = GPS_DROPOUT_STEEPNESS,
        dr_drift_sigma_m_per_rtsec: float = GPS_DR_DRIFT_SIGMA_M_PER_RTSEC,
    ) -> None:
        if sigma_xy_m < 0:
            raise ValueError(f"sigma_xy_m must be ≥ 0, got {sigma_xy_m}")
        if sigma_heading_deg < 0:
            raise ValueError(
                f"sigma_heading_deg must be ≥ 0, got {sigma_heading_deg}"
            )
        if not 0.0 <= dropout_threshold <= 1.0:
            raise ValueError(
                f"dropout_threshold must be in [0, 1], got {dropout_threshold}"
            )
        if dropout_steepness < 0:
            raise ValueError(
                f"dropout_steepness must be ≥ 0, got {dropout_steepness}"
            )
        if dr_drift_sigma_m_per_rtsec < 0:
            raise ValueError(
                f"dr_drift_sigma_m_per_rtsec must be ≥ 0, "
                f"got {dr_drift_sigma_m_per_rtsec}"
            )

        self._rng = rng
        self._sigma_xy = float(sigma_xy_m)
        self._sigma_h = math.radians(float(sigma_heading_deg))
        self._drop_thr = float(dropout_threshold)
        self._drop_k = float(dropout_steepness)
        self._dr_sigma = float(dr_drift_sigma_m_per_rtsec)

        # State
        self._last_good_x = 0.0
        self._last_good_y = 0.0
        self._last_good_h = 0.0
        self._dropout_seconds = 0.0
        self._initialized = False

    # -- Public ----------------------------------------------------------

    @property
    def sigma_xy_m(self) -> float:
        return self._sigma_xy

    @property
    def sigma_heading_rad(self) -> float:
        return self._sigma_h

    @property
    def dropout_seconds(self) -> float:
        return self._dropout_seconds

    def reset(
        self,
        true_x_m: float,
        true_y_m: float,
        true_heading: float,
    ) -> None:
        """Initialize / reset state. Call at episode start with ground truth."""
        self._last_good_x = float(true_x_m)
        self._last_good_y = float(true_y_m)
        self._last_good_h = float(true_heading)
        self._dropout_seconds = 0.0
        self._initialized = True

    def dropout_probability(self, building_density: float) -> float:
        """Logistic ``p_drop = σ(k·(density − θ))``."""
        z = self._drop_k * (float(building_density) - self._drop_thr)
        # Clip exponent for numerical stability.
        if z > 50.0:
            return 1.0
        if z < -50.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def update(
        self,
        true_x_m: float,
        true_y_m: float,
        true_heading: float,
        dt: float,
        building_density: float,
    ) -> tuple[float, float, float, bool]:
        """One sensor update.

        Returns ``(x_meas, y_meas, ψ_meas, dropout_flag)``. ``ψ_meas`` is
        in radians (the same units as the input heading).
        """
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if not 0.0 <= building_density <= 1.0:
            raise ValueError(
                f"building_density must be in [0, 1], got {building_density}"
            )
        if not self._initialized:
            self.reset(true_x_m, true_y_m, true_heading)

        p_drop = self.dropout_probability(building_density)
        dropout = bool(self._rng.random() < p_drop)

        if dropout:
            self._dropout_seconds += dt
            # Wiener increment: each step contributes σ_dr · √dt of drift.
            drift_step = self._dr_sigma * math.sqrt(dt)
            self._last_good_x += float(
                self._rng.normal(0.0, drift_step)
            )
            self._last_good_y += float(
                self._rng.normal(0.0, drift_step)
            )
            # Heading also drifts during dropout (HMC5883L is independent
            # of GNSS, but we model joint outage; small per-step jitter).
            self._last_good_h += float(self._rng.normal(0.0, self._sigma_h))
            return (
                self._last_good_x,
                self._last_good_y,
                self._last_good_h,
                True,
            )

        # Clean fix: i.i.d. Gaussian on truth, then update last_good.
        self._dropout_seconds = 0.0
        x_meas = true_x_m + float(self._rng.normal(0.0, self._sigma_xy))
        y_meas = true_y_m + float(self._rng.normal(0.0, self._sigma_xy))
        h_meas = true_heading + float(self._rng.normal(0.0, self._sigma_h))
        self._last_good_x = x_meas
        self._last_good_y = y_meas
        self._last_good_h = h_meas
        return x_meas, y_meas, h_meas, False
