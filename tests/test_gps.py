"""tests/test_gps.py — BN-880 GPS + HMC5883L compass tests.

Covers:
- Constructor input validation.
- Default constants match the spec.
- ``dropout_probability`` follows the logistic curve.
- Clean-sky noise is Gaussian with the spec σ.
- Heavy-dropout regime sets the flag and accumulates dead-reckoning drift.
- ``reset`` clears state.
- Determinism (Invariant I2).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from flare.sensors.gps import (
    COMPASS_SIGMA_DEG,
    GPS_CEP_M,
    GPS_DR_DRIFT_SIGMA_M_PER_RTSEC,
    GPS_DROPOUT_STEEPNESS,
    GPS_DROPOUT_THRESHOLD,
    GPS_SIGMA_PER_AXIS_M,
    GPSSensor,
)


def _make(seed: int = 0, **kwargs) -> GPSSensor:
    rng = np.random.default_rng(seed)
    s = GPSSensor(rng=rng, **kwargs)
    s.reset(0.0, 0.0, 0.0)
    return s


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_negative_sigma_xy_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="sigma_xy"):
            GPSSensor(rng=rng, sigma_xy_m=-1.0)

    def test_negative_sigma_h_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="sigma_heading"):
            GPSSensor(rng=rng, sigma_heading_deg=-1.0)

    def test_threshold_outside_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="dropout_threshold"):
            GPSSensor(rng=rng, dropout_threshold=1.5)

    def test_negative_steepness_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="dropout_steepness"):
            GPSSensor(rng=rng, dropout_steepness=-2.0)

    def test_zero_dt_rejected_at_update(self) -> None:
        s = _make()
        with pytest.raises(ValueError, match="dt"):
            s.update(0.0, 0.0, 0.0, dt=0.0, building_density=0.0)

    def test_density_outside_unit_interval_rejected(self) -> None:
        s = _make()
        with pytest.raises(ValueError, match="building_density"):
            s.update(0.0, 0.0, 0.0, dt=1.0, building_density=1.5)


# ===========================================================================
# Defaults
# ===========================================================================


class TestDefaults:
    def test_constants(self) -> None:
        assert GPS_CEP_M == 2.0
        assert GPS_SIGMA_PER_AXIS_M == pytest.approx(1.4, rel=0.05)
        assert GPS_SIGMA_PER_AXIS_M == pytest.approx(GPS_CEP_M / math.sqrt(2), rel=0.05)
        assert COMPASS_SIGMA_DEG == 1.5
        assert GPS_DROPOUT_THRESHOLD == 0.4
        assert GPS_DROPOUT_STEEPNESS == 10.0
        assert GPS_DR_DRIFT_SIGMA_M_PER_RTSEC == 0.5


# ===========================================================================
# Dropout probability
# ===========================================================================


class TestDropoutProbability:
    def test_at_threshold_one_half(self) -> None:
        s = _make()
        assert s.dropout_probability(GPS_DROPOUT_THRESHOLD) == pytest.approx(
            0.5, abs=1e-6
        )

    def test_low_density_no_dropout(self) -> None:
        s = _make()
        assert s.dropout_probability(0.0) < 0.05

    def test_high_density_almost_always_dropout(self) -> None:
        s = _make()
        assert s.dropout_probability(1.0) > 0.95

    def test_monotonic_in_density(self) -> None:
        s = _make()
        densities = np.linspace(0.0, 1.0, 11)
        probs = [s.dropout_probability(float(d)) for d in densities]
        assert all(b >= a for a, b in zip(probs, probs[1:]))


# ===========================================================================
# Clean-sky noise statistics
# ===========================================================================


class TestCleanSkyNoise:
    def test_xy_noise_matches_sigma(self) -> None:
        """In a clear-sky regime, aggregate noise std on non-dropout samples
        ≈ σ_xy. (At density=0 the residual dropout probability is the
        sigmoid tail σ(−4) ≈ 1.8 %, which we filter out.)"""
        s = _make(seed=0)
        n_samples = 5000
        dx_samples: list[float] = []
        dy_samples: list[float] = []
        for _ in range(n_samples):
            x, y, _, drop = s.update(
                0.0, 0.0, 0.0, dt=1.0, building_density=0.0
            )
            if not drop:
                dx_samples.append(x)
                dy_samples.append(y)
        assert len(dx_samples) > 4500
        sigma_x = float(np.std(dx_samples))
        sigma_y = float(np.std(dy_samples))
        assert sigma_x == pytest.approx(GPS_SIGMA_PER_AXIS_M, rel=0.10)
        assert sigma_y == pytest.approx(GPS_SIGMA_PER_AXIS_M, rel=0.10)

    def test_heading_noise_matches_sigma(self) -> None:
        s = _make(seed=1)
        n_samples = 5000
        psi_samples: list[float] = []
        for _ in range(n_samples):
            _, _, psi, drop = s.update(
                0.0, 0.0, 0.0, dt=1.0, building_density=0.0
            )
            if not drop:
                psi_samples.append(psi)
        sigma_psi = float(np.std(psi_samples))
        assert sigma_psi == pytest.approx(
            math.radians(COMPASS_SIGMA_DEG), rel=0.10
        )


# ===========================================================================
# Dropout regime
# ===========================================================================


class TestDropoutRegime:
    def test_dropout_flag_set_in_canyon(self) -> None:
        s = _make(seed=2)
        n_drops = 0
        for _ in range(500):
            _, _, _, drop = s.update(
                0.0, 0.0, 0.0, dt=1.0, building_density=0.95
            )
            if drop:
                n_drops += 1
        # p ≈ σ(10·(0.95−0.4)) = σ(5.5) ≈ 0.996. Nearly all should drop.
        assert n_drops > 480

    def test_dropout_seconds_resets_on_clean_fix(self) -> None:
        s = _make(seed=3)
        # Force a few dropout steps, then a clean step
        for _ in range(5):
            s.update(0.0, 0.0, 0.0, dt=1.0, building_density=0.95)
        # We don't know how many actually dropped — read counter
        ds_after_canyon = s.dropout_seconds
        # Clean for several to definitely catch a non-dropout step
        for _ in range(50):
            _, _, _, drop = s.update(
                0.0, 0.0, 0.0, dt=1.0, building_density=0.0
            )
            if not drop:
                break
        # Counter must reset to 0 once a clean fix lands.
        assert s.dropout_seconds == 0.0
        # And it had been ≥ 0 (could be 0 if no dropout in first 5 calls).
        assert ds_after_canyon >= 0.0

    def test_drift_grows_with_dropout_duration(self) -> None:
        """During sustained dropout the dead-reckoned position drifts away
        from truth. Variance should grow ~ linearly in dropout time."""
        # Many independent runs to estimate variance after N dropout steps.
        N_short = 4
        N_long = 64
        n_runs = 200
        true_x = 0.0
        deltas_short = []
        deltas_long = []
        for run in range(n_runs):
            s = _make(seed=run, dropout_steepness=1000.0)  # near-binary
            # Force dropouts via density=1.0 (p≈1).
            for _ in range(N_short):
                x, y, _, drop = s.update(
                    true_x, 0.0, 0.0, dt=1.0, building_density=1.0
                )
            deltas_short.append(x - true_x)
            for _ in range(N_long - N_short):
                x, y, _, drop = s.update(
                    true_x, 0.0, 0.0, dt=1.0, building_density=1.0
                )
            deltas_long.append(x - true_x)
        var_short = float(np.var(deltas_short))
        var_long = float(np.var(deltas_long))
        assert var_long > var_short, (
            f"Drift should grow: var(N={N_short})={var_short:.3f}, "
            f"var(N={N_long})={var_long:.3f}"
        )


# ===========================================================================
# Reset
# ===========================================================================


class TestReset:
    def test_reset_clears_dropout_seconds(self) -> None:
        s = _make()
        # Force dropouts
        for _ in range(20):
            s.update(0.0, 0.0, 0.0, dt=1.0, building_density=1.0)
        assert s.dropout_seconds > 0
        s.reset(10.0, 20.0, 0.5)
        assert s.dropout_seconds == 0.0

    def test_reset_sets_last_good_to_truth(self) -> None:
        s = _make()
        s.reset(10.0, 20.0, 0.5)
        # First update at dropout regime should drift from (10, 20, 0.5).
        x, y, psi, drop = s.update(
            0.0, 0.0, 0.0, dt=1.0, building_density=1.0
        )
        assert drop
        # Drift std at dt=1 is 0.5 — within 5σ of truth almost surely.
        assert abs(x - 10.0) < 5.0
        assert abs(y - 20.0) < 5.0


# ===========================================================================
# Determinism
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_seed_same_trajectory(self) -> None:
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        a = GPSSensor(rng=rng_a)
        b = GPSSensor(rng=rng_b)
        a.reset(0.0, 0.0, 0.0)
        b.reset(0.0, 0.0, 0.0)
        for k in range(50):
            density = 0.5
            ra = a.update(float(k), 0.0, 0.0, 1.0, density)
            rb = b.update(float(k), 0.0, 0.0, 1.0, density)
            assert ra == rb
