"""flare/core/hazards/smoke.py — anisotropic Gaussian smoke plume.

Per-step smoke density field built from a 2-D Gaussian convolution kernel
elongated along the wind direction. The field couples to sensor visibility
(LiDAR/camera, Phase 2) through the mass-specific extinction coefficient
``α_m = 8700 m² / kg``.

Mathematical model
------------------
For each burning cell at world position ``(x_b, y_b)`` with emission ``Q``
(kg per step), the contributed density at ``(x, y)`` is::

    ρ_b(x, y) = Q · g(x − x_b, y − y_b)

where the Gaussian pdf ``g`` (units ``1/m²``) is::

    g(x', y') = (1 / (2π σ_x σ_y)) · exp(−(u² / (2σ_x²) + v² / (2σ_y²)))

with ``(u, v)`` the wind-aligned rotation of ``(x', y')`` (``u`` downwind,
``v`` crosswind) and ``σ_x ≥ σ_y`` controlling along- vs cross-dispersion.

Total density (linear superposition) is computed as a single convolution::

    ρ_smoke(t) = Q · (burning(t) ⊛ g)

with optional persistence::

    ρ_smoke(t) = decay · ρ_smoke(t − 1) + Q · (burning(t) ⊛ g)

Extinction coefficient (couples to sensor visibility — Phase 2)::

    σ_ext = α_m · ρ_smoke         (units: m⁻¹)

Visibility (Jin 1971)::

    V = K / σ_ext                 (units: m;  K=3 reflective, K=8 emissive)

Citations
---------
- Mulholland, G. W. (1995). "Smoke production and properties." SFPE Handbook
  of Fire Protection Engineering. (α_m ≈ 8700 m²/kg for fuel-rich smoke.)
- McGrattan, K. et al. *NIST Fire Dynamics Simulator (FDS) Technical
  Reference Guide*, Vol. 1. (Mass-specific extinction formulation.)
- Jin, T. (1971). "Visibility through fire smoke." *Report of the Fire
  Research Institute of Japan*, No. 33.

Determinism (Invariant I2): no randomness — deterministic field given
``(burning_mask, wind, parameters)``. Stochastic ignition lives upstream
in the fire CA.

Performance
-----------
A single FFT-based convolution per step: ``O(H W log(H W))``, independent
of the number of burning cells. The kernel is rebuilt only when wind
changes (``set_wind``).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

# Mulholland 1995 / FDS — mass-specific extinction (m² / kg). Default value
# used to map smoke density to the extinction coefficient σ_ext.
ALPHA_MASS_EXTINCTION_M2_PER_KG: float = 8700.0


class GaussianSmokePlume:
    """Anisotropic Gaussian smoke plume on a 2-D grid.

    Parameters
    ----------
    grid_shape:
        ``(H, W)`` grid dimensions in cells.
    cell_size_m:
        Side length of one cell in metres. Default ``3.0`` (Paper 1).
    emission_rate_kg:
        Smoke mass emitted per burning cell per step (``Q``). Default
        ``1e-3`` kg.
    sigma_along_m, sigma_cross_m:
        Plume standard deviations along- and cross-wind, in metres.
        Defaults ``(10.0, 5.0)`` (moderate dispersion at the urban scale).
    wind_speed:
        Wind speed (m/s). At ``0.0`` the plume is isotropic with
        ``σ = (σ_along + σ_cross) / 2``.
    wind_direction:
        Wind TO-direction in radians, x = east, CCW (matches
        :mod:`flare.core.wind`).
    alpha_mass_extinction:
        Mass-specific extinction coefficient (m² kg⁻¹). Default 8700.
    decay:
        Persistence factor in ``[0, 1)``. ``0.0`` (default) ⇒ no memory
        (steady-state from the current step). Larger values retain a
        fraction of the previous-step density.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        cell_size_m: float = 3.0,
        emission_rate_kg: float = 1e-3,
        sigma_along_m: float = 10.0,
        sigma_cross_m: float = 5.0,
        wind_speed: float = 0.0,
        wind_direction: float = 0.0,
        alpha_mass_extinction: float = ALPHA_MASS_EXTINCTION_M2_PER_KG,
        decay: float = 0.0,
    ) -> None:
        if cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be > 0, got {cell_size_m}")
        if emission_rate_kg <= 0:
            raise ValueError(
                f"emission_rate_kg must be > 0, got {emission_rate_kg}"
            )
        if sigma_along_m <= 0 or sigma_cross_m <= 0:
            raise ValueError("sigma_along_m and sigma_cross_m must be > 0")
        if alpha_mass_extinction <= 0:
            raise ValueError("alpha_mass_extinction must be > 0")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")

        self._H, self._W = grid_shape
        self._cell = float(cell_size_m)
        self._Q = float(emission_rate_kg)
        self._sigma_along = float(sigma_along_m)
        self._sigma_cross = float(sigma_cross_m)
        self._wind_speed = float(wind_speed)
        self._wind_direction = float(wind_direction)
        self._alpha_m = float(alpha_mass_extinction)
        self._decay = float(decay)

        self._density = np.zeros(grid_shape, dtype=np.float32)
        self._kernel = self._build_kernel()

    # -- Public ----------------------------------------------------------

    @property
    def density(self) -> np.ndarray:
        """Smoke density field ``ρ_smoke`` (kg / m²). Returns a copy."""
        return self._density.copy()

    @property
    def extinction(self) -> np.ndarray:
        """Extinction coefficient ``σ_ext = α_m · ρ_smoke`` (1 / m)."""
        return self._alpha_m * self._density

    @property
    def alpha_mass_extinction(self) -> float:
        return self._alpha_m

    @property
    def kernel(self) -> np.ndarray:
        """Read-only view of the current convolution kernel."""
        return self._kernel.copy()

    def visibility(self, K: float = 3.0, eps: float = 1e-6) -> np.ndarray:
        """Jin (1971) visibility ``V = K / σ_ext`` (metres).

        ``K = 3`` for reflective targets (default), ``K = 8`` for
        light-emitting. Cells where ``σ_ext < eps`` return ``inf``.
        """
        sigma = self.extinction
        return np.where(sigma > eps, K / np.maximum(sigma, eps), np.inf)

    def update(self, burning_mask: np.ndarray) -> None:
        """Advance the smoke field by one step from ``burning_mask``.

        ``burning_mask`` is a 2-D boolean (or 0/1 numeric) array of shape
        ``(H, W)``. After the call::

            ρ(t) = decay · ρ(t − 1) + Q · (burning ⊛ g)

        Negative noise from FFT convolution is clipped to zero.
        """
        if burning_mask.shape != (self._H, self._W):
            raise ValueError(
                f"burning_mask shape {burning_mask.shape} ≠ "
                f"grid_shape {(self._H, self._W)}"
            )
        source = burning_mask.astype(np.float32)
        contrib = self._Q * fftconvolve(source, self._kernel, mode="same")
        contrib = np.maximum(contrib, 0.0).astype(np.float32)
        self._density = (
            self._decay * self._density + contrib
        ).astype(np.float32)

    def reset(self) -> None:
        """Zero the density field."""
        self._density.fill(0.0)

    def set_wind(self, wind_speed: float, wind_direction: float) -> None:
        """Update wind and rebuild the convolution kernel."""
        self._wind_speed = float(wind_speed)
        self._wind_direction = float(wind_direction)
        self._kernel = self._build_kernel()

    # -- Internal --------------------------------------------------------

    def _build_kernel(self) -> np.ndarray:
        """Discretized Gaussian kernel oriented downwind.

        Half-width is ``4σ`` in the longest direction (captures > 99.96 %
        of mass). Returns a float32 array shape ``(2 r + 1, 2 r + 1)``
        with the Gaussian pdf in units of ``1 / m²``.
        """
        sigma_max = max(self._sigma_along, self._sigma_cross)
        radius_cells = max(1, int(math.ceil(4.0 * sigma_max / self._cell)))
        size = 2 * radius_cells + 1

        offsets = (np.arange(size) - radius_cells) * self._cell
        xx_m, yy_m = np.meshgrid(offsets, offsets, indexing="xy")

        if self._wind_speed > 0.0:
            cos_t = math.cos(self._wind_direction)
            sin_t = math.sin(self._wind_direction)
            u = xx_m * cos_t + yy_m * sin_t
            v = -xx_m * sin_t + yy_m * cos_t
            sigma_u = self._sigma_along
            sigma_v = self._sigma_cross
        else:
            u, v = xx_m, yy_m
            sigma_u = sigma_v = 0.5 * (self._sigma_along + self._sigma_cross)

        kernel = (
            1.0 / (2.0 * math.pi * sigma_u * sigma_v)
        ) * np.exp(
            -(u**2 / (2.0 * sigma_u**2) + v**2 / (2.0 * sigma_v**2))
        )
        return kernel.astype(np.float32)
