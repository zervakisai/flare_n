"""tests/test_smoke.py — Gaussian smoke plume tests.

Covers:
- Constructor input validation.
- Empty burning mask → zero density.
- Isotropic plume symmetry at ``wind_speed = 0``.
- Wind elongation (max density downwind).
- Extinction coupling and Jin visibility.
- Persistence (decay).
- ``set_wind`` rebuilds kernel; ``reset`` zeros density.
- Approximate mass conservation: Σ ρ · cell_area ≈ Q · n_burning.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from flare.core.hazards.smoke import (
    ALPHA_MASS_EXTINCTION_M2_PER_KG,
    GaussianSmokePlume,
)


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_zero_cell_size(self) -> None:
        with pytest.raises(ValueError, match="cell_size_m"):
            GaussianSmokePlume(grid_shape=(10, 10), cell_size_m=0.0)

    def test_zero_emission_rejected(self) -> None:
        with pytest.raises(ValueError, match="emission_rate_kg"):
            GaussianSmokePlume(grid_shape=(10, 10), emission_rate_kg=0.0)

    def test_zero_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            GaussianSmokePlume(grid_shape=(10, 10), sigma_along_m=0.0)

    def test_decay_outside_range(self) -> None:
        with pytest.raises(ValueError, match="decay"):
            GaussianSmokePlume(grid_shape=(10, 10), decay=1.0)
        with pytest.raises(ValueError, match="decay"):
            GaussianSmokePlume(grid_shape=(10, 10), decay=-0.1)

    def test_burning_mask_shape_mismatch(self) -> None:
        s = GaussianSmokePlume(grid_shape=(10, 10))
        with pytest.raises(ValueError, match="shape"):
            s.update(np.zeros((20, 20), dtype=bool))


# ===========================================================================
# Empty mask
# ===========================================================================


class TestEmptyMask:
    def test_no_density_when_no_fire(self) -> None:
        s = GaussianSmokePlume(grid_shape=(50, 50))
        s.update(np.zeros((50, 50), dtype=bool))
        assert s.density.sum() == pytest.approx(0.0)

    def test_dtype_is_float32(self) -> None:
        s = GaussianSmokePlume(grid_shape=(20, 20))
        s.update(np.zeros((20, 20), dtype=bool))
        assert s.density.dtype == np.float32


# ===========================================================================
# Isotropic plume at zero wind
# ===========================================================================


class TestIsotropicGaussian:
    def test_symmetric_around_source(self) -> None:
        s = GaussianSmokePlume(grid_shape=(51, 51), wind_speed=0.0)
        m = np.zeros((51, 51), dtype=bool)
        m[25, 25] = True
        s.update(m)
        d = s.density
        for k in range(1, 10):
            assert d[25 - k, 25] == pytest.approx(d[25 + k, 25], rel=1e-5)
            assert d[25, 25 - k] == pytest.approx(d[25, 25 + k], rel=1e-5)
            assert d[25 - k, 25 - k] == pytest.approx(
                d[25 + k, 25 + k], rel=1e-5
            )

    def test_peak_at_source(self) -> None:
        s = GaussianSmokePlume(grid_shape=(51, 51), wind_speed=0.0)
        m = np.zeros((51, 51), dtype=bool)
        m[25, 25] = True
        s.update(m)
        d = s.density
        assert d[25, 25] == d.max()


# ===========================================================================
# Anisotropy
# ===========================================================================


class TestAnisotropicPlume:
    def test_max_density_downwind(self) -> None:
        """Wind blowing east (TO convention) → density at +x (east) of
        source exceeds density at +y (north) at the same Euclidean
        distance, assuming σ_along > σ_cross."""
        s = GaussianSmokePlume(
            grid_shape=(101, 101),
            sigma_along_m=18.0,
            sigma_cross_m=4.0,
            wind_speed=5.0,
            wind_direction=0.0,  # TO east
        )
        m = np.zeros((101, 101), dtype=bool)
        m[50, 50] = True
        s.update(m)
        d = s.density
        # 5 cells (15 m) east of source vs 5 cells north at the same dist
        east_5 = d[50, 55]
        north_5 = d[45, 50]
        assert east_5 > north_5, (
            f"Expected stronger plume east (downwind): "
            f"east={east_5:.4e}, north={north_5:.4e}"
        )

    def test_isotropy_when_sigmas_equal(self) -> None:
        """If σ_along == σ_cross, plume is isotropic regardless of wind."""
        s = GaussianSmokePlume(
            grid_shape=(51, 51),
            sigma_along_m=8.0,
            sigma_cross_m=8.0,
            wind_speed=5.0,
            wind_direction=math.pi / 3,
        )
        m = np.zeros((51, 51), dtype=bool)
        m[25, 25] = True
        s.update(m)
        d = s.density
        for k in range(1, 8):
            assert d[25 - k, 25] == pytest.approx(d[25 + k, 25], rel=1e-5)
            assert d[25, 25 - k] == pytest.approx(d[25, 25 + k], rel=1e-5)


# ===========================================================================
# Extinction + visibility
# ===========================================================================


class TestExtinctionCoupling:
    def test_extinction_scales_density(self) -> None:
        s = GaussianSmokePlume(grid_shape=(20, 20))
        s.update(np.eye(20, dtype=bool))
        np.testing.assert_allclose(
            s.extinction,
            ALPHA_MASS_EXTINCTION_M2_PER_KG * s.density,
            rtol=1e-6,
        )

    def test_default_alpha_mass(self) -> None:
        s = GaussianSmokePlume(grid_shape=(10, 10))
        assert s.alpha_mass_extinction == 8700.0

    def test_visibility_inverse_extinction(self) -> None:
        s = GaussianSmokePlume(grid_shape=(20, 20))
        s.update(np.ones((20, 20), dtype=bool))
        v = s.visibility(K=3.0)
        sig = s.extinction
        mask = sig > 1e-3
        assert mask.any()
        np.testing.assert_allclose(v[mask], 3.0 / sig[mask], rtol=1e-5)

    def test_visibility_inf_in_clear_air(self) -> None:
        """Cells far from any fire have negligible extinction → V = inf."""
        s = GaussianSmokePlume(
            grid_shape=(101, 101), sigma_along_m=2.0, sigma_cross_m=2.0,
        )
        m = np.zeros((101, 101), dtype=bool)
        m[50, 50] = True
        s.update(m)
        v = s.visibility(K=3.0)
        # Corner cells are far enough to be effectively zero σ_ext.
        assert math.isinf(float(v[0, 0]))


# ===========================================================================
# Persistence (decay)
# ===========================================================================


class TestPersistence:
    def test_decay_zero_no_memory(self) -> None:
        s = GaussianSmokePlume(grid_shape=(20, 20), decay=0.0)
        m1 = np.zeros((20, 20), dtype=bool); m1[10, 10] = True
        m2 = np.zeros((20, 20), dtype=bool)
        s.update(m1)
        s.update(m2)
        assert s.density.max() == pytest.approx(0.0)

    def test_decay_persists(self) -> None:
        s = GaussianSmokePlume(grid_shape=(30, 30), decay=0.7)
        m1 = np.zeros((30, 30), dtype=bool); m1[15, 15] = True
        m2 = np.zeros((30, 30), dtype=bool)
        s.update(m1)
        d1 = s.density.copy()
        s.update(m2)
        d2 = s.density
        np.testing.assert_allclose(d2, 0.7 * d1, rtol=1e-5)


# ===========================================================================
# set_wind / reset
# ===========================================================================


class TestSetWind:
    def test_kernel_changes_when_wind_changes(self) -> None:
        s = GaussianSmokePlume(grid_shape=(50, 50), wind_speed=0.0)
        k0 = s.kernel
        s.set_wind(wind_speed=5.0, wind_direction=math.pi / 4)
        k1 = s.kernel
        assert k0.shape == k1.shape
        assert not np.array_equal(k0, k1)


class TestReset:
    def test_density_zeroed(self) -> None:
        s = GaussianSmokePlume(grid_shape=(20, 20))
        s.update(np.ones((20, 20), dtype=bool))
        assert s.density.sum() > 0
        s.reset()
        assert s.density.sum() == pytest.approx(0.0)


# ===========================================================================
# Mass conservation
# ===========================================================================


class TestMassConservation:
    """Σ ρ · cell_area ≈ Q · n_burning when the kernel reach is ≥ ~4σ."""

    def test_single_emitter(self) -> None:
        cell = 3.0
        Q = 1.0
        s = GaussianSmokePlume(
            grid_shape=(101, 101),
            cell_size_m=cell,
            emission_rate_kg=Q,
            sigma_along_m=6.0,
            sigma_cross_m=6.0,
            wind_speed=0.0,
            decay=0.0,
        )
        m = np.zeros((101, 101), dtype=bool)
        m[50, 50] = True
        s.update(m)
        total = float(s.density.sum()) * (cell**2)
        assert total == pytest.approx(Q, rel=0.05)

    def test_linear_in_burning_count(self) -> None:
        """Doubling the burning count doubles the total mass (within tol)."""
        cell = 3.0
        Q = 1.0
        s = GaussianSmokePlume(
            grid_shape=(81, 81),
            cell_size_m=cell,
            emission_rate_kg=Q,
            sigma_along_m=4.0,
            sigma_cross_m=4.0,
            wind_speed=0.0,
            decay=0.0,
        )
        m1 = np.zeros((81, 81), dtype=bool)
        m1[40, 40] = True
        s.update(m1)
        total_1 = float(s.density.sum()) * (cell**2)

        s.reset()
        m2 = np.zeros((81, 81), dtype=bool)
        m2[40, 40] = True
        m2[40, 60] = True  # second emitter, far enough not to leak off-grid
        s.update(m2)
        total_2 = float(s.density.sum()) * (cell**2)

        assert total_2 == pytest.approx(2.0 * total_1, rel=0.05)


# ===========================================================================
# Determinism
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_inputs_same_output(self) -> None:
        a = GaussianSmokePlume(
            grid_shape=(40, 40), wind_speed=3.0, wind_direction=0.5,
        )
        b = GaussianSmokePlume(
            grid_shape=(40, 40), wind_speed=3.0, wind_direction=0.5,
        )
        rng = np.random.default_rng(0)
        m = (rng.random((40, 40)) > 0.95).astype(bool)
        a.update(m)
        b.update(m)
        np.testing.assert_array_equal(a.density, b.density)
