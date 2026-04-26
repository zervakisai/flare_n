"""tests/test_wind.py — unit tests for flare.core.wind.

Covers:
- Isotropic fallback at ``wind_speed = 0`` (both kernels).
- Alexandridis kernel reproduces Paper 1 numerics bit-for-bit.
- Cosine kernel matches the analytical formula.
- Coordinate convention: downwind direction yields the maximum factor.
- 180° rotation reverses anisotropy.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from flare.core.wind import (
    ALEXANDRIDIS_C1,
    ALEXANDRIDIS_C2,
    NEIGHBORS_DY_DX,
    alexandridis_wind_factors,
    cosine_wind_factors,
)


# ===========================================================================
# Neighbour ordering
# ===========================================================================


class TestNeighborhood:
    def test_eight_neighbours(self) -> None:
        assert len(NEIGHBORS_DY_DX) == 8

    def test_no_self_offset(self) -> None:
        assert (0, 0) not in NEIGHBORS_DY_DX

    def test_all_unit_steps(self) -> None:
        for dy, dx in NEIGHBORS_DY_DX:
            assert -1 <= dy <= 1 and -1 <= dx <= 1


# ===========================================================================
# Isotropic fallback
# ===========================================================================


class TestIsotropicFallback:
    def test_alexandridis_zero_wind(self) -> None:
        f = alexandridis_wind_factors(wind_speed=0.0, wind_direction=1.234)
        np.testing.assert_array_equal(f, np.ones(8, dtype=np.float32))

    def test_alexandridis_negative_wind(self) -> None:
        f = alexandridis_wind_factors(wind_speed=-2.0, wind_direction=0.5)
        np.testing.assert_array_equal(f, np.ones(8, dtype=np.float32))

    def test_cosine_zero_wind(self) -> None:
        f = cosine_wind_factors(wind_speed=0.0, wind_direction=0.7, coupling=1.0)
        np.testing.assert_array_equal(f, np.ones(8, dtype=np.float32))


# ===========================================================================
# Alexandridis numerics — bit-identity with the verbatim Paper 1 formula
# ===========================================================================


class TestAlexandridisNumerics:
    """Re-derive the factor inline and compare element-wise."""

    @pytest.mark.parametrize("speed", [1.0, 3.5, 8.0])
    @pytest.mark.parametrize("direction", [0.0, math.pi / 4, math.pi / 2,
                                           math.pi, -math.pi / 3])
    def test_matches_inline_formula(
        self, speed: float, direction: float
    ) -> None:
        got = alexandridis_wind_factors(speed, direction)
        expected = np.empty(8, dtype=np.float32)
        for i, (dy, dx) in enumerate(NEIGHBORS_DY_DX):
            spread_angle = math.atan2(dy, -dx)
            theta = spread_angle - direction
            expected[i] = math.exp(
                speed * (ALEXANDRIDIS_C1
                         + ALEXANDRIDIS_C2 * (math.cos(theta) - 1.0))
            )
        np.testing.assert_array_equal(got, expected)

    def test_dtype_is_float32(self) -> None:
        f = alexandridis_wind_factors(5.0, 0.0)
        assert f.dtype == np.float32

    def test_shape_is_eight(self) -> None:
        f = alexandridis_wind_factors(5.0, 0.0)
        assert f.shape == (8,)


# ===========================================================================
# Anisotropy direction (Alexandridis)
# ===========================================================================


class TestAlexandridisAnisotropy:
    """Spread is enhanced downwind, suppressed upwind.

    ``NEIGHBORS_DY_DX`` indexes by the burning neighbour's offset from the
    candidate. For spread heading east, the candidate sits east of the
    burning cell — so its burning neighbour offset is ``(0, -1)``.
    For spread heading west, ``(0, +1)``.
    """

    def test_max_factor_in_downwind_direction(self) -> None:
        # wind_direction=0 (TO east) → downwind = east → spread east is max.
        f = alexandridis_wind_factors(wind_speed=5.0, wind_direction=0.0)
        idx_spread_east = NEIGHBORS_DY_DX.index((0, -1))
        idx_spread_west = NEIGHBORS_DY_DX.index((0, +1))
        assert f[idx_spread_east] == f.max()
        assert f[idx_spread_west] == f.min()

    def test_180_rotation_reverses_pattern(self) -> None:
        f0 = alexandridis_wind_factors(wind_speed=5.0, wind_direction=0.0)
        fpi = alexandridis_wind_factors(wind_speed=5.0, wind_direction=math.pi)
        idx_spread_east = NEIGHBORS_DY_DX.index((0, -1))
        idx_spread_west = NEIGHBORS_DY_DX.index((0, +1))
        # When the wind reverses, the max swaps direction.
        assert f0[idx_spread_east] == pytest.approx(
            fpi[idx_spread_west], rel=1e-6
        )
        assert f0[idx_spread_west] == pytest.approx(
            fpi[idx_spread_east], rel=1e-6
        )


# ===========================================================================
# Cosine kernel
# ===========================================================================


class TestCosineKernel:
    @pytest.mark.parametrize("coupling", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("direction", [0.0, math.pi / 2, math.pi])
    def test_matches_analytical_formula(
        self, coupling: float, direction: float
    ) -> None:
        got = cosine_wind_factors(
            wind_speed=5.0, wind_direction=direction, coupling=coupling
        )
        expected = np.empty(8, dtype=np.float32)
        for i, (dy, dx) in enumerate(NEIGHBORS_DY_DX):
            spread_angle = math.atan2(dy, -dx)
            theta = direction - spread_angle
            expected[i] = max(0.0, 1.0 + coupling * math.cos(theta))
        np.testing.assert_array_equal(got, expected)

    def test_clipped_at_zero(self) -> None:
        """With coupling > 1 the upwind factor would go negative; clipped."""
        f = cosine_wind_factors(
            wind_speed=5.0, wind_direction=0.0, coupling=2.0
        )
        assert (f >= 0.0).all()

    def test_max_in_downwind_direction(self) -> None:
        """Same anisotropy convention as Alexandridis — peak downwind."""
        f = cosine_wind_factors(
            wind_speed=5.0, wind_direction=0.0, coupling=1.0
        )
        idx_spread_east = NEIGHBORS_DY_DX.index((0, -1))
        idx_spread_west = NEIGHBORS_DY_DX.index((0, +1))
        assert f[idx_spread_east] == f.max()
        assert f[idx_spread_west] == f.min()
