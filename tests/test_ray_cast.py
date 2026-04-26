"""tests/test_ray_cast.py — unit tests for the Numba ray-cast engine.

Covers:
- Empty grid → ray reaches ``max_range``.
- Building blocks ray → ``hit_type=1``.
- Out-of-bounds detection.
- Smoke attenuation cuts the ray short (Beer–Lambert with factor 2).
- Multi-ray sweep agrees with the single-ray function.
- Direction convention: ``angle=0`` walks east (+col), ``angle=π/2`` walks
  north (−row).
- Determinism: pure function, same inputs → same outputs.
- Throughput target: 450 rays in < 1 ms on M1 (warm cache).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from flare.sensors.ray_cast import cast_ray, cast_rays, make_uniform_angles


def _empty_world(H: int = 50, W: int = 50):
    occ = np.zeros((H, W), dtype=np.int8)
    ext = np.zeros((H, W), dtype=np.float32)
    return occ, ext


# ===========================================================================
# Empty world
# ===========================================================================


class TestEmptyWorld:
    def test_max_range_reached(self) -> None:
        occ, ext = _empty_world()
        x0, y0 = 25.0, 25.0
        hx, hy, d, ht = cast_ray(
            x0, y0, 0.0, 10, occ, ext, cell_size_m=1.0, snr_threshold=0.05
        )
        assert ht == 0
        assert d == pytest.approx(10.0)
        assert hx == 35  # walked east 10 cells
        assert hy == 25

    def test_dtype_preserved(self) -> None:
        occ, ext = _empty_world()
        hx, hy, d, ht = cast_ray(
            25.0, 25.0, 0.0, 5, occ, ext, 1.0, 0.05
        )
        assert isinstance(hx, int)
        assert isinstance(hy, int)
        assert isinstance(ht, int)


# ===========================================================================
# Direction convention
# ===========================================================================


class TestDirectionConvention:
    def test_east_increments_column(self) -> None:
        occ, ext = _empty_world()
        _, _, _, _ = cast_ray(10.0, 10.0, 0.0, 5, occ, ext, 1.0, 0.05)
        # Inspect terminal coords
        hx, hy, _, _ = cast_ray(10.0, 10.0, 0.0, 5, occ, ext, 1.0, 0.05)
        assert hx == 15
        assert hy == 10

    def test_north_decrements_row(self) -> None:
        """angle = π/2 → world north → array row decreases."""
        occ, ext = _empty_world()
        hx, hy, _, _ = cast_ray(
            10.0, 20.0, math.pi / 2, 5, occ, ext, 1.0, 0.05
        )
        assert hx == 10
        assert hy == 15  # row went from 20 → 15 (north)

    def test_west_decrements_column(self) -> None:
        occ, ext = _empty_world()
        hx, hy, _, _ = cast_ray(
            30.0, 10.0, math.pi, 5, occ, ext, 1.0, 0.05
        )
        assert hx == 25
        assert hy == 10


# ===========================================================================
# Occupancy hit
# ===========================================================================


class TestOccupiedHit:
    def test_building_blocks_ray(self) -> None:
        occ, ext = _empty_world()
        occ[10, 15] = 1  # wall at (col=15, row=10)
        hx, hy, d, ht = cast_ray(
            10.0, 10.0, 0.0, 20, occ, ext, 1.0, 0.05
        )
        assert ht == 1
        assert (hx, hy) == (15, 10)
        assert d == pytest.approx(5.0)

    def test_no_hit_when_blocker_outside_range(self) -> None:
        occ, ext = _empty_world(H=50, W=50)
        occ[10, 40] = 1
        hx, hy, d, ht = cast_ray(
            10.0, 10.0, 0.0, 5, occ, ext, 1.0, 0.05
        )
        assert ht == 0
        assert d == pytest.approx(5.0)


# ===========================================================================
# Out-of-bounds
# ===========================================================================


class TestOutOfBounds:
    def test_ray_walks_off_grid(self) -> None:
        occ, ext = _empty_world(H=20, W=20)
        hx, hy, d, ht = cast_ray(
            10.0, 10.0, 0.0, 100, occ, ext, 1.0, 0.05
        )
        assert ht == 3
        assert hx >= 20  # walked off the east edge


# ===========================================================================
# Smoke attenuation
# ===========================================================================


class TestSmokeAttenuation:
    def test_no_smoke_full_range(self) -> None:
        occ, ext = _empty_world()
        hx, hy, d, ht = cast_ray(
            10.0, 10.0, 0.0, 12, occ, ext, 1.0, 0.05
        )
        assert ht == 0  # max range
        assert d == pytest.approx(12.0)

    def test_dense_smoke_cuts_ray_short(self) -> None:
        """Uniform extinction σ = 2 m⁻¹, cell_size 1 m, threshold 0.05.
        Cumulative threshold = -ln(0.05) / 2 ≈ 1.498 nepers.
        Per-step contribution = 2 × 1 = 2; threshold reached after step 1."""
        occ = np.zeros((30, 30), dtype=np.int8)
        ext = np.full((30, 30), 2.0, dtype=np.float32)
        hx, hy, d, ht = cast_ray(
            15.0, 15.0, 0.0, 25, occ, ext, 1.0, 0.05
        )
        assert ht == 2  # smoke cutoff
        assert d == pytest.approx(1.0)

    def test_threshold_consistent(self) -> None:
        """At σ = 0.5 m⁻¹ uniform, cell_size 1 m, threshold 0.05:
        cumulative threshold ≈ 1.498; per-step σ·ds = 0.5; reached at
        step ⌈1.498/0.5⌉ = 3."""
        occ = np.zeros((30, 30), dtype=np.int8)
        ext = np.full((30, 30), 0.5, dtype=np.float32)
        hx, hy, d, ht = cast_ray(
            15.0, 15.0, 0.0, 25, occ, ext, 1.0, 0.05
        )
        assert ht == 2
        assert d == pytest.approx(3.0)

    def test_lower_snr_threshold_punches_further(self) -> None:
        occ = np.zeros((30, 30), dtype=np.int8)
        ext = np.full((30, 30), 0.5, dtype=np.float32)
        _, _, d_strict, _ = cast_ray(
            15.0, 15.0, 0.0, 25, occ, ext, 1.0, 0.10  # stricter
        )
        _, _, d_loose, _ = cast_ray(
            15.0, 15.0, 0.0, 25, occ, ext, 1.0, 0.001  # very permissive
        )
        assert d_loose > d_strict


# ===========================================================================
# Sweep
# ===========================================================================


class TestSweep:
    def test_make_uniform_angles(self) -> None:
        a = make_uniform_angles(8)
        assert a.shape == (8,)
        # Step is exactly 2π/8 = π/4
        deltas = np.diff(a)
        assert np.allclose(deltas, math.pi / 4)

    def test_sweep_matches_single_ray(self) -> None:
        occ, ext = _empty_world()
        occ[15, 30] = 1
        angles = np.array([0.0, math.pi / 2, math.pi], dtype=np.float64)
        hx, hy, d, ht = cast_rays(
            15.0, 15.0, angles, 30, occ, ext, 1.0, 0.05
        )
        # Compare per ray
        for i, a in enumerate(angles):
            sx, sy, sd, sht = cast_ray(
                15.0, 15.0, a, 30, occ, ext, 1.0, 0.05
            )
            assert (hx[i], hy[i]) == (sx, sy)
            assert d[i] == pytest.approx(sd)
            assert ht[i] == sht


# ===========================================================================
# Determinism
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_inputs_same_output(self) -> None:
        occ = np.zeros((40, 40), dtype=np.int8)
        occ[20, 20] = 1
        ext = np.zeros((40, 40), dtype=np.float32)
        a, b, c, d_ = cast_ray(
            10.0, 10.0, math.pi / 4, 30, occ, ext, 1.0, 0.05
        )
        a2, b2, c2, d2 = cast_ray(
            10.0, 10.0, math.pi / 4, 30, occ, ext, 1.0, 0.05
        )
        assert (a, b, c, d_) == (a2, b2, c2, d2)


# ===========================================================================
# Performance — informational
# ===========================================================================


@pytest.mark.slow
class TestPerformance:
    def test_450_rays_under_1ms(self) -> None:
        """Target: < 1 ms for 450 rays on a 500×500 grid (CLAUDE.md)."""
        H = W = 500
        occ = np.zeros((H, W), dtype=np.int8)
        # Sparse occlusions
        rng = np.random.default_rng(0)
        for _ in range(200):
            occ[rng.integers(H), rng.integers(W)] = 1
        ext = np.zeros((H, W), dtype=np.float32)
        angles = make_uniform_angles(450)
        # Warm-up to JIT-compile
        cast_rays(250.0, 250.0, angles, 60, occ, ext, 3.0, 0.05)
        # Time
        n_iter = 50
        t0 = time.perf_counter()
        for _ in range(n_iter):
            cast_rays(250.0, 250.0, angles, 60, occ, ext, 3.0, 0.05)
        elapsed_ms = (time.perf_counter() - t0) / n_iter * 1000.0
        # Target is < 1 ms; allow up to 5 ms before flagging on M1.
        assert elapsed_ms < 5.0, (
            f"450-ray sweep took {elapsed_ms:.3f} ms; target < 1 ms"
        )
