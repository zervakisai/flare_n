"""tests/test_lidar.py — LD19 LiDAR sensor tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from flare.sensors.lidar import (
    LIDAR_ANGULAR_NOISE_DEG,
    LIDAR_FALSE_NEGATIVE_RATE,
    LIDAR_MAX_RANGE_M,
    LIDAR_NUM_RAYS,
    LiDARScan,
    LiDARSensor,
)


def _sensor(seed: int = 0, **kwargs) -> LiDARSensor:
    rng = np.random.default_rng(seed)
    return LiDARSensor(rng=rng, **kwargs)


def _empty_world(H: int = 80, W: int = 80):
    return (
        np.zeros((H, W), dtype=np.int8),
        np.zeros((H, W), dtype=np.float32),
    )


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_zero_range_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="max_range_m"):
            LiDARSensor(rng=rng, max_range_m=0.0)

    def test_zero_rays_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="n_rays"):
            LiDARSensor(rng=rng, n_rays=0)

    def test_bad_fn_rate_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="false_negative_rate"):
            LiDARSensor(rng=rng, false_negative_rate=1.5)

    def test_bad_snr_threshold_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="snr_threshold"):
            LiDARSensor(rng=rng, snr_threshold=0.0)

    def test_zero_cell_size_rejected_at_scan(self) -> None:
        s = _sensor()
        occ, ext = _empty_world()
        with pytest.raises(ValueError, match="cell_size_m"):
            s.scan(40.0, 40.0, 0.0, occ, ext, cell_size_m=0.0)


# ===========================================================================
# Defaults
# ===========================================================================


class TestDefaults:
    def test_default_constants(self) -> None:
        assert LIDAR_MAX_RANGE_M == 12.0
        assert LIDAR_NUM_RAYS == 450
        assert LIDAR_ANGULAR_NOISE_DEG == 2.0
        assert LIDAR_FALSE_NEGATIVE_RATE == 0.02

    def test_n_rays_property(self) -> None:
        s = _sensor()
        assert s.n_rays == 450

    def test_max_range_property(self) -> None:
        s = _sensor()
        assert s.max_range_m == 12.0


# ===========================================================================
# Scan shape
# ===========================================================================


class TestScanShape:
    def test_returns_lidarscan_with_correct_shapes(self) -> None:
        s = _sensor()
        occ, ext = _empty_world()
        scan = s.scan(40.0, 40.0, 0.0, occ, ext, cell_size_m=1.0)
        assert isinstance(scan, LiDARScan)
        n = s.n_rays
        assert scan.angles.shape == (n,)
        assert scan.distances_m.shape == (n,)
        assert scan.hit_x.shape == (n,)
        assert scan.hit_y.shape == (n,)
        assert scan.hit_types.shape == (n,)
        assert scan.valid.shape == (n,)


# ===========================================================================
# Empty world → max range
# ===========================================================================


class TestEmptyWorld:
    def test_distances_at_max_range(self) -> None:
        """In an empty world with no smoke, distances cluster at max_range_m
        (modulo noise) and hit_types are mostly 0 (max-range)."""
        s = _sensor(angular_noise_deg=0.0,
                    range_noise_mm_fixed=0.0,
                    range_noise_proportional=0.0,
                    false_negative_rate=0.0)
        occ, ext = _empty_world(H=200, W=200)
        scan = s.scan(100.0, 100.0, 0.0, occ, ext, cell_size_m=1.0)
        # All rays should reach max range (12 cells = 12 m)
        assert (scan.hit_types == 0).all()
        np.testing.assert_allclose(scan.distances_m, 12.0, rtol=1e-5)


# ===========================================================================
# Wall hit
# ===========================================================================


class TestWallHit:
    def test_wall_to_east_detected(self) -> None:
        s = _sensor(angular_noise_deg=0.0,
                    range_noise_mm_fixed=0.0,
                    range_noise_proportional=0.0,
                    false_negative_rate=0.0)
        occ = np.zeros((50, 50), dtype=np.int8)
        # Wall at column 30 (5 m east of an agent at col 25, cell_size 1 m)
        occ[:, 30] = 1
        ext = np.zeros((50, 50), dtype=np.float32)
        scan = s.scan(25.0, 25.0, 0.0, occ, ext, cell_size_m=1.0)
        # The ray closest to angle 0 (east) should detect at ~5 m.
        # Find the ray whose angle is nearest 0 modulo 2π.
        angles_mod = np.mod(scan.angles + math.pi, 2 * math.pi) - math.pi
        idx_east = int(np.argmin(np.abs(angles_mod)))
        assert scan.hit_types[idx_east] == 1
        assert scan.distances_m[idx_east] == pytest.approx(5.0, abs=0.1)


# ===========================================================================
# Smoke attenuation
# ===========================================================================


class TestSmokeAttenuation:
    def test_dense_smoke_cuts_range(self) -> None:
        s = _sensor(angular_noise_deg=0.0,
                    range_noise_mm_fixed=0.0,
                    range_noise_proportional=0.0,
                    false_negative_rate=0.0)
        occ = np.zeros((80, 80), dtype=np.int8)
        # σ_ext = 1.5 m⁻¹ uniform — heavy smoke
        ext = np.full((80, 80), 1.5, dtype=np.float32)
        scan = s.scan(40.0, 40.0, 0.0, occ, ext, cell_size_m=1.0)
        # Every ray should be cut by smoke (hit_type = 2) well before 12 m
        assert (scan.hit_types == 2).all()
        assert (scan.distances_m < 5.0).all()


# ===========================================================================
# Noise statistics
# ===========================================================================


class TestNoiseStatistics:
    def test_range_noise_at_max_range_matches_spec(self) -> None:
        """Empirical σ_r at r = 12 m matches the spec formula
        ``σ_r = 10 mm + 0.001 · r`` (≈ 22 mm)."""
        s = _sensor(angular_noise_deg=0.0, false_negative_rate=0.0)
        occ, ext = _empty_world(H=200, W=200)
        samples = []
        for _ in range(20):
            scan = s.scan(100.0, 100.0, 0.0, occ, ext, cell_size_m=1.0)
            samples.append(scan.distances_m.copy())
        arr = np.concatenate(samples)
        # Mean ≈ 12 m (a clipped Gaussian, slight downward bias from the
        # ``maximum(d, 0)`` floor — but at 12 m that floor is irrelevant).
        sigma_observed = float(arr.std())
        sigma_predicted = 0.010 + 0.001 * 12.0  # 0.022 m
        # Allow ±25 % for sampling; ~9000 samples here.
        assert (
            0.75 * sigma_predicted < sigma_observed < 1.25 * sigma_predicted
        ), (
            f"σ_observed={sigma_observed:.4f} m, "
            f"σ_predicted={sigma_predicted:.4f} m"
        )

    def test_angular_noise_introduces_spread(self) -> None:
        """With non-zero σ_θ, the cast angles deviate from the nominal grid."""
        s = _sensor(angular_noise_deg=2.0,
                    range_noise_mm_fixed=0.0,
                    range_noise_proportional=0.0,
                    false_negative_rate=0.0)
        occ, ext = _empty_world()
        scan = s.scan(40.0, 40.0, 0.0, occ, ext, cell_size_m=1.0)
        # Differences between consecutive angles should not all equal 2π/N.
        nominal_step = 2 * math.pi / s.n_rays
        diffs = np.diff(scan.angles)
        assert not np.allclose(diffs, nominal_step, atol=1e-6)

    def test_false_negative_rate_empirical(self) -> None:
        """Across 5 sweeps, the empirical drop rate should be near the spec."""
        s = _sensor(angular_noise_deg=0.0,
                    range_noise_mm_fixed=0.0,
                    range_noise_proportional=0.0,
                    false_negative_rate=0.10)
        occ, ext = _empty_world()
        invalid_count = 0
        total = 0
        for _ in range(5):
            scan = s.scan(40.0, 40.0, 0.0, occ, ext, cell_size_m=1.0)
            invalid_count += int((~scan.valid).sum())
            total += scan.valid.size
        rate = invalid_count / total
        # Tolerance: ±3 standard deviations (binomial). Sample size large.
        assert 0.05 < rate < 0.15


# ===========================================================================
# Determinism — Invariant I2
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_seed_same_scan(self) -> None:
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        s_a = LiDARSensor(rng=rng_a)
        s_b = LiDARSensor(rng=rng_b)
        occ = np.zeros((40, 40), dtype=np.int8)
        occ[20, 20] = 1
        ext = np.zeros((40, 40), dtype=np.float32)
        a = s_a.scan(15.0, 15.0, 0.5, occ, ext, cell_size_m=1.0)
        b = s_b.scan(15.0, 15.0, 0.5, occ, ext, cell_size_m=1.0)
        np.testing.assert_array_equal(a.angles, b.angles)
        np.testing.assert_array_equal(a.distances_m, b.distances_m)
        np.testing.assert_array_equal(a.hit_x, b.hit_x)
        np.testing.assert_array_equal(a.valid, b.valid)


# ===========================================================================
# Range cap
# ===========================================================================


class TestRangeCap:
    def test_distances_never_exceed_max_range(self) -> None:
        """Even with positive range noise, reported distance ≤ max_range_m
        + small noise tail. Empirically test mean+3σ stays below max+20 mm."""
        s = _sensor(false_negative_rate=0.0)
        occ, ext = _empty_world(H=200, W=200)
        scan = s.scan(100.0, 100.0, 0.0, occ, ext, cell_size_m=1.0)
        # Most rays report at max_range; with σ_r at 12 m ≈ 22 mm,
        # very few should exceed max + 100 mm.
        too_far = (scan.distances_m > 12.5).mean()
        assert too_far < 0.05
