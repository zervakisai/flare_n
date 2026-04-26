"""tests/test_camera.py — fire-detection camera tests.

Covers:
- Constructor input validation.
- No-fire world → empty detection.
- Fire ahead in clean air → detected with confidence ∈ (0, 1].
- Fire behind agent (outside HFOV) → not detected.
- Fire beyond range → not detected.
- Wall between agent and fire → not detected.
- Smoke between agent and fire → not detected.
- False-negative dropout matches the spec rate empirically.
- Determinism (Invariant I2).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from flare.sensors.camera import (
    CAMERA_FALSE_NEGATIVE_RATE,
    CAMERA_HFOV_DEG,
    CAMERA_MAX_RANGE_M,
    CameraScan,
    CameraSensor,
)


def _make(seed: int = 0, **kwargs) -> CameraSensor:
    rng = np.random.default_rng(seed)
    return CameraSensor(rng=rng, **kwargs)


def _empty_world(H: int = 80, W: int = 80):
    occ = np.zeros((H, W), dtype=np.int8)
    fire = np.zeros((H, W), dtype=bool)
    ext = np.zeros((H, W), dtype=np.float32)
    return occ, fire, ext


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_zero_hfov_rejected(self) -> None:
        with pytest.raises(ValueError, match="hfov_deg"):
            _make(hfov_deg=0.0)

    def test_too_wide_hfov_rejected(self) -> None:
        with pytest.raises(ValueError, match="hfov_deg"):
            _make(hfov_deg=400.0)

    def test_zero_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_range_m"):
            _make(max_range_m=0.0)

    def test_bad_fn_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="false_negative_rate"):
            _make(false_negative_rate=2.0)

    def test_bad_snr_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="snr_threshold"):
            _make(snr_threshold=1.0)

    def test_zero_cell_size_rejected_at_scan(self) -> None:
        s = _make()
        occ, fire, ext = _empty_world()
        with pytest.raises(ValueError, match="cell_size_m"):
            s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=0.0)

    def test_shape_mismatch_rejected(self) -> None:
        s = _make()
        occ, fire, ext = _empty_world(50, 50)
        bad_fire = np.zeros((40, 40), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            s.scan(25.0, 25.0, 0.0, occ, bad_fire, ext, cell_size_m=1.0)


# ===========================================================================
# Defaults
# ===========================================================================


class TestDefaults:
    def test_constants(self) -> None:
        assert CAMERA_HFOV_DEG == 130.0
        assert CAMERA_MAX_RANGE_M == 30.0
        assert CAMERA_FALSE_NEGATIVE_RATE == 0.05

    def test_properties(self) -> None:
        s = _make()
        assert s.hfov_deg == 130.0
        assert s.max_range_m == 30.0


# ===========================================================================
# No fire
# ===========================================================================


class TestNoFire:
    def test_empty_detections_when_no_fire(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world()
        scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        assert isinstance(scan, CameraScan)
        assert scan.detected_y.size == 0
        assert scan.detected_x.size == 0
        assert scan.confidences.size == 0


# ===========================================================================
# Detection in clean air
# ===========================================================================


class TestCleanAirDetection:
    def test_fire_ahead_detected(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world()
        # Fire 10 cells east of agent
        fire[40, 50] = True
        scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        assert (scan.detected_y == 40).any()
        assert (scan.detected_x == 50).any()

    def test_confidence_in_range(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world()
        fire[40, 50] = True
        scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        idx = int(np.where(scan.detected_x == 50)[0][0])
        c = float(scan.confidences[idx])
        assert 0.0 < c <= 1.0

    def test_closer_fire_higher_confidence(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world(80, 80)
        fire[40, 45] = True   # 5 cells east
        fire[40, 60] = True   # 20 cells east
        scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        # Find each detection
        idx_close = int(np.where(scan.detected_x == 45)[0][0])
        idx_far = int(np.where(scan.detected_x == 60)[0][0])
        assert scan.confidences[idx_close] > scan.confidences[idx_far]


# ===========================================================================
# HFOV gating
# ===========================================================================


class TestFOVGating:
    def test_fire_behind_not_detected(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world()
        # Fire 10 cells west of agent — agent looks east
        fire[40, 30] = True
        scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        assert scan.detected_y.size == 0

    def test_fire_at_fov_edge(self) -> None:
        """At HFOV/2 = 65°, a fire just inside should be detected, just
        outside should not."""
        s = _make(false_negative_rate=0.0, hfov_deg=130.0)
        occ, fire, ext = _empty_world(80, 80)
        # Inside FOV: fire at +60° from heading=0, distance 5
        # World coords: dx = 5 cos 60° = 2.5, dy = 5 sin 60° ≈ 4.33 (north)
        # array: col = 40 + 2 ≈ 42 (round to 3 → 43), row = 40 - 4 ≈ 36
        fire[36, 43] = True
        scan_in = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)

        fire2 = np.zeros_like(fire)
        # Outside FOV: +85° from heading=0
        # dx = 5 cos 85° ≈ 0.44, dy = 5 sin 85° ≈ 4.98
        fire2[35, 40] = True
        scan_out = s.scan(40.0, 40.0, 0.0, occ, fire2, ext, cell_size_m=1.0)

        assert scan_in.detected_y.size > 0
        assert scan_out.detected_y.size == 0


# ===========================================================================
# Range gating
# ===========================================================================


class TestRangeGating:
    def test_far_fire_not_detected(self) -> None:
        s = _make(false_negative_rate=0.0, max_range_m=10.0)
        occ, fire, ext = _empty_world(60, 60)
        # 20 cells east, beyond 10-cell max range
        fire[30, 50] = True
        scan = s.scan(30.0, 30.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        assert scan.detected_y.size == 0


# ===========================================================================
# Occlusion
# ===========================================================================


class TestOcclusion:
    def test_wall_blocks_fire(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world(60, 60)
        # Fire at 10 cells east
        fire[30, 40] = True
        # Wall in between (5 cells east)
        occ[30, 35] = 1
        scan = s.scan(30.0, 30.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        assert scan.detected_y.size == 0


# ===========================================================================
# Smoke attenuation
# ===========================================================================


class TestSmokeAttenuation:
    def test_dense_smoke_blocks_fire(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world(80, 80)
        fire[40, 55] = True  # 15 cells east
        # Heavy smoke σ_ext = 0.5/m → attenuates the LOS over 15 m
        ext[:, :] = 0.5
        scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        # SNR cutoff threshold: cum_ext = 0.5·15 = 7.5 > 1.498 → blocked.
        assert scan.detected_y.size == 0

    def test_thin_smoke_allows_detection(self) -> None:
        s = _make(false_negative_rate=0.0)
        occ, fire, ext = _empty_world(80, 80)
        fire[40, 50] = True  # 10 cells east
        # Thin smoke σ_ext = 0.05 → cum_ext ≈ 0.5 < 1.498
        ext[:, :] = 0.05
        scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        assert scan.detected_y.size == 1


# ===========================================================================
# False-negative empirical rate
# ===========================================================================


class TestFalseNegative:
    def test_empirical_fn_rate(self) -> None:
        """Detection probability ≈ 1 - FN. Place many fires, count detected."""
        s = _make(false_negative_rate=0.20)
        occ = np.zeros((80, 80), dtype=np.int8)
        fire = np.zeros((80, 80), dtype=bool)
        # Place 50 fires in front of the agent (heading east)
        rng = np.random.default_rng(0)
        for _ in range(50):
            r = int(rng.integers(2, 25))
            ang = float(rng.uniform(-1.0, 1.0))  # narrow forward spread
            dy = -int(round(r * math.sin(ang)))
            dx = int(round(r * math.cos(ang)))
            fire[40 + dy, 40 + dx] = True
        ext = np.zeros((80, 80), dtype=np.float32)
        # Aggregate over many sweeps
        n_total_fires = int(fire.sum())
        n_detected = 0
        for _ in range(20):
            scan = s.scan(40.0, 40.0, 0.0, occ, fire, ext, cell_size_m=1.0)
            n_detected += scan.detected_y.size
        rate_seen = n_detected / (20 * n_total_fires)
        # Expect ~ 0.80 ± tolerance
        assert 0.65 < rate_seen < 0.92


# ===========================================================================
# Determinism
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_seed_same_scan(self) -> None:
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        a = CameraSensor(rng=rng_a)
        b = CameraSensor(rng=rng_b)
        occ = np.zeros((40, 40), dtype=np.int8)
        fire = np.zeros((40, 40), dtype=bool)
        for (yy, xx) in [(15, 25), (20, 30), (10, 22)]:
            fire[yy, xx] = True
        ext = np.full((40, 40), 0.02, dtype=np.float32)
        sa = a.scan(15.0, 15.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        sb = b.scan(15.0, 15.0, 0.0, occ, fire, ext, cell_size_m=1.0)
        np.testing.assert_array_equal(sa.detected_y, sb.detected_y)
        np.testing.assert_array_equal(sa.detected_x, sb.detected_x)
        np.testing.assert_array_equal(sa.confidences, sb.confidences)
