"""tests/test_nfz.py — dynamic NFZ tests.

Covers:
- Constructor input validation.
- Initial state (no zones).
- ``nfz_mask`` shape and dtype.
- Determinism (Invariant I2): identical seeds → identical trajectories.
- ``max_zones`` cap is never exceeded.
- Spawned zone radii are within range.
- Spawned zone centres lie inside the grid.
- Mean lifetime ≈ ``mean_duration`` over many spawns.
- Steady-state mean active count ≈ ``arrival_rate · mean_duration`` (capped).
- ``reset`` clears state.
- ``arrival_rate=0`` ⇒ no zones ever appear.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import numpy as np
import pytest

from flare.core.hazards.nfz import DynamicNFZ, NFZZone


def _make(
    seed: int = 0,
    grid_shape: tuple[int, int] = (50, 50),
    **kwargs,
) -> DynamicNFZ:
    rng = np.random.default_rng(seed)
    return DynamicNFZ(grid_shape=grid_shape, rng=rng, **kwargs)


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_negative_max_zones_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_zones"):
            _make(max_zones=-1)

    def test_zero_mean_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="mean_duration"):
            _make(mean_duration=0)

    def test_negative_arrival_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="arrival_rate"):
            _make(arrival_rate=-0.1)

    def test_radius_range_inverted_rejected(self) -> None:
        with pytest.raises(ValueError, match="radius_cells_range"):
            _make(radius_cells_range=(10, 5))

    def test_radius_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="radius_cells_range"):
            _make(radius_cells_range=(0, 5))


# ===========================================================================
# Initial state
# ===========================================================================


class TestInitialState:
    def test_no_zones_at_start(self) -> None:
        n = _make()
        assert n.n_active == 0
        assert n.zones == []

    def test_default_arrival_rate(self) -> None:
        n = _make(max_zones=3, mean_duration=50)
        assert n.arrival_rate == pytest.approx(3 / 50)

    def test_explicit_arrival_rate_overrides(self) -> None:
        n = _make(max_zones=3, mean_duration=50, arrival_rate=0.2)
        assert n.arrival_rate == 0.2

    def test_mask_dtype_and_shape(self) -> None:
        n = _make(grid_shape=(20, 30))
        m = n.nfz_mask
        assert m.shape == (20, 30)
        assert m.dtype == bool
        assert not m.any()


# ===========================================================================
# Determinism — Invariant I2
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_seed_same_trajectory(self) -> None:
        a = _make(seed=42, max_zones=5, mean_duration=20)
        b = _make(seed=42, max_zones=5, mean_duration=20)
        for _ in range(100):
            a.step()
            b.step()
            assert a.zones == b.zones

    def test_different_seed_diverges(self) -> None:
        a = _make(seed=1, max_zones=5, mean_duration=20)
        b = _make(seed=2, max_zones=5, mean_duration=20)
        for _ in range(100):
            a.step()
            b.step()
        # Almost certainly differ at some point in 100 steps
        assert a.zones != b.zones


# ===========================================================================
# Cap on max zones
# ===========================================================================


class TestMaxZonesCap:
    def test_never_exceeds_cap(self) -> None:
        n = _make(
            seed=0, max_zones=4, mean_duration=200, arrival_rate=2.0
        )  # forced overload
        for _ in range(500):
            n.step()
            assert n.n_active <= 4

    def test_zero_max_zones_means_no_zones(self) -> None:
        n = _make(seed=0, max_zones=0, mean_duration=10, arrival_rate=1.0)
        for _ in range(50):
            n.step()
            assert n.n_active == 0

    def test_zero_arrival_rate_no_zones(self) -> None:
        n = _make(seed=0, max_zones=10, mean_duration=10, arrival_rate=0.0)
        for _ in range(200):
            n.step()
            assert n.n_active == 0


# ===========================================================================
# Spawn geometry
# ===========================================================================


class TestSpawnGeometry:
    def test_radius_within_range(self) -> None:
        n = _make(
            seed=0, max_zones=20, mean_duration=10,
            arrival_rate=2.0, radius_cells_range=(3, 7),
        )
        radii: list[int] = []
        for _ in range(200):
            n.step()
            radii.extend(z.radius for z in n.zones)
        assert min(radii) >= 3
        assert max(radii) <= 7

    def test_centres_inside_grid(self) -> None:
        n = _make(seed=0, max_zones=20, mean_duration=20, arrival_rate=2.0)
        H, W = 50, 50
        for _ in range(200):
            n.step()
            for z in n.zones:
                assert 0 <= z.cy < H
                assert 0 <= z.cx < W


# ===========================================================================
# Lifetime statistics
# ===========================================================================


class TestLifetimeStatistics:
    def test_mean_lifetime_matches_mean_duration(self) -> None:
        """Average lifetime of spawned zones ≈ mean_duration."""
        n = _make(
            seed=0,
            grid_shape=(200, 200),
            max_zones=200,         # large cap so spawns are rarely dropped
            mean_duration=30,
            arrival_rate=5.0,
            radius_cells_range=(3, 6),
        )
        ages: list[int] = []
        prev_ids: dict[tuple[int, int, int], int] = {}
        # We don't track zone IDs, so approximate: snapshot remaining at
        # spawn time. Track all (cy, cx, radius, max_remaining_seen)
        # and report the max_remaining as the lifetime.
        seen: dict[tuple[int, int, int], int] = {}
        prev_zones: set[tuple[int, int, int]] = set()
        for _ in range(300):
            cur = n.zones
            cur_ids = {(z.cy, z.cx, z.radius): z.remaining for z in cur}
            new_ids = set(cur_ids) - prev_zones
            for k in new_ids:
                seen[k] = cur_ids[k]
            prev_zones = set(cur_ids)
            n.step()
        ages = list(seen.values())
        assert len(ages) >= 50, (
            f"Need ≥ 50 spawns for stable mean, got {len(ages)}"
        )
        sample_mean = float(np.mean(ages))
        # Geometric(p) has mean 1/p = mean_duration. Allow ±25% tol since
        # geometric has high variance and we have moderate sample size.
        assert sample_mean == pytest.approx(30, rel=0.25), (
            f"Mean spawned lifetime {sample_mean:.2f} vs mean_duration=30"
        )


# ===========================================================================
# Steady-state occupancy
# ===========================================================================


class TestSteadyState:
    def test_average_active_below_cap(self) -> None:
        """With ``arrival_rate · mean_duration ≤ max_zones`` the system
        should settle at occupancy ≤ max_zones."""
        n = _make(
            seed=0, max_zones=8, mean_duration=20, arrival_rate=0.4
        )
        # Burn-in, then sample
        for _ in range(100):
            n.step()
        counts: list[int] = []
        for _ in range(500):
            n.step()
            counts.append(n.n_active)
        avg = float(np.mean(counts))
        assert avg <= 8.0
        # Theoretical mean (M/M/∞ approximation) is λ * E[lifetime] = 8.
        # Cap at 8 makes actual ≤ 8; loose lower bound.
        assert avg > 1.0


# ===========================================================================
# Mask correctness
# ===========================================================================


class TestMask:
    def test_mask_inside_zone(self) -> None:
        n = _make(seed=0, grid_shape=(40, 40), max_zones=1)
        # Inject a known zone
        n._zones = [NFZZone(cy=20, cx=20, radius=5, remaining=10)]
        mask = n.nfz_mask
        assert mask[20, 20]                    # centre
        assert mask[20, 25]                    # boundary
        assert not mask[20, 26]                # just outside
        assert mask[15, 20] and not mask[14, 20]

    def test_mask_off_grid_clipped(self) -> None:
        """A zone whose disk extends off-grid is clipped to grid bounds."""
        n = _make(seed=0, grid_shape=(20, 20), max_zones=1)
        n._zones = [NFZZone(cy=0, cx=0, radius=10, remaining=10)]
        mask = n.nfz_mask
        # No errors, mask shape preserved, contains the in-grid quarter.
        assert mask.shape == (20, 20)
        assert mask[0, 0]
        # Top-left quadrant largely covered; lower-right corner not.
        assert not mask[19, 19]


# ===========================================================================
# Reset
# ===========================================================================


class TestReset:
    def test_reset_clears_zones(self) -> None:
        n = _make(seed=0, max_zones=10, mean_duration=20, arrival_rate=2.0)
        for _ in range(50):
            n.step()
        assert n.n_active > 0
        n.reset()
        assert n.n_active == 0
        assert not n.nfz_mask.any()
