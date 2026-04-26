"""tests/test_fire_deterministic.py — unit tests for fire_deterministic.

Mirrors Paper 1's ``unit_test_fire_ca.py`` against the Paper 2 module path.
Covers determinism, spread rules, smoke generation, burnout, and the
``force_cell_state`` test hook.

Marker ``@pytest.mark.invariant`` is applied to determinism tests because
they underpin Invariant I1 (MDP continuity) and I2 (deterministic replay).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import numpy as np
import pytest

from flare.core.hazards.fire_deterministic import (
    BURNED_OUT,
    BURNING,
    UNBURNED,
    FireSpreadModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fire_model(
    map_size: int = 20,
    seed: int = 42,
    n_ignition: int = 2,
    **kwargs,
) -> FireSpreadModel:
    """Construct a no-wind FireSpreadModel for testing."""
    rng = np.random.default_rng(seed)
    return FireSpreadModel(
        map_shape=(map_size, map_size),
        rng=rng,
        n_ignition=n_ignition,
        **kwargs,
    )


# ===========================================================================
# Determinism — Invariants I1, I2
# ===========================================================================


@pytest.mark.invariant
class TestFireCADeterminism:
    """Fire CA produces identical results for identical seeds (DC-1)."""

    def test_same_seed_same_state(self) -> None:
        a = _make_fire_model(seed=42)
        b = _make_fire_model(seed=42)
        for _ in range(10):
            a.step()
            b.step()
        np.testing.assert_array_equal(
            a.fire_mask, b.fire_mask,
            err_msg="Fire CA must be deterministic for same seed",
        )
        np.testing.assert_array_equal(
            a.smoke_mask, b.smoke_mask,
            err_msg="Smoke must be deterministic for same seed",
        )

    def test_different_seed_different_state(self) -> None:
        a = _make_fire_model(seed=42)
        b = _make_fire_model(seed=99)
        for _ in range(10):
            a.step()
            b.step()
        assert not np.array_equal(a.fire_mask, b.fire_mask), (
            "Different seeds should produce different fire states"
        )

    def test_wind_path_deterministic(self) -> None:
        """Wind-modulated spread is also deterministic for same seed."""
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        a = FireSpreadModel(
            map_shape=(30, 30), rng=rng_a, n_ignition=3,
            wind_speed=5.0, wind_direction=np.pi / 4,
        )
        b = FireSpreadModel(
            map_shape=(30, 30), rng=rng_b, n_ignition=3,
            wind_speed=5.0, wind_direction=np.pi / 4,
        )
        for _ in range(20):
            a.step()
            b.step()
        np.testing.assert_array_equal(a.fire_mask, b.fire_mask)


# ===========================================================================
# Spread rules
# ===========================================================================


class TestFireSpreadRules:
    def test_initial_ignition(self) -> None:
        m = _make_fire_model(n_ignition=3)
        assert m.fire_mask.any()

    def test_fire_spreads(self) -> None:
        m = _make_fire_model(n_ignition=2, map_size=30)
        initial_count = int(m.fire_mask.sum())
        for _ in range(20):
            m.step()
        assert m.total_affected >= initial_count

    def test_water_does_not_burn(self) -> None:
        rng = np.random.default_rng(42)
        landuse = np.ones((20, 20), dtype=np.int8)
        landuse[:, :10] = 4  # water
        m = FireSpreadModel(
            map_shape=(20, 20), rng=rng, n_ignition=3, landuse_map=landuse,
        )
        for _ in range(30):
            m.step()
        assert not m.fire_mask[:, :10].any()


# ===========================================================================
# Wind anisotropy (FD-5b)
# ===========================================================================


class TestWindAnisotropy:
    """With strong wind, spread skews downwind."""

    def test_wind_breaks_isotropy(self) -> None:
        """Two models, identical seed; one with wind, one without — must differ."""
        rng_a = np.random.default_rng(13)
        rng_b = np.random.default_rng(13)
        no_wind = FireSpreadModel(
            map_shape=(30, 30), rng=rng_a, n_ignition=2, wind_speed=0.0,
        )
        windy = FireSpreadModel(
            map_shape=(30, 30), rng=rng_b, n_ignition=2,
            wind_speed=8.0, wind_direction=0.0,
        )
        for _ in range(30):
            no_wind.step()
            windy.step()
        # The two ignition sets are identical (same seed, same RNG order in
        # _ignite_random) but spread paths diverge once wind kicks in.
        assert not np.array_equal(no_wind.fire_mask, windy.fire_mask)


# ===========================================================================
# Smoke
# ===========================================================================


class TestSmokeGeneration:
    def test_smoke_mask_shape(self) -> None:
        m = _make_fire_model(map_size=15)
        assert m.smoke_mask.shape == (15, 15)

    def test_smoke_near_fire(self) -> None:
        m = _make_fire_model(map_size=20, n_ignition=3)
        for _ in range(10):
            m.step()
        assert m.smoke_mask.max() > 0.0

    def test_smoke_values_bounded(self) -> None:
        m = _make_fire_model(map_size=20, n_ignition=3)
        for _ in range(20):
            m.step()
        assert m.smoke_mask.min() >= 0.0
        assert m.smoke_mask.max() <= 1.0


# ===========================================================================
# Burnout
# ===========================================================================


class TestBurnout:
    def test_burnout_occurs(self) -> None:
        m = _make_fire_model(map_size=20, n_ignition=5)
        for _ in range(250):
            m.step()
        assert m.burned_mask.any()

    def test_burned_cells_not_burning(self) -> None:
        m = _make_fire_model(map_size=20, n_ignition=5)
        for _ in range(250):
            m.step()
        overlap = m.fire_mask & m.burned_mask
        assert not overlap.any()


# ===========================================================================
# force_cell_state test hook
# ===========================================================================


class TestForceState:
    def test_force_burning(self) -> None:
        rng = np.random.default_rng(42)
        m = FireSpreadModel(map_shape=(10, 10), rng=rng, n_ignition=0)
        assert not m.fire_mask[5, 3]
        m.force_cell_state(3, 5, BURNING)
        assert m.fire_mask[5, 3]

    def test_force_does_not_affect_others(self) -> None:
        rng = np.random.default_rng(42)
        m = FireSpreadModel(map_shape=(10, 10), rng=rng, n_ignition=0)
        initial = m.fire_mask.copy()
        m.force_cell_state(5, 5, BURNING)
        diff = m.fire_mask != initial
        assert diff.sum() == 1
        assert diff[5, 5]


# ===========================================================================
# Property surface
# ===========================================================================


class TestFireProperties:
    def test_fire_mask_is_bool(self) -> None:
        assert _make_fire_model().fire_mask.dtype == bool

    def test_burned_mask_is_bool(self) -> None:
        assert _make_fire_model().burned_mask.dtype == bool

    def test_smoke_mask_is_float(self) -> None:
        assert _make_fire_model().smoke_mask.dtype == np.float32

    def test_total_affected_is_int(self) -> None:
        assert isinstance(_make_fire_model().total_affected, (int, np.integer))

    def test_masks_are_copies(self) -> None:
        m = _make_fire_model()
        a = m.fire_mask
        b = m.fire_mask
        assert a is not b


# ===========================================================================
# Roads firebreak
# ===========================================================================


class TestRoadsFirebreak:
    def test_roads_halve_probability(self) -> None:
        """A model with all-roads should produce strictly fewer ignitions
        on average than a model with no roads, given identical inputs."""
        # Average over a small batch of seeds to suppress variance.
        no_roads_total = 0
        roads_total = 0
        for s in range(20):
            rng_a = np.random.default_rng(s)
            rng_b = np.random.default_rng(s)
            no_roads = FireSpreadModel(
                map_shape=(30, 30), rng=rng_a, n_ignition=3,
            )
            roads = FireSpreadModel(
                map_shape=(30, 30), rng=rng_b, n_ignition=3,
                roads_mask=np.ones((30, 30), dtype=bool),
            )
            for _ in range(15):
                no_roads.step()
                roads.step()
            no_roads_total += no_roads.total_affected
            roads_total += roads.total_affected
        assert roads_total < no_roads_total, (
            f"Roads firebreak should suppress spread "
            f"(no_roads={no_roads_total}, roads={roads_total})"
        )


# ===========================================================================
# State-machine sanity
# ===========================================================================


class TestStateMachine:
    def test_state_constants_distinct(self) -> None:
        assert UNBURNED == 0
        assert BURNING == 1
        assert BURNED_OUT == 2

    def test_no_cell_skips_burning(self) -> None:
        """A cell cannot transition UNBURNED → BURNED_OUT directly within one
        step (must pass through BURNING). Burnout requires ``burn_timer``
        to exceed the cell's burnout threshold, which means at least one
        prior step in BURNING."""
        m = _make_fire_model(map_size=20, n_ignition=5)
        for _ in range(200):
            prev_burning = m.fire_mask
            prev_burned = m.burned_mask
            m.step()
            newly_burned = m.burned_mask & ~prev_burned
            assert not (newly_burned & ~prev_burning).any(), (
                "Detected cell that became BURNED_OUT without prior BURNING"
            )
