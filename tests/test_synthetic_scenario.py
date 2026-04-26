"""tests/test_synthetic_scenario.py — synthetic scenario tests."""
from __future__ import annotations

import numpy as np
import pytest

from flare.scenarios.synthetic import Scenario, make_synthetic


class TestValidation:
    def test_too_small_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match="grid_shape"):
            make_synthetic(grid_shape=(5, 5))

    def test_density_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="building_density"):
            make_synthetic(building_density=1.0)

    def test_zero_cell_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="cell_size_m"):
            make_synthetic(cell_size_m=0.0)


class TestShape:
    def test_default_60x60(self) -> None:
        s = make_synthetic()
        assert s.grid_shape == (60, 60)
        assert s.occupancy.shape == (60, 60)
        assert s.landuse.shape == (60, 60)
        assert s.roads_mask.shape == (60, 60)

    def test_custom_shape(self) -> None:
        s = make_synthetic(grid_shape=(40, 80))
        assert s.grid_shape == (40, 80)
        assert s.occupancy.shape == (40, 80)


class TestStartGoal:
    def test_start_and_goal_clear(self) -> None:
        s = make_synthetic(seed=0)
        sx, sy = s.start
        gx, gy = s.goal
        assert s.occupancy[sy - 2 : sy + 3, sx - 2 : sx + 3].sum() == 0
        assert s.occupancy[gy - 2 : gy + 3, gx - 2 : gx + 3].sum() == 0

    def test_start_left_goal_right(self) -> None:
        s = make_synthetic()
        assert s.start[0] < s.goal[0]


class TestLanduseAlignment:
    def test_buildings_get_urban_landuse(self) -> None:
        s = make_synthetic(seed=42)
        assert (s.landuse[s.occupancy == 1] == 2).all()
        assert (s.landuse[s.occupancy == 0] == 1).all()


class TestDeterminism:
    def test_same_seed_same_scenario(self) -> None:
        a = make_synthetic(seed=7)
        b = make_synthetic(seed=7)
        np.testing.assert_array_equal(a.occupancy, b.occupancy)
        np.testing.assert_array_equal(a.landuse, b.landuse)
        assert a.start == b.start
        assert a.goal == b.goal

    def test_different_seed_different_grid(self) -> None:
        a = make_synthetic(seed=1, building_density=0.20)
        b = make_synthetic(seed=2, building_density=0.20)
        assert not np.array_equal(a.occupancy, b.occupancy)


class TestDataclassFrozen:
    def test_scenario_is_frozen(self) -> None:
        s = make_synthetic()
        with pytest.raises((AttributeError, Exception)):
            s.name = "mutated"  # type: ignore[misc]


class TestBuildingDensity:
    def test_density_is_approximately_realized(self) -> None:
        """After clearing start/goal, the density is a bit lower than
        the input; with default density and 60×60 = 3600 cells minus
        50 cleared, the fraction should be within ±3 percentage points."""
        target = 0.20
        s = make_synthetic(building_density=target, seed=3)
        actual = float(s.occupancy.mean())
        assert abs(actual - target) < 0.03
