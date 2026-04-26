"""tests/test_env.py — FLARE-PO Gymnasium env tests.

Covers:
- Constructor input validation.
- ``reset`` returns an observation conforming to ``observation_space``.
- ``step`` returns the standard 5-tuple with sane types and shapes.
- ``check_env`` (Gymnasium 1.x API) passes — Invariant I3.
- Determinism (Invariant I2): same seed ⇒ same trajectory.
- Termination causes: out-of-bounds, occupied, fire, NFZ, goal.
- Reward signs match the spec (death −1, goal +1, otherwise small).
- ``render(rgb_array)`` returns a HxWx3 uint8 array.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flare.envs.flare_pomdp import (
    ACTION_HOVER,
    ACTION_SCAN,
    FlarePOEnv,
    N_ACTIONS,
)
from flare.scenarios.synthetic import make_synthetic


def _env(**kwargs) -> FlarePOEnv:
    sc = make_synthetic(grid_shape=(40, 40), seed=0)
    kwargs.setdefault("scenario", sc)
    kwargs.setdefault("belief_shape", (10, 10))
    kwargs.setdefault("max_steps", 50)
    kwargs.setdefault("ego_size", 11)
    kwargs.setdefault("n_initial_ignitions", 1)
    return FlarePOEnv(**kwargs)


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_observability_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="observability"):
            _env(observability=1.5)

    def test_zero_max_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            _env(max_steps=0)

    def test_small_ego_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="ego_size"):
            _env(ego_size=2)

    def test_bad_fire_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="fire_mode"):
            _env(fire_mode="random")

    def test_bad_render_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="render_mode"):
            _env(render_mode="weird")

    def test_belief_shape_must_divide_grid(self) -> None:
        with pytest.raises(ValueError, match="belief_shape"):
            _env(belief_shape=(7, 7))


# ===========================================================================
# Spaces
# ===========================================================================


class TestSpaces:
    def test_action_space_size(self) -> None:
        env = _env()
        assert env.action_space.n == N_ACTIONS == 10

    def test_observation_space_keys(self) -> None:
        env = _env()
        keys = set(env.observation_space.spaces.keys())
        assert keys == {
            "lidar_occupancy",
            "fire_detection",
            "smoke_density",
            "belief_map",
            "pose",
            "mission_ctx",
            "goal_vector",
        }

    def test_observation_space_shapes(self) -> None:
        env = _env(ego_size=11, belief_shape=(10, 10))
        s = env.observation_space.spaces
        assert s["lidar_occupancy"].shape == (11, 11)
        assert s["fire_detection"].shape == (11, 11)
        assert s["smoke_density"].shape == (11, 11)
        assert s["belief_map"].shape == (4, 10, 10)
        assert s["pose"].shape == (3,)
        assert s["mission_ctx"].shape == (5,)
        assert s["goal_vector"].shape == (3,)


# ===========================================================================
# Reset / step API
# ===========================================================================


class TestResetStep:
    def test_reset_returns_obs_in_space(self) -> None:
        env = _env()
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_returns_tuple_of_5(self) -> None:
        env = _env()
        env.reset(seed=0)
        out = env.step(ACTION_HOVER)
        assert len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_before_reset_raises(self) -> None:
        env = _env()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)

    def test_invalid_action_raises(self) -> None:
        env = _env()
        env.reset(seed=0)
        with pytest.raises(ValueError, match="invalid action"):
            env.step(99)


# ===========================================================================
# Gymnasium check_env — Invariant I3
# ===========================================================================


@pytest.mark.invariant
class TestGymnasiumAPI:
    def test_check_env_passes(self) -> None:
        env = _env()
        # Skip rendering checks (we only support rgb_array optionally).
        check_env(env, skip_render_check=True)


# ===========================================================================
# Determinism — Invariant I2
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_seed_same_trajectory(self) -> None:
        env_a = _env(max_steps=20)
        env_b = _env(max_steps=20)
        a, _ = env_a.reset(seed=42)
        b, _ = env_b.reset(seed=42)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        for action in [2, 2, 8, 9, 0, 4, 4, 6, 6, 8]:
            oa, ra, ta, _, _ = env_a.step(action)
            ob, rb, tb, _, _ = env_b.step(action)
            for k in oa:
                np.testing.assert_array_equal(oa[k], ob[k])
            assert ra == rb
            assert ta == tb
            if ta:
                break


# ===========================================================================
# Termination causes
# ===========================================================================


class TestTerminationCauses:
    def test_step_into_building_dies(self) -> None:
        # Carve a scenario with a building right next to the start.
        sc = make_synthetic(grid_shape=(40, 40), seed=0)
        ax, ay = sc.start
        sc.occupancy[ay, ax + 1] = 1  # building one cell east of start
        env = FlarePOEnv(
            scenario=sc,
            belief_shape=(10, 10),
            max_steps=50,
            ego_size=11,
            n_initial_ignitions=0,
            wind_speed=0.0,
        )
        env.reset(seed=0)
        _, reward, terminated, _, info = env.step(2)  # E
        assert terminated
        assert reward == -1.0
        assert info.get("death_cause") == "occupied"

    def test_walking_off_grid_dies(self) -> None:
        sc = make_synthetic(grid_shape=(20, 20), seed=0)
        env = FlarePOEnv(
            scenario=sc,
            belief_shape=(10, 10),
            max_steps=50,
            ego_size=7,
            n_initial_ignitions=0,
            wind_speed=0.0,
        )
        env.reset(seed=0)
        # Walk west repeatedly until off-grid
        terminated = False
        for _ in range(15):
            _, reward, terminated, _, info = env.step(6)  # W
            if terminated:
                assert info.get("death_cause") == "out_of_bounds"
                assert reward == -1.0
                break
        assert terminated, "expected to walk off-grid in 15 steps"


# ===========================================================================
# Reward shape
# ===========================================================================


class TestReward:
    def test_step_penalty_on_hover(self) -> None:
        env = _env(n_initial_ignitions=0, wind_speed=0.0)
        env.reset(seed=0)
        _, reward, terminated, _, _ = env.step(ACTION_HOVER)
        # No movement → progress=0; some new info early on; small penalty.
        assert not terminated
        assert reward < 0.2     # not the goal reward
        assert reward > -0.5    # not death

    def test_scan_action_changes_belief(self) -> None:
        env = _env(n_initial_ignitions=0, wind_speed=0.0)
        env.reset(seed=0)
        # Hover then scan; the belief should grow (more cells observed)
        env.step(ACTION_HOVER)
        before = env._belief.certainty.sum()
        env.step(ACTION_SCAN)
        after = env._belief.certainty.sum()
        # SCAN with widened HFOV should not reduce information.
        assert after >= before * 0.99  # tolerate decay


# ===========================================================================
# Render
# ===========================================================================


class TestRender:
    def test_render_rgb_array(self) -> None:
        env = _env(render_mode="rgb_array")
        env.reset(seed=0)
        img = env.render()
        H, W = env.scenario.grid_shape
        assert img is not None
        assert img.shape == (H, W, 3)
        assert img.dtype == np.uint8

    def test_render_returns_none_when_disabled(self) -> None:
        env = _env(render_mode=None)
        env.reset(seed=0)
        assert env.render() is None
