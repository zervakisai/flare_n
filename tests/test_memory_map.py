"""tests/test_memory_map.py — BeliefMap unit tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from flare.belief.memory_map import (
    CHANNEL_CERTAINTY,
    CHANNEL_FIRE,
    CHANNEL_OCCUPANCY,
    CHANNEL_TRAFFIC,
    N_CHANNELS,
    BeliefMap,
)


def _make(
    world: tuple[int, int] = (100, 100),
    belief: tuple[int, int] = (10, 10),
    osm: np.ndarray | None = None,
    prior_noise: float = 0.0,
    tau: float = 50.0,
    seed: int = 0,
) -> BeliefMap:
    rng = np.random.default_rng(seed)
    return BeliefMap(
        world_shape=world,
        belief_shape=belief,
        osm_prior=osm,
        prior_noise=prior_noise,
        tau_decay_steps=tau,
        rng=rng,
    )


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_indivisible_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            _make(world=(100, 100), belief=(7, 10))

    def test_non_square_downsample_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-square"):
            _make(world=(100, 200), belief=(10, 10))

    def test_prior_noise_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="prior_noise"):
            _make(prior_noise=2.0)

    def test_zero_tau_rejected(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            _make(tau=0.0)

    def test_osm_shape_mismatch_rejected(self) -> None:
        bad_prior = np.zeros((50, 50), dtype=np.float32)
        with pytest.raises(ValueError, match="osm_prior shape"):
            _make(world=(100, 100), osm=bad_prior)

    def test_observation_shape_mismatch_rejected(self) -> None:
        b = _make(world=(100, 100), belief=(10, 10))
        bad_obs = np.zeros((50, 50), dtype=bool)
        with pytest.raises(ValueError, match="observed_mask_world"):
            b.update_observation(bad_obs)


# ===========================================================================
# Initialization
# ===========================================================================


class TestInit:
    def test_default_zeros(self) -> None:
        b = _make()
        assert b.shape == (N_CHANNELS, 10, 10)
        np.testing.assert_array_equal(b.map, np.zeros((N_CHANNELS, 10, 10)))

    def test_downsample_factor(self) -> None:
        b = _make(world=(100, 100), belief=(10, 10))
        assert b.downsample_factor == 10

    def test_osm_prior_no_noise_perfect_copy(self) -> None:
        prior = np.zeros((100, 100), dtype=np.float32)
        prior[:50, :50] = 1.0  # top-left quadrant occupied
        b = _make(osm=prior, prior_noise=0.0)
        # Block-mean of a region of all 1s ⇒ 1.0; all 0s ⇒ 0.
        occ = b.occupancy
        assert occ[:5, :5].mean() == pytest.approx(1.0)
        assert occ[5:, 5:].mean() == pytest.approx(0.0)

    def test_osm_prior_with_noise_close_to_truth(self) -> None:
        """5 % bit-flip ⇒ block mean stays close to the deterministic value."""
        prior = np.ones((100, 100), dtype=np.float32)
        b = _make(osm=prior, prior_noise=0.05)
        # Each 10×10 block has 100 cells, ~5 flipped on average ⇒ mean ≈ 0.95.
        occ = b.occupancy
        assert occ.mean() == pytest.approx(0.95, abs=0.03)


# ===========================================================================
# Observation update
# ===========================================================================


class TestUpdateObservation:
    def test_only_observed_blocks_change(self) -> None:
        b = _make(world=(100, 100), belief=(10, 10))
        # Observe a 10×10 patch in the top-left
        mask = np.zeros((100, 100), dtype=bool)
        mask[:10, :10] = True
        fire = np.zeros((100, 100), dtype=bool)
        fire[5, 5] = True  # one fire cell inside patch
        b.update_observation(mask, observed_fire_world=fire)
        assert b.fire[0, 0] == pytest.approx(1.0 / 100)  # 1/100 of block
        # All other belief cells remain 0
        rest = b.fire.copy()
        rest[0, 0] = 0
        assert (rest == 0).all()

    def test_certainty_one_inside_observed_blocks(self) -> None:
        b = _make()
        mask = np.zeros((100, 100), dtype=bool)
        mask[:10, :10] = True  # one belief cell observed
        b.update_observation(mask)
        assert b.certainty[0, 0] == 1.0
        # Uninformed cells stay at 0
        rest = b.certainty.copy()
        rest[0, 0] = 0
        assert (rest == 0).all()

    def test_partial_block_observation_marks_block(self) -> None:
        """Even one observed cell in a 10×10 block marks the entire belief
        cell as observed."""
        b = _make()
        mask = np.zeros((100, 100), dtype=bool)
        mask[3, 7] = True  # single cell within block (0, 0)
        b.update_observation(mask)
        assert b.certainty[0, 0] == 1.0

    def test_missing_channels_untouched(self) -> None:
        prior = np.ones((100, 100), dtype=np.float32)
        b = _make(osm=prior)
        # Snapshot occupancy before observation
        occ_before = b.occupancy.copy()
        # Observe but don't pass observed_occupancy
        mask = np.ones((100, 100), dtype=bool)
        b.update_observation(
            mask, observed_fire_world=np.zeros((100, 100), dtype=bool)
        )
        np.testing.assert_array_equal(b.occupancy, occ_before)


# ===========================================================================
# Decay
# ===========================================================================


class TestDecay:
    def test_certainty_decays(self) -> None:
        b = _make(tau=10.0)
        mask = np.ones((100, 100), dtype=bool)
        b.update_observation(mask)
        np.testing.assert_array_equal(b.certainty, np.ones((10, 10)))
        b.decay(dt=10.0)
        np.testing.assert_allclose(
            b.certainty, np.ones((10, 10)) * math.exp(-1.0), rtol=1e-5
        )

    def test_decay_does_not_touch_other_channels(self) -> None:
        prior = np.ones((100, 100), dtype=np.float32)
        b = _make(osm=prior, prior_noise=0.0, tau=5.0)
        occ_before = b.occupancy.copy()
        mask = np.ones((100, 100), dtype=bool)
        fire = np.zeros((100, 100), dtype=bool)
        fire[:50, :50] = True
        b.update_observation(mask, observed_fire_world=fire)
        fire_before = b.fire.copy()
        b.decay(dt=20.0)
        np.testing.assert_array_equal(b.occupancy, occ_before)
        np.testing.assert_array_equal(b.fire, fire_before)


# ===========================================================================
# Reset
# ===========================================================================


class TestReset:
    def test_reset_zeros_all_channels(self) -> None:
        prior = np.ones((100, 100), dtype=np.float32)
        b = _make(osm=prior)
        mask = np.ones((100, 100), dtype=bool)
        fire = np.zeros((100, 100), dtype=bool)
        fire[0, 0] = True
        b.update_observation(mask, observed_fire_world=fire)
        b.reset()
        np.testing.assert_array_equal(b.map, np.zeros_like(b.map))


# ===========================================================================
# Determinism
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_seed_same_initial_belief(self) -> None:
        prior = np.zeros((100, 100), dtype=np.float32)
        prior[::3, ::3] = 1.0  # arbitrary pattern
        a = _make(osm=prior, prior_noise=0.05, seed=42)
        b = _make(osm=prior, prior_noise=0.05, seed=42)
        np.testing.assert_array_equal(a.map, b.map)

    def test_update_is_deterministic(self) -> None:
        a = _make()
        b = _make()
        rng = np.random.default_rng(7)
        for _ in range(5):
            mask = rng.random((100, 100)) < 0.1
            fire = rng.random((100, 100)) < 0.05
            a.update_observation(mask, observed_fire_world=fire)
            b.update_observation(mask, observed_fire_world=fire)
            a.decay(dt=1.0)
            b.decay(dt=1.0)
        np.testing.assert_array_equal(a.map, b.map)


# ===========================================================================
# Channel constants
# ===========================================================================


class TestChannelConstants:
    def test_distinct_indices(self) -> None:
        ch = {
            CHANNEL_OCCUPANCY,
            CHANNEL_FIRE,
            CHANNEL_TRAFFIC,
            CHANNEL_CERTAINTY,
        }
        assert len(ch) == 4
        assert max(ch) == 3
        assert min(ch) == 0

    def test_n_channels_is_four(self) -> None:
        assert N_CHANNELS == 4
