"""tests/test_fire_stochastic.py — unit tests for fire_stochastic.

Covers:
- Determinism for matching seeds (Invariant I2).
- Variance scaling: low concentration → high variance, high → low variance.
- High-concentration convergence to Paper 1 means (Invariant I1 limit, in
  expectation; not bit-identical since Beta draws consume extra RNG).
- Per-landuse Beta parameter validity.
- Input validation.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import numpy as np
import pytest

from flare.core.hazards.fire_deterministic import (
    _LANDUSE_PROB,
    FireSpreadModel,
)
from flare.core.hazards.fire_stochastic import StochasticFireSpreadModel


def _make_stochastic(
    map_size: int = 30,
    seed: int = 0,
    concentration: float = 100.0,
    n_ignition: int = 3,
    **kwargs,
) -> StochasticFireSpreadModel:
    rng = np.random.default_rng(seed)
    return StochasticFireSpreadModel(
        map_shape=(map_size, map_size),
        rng=rng,
        n_ignition=n_ignition,
        concentration=concentration,
        **kwargs,
    )


def _make_deterministic(
    map_size: int = 30,
    seed: int = 0,
    n_ignition: int = 3,
    **kwargs,
) -> FireSpreadModel:
    rng = np.random.default_rng(seed)
    return FireSpreadModel(
        map_shape=(map_size, map_size),
        rng=rng,
        n_ignition=n_ignition,
        **kwargs,
    )


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_zero_concentration_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="concentration"):
            StochasticFireSpreadModel(
                map_shape=(10, 10), rng=rng, n_ignition=1, concentration=0.0,
            )

    def test_negative_concentration_rejected(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="concentration"):
            StochasticFireSpreadModel(
                map_shape=(10, 10), rng=rng, n_ignition=1, concentration=-1.0,
            )


# ===========================================================================
# Beta parameters
# ===========================================================================


class TestBetaParameters:
    """Per-landuse α, β satisfy α/(α+β) = μ_lu and α+β = concentration."""

    def test_alpha_beta_match_mean(self) -> None:
        c = 50.0
        m = _make_stochastic(concentration=c)
        for lu_val, mu in _LANDUSE_PROB.items():
            if not (0.0 < mu < 1.0):
                continue
            a = m._alpha_per_lu[lu_val]
            b = m._beta_per_lu[lu_val]
            assert a + b == pytest.approx(c)
            assert a / (a + b) == pytest.approx(mu)


# ===========================================================================
# Determinism — Invariant I2
# ===========================================================================


@pytest.mark.invariant
class TestDeterminism:
    def test_same_seed_same_state(self) -> None:
        a = _make_stochastic(seed=7, concentration=10.0)
        b = _make_stochastic(seed=7, concentration=10.0)
        for _ in range(20):
            a.step()
            b.step()
        np.testing.assert_array_equal(a.fire_mask, b.fire_mask)
        np.testing.assert_array_equal(a.smoke_mask, b.smoke_mask)

    def test_different_seed_different_state(self) -> None:
        a = _make_stochastic(seed=1, concentration=10.0)
        b = _make_stochastic(seed=2, concentration=10.0)
        for _ in range(20):
            a.step()
            b.step()
        assert not np.array_equal(a.fire_mask, b.fire_mask)


# ===========================================================================
# Variance scaling
# ===========================================================================


class TestVarianceScaling:
    """Low concentration → high spread-pattern variance across seeds."""

    @staticmethod
    def _spread_signature(
        map_size: int, concentration: float, seeds: list[int], n_steps: int
    ) -> np.ndarray:
        """Run for n_steps, return total_affected per seed."""
        out = np.zeros(len(seeds), dtype=np.int64)
        for i, s in enumerate(seeds):
            m = _make_stochastic(
                map_size=map_size,
                seed=s,
                concentration=concentration,
                n_ignition=2,
            )
            for _ in range(n_steps):
                m.step()
            out[i] = m.total_affected
        return out

    def test_low_concentration_more_variance(self) -> None:
        """Variance of total_affected across seeds is larger at low c."""
        seeds = list(range(20))
        n_steps = 30
        low_c = self._spread_signature(40, 0.5, seeds, n_steps)
        high_c = self._spread_signature(40, 1000.0, seeds, n_steps)
        assert low_c.std() > high_c.std(), (
            f"Expected higher variance at low concentration: "
            f"low_c std={low_c.std():.2f}, high_c std={high_c.std():.2f}"
        )


# ===========================================================================
# High-concentration limit — Invariant I1 (statistical)
# ===========================================================================


@pytest.mark.invariant
class TestHighConcentrationConvergence:
    """At very high concentration, the stochastic CA matches the deterministic
    CA in expectation (means converge). Bit-identity is not expected: the
    Beta sampler consumes RNG draws that are absent from the deterministic
    path, so per-seed trajectories will differ.
    """

    def test_mean_total_affected_matches(self) -> None:
        """Average total_affected over many seeds: stochastic ≈ deterministic
        when concentration is large."""
        seeds = list(range(30))
        n_steps = 25
        size = 40

        det_means: list[int] = []
        sto_means: list[int] = []
        for s in seeds:
            d = _make_deterministic(map_size=size, seed=s, n_ignition=2)
            for _ in range(n_steps):
                d.step()
            det_means.append(d.total_affected)

            t = _make_stochastic(
                map_size=size,
                seed=s,
                concentration=10_000.0,
                n_ignition=2,
            )
            for _ in range(n_steps):
                t.step()
            sto_means.append(t.total_affected)

        det_avg = float(np.mean(det_means))
        sto_avg = float(np.mean(sto_means))
        # Allow ±10% relative deviation. Empirically tight; mostly tests
        # that the per-landuse Beta indeed has the right mean.
        assert abs(sto_avg - det_avg) / max(det_avg, 1.0) < 0.10, (
            f"At high concentration the stochastic and deterministic means "
            f"should agree: det={det_avg:.1f}, sto={sto_avg:.1f}"
        )

    def test_beta_samples_concentrate(self) -> None:
        """Direct check on the sampler: at high c, Beta samples for each
        landuse have variance ≈ 0 and mean ≈ μ_lu."""
        c = 100_000.0
        m = _make_stochastic(concentration=c)
        # Sample a fresh prob map and check per-landuse statistics.
        n_samples = 5
        samples = [m._sample_prob_map() for _ in range(n_samples)]
        for lu_val, mu in _LANDUSE_PROB.items():
            if mu <= 0.0:
                continue
            mask = m._landuse == lu_val
            if not mask.any():
                continue
            # Roads firebreak halves probability for road cells; exclude.
            mask = mask & ~m._roads
            if not mask.any():
                continue
            # Concatenate all samples within this landuse mask
            vals = np.concatenate([s[mask] for s in samples])
            assert vals.mean() == pytest.approx(mu, rel=0.05), (
                f"landuse={lu_val}: sample mean {vals.mean():.4f} ≠ μ {mu}"
            )
            # At c=1e5 the std should be tiny (~ sqrt(μ(1−μ)/(c+1)))
            expected_std = np.sqrt(mu * (1 - mu) / (c + 1))
            assert vals.std() < 5.0 * expected_std + 1e-6


# ===========================================================================
# Smoke + burnout still functional
# ===========================================================================


class TestInheritedBehaviour:
    def test_fire_spreads(self) -> None:
        m = _make_stochastic(map_size=30, seed=0, concentration=50.0, n_ignition=3)
        initial = int(m.fire_mask.sum())
        for _ in range(20):
            m.step()
        assert m.total_affected >= initial

    def test_smoke_generated(self) -> None:
        m = _make_stochastic(seed=0, concentration=50.0, n_ignition=3)
        for _ in range(15):
            m.step()
        assert m.smoke_mask.max() > 0.0

    def test_burnout_eventually(self) -> None:
        m = _make_stochastic(seed=0, concentration=50.0, n_ignition=5)
        for _ in range(250):
            m.step()
        assert m.burned_mask.any()

    def test_concentration_property(self) -> None:
        m = _make_stochastic(concentration=42.0)
        assert m.concentration == 42.0
