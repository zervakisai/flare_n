"""tests/test_certainty_decay.py — exponential certainty decay tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from flare.belief.certainty_decay import (
    decay_certainty,
    decay_factor,
    half_life_to_tau,
    tau_to_half_life,
)


class TestDecayFactor:
    def test_zero_dt_returns_one(self) -> None:
        assert decay_factor(0.0, 50.0) == pytest.approx(1.0)

    def test_one_tau_returns_inv_e(self) -> None:
        assert decay_factor(50.0, 50.0) == pytest.approx(math.exp(-1.0))

    def test_monotonic_decreasing(self) -> None:
        f = [decay_factor(t, 10.0) for t in (0.0, 5.0, 10.0, 20.0, 100.0)]
        assert all(b <= a for a, b in zip(f, f[1:]))

    def test_negative_dt_rejected(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            decay_factor(-1.0, 50.0)

    def test_zero_tau_rejected(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            decay_factor(1.0, 0.0)

    def test_negative_tau_rejected(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            decay_factor(1.0, -1.0)


class TestDecayCertainty:
    def test_array_decays(self) -> None:
        c = np.array([1.0, 0.5, 0.1], dtype=np.float32)
        out = decay_certainty(c, dt=10.0, tau=10.0)
        np.testing.assert_allclose(out, c * math.exp(-1.0), rtol=1e-6)

    def test_dtype_preserved(self) -> None:
        c = np.ones((3, 3), dtype=np.float32)
        out = decay_certainty(c, dt=1.0, tau=10.0)
        assert out.dtype == np.float32


class TestHalfLifeConversion:
    def test_half_life_round_trip(self) -> None:
        for hl in (1.0, 5.0, 100.0):
            assert tau_to_half_life(half_life_to_tau(hl)) == pytest.approx(hl)

    def test_decay_at_half_life_equals_one_half(self) -> None:
        hl = 25.0
        tau = half_life_to_tau(hl)
        assert decay_factor(hl, tau) == pytest.approx(0.5, rel=1e-6)

    def test_negative_half_life_rejected(self) -> None:
        with pytest.raises(ValueError, match="half_life"):
            half_life_to_tau(-1.0)
