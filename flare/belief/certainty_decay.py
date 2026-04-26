"""flare/belief/certainty_decay.py — exponential certainty decay.

Pure-function module. The belief map's recency channel ``c(x, t)``
ages between observations as::

    c(x, t) = c(x, t_obs) · exp(−(t − t_obs) / τ)

so a cell observed at time ``t_obs`` has certainty 1, and that drops
back to ``e^(-Δt/τ)`` by time ``t_obs + Δt``. The same factor is
applied to every cell each step in :class:`BeliefMap`.

This formulation is the same one used in PyroTrack-style fire-tracking
literature (e.g. exponential staleness on belief grids).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np


def decay_factor(dt: float, tau: float) -> float:
    """Return the per-step multiplicative decay factor ``exp(−dt / τ)``.

    Parameters
    ----------
    dt:
        Time elapsed (steps or seconds — must be consistent with ``tau``).
    tau:
        Time constant. ``tau > 0`` required.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if dt < 0:
        raise ValueError(f"dt must be ≥ 0, got {dt}")
    return math.exp(-dt / tau)


def decay_certainty(
    certainty: np.ndarray,
    dt: float,
    tau: float,
) -> np.ndarray:
    """Multiply ``certainty`` by ``exp(−dt / τ)`` element-wise.

    Returns a new array; the input is not mutated. Use the in-place
    operator ``certainty *= decay_factor(dt, tau)`` if mutation is wanted.
    """
    return (certainty * decay_factor(dt, tau)).astype(certainty.dtype, copy=False)


def half_life_to_tau(half_life: float) -> float:
    """Convert a half-life to its equivalent time constant.

    ``c(t_½) = c₀ / 2``  ⇒  ``τ = t_½ / ln 2``.
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life}")
    return half_life / math.log(2.0)


def tau_to_half_life(tau: float) -> float:
    """Inverse of :func:`half_life_to_tau`."""
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return tau * math.log(2.0)
