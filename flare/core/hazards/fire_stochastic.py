"""flare/core/hazards/fire_stochastic.py — probabilistic Alexandridis CA.

Extends the Paper 1 deterministic CA by treating the per-landuse spread
probability as a Beta-distributed random variable resampled each step.
This injects step-to-step variance representing fuel-moisture and weather
fluctuations not captured by the deterministic model.

Mathematical model
------------------
For each landuse code ``lu`` with calibrated mean ``μ_lu`` (from Paper 1's
``_LANDUSE_PROB``), the per-step spread probability is::

    p_lu(t) ~ Beta(α_lu, β_lu)        (1)

where ``α_lu = c · μ_lu`` and ``β_lu = c · (1 − μ_lu)``, with the
**concentration** parameter ``c = α + β > 0``.

Mean and variance::

    E[p_lu(t)] = μ_lu
    Var[p_lu(t)] = μ_lu (1 − μ_lu) / (c + 1)

Limit behaviour::

    c → ∞:  Var → 0,  p_lu(t) → μ_lu (Dirac).  Recovers the Paper 1
            ``FireSpreadModel`` *in expectation* — Invariant I1 high-
            concentration limit. Not bit-identical because the Beta
            sampling consumes additional RNG draws.

    c → 0:  Var → μ_lu(1 − μ_lu).  p_lu(t) is bimodal at {0, 1}.

Note on the CLAUDE.md ``(α, β)`` shorthand
------------------------------------------
The ``StochasticFireCA`` pseudo-code in CLAUDE.md uses a single ``(α, β)``
pair. That maps onto this implementation per landuse via the equations
above; the single tunable here is ``concentration``.

Determinism (DC-1, Invariant I2): all randomness — including Beta
sampling — flows through the caller-supplied ``np.random.Generator``.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import numpy as np

from flare.core.hazards.fire_deterministic import (
    _LANDUSE_PROB,
    FireSpreadModel,
)


class StochasticFireSpreadModel(FireSpreadModel):
    """Stochastic Alexandridis fire CA with Beta-distributed spread rates.

    Subclasses :class:`FireSpreadModel`; identical state machine, ignition,
    burnout, and smoke logic. Only the per-step spread probability map is
    resampled — see module docstring for the model.

    Parameters
    ----------
    *args, **kwargs:
        Forwarded to :class:`FireSpreadModel`.
    concentration:
        Beta concentration ``c = α + β > 0``. Larger → tighter around the
        Paper 1 mean (deterministic limit); smaller → higher per-step
        variance. Default ``100.0``.
    """

    def __init__(
        self,
        *args,
        concentration: float = 100.0,
        **kwargs,
    ) -> None:
        if concentration <= 0:
            raise ValueError(
                f"concentration must be > 0, got {concentration}"
            )
        super().__init__(*args, **kwargs)
        self._concentration = float(concentration)

        # Per-landuse Beta shape parameters. For degenerate means (μ=0 or
        # μ=1) we mark with sentinel zero values and short-circuit at
        # sample time — np.random.Generator.beta requires α,β > 0.
        self._alpha_per_lu: dict[int, float] = {}
        self._beta_per_lu: dict[int, float] = {}
        for lu_val, mu in _LANDUSE_PROB.items():
            if 0.0 < mu < 1.0:
                self._alpha_per_lu[lu_val] = concentration * mu
                self._beta_per_lu[lu_val] = concentration * (1.0 - mu)
            else:
                # Degenerate: μ ∈ {0, 1}. Use the deterministic value
                # directly (no Beta sampling needed).
                self._alpha_per_lu[lu_val] = 0.0
                self._beta_per_lu[lu_val] = 0.0

    # -- Public ----------------------------------------------------------

    @property
    def concentration(self) -> float:
        """Beta concentration ``α + β``. Read-only."""
        return self._concentration

    def step(self, dt: float = 1.0) -> None:
        """Advance fire by one step.

        Resamples ``self._prob_map`` from per-landuse Beta(α_lu, β_lu) and
        delegates the rest of the update (spread, burnout, events, smoke)
        to :meth:`FireSpreadModel.step`.
        """
        # Snapshot the deterministic mean map so we can restore it after the
        # super's step completes. (Defensive: if any caller keeps a reference
        # to ``_prob_map`` between steps they'd see the sampled values.)
        original_prob_map = self._prob_map
        self._prob_map = self._sample_prob_map()
        try:
            super().step(dt=dt)
        finally:
            self._prob_map = original_prob_map

    # -- Internal --------------------------------------------------------

    def _sample_prob_map(self) -> np.ndarray:
        """Return a freshly Beta-sampled spread probability map.

        Same shape as ``_landuse``. Per-landuse Beta sampling, then the
        50 % road firebreak applied multiplicatively (matching Paper 1).
        """
        prob_map = np.zeros((self._H, self._W), dtype=np.float32)
        for lu_val, mu in _LANDUSE_PROB.items():
            mask = self._landuse == lu_val
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            if mu <= 0.0:
                # μ = 0 → never burns (water).
                continue
            if mu >= 1.0:
                # μ = 1 → always ignites (no entry currently has this).
                prob_map[mask] = 1.0
                continue
            alpha = self._alpha_per_lu[lu_val]
            beta = self._beta_per_lu[lu_val]
            samples = self._rng.beta(alpha, beta, size=n_cells).astype(
                np.float32
            )
            prob_map[mask] = samples

        # Roads firebreak: 50 % multiplicative reduction (Paper 1).
        prob_map[self._roads] *= 0.5
        return prob_map
