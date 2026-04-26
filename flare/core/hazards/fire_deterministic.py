"""flare/core/hazards/fire_deterministic.py — Paper 1 Alexandridis fire CA.

Verbatim port of the deterministic Alexandridis CA from FLARE (Paper 1).
This module is the *reference implementation* for Invariant I1 (MDP continuity):
the Paper 2 stochastic CA in :mod:`flare.core.hazards.fire_stochastic` must
reduce to this implementation in the limit β → ∞ (the Beta(α,β) base-spread
distribution collapsing to its mean recovers Paper 1 behaviour).

Cell states (FD-1)::

    UNBURNED(0) → BURNING(1) → BURNED_OUT(2)

Determinism rule (DC-1, Invariant I2): every random draw flows through the
caller-supplied ``np.random.Generator``. No module-level RNG, no global state.

Backward-compat (FD-5b): ``wind_speed == 0`` → isotropic ``binary_dilation``
path, bit-identical to FLARE v2. ``wind_speed > 0`` → per-direction loop with
the Alexandridis et al. (2008) wind factor.

References
----------
- Source port: `/Users/konstantinos/Dev/planning/uavbench/src/flare/dynamics/fire_ca.py`
- Wind modulation: Alexandridis, A., Vakalis, D., Siettos, C.I., Bafas, G.V.
  (2008). "A cellular automata model for forest fire spread prediction: The
  case of the wildfire that swept through Spetses Island in 1990."
  *Applied Mathematics and Computation*, 204(1), 191-201. (Eq. 4.)
- Per-landuse base probabilities: Paper 1 calibration (CC contracts).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from flare.core.wind import NEIGHBORS_DY_DX, alexandridis_wind_factors

# Cell states (FD-1)
UNBURNED: int = 0
BURNING: int = 1
BURNED_OUT: int = 2

# Landuse spread probabilities (Paper 1 CC-calibrated; tuned for partial
# blockages with navigable detours on dense OSM grids).
_LANDUSE_PROB: dict[int, float] = {
    0: 0.02,  # empty
    1: 0.15,  # forest
    2: 0.06,  # urban
    3: 0.03,  # industrial
    4: 0.00,  # water — never burns
}

# 8-connected Moore neighbourhood, re-exported from :mod:`flare.core.wind`
# for backwards compatibility with code that imports ``_NEIGHBORS`` from
# this module.
_NEIGHBORS: tuple[tuple[int, int], ...] = NEIGHBORS_DY_DX

# Pre-computed Moore dilation structuring element.
_MOORE_STRUCT: np.ndarray = np.ones((3, 3), dtype=bool)


class FireSpreadModel:
    """Deterministic Alexandridis fire CA on a 2D grid.

    Parameters
    ----------
    map_shape:
        ``(H, W)`` grid dimensions in cells.
    rng:
        Caller-supplied ``np.random.Generator`` — sole source of randomness.
    n_ignition:
        Number of initial random ignitions (default 3).
    landuse_map:
        Optional ``int8[H, W]`` landuse code per cell. ``None`` → uniform
        urban (code 2). Codes follow ``_LANDUSE_PROB``.
    roads_mask:
        Optional ``bool[H, W]``. Roads act as a 50 % firebreak (multiplicative
        on per-cell spread probability).
    corridor_cells:
        Optional list of ``(x, y)`` cells defining a planning corridor. Used
        only by ``_ignite_approach_fires`` / ``_ignite_near_corridor``.
    guarantee_targets:
        Optional list of ``(x, y)`` cells that must be burning by
        ``guarantee_step`` (force-ignition safety net).
    guarantee_step:
        Step at which to force-ignite any unmet ``guarantee_targets``.
    wind_speed:
        Wind speed in m/s. ``0.0`` (default) disables wind for FD-5b backward
        compatibility (bit-identical to FLARE v2).
    wind_direction:
        Wind direction in radians (TO-direction in world frame; see
        :mod:`flare.core.wind` module docstring). Ignored when
        ``wind_speed == 0``.

    Notes
    -----
    Coordinates: the public ``force_cell_state`` and ``corridor_cells`` API
    use ``(x, y)`` (col, row); internal arrays are accessed as ``[y, x]``.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        rng: np.random.Generator,
        n_ignition: int = 3,
        landuse_map: np.ndarray | None = None,
        roads_mask: np.ndarray | None = None,
        corridor_cells: list[tuple[int, int]] | None = None,
        guarantee_targets: list[tuple[int, int]] | None = None,
        guarantee_step: int | None = None,
        wind_speed: float = 0.0,
        wind_direction: float = 0.0,
    ) -> None:
        self._rng = rng
        self._H, self._W = map_shape

        # State arrays
        self._state = np.full((self._H, self._W), UNBURNED, dtype=np.int8)
        self._burn_timer = np.zeros((self._H, self._W), dtype=np.float32)
        self._burnout_time = rng.uniform(
            100.0, 200.0, size=(self._H, self._W)
        ).astype(np.float32)
        self._smoke = np.zeros((self._H, self._W), dtype=np.float32)

        self._step_count = 0
        self._fire_events: list[dict] = []

        # Fire corridor guarantee: corridor cells that must be burning by
        # guarantee_step. Approach ignitions try natural spread; the safety
        # net force-ignites any missed targets.
        self._guarantee_targets = guarantee_targets or []
        self._guarantee_step = guarantee_step

        # Guarantee targets use extended burnout — effectively infinite so
        # they never transition to BURNED_OUT within episode duration. Keeps
        # the CA state machine consistent (no re-ignition hack).
        for gx, gy in self._guarantee_targets:
            if 0 <= gy < self._H and 0 <= gx < self._W:
                self._burnout_time[gy, gx] = 999_999.0

        # Landuse (default: urban=2 for realistic spread in urban grids).
        if landuse_map is not None:
            self._landuse = landuse_map.astype(np.int8)
        else:
            self._landuse = np.full((self._H, self._W), 2, dtype=np.int8)

        # Pre-compute per-cell base spread probability (avoids rebuilding it
        # every step).
        self._prob_map = np.zeros((self._H, self._W), dtype=np.float32)
        for lu_val, prob in _LANDUSE_PROB.items():
            self._prob_map[self._landuse == lu_val] = prob

        # Roads act as a 50 % firebreak.
        self._roads = (
            roads_mask.astype(bool) if roads_mask is not None
            else np.zeros((self._H, self._W), dtype=bool)
        )
        self._prob_map[self._roads] *= 0.5

        # Wind modulation (FD-5b, WD-1: deterministic given identical params).
        # The Alexandridis kernel lives in :mod:`flare.core.wind`; this class
        # only stores the per-direction multipliers needed by the inner loop.
        self._wind_speed = wind_speed
        self._wind_factors = alexandridis_wind_factors(
            wind_speed, wind_direction
        )

        # Initial random ignitions for environmental hazards.
        if n_ignition > 0:
            self._ignite(n_ignition, corridor_cells)

        # Approach ignitions seed cells near each guarantee target so that
        # natural spread reaches the corridor by ``guarantee_step``.
        if self._guarantee_targets and corridor_cells:
            self._ignite_approach_fires(corridor_cells)

    # -- Public properties -----------------------------------------------

    @property
    def fire_mask(self) -> np.ndarray:
        """``bool[H, W]``: ``True`` where cell is currently burning."""
        return (self._state == BURNING).copy()

    @property
    def burned_mask(self) -> np.ndarray:
        """``bool[H, W]``: ``True`` where cell has burned out."""
        return (self._state == BURNED_OUT).copy()

    @property
    def smoke_mask(self) -> np.ndarray:
        """``float32[H, W]`` in ``[0, 1]``: smoke concentration."""
        return self._smoke.copy()

    @property
    def total_affected(self) -> int:
        """Count of cells that are burning or burned out."""
        return int((self._state > UNBURNED).sum())

    def pop_events(self) -> list[dict]:
        """Return and clear pending fire events.

        Events are emitted when fire reaches building cells (landuse 2/3).
        Used downstream by mission engines for dynamic casualty injection.
        """
        events = self._fire_events
        self._fire_events = []
        return events

    # -- Step ------------------------------------------------------------

    def step(self, dt: float = 1.0) -> None:
        """Advance fire by one timestep (FD-1, FD-4).

        1. Spread: burning cells attempt to ignite neighbours. ``wind_speed=0``
           uses the isotropic ``binary_dilation`` path (single random roll per
           candidate); otherwise per-direction loop with the Alexandridis wind
           factor.
        2. Building-fire events: cells that just became BURNING in landuse
           {2, 3} emit a ``building_fire`` event.
        3. Guarantee safety net: at ``guarantee_step``, force-ignite any
           ``guarantee_targets`` not yet burning.
        4. Burnout: cells whose ``burn_timer`` exceeded ``burnout_time``
           transition to BURNED_OUT. Guarantee targets have an extended
           burnout time set in ``__init__`` so they never burn out.
        5. Smoke: 3×3 box-blur source field every other step.
        """
        burning = self._state == BURNING

        # Increment burn timers for currently burning cells.
        self._burn_timer[burning] += dt

        # --- Spread (FD-2 / FD-5b) -------------------------------------
        if burning.any():
            if self._wind_speed == 0.0:
                # Isotropic: single binary_dilation, bit-identical to FLARE v2.
                spread_candidates = (
                    binary_dilation(burning, structure=_MOORE_STRUCT)
                    & (self._state == UNBURNED)
                )
                if spread_candidates.any():
                    candidate_ys, candidate_xs = np.where(spread_candidates)
                    rolls = self._rng.random(len(candidate_ys))
                    probs = self._prob_map[candidate_ys, candidate_xs]
                    ignite = rolls < probs
                    ignite_ys = candidate_ys[ignite]
                    ignite_xs = candidate_xs[ignite]
                    if len(ignite_ys) > 0:
                        self._state[ignite_ys, ignite_xs] = BURNING
            else:
                # Wind-modulated: per-direction spread (Alexandridis 2008).
                unburned = self._state == UNBURNED
                for i, (dy, dx) in enumerate(_NEIGHBORS):
                    # Shift burning mask: find cells with a burning neighbour
                    # in direction (dy, dx). np.roll wraps; fix boundaries.
                    shifted = np.roll(np.roll(burning, -dy, axis=0), -dx, axis=1)
                    if dy < 0:
                        shifted[dy:, :] = False
                    elif dy > 0:
                        shifted[:dy, :] = False
                    if dx < 0:
                        shifted[:, dx:] = False
                    elif dx > 0:
                        shifted[:, :dx] = False
                    candidates = shifted & unburned
                    if not candidates.any():
                        continue
                    ys, xs = np.where(candidates)
                    rolls = self._rng.random(len(ys))
                    probs = self._prob_map[ys, xs] * self._wind_factors[i]
                    ignite_mask = rolls < probs
                    if ignite_mask.any():
                        self._state[ys[ignite_mask], xs[ignite_mask]] = BURNING
                        unburned = self._state == UNBURNED  # avoid double-count

        # --- Building-fire events --------------------------------------
        newly_burning = (self._state == BURNING) & ~burning
        if newly_burning.any():
            building_fire = newly_burning & np.isin(self._landuse, [2, 3])
            if building_fire.any():
                bfy, bfx = np.where(building_fire)
                for y, x in zip(bfy, bfx):
                    self._fire_events.append({
                        "type": "building_fire",
                        "x": int(x),
                        "y": int(y),
                        "step": self._step_count,
                    })

        # --- Guarantee safety net --------------------------------------
        if (self._guarantee_targets
                and self._guarantee_step is not None
                and self._step_count >= self._guarantee_step):
            for gx, gy in self._guarantee_targets:
                if (0 <= gy < self._H and 0 <= gx < self._W
                        and self._state[gy, gx] == UNBURNED):
                    self._state[gy, gx] = BURNING

        # --- Burnout ----------------------------------------------------
        burnout_mask = burning & (self._burn_timer >= self._burnout_time)
        self._state[burnout_mask] = BURNED_OUT

        # --- Smoke (every 2 steps — smoke evolves slowly) --------------
        if self._step_count % 2 == 0:
            self._update_smoke()
        self._step_count += 1

    # -- Test hook -------------------------------------------------------

    def force_cell_state(self, x: int, y: int, state: int) -> None:
        """Set cell state directly. FOR TESTS ONLY.

        Coordinates: ``x = col``, ``y = row``. Internal access uses
        ``_state[y, x]``.
        """
        self._state[y, x] = state

    # -- Internal --------------------------------------------------------

    def _ignite(
        self,
        n: int,
        corridor_cells: list[tuple[int, int]] | None = None,
    ) -> None:
        """Ignite n cells at random locations.

        Random placement creates distributed environmental hazards without
        overwhelming chokepoints on dense OSM maps. Corridor interdiction is
        handled by the corridor guarantee mechanism (approach ignitions plus
        force-ignite safety net).
        """
        self._ignite_random(n)

    def _ignite_approach_fires(
        self,
        corridor_cells: list[tuple[int, int]],
    ) -> None:
        """Place approach ignitions near each ``guarantee_target``.

        For each guarantee target, ignite one cell at Manhattan distance
        ``[8, 15]``. Natural spread should reach the corridor by
        ``guarantee_step``; the safety net in :meth:`step` catches misses.
        """
        for gx, gy in self._guarantee_targets:
            candidates_y: list[int] = []
            candidates_x: list[int] = []
            r_min, r_max = 8, 15
            for dy in range(-r_max, r_max + 1):
                for dx in range(-r_max, r_max + 1):
                    dist = abs(dy) + abs(dx)
                    if dist < r_min or dist > r_max:
                        continue
                    cy, cx = gy + dy, gx + dx
                    if not (0 <= cy < self._H and 0 <= cx < self._W):
                        continue
                    if self._state[cy, cx] != UNBURNED:
                        continue
                    if self._landuse[cy, cx] == 4:  # water
                        continue
                    candidates_y.append(cy)
                    candidates_x.append(cx)

            if candidates_y:
                idx = self._rng.integers(len(candidates_y))
                self._state[candidates_y[idx], candidates_x[idx]] = BURNING

    def _ignite_near_corridor(
        self,
        n: int,
        corridor_cells: list[tuple[int, int]],
    ) -> None:
        """Ignite some cells near corridor, rest randomly (Paper 1 CC-2).

        Half of ``n`` (at least 1) are placed in the ``[15, 25]`` Manhattan
        ring around evenly-spaced corridor anchors; the rest are random.
        """
        interior = corridor_cells[1:-1]
        if len(interior) == 0:
            self._ignite_random(n)
            return

        n_corridor = max(1, n // 2)

        step = max(1, len(interior) // n_corridor)
        placed = 0

        for i in range(n_corridor):
            anchor_idx = min(i * step, len(interior) - 1)
            ax, ay = interior[anchor_idx]

            candidates_y: list[int] = []
            candidates_x: list[int] = []
            r_min, r_max = 15, 25
            for dy in range(-r_max, r_max + 1):
                for dx in range(-r_max, r_max + 1):
                    dist = abs(dy) + abs(dx)
                    if dist < r_min or dist > r_max:
                        continue
                    cy, cx = ay + dy, ax + dx
                    if not (0 <= cy < self._H and 0 <= cx < self._W):
                        continue
                    if self._state[cy, cx] != UNBURNED:
                        continue
                    if self._landuse[cy, cx] == 4:
                        continue
                    candidates_y.append(cy)
                    candidates_x.append(cx)

            if candidates_y:
                idx = self._rng.integers(len(candidates_y))
                self._state[candidates_y[idx], candidates_x[idx]] = BURNING
                placed += 1

        remaining = n - placed
        if remaining > 0:
            self._ignite_random(remaining)

    def _ignite_random(self, n: int) -> None:
        """Ignite ``n`` cells at random, preferring forest (landuse=1)."""
        forest_ys, forest_xs = np.where(
            (self._landuse == 1) & (self._state == UNBURNED),
        )
        if len(forest_ys) >= n:
            indices = self._rng.choice(len(forest_ys), size=n, replace=False)
            for idx in indices:
                self._state[forest_ys[idx], forest_xs[idx]] = BURNING
        else:
            valid_ys, valid_xs = np.where(
                (self._landuse != 4) & (self._state == UNBURNED),
            )
            if len(valid_ys) > 0:
                k = min(n, len(valid_ys))
                indices = self._rng.choice(len(valid_ys), size=k, replace=False)
                for idx in indices:
                    self._state[valid_ys[idx], valid_xs[idx]] = BURNING

    def _update_smoke(self) -> None:
        """Update smoke field: source from fire, diffuse (no wind advection)."""
        # Source: burning=1.0, burned_out=0.3.
        source = np.zeros_like(self._smoke)
        source[self._state == BURNING] = 1.0
        source[self._state == BURNED_OUT] = 0.3

        # Diffuse with two passes of a 3×3 box blur.
        blurred = uniform_filter(source, size=3, mode="constant", cval=0.0)
        blurred = uniform_filter(blurred, size=3, mode="constant", cval=0.0)

        # Persistence (Paper 1 calibration: thinner halos with navigable gaps).
        self._smoke = np.clip(
            0.7 * self._smoke + 0.45 * blurred, 0.0, 1.0
        ).astype(np.float32)
