"""flare/core/hazards/nfz.py — dynamic no-fly zones.

Stochastic circular no-fly zones (NFZs) appearing and disappearing over
time. Models temporary airspace closures created during emergency
response (helicopter corridors, water-bomber paths, manned-aircraft
operations) that the UAV must avoid but cannot foresee.

Process model
-------------
- **Arrivals:** ``N_t ~ Poisson(λ)`` new zones per step, capped at
  ``max_zones`` simultaneously active (excess arrivals are dropped).
- **Lifetime:** each new zone has remaining steps drawn from
  ``Geometric(1 / mean_duration)``, giving ``E[lifetime] = mean_duration``.
- **Geometry:** each zone is a disk centered at a uniform-random cell,
  with radius uniform in ``radius_cells_range`` (cells).

By default ``arrival_rate = max_zones / mean_duration`` so the steady-state
mean number of active zones (without the cap) equals ``max_zones`` —
adjust if you need a softer or harder occupancy.

Determinism (Invariant I2): all randomness flows through the
caller-supplied ``np.random.Generator``. The agent observes the NFZ mask
only inside its sensor footprint (Phase 2/3 wiring), but the truth state
is fully reproducible from ``(seed, parameters)``.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NFZZone:
    """Snapshot of a single active NFZ.

    Coordinates are in array space (``cy = row``, ``cx = col``). ``radius``
    is in cells. ``remaining`` is the number of steps the zone has left
    *as of the last :meth:`DynamicNFZ.step`*.
    """

    cy: int
    cx: int
    radius: int
    remaining: int


class DynamicNFZ:
    """Manages a population of dynamic circular no-fly zones.

    Parameters
    ----------
    grid_shape:
        ``(H, W)`` grid dimensions in cells.
    rng:
        Caller-supplied ``np.random.Generator``.
    max_zones:
        Hard cap on simultaneously active zones. Default ``3``.
    mean_duration:
        Mean zone lifetime in steps (geometric distribution parameter is
        ``1 / mean_duration``). Default ``50``.
    arrival_rate:
        Poisson rate ``λ`` of new arrivals per step. ``None`` (default)
        sets ``λ = max_zones / mean_duration``.
    radius_cells_range:
        Inclusive ``(r_min, r_max)`` range for zone radius in cells.
        Default ``(4, 12)``.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        rng: np.random.Generator,
        max_zones: int = 3,
        mean_duration: int = 50,
        arrival_rate: float | None = None,
        radius_cells_range: tuple[int, int] = (4, 12),
    ) -> None:
        if max_zones < 0:
            raise ValueError(f"max_zones must be ≥ 0, got {max_zones}")
        if mean_duration < 1:
            raise ValueError(
                f"mean_duration must be ≥ 1, got {mean_duration}"
            )
        if arrival_rate is not None and arrival_rate < 0:
            raise ValueError(
                f"arrival_rate must be ≥ 0, got {arrival_rate}"
            )
        r_min, r_max = radius_cells_range
        if not (1 <= r_min <= r_max):
            raise ValueError(
                f"radius_cells_range must satisfy 1 ≤ r_min ≤ r_max, "
                f"got {radius_cells_range}"
            )

        self._H, self._W = grid_shape
        self._rng = rng
        self._max_zones = int(max_zones)
        self._mean_duration = int(mean_duration)
        self._arrival_rate = float(
            arrival_rate
            if arrival_rate is not None
            else max_zones / mean_duration
        )
        self._r_min = int(r_min)
        self._r_max = int(r_max)
        self._zones: list[NFZZone] = []

        # Pre-build coordinate grids for fast circular masking.
        self._yy, self._xx = np.mgrid[: self._H, : self._W]

    # -- Public ----------------------------------------------------------

    @property
    def n_active(self) -> int:
        """Number of currently active zones."""
        return len(self._zones)

    @property
    def zones(self) -> list[NFZZone]:
        """Snapshot list of active zones."""
        return list(self._zones)

    @property
    def max_zones(self) -> int:
        return self._max_zones

    @property
    def mean_duration(self) -> int:
        return self._mean_duration

    @property
    def arrival_rate(self) -> float:
        return self._arrival_rate

    @property
    def nfz_mask(self) -> np.ndarray:
        """``bool[H, W]`` true inside any active NFZ."""
        mask = np.zeros((self._H, self._W), dtype=bool)
        for z in self._zones:
            d2 = (self._yy - z.cy) ** 2 + (self._xx - z.cx) ** 2
            mask |= d2 <= z.radius * z.radius
        return mask

    def step(self) -> None:
        """Advance one timestep: age zones, retire expired, spawn arrivals.

        Order: decrement lifetimes first (so a zone with ``remaining == 1``
        survives the current step's mask but dies before the next), then
        sample Poisson arrivals capped at ``max_zones``.
        """
        # Age and cull
        self._zones = [
            NFZZone(z.cy, z.cx, z.radius, z.remaining - 1)
            for z in self._zones
            if z.remaining - 1 > 0
        ]
        # Arrivals
        n_arrivals = int(self._rng.poisson(self._arrival_rate))
        for _ in range(n_arrivals):
            if len(self._zones) >= self._max_zones:
                break
            self._zones.append(self._spawn_zone())

    def reset(self) -> None:
        """Clear all active zones."""
        self._zones = []

    # -- Internal --------------------------------------------------------

    def _spawn_zone(self) -> NFZZone:
        cy = int(self._rng.integers(0, self._H))
        cx = int(self._rng.integers(0, self._W))
        radius = int(self._rng.integers(self._r_min, self._r_max + 1))
        # Generator.geometric(p) returns values in {1, 2, ...} with mean 1/p.
        remaining = int(self._rng.geometric(p=1.0 / self._mean_duration))
        return NFZZone(cy, cx, radius, remaining)
