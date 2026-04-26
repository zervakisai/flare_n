"""flare/scenarios/synthetic.py — minimal scenario for env testing.

Lightweight, OSM-free scenario generator. Used by the FLARE-PO env's
unit tests so they can construct a known-good world without round-tripping
through ``osmnx``. Real Paper 2 scenarios (Penteli, Piraeus, Downtown,
Mati, multi-triage) live in their own modules and share the same
``Scenario`` dataclass.

The grid uses Paper 1's landuse codes::

    0 — empty            (rare; very low spread)
    1 — forest           (dominant; high spread)
    2 — urban (building) (occupied for path-planning; medium spread)
    3 — industrial       (medium-low spread)
    4 — water            (never burns)

Determinism (Invariant I2): all randomness via the caller-supplied
``seed``. Same seed ⇒ identical grid, start, and goal.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """A single scenario — grid + start/goal + metadata.

    Coordinates use the Paper 1 convention: ``(x, y) = (col, row)`` with
    array access ``[y, x]``. ``cell_size_m`` is metres per cell.
    """

    name: str
    grid_shape: tuple[int, int]   # (H, W)
    cell_size_m: float
    occupancy: np.ndarray          # int8[H, W]; 1 ⇒ blocking obstacle
    landuse: np.ndarray            # int8[H, W]; codes per module docstring
    roads_mask: np.ndarray         # bool[H, W]
    start: tuple[int, int]         # (x, y) cells
    goal: tuple[int, int]          # (x, y) cells
    seed: int


def make_synthetic(
    name: str = "synthetic",
    grid_shape: tuple[int, int] = (60, 60),
    cell_size_m: float = 3.0,
    building_density: float = 0.10,
    seed: int = 0,
) -> Scenario:
    """Generate a synthetic scenario.

    Buildings are placed uniformly at random with the given density,
    then a small clearing is carved around the start (5, H/2) and goal
    (W-6, H/2) so the agent can spawn freely. Landuse defaults to
    forest (code 1) outside buildings (which take code 2).

    Parameters
    ----------
    name:
        Human-readable label, copied into the returned dataclass.
    grid_shape:
        ``(H, W)`` in cells. Default ``(60, 60)``.
    cell_size_m:
        World cell size in metres. Default ``3.0`` (matches Paper 1).
    building_density:
        Probability a cell becomes a building, before clearing. Must
        be in ``[0, 1)``. Default ``0.10``.
    seed:
        Reproducibility seed.
    """
    H, W = grid_shape
    if H < 12 or W < 12:
        raise ValueError(
            f"grid_shape must be at least 12×12 to fit start/goal "
            f"clearings, got {grid_shape}"
        )
    if not 0.0 <= building_density < 1.0:
        raise ValueError(
            f"building_density must be in [0, 1), got {building_density}"
        )
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be > 0, got {cell_size_m}")

    rng = np.random.default_rng(seed)
    occ = (rng.random((H, W)) < building_density).astype(np.int8)

    sx, sy = 5, H // 2
    gx, gy = W - 6, H // 2

    # Clear 5×5 squares around start and goal so the agent can spawn
    # without immediately being trapped or inside a building.
    occ[sy - 2 : sy + 3, sx - 2 : sx + 3] = 0
    occ[gy - 2 : gy + 3, gx - 2 : gx + 3] = 0

    landuse = np.ones((H, W), dtype=np.int8)  # forest everywhere
    landuse[occ == 1] = 2  # urban codes overlap buildings

    # No roads in synthetic; callers that need a road firebreak can
    # pass their own ``roads_mask`` to FireSpreadModel directly.
    roads = np.zeros((H, W), dtype=bool)

    return Scenario(
        name=name,
        grid_shape=(H, W),
        cell_size_m=float(cell_size_m),
        occupancy=occ,
        landuse=landuse,
        roads_mask=roads,
        start=(sx, sy),
        goal=(gx, gy),
        seed=int(seed),
    )
