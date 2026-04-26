"""flare/sensors/ray_cast.py — Numba ray-cast with smoke attenuation.

Single-ray and N-ray sweeps that walk a 2-D occupancy grid one cell at
a time, stopping on the first occupied cell or when cumulative smoke
extinction has dropped the round-trip transmission below an
SNR threshold. Used by the LiDAR (Phase 2.1) and Camera (Phase 2.3)
sensor models.

Coordinate convention
---------------------
Ray endpoints and direction are given in **array space**:

* ``x0, y0`` are floats — column, row of the source (sub-cell allowed).
* ``angle`` is in radians, CCW from the +x (east, +column) axis. Increment
  per cell-step is ``(Δcol, Δrow) = (cos α, −sin α)`` so that +y in the
  world (north) corresponds to −row.

Hit types
---------
==  =============================================================
0   ``max-range`` — ray reached ``max_range_cells`` without hitting
    anything.
1   ``occupied`` — first cell with ``occupancy > 0`` (building, etc.).
2   ``smoke-cutoff`` — round-trip transmission ``T(s) = exp(−2·∫σ_ext ds)``
    has fallen below ``snr_threshold``; the return signal is too weak.
3   ``out-of-bounds`` — ray walked off the grid.
==  =============================================================

The smoke-cutoff condition uses the Beer–Lambert law for an *active*
sensor (LiDAR pulse goes out and back), hence the factor of 2. Stop when
``2 · cumulative_extinction_metres ≥ −ln(snr_threshold)``.

Performance
-----------
``cast_ray`` and ``cast_rays`` are decorated ``@njit(cache=True)``. First
call compiles (~1 s); subsequent calls reuse the on-disk cache. Target on
M1: ``< 1 ms`` for 450 rays on a 500×500 grid (CLAUDE.md Phase 2.4).

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def cast_ray(
    x0: float,
    y0: float,
    angle: float,
    max_range_cells: int,
    occupancy: np.ndarray,
    extinction: np.ndarray,
    cell_size_m: float,
    snr_threshold: float,
) -> tuple[int, int, float, int]:
    """Cast a single ray from ``(x0, y0)`` along ``angle``.

    Parameters
    ----------
    x0, y0:
        Source position in array space (column, row), float, sub-cell ok.
    angle:
        Heading in radians, CCW from +x.
    max_range_cells:
        Maximum number of cell-steps to walk.
    occupancy:
        ``int8[H, W]`` grid; non-zero ⇒ ray stops with hit_type=1.
    extinction:
        ``float32[H, W]`` field of ``σ_ext`` in 1/m (e.g. from
        :class:`flare.core.hazards.smoke.GaussianSmokePlume.extinction`).
    cell_size_m:
        Side length of one cell in metres (for converting σ_ext × ds).
    snr_threshold:
        Minimum round-trip transmission considered detectable. Smaller
        values let the ray punch through more smoke (e.g. 0.05).

    Returns
    -------
    (hit_x, hit_y, distance_cells, hit_type)
        See module docstring for hit_type semantics.
    """
    H, W = occupancy.shape
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # World x → +col; world y → -row (array row increases southward).
    dx = cos_a
    dy = -sin_a
    x = x0
    y = y0
    cum_ext = 0.0
    cum_threshold = -0.5 * math.log(snr_threshold)
    for step in range(1, max_range_cells + 1):
        x += dx
        y += dy
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            return ix, iy, float(step), 3
        if occupancy[iy, ix] > 0:
            return ix, iy, float(step), 1
        cum_ext += extinction[iy, ix] * cell_size_m
        if cum_ext >= cum_threshold:
            return ix, iy, float(step), 2
    return int(round(x)), int(round(y)), float(max_range_cells), 0


@njit(cache=True)
def cast_rays(
    x0: float,
    y0: float,
    angles: np.ndarray,
    max_range_cells: int,
    occupancy: np.ndarray,
    extinction: np.ndarray,
    cell_size_m: float,
    snr_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cast ``N`` rays from a common source.

    Returns four parallel arrays of length ``N``:

    * ``hits_x : int32[N]`` — terminal column per ray
    * ``hits_y : int32[N]`` — terminal row per ray
    * ``distances : float32[N]`` — cell-distance walked
    * ``hit_types : int8[N]`` — see module docstring

    Calls :func:`cast_ray` per element; both functions are JIT-compiled
    so the per-ray overhead is small.
    """
    n = angles.shape[0]
    hits_x = np.empty(n, dtype=np.int32)
    hits_y = np.empty(n, dtype=np.int32)
    distances = np.empty(n, dtype=np.float32)
    hit_types = np.empty(n, dtype=np.int8)
    for i in range(n):
        hx, hy, d, ht = cast_ray(
            x0,
            y0,
            angles[i],
            max_range_cells,
            occupancy,
            extinction,
            cell_size_m,
            snr_threshold,
        )
        hits_x[i] = hx
        hits_y[i] = hy
        distances[i] = d
        hit_types[i] = ht
    return hits_x, hits_y, distances, hit_types


def make_uniform_angles(n_rays: int, start_angle: float = 0.0) -> np.ndarray:
    """Convenience: ``n_rays`` evenly spaced angles in ``[start, start+2π)``.

    Plain Python — fine for one-off setup outside hot loops.
    """
    return (start_angle + np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)).astype(
        np.float64
    )
