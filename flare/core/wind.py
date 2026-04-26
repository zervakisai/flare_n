"""flare/core/wind.py — anisotropic wind kernels for fire CA spread.

Two wind factor models are provided as pure functions, each returning an
8-element array of multiplicative factors aligned with the Moore
neighbourhood ``NEIGHBORS_DY_DX``:

1. :func:`alexandridis_wind_factors` — Paper 1 / Alexandridis et al. 2008
   exponential kernel (Eq. 4)::

        ρ_w(θ) = exp(V · (c1 + c2 · (cos θ − 1)))

   where ``V`` is wind speed, ``θ = spread_angle − wind_direction``, and
   ``c1, c2`` are calibration constants (defaults 0.045, 0.131 from the
   Spetses Island fit). This is the function used at ``wind_speed > 0`` in
   :class:`flare.core.hazards.fire_deterministic.FireSpreadModel` so the
   Invariant I1 bit-identity with Paper 1 is preserved.

2. :func:`cosine_wind_factors` — the simpler linear-cosine kernel
   referenced in CLAUDE.md Phase 1.2::

        ρ(θ) = 1 + c_w · cos(θ_wind − θ_ij),       c_w ∈ [0.5, 2.0]

   Cheaper, smoother, and easier to interpret. Available for comparative
   experiments and ablations. Not used by the deterministic CA at d=0
   (would break I1).

Coordinate convention
---------------------
Array indices are ``[row, col] = [y, x]`` with row increasing downward.
The world frame uses ``x`` east, ``y`` north. ``NEIGHBORS_DY_DX[i] = (dy, dx)``
is the array offset *to the burning neighbour from a candidate cell*; the
actual fire spread direction is therefore from ``(y+dy, x+dx)`` (burning)
to ``(y, x)`` (candidate), i.e. world vector ``(north=dy, east=-dx)`` and
angle ``atan2(dy, -dx)``.

``wind_direction`` is given **in the TO-direction convention** (radians,
CCW from east) — the direction the wind blows *toward*. This deviates
from the meteorological FROM-direction standard but matches Paper 1
(``src/flare/dynamics/fire_ca.py``); changing it would break Invariant
I1 bit-identity. Spread is enhanced when the spread angle aligns with
``wind_direction`` (i.e. ``θ = spread_angle − wind_direction → 0``).

Citation
--------
Alexandridis, A., Vakalis, D., Siettos, C.I., Bafas, G.V. (2008).
"A cellular automata model for forest fire spread prediction: The case
of the wildfire that swept through Spetses Island in 1990."
*Applied Mathematics and Computation*, 204(1), 191-201.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math

import numpy as np

# 8-connected Moore neighbourhood. Order matters — the wind factor index
# corresponds 1-to-1 with the offset entry, and downstream callers (e.g.
# the per-direction spread loop in :mod:`flare.core.hazards.fire_deterministic`)
# rely on this ordering for correctness.
NEIGHBORS_DY_DX: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# Alexandridis et al. (2008) wind constants — Eq. 4 of the paper.
ALEXANDRIDIS_C1: float = 0.045
ALEXANDRIDIS_C2: float = 0.131


def alexandridis_wind_factors(
    wind_speed: float,
    wind_direction: float,
    c1: float = ALEXANDRIDIS_C1,
    c2: float = ALEXANDRIDIS_C2,
) -> np.ndarray:
    """Compute Alexandridis 2008 wind factors for the Moore neighbourhood.

    Parameters
    ----------
    wind_speed:
        Wind speed (m/s). ``0.0`` returns all-ones (isotropic).
    wind_direction:
        Wind direction in radians (TO-direction in world frame, CCW from
        east) — see module docstring for the convention.
    c1, c2:
        Alexandridis calibration constants. Defaults reproduce Paper 1.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(8,)`` indexed in the same order as
        :data:`NEIGHBORS_DY_DX`. Multiply per-cell base spread probability
        by ``factors[i]`` for the spread direction ``i``.
    """
    factors = np.ones(8, dtype=np.float32)
    if wind_speed <= 0.0:
        return factors
    for i, (dy, dx) in enumerate(NEIGHBORS_DY_DX):
        spread_angle = math.atan2(dy, -dx)
        theta = spread_angle - wind_direction
        factors[i] = math.exp(
            wind_speed * (c1 + c2 * (math.cos(theta) - 1.0))
        )
    return factors


def cosine_wind_factors(
    wind_speed: float,
    wind_direction: float,
    coupling: float = 0.5,
) -> np.ndarray:
    """Linear-cosine wind kernel ``ρ(θ) = 1 + c_w · cos(θ_wind − θ_ij)``.

    Cheaper alternative to :func:`alexandridis_wind_factors`. Output is
    clipped at zero (a strong upwind component otherwise yields negative
    factors, which is unphysical as a multiplicative spread modifier).

    Parameters
    ----------
    wind_speed:
        Wind speed (m/s). ``0.0`` returns all-ones (isotropic). The kernel
        itself is intensity-independent; ``wind_speed`` is used only as
        the on/off gate for backward compatibility with the Alexandridis
        signature.
    wind_direction:
        Wind direction in radians (TO-direction, world frame). Same
        convention as :func:`alexandridis_wind_factors`.
    coupling:
        Anisotropy strength ``c_w``. CLAUDE.md range: ``[0.5, 2.0]``
        (moderate to strong wind). Default ``0.5``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(8,)`` indexed in :data:`NEIGHBORS_DY_DX`
        order, clipped to ``≥ 0``.
    """
    factors = np.ones(8, dtype=np.float32)
    if wind_speed <= 0.0:
        return factors
    for i, (dy, dx) in enumerate(NEIGHBORS_DY_DX):
        spread_angle = math.atan2(dy, -dx)
        theta = wind_direction - spread_angle
        factors[i] = max(0.0, 1.0 + coupling * math.cos(theta))
    return factors
