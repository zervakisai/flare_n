"""flare/belief/memory_map.py — 4-channel global belief memory.

The agent's belief about the world is a downsampled grid stored as
``float32[4, H_b, W_b]`` where::

    H_b = H_world // downsample_factor
    W_b = W_world // downsample_factor

Channels
--------
* **0 — occupancy**: probability the cell holds a static obstacle
  (building, permanent debris). Initialized from the caller-supplied
  OSM prior with optional bit-flip corruption (``prior_noise``).
* **1 — fire**: probability the cell is burning, refreshed where the
  agent's sensors observe.
* **2 — traffic**: probability the cell is blocked by traffic /
  collapse debris (analogous to Paper 1's ``traffic`` mask).
* **3 — certainty**: a recency score in ``[0, 1]``. ``1`` means the
  cell was observed *this step*; the value decays multiplicatively
  toward 0 by ``exp(-dt / τ)`` every :meth:`decay` call.

Update model (v1 — non-Bayesian)
--------------------------------
Inside the sensor footprint each step, channels 0–2 are *replaced*
with the empirical mean over the corresponding world block, and
channel 3 is reset to 1. Outside the footprint, channels 0–2 keep
their previous value while channel 3 decays. This is intentionally
simple — a full Bayesian log-odds update can replace the in-place
write later without changing callers.

Determinism (Invariant I2)
--------------------------
Initial OSM-prior corruption uses the caller-supplied
``np.random.Generator``. After ``__init__`` the class is deterministic
given the same observation sequence.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import numpy as np

from flare.belief.certainty_decay import decay_factor

# Channel indices — exported for callers that prefer named access.
CHANNEL_OCCUPANCY: int = 0
CHANNEL_FIRE: int = 1
CHANNEL_TRAFFIC: int = 2
CHANNEL_CERTAINTY: int = 3
N_CHANNELS: int = 4


class BeliefMap:
    """Global downsampled belief over the world.

    Parameters
    ----------
    world_shape:
        ``(H_w, W_w)`` ground-truth grid dimensions.
    belief_shape:
        ``(H_b, W_b)`` belief-grid dimensions. ``H_w / H_b`` and
        ``W_w / W_b`` must each be a positive integer (downsample
        factor); both must match.
    osm_prior:
        Optional ``[H_w, W_w]`` array with the OSM occupancy prior
        (any numeric type; cast to float32). ``None`` ⇒ start with all
        zeros (no prior knowledge).
    prior_noise:
        Probability of independently bit-flipping each *world cell* of
        the OSM prior before downsampling. Models stale / inaccurate
        maps. Default ``0.05``.
    tau_decay_steps:
        Time constant for the certainty channel (steps). Default
        ``50.0`` — half-life ≈ 35 steps.
    rng:
        Caller-supplied ``np.random.Generator`` for the OSM-prior bit-flip
        corruption. ``None`` ⇒ use a fresh default generator (NOT
        recommended in production: breaks Invariant I2).
    """

    def __init__(
        self,
        world_shape: tuple[int, int],
        belief_shape: tuple[int, int] = (50, 50),
        osm_prior: np.ndarray | None = None,
        prior_noise: float = 0.05,
        tau_decay_steps: float = 50.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        H_w, W_w = world_shape
        H_b, W_b = belief_shape
        if H_w % H_b != 0 or W_w % W_b != 0:
            raise ValueError(
                f"world_shape {world_shape} must be divisible by "
                f"belief_shape {belief_shape}"
            )
        ds_h = H_w // H_b
        ds_w = W_w // W_b
        if ds_h != ds_w:
            raise ValueError(
                f"non-square downsample factor (H={ds_h}, W={ds_w}) "
                f"is not supported"
            )
        if not 0.0 <= prior_noise <= 1.0:
            raise ValueError(
                f"prior_noise must be in [0, 1], got {prior_noise}"
            )
        if tau_decay_steps <= 0:
            raise ValueError(
                f"tau_decay_steps must be > 0, got {tau_decay_steps}"
            )

        self._world_shape = (H_w, W_w)
        self._belief_shape = (H_b, W_b)
        self._ds = ds_h
        self._tau = float(tau_decay_steps)
        self._rng = rng if rng is not None else np.random.default_rng()

        self._map = np.zeros((N_CHANNELS, H_b, W_b), dtype=np.float32)

        if osm_prior is not None:
            if osm_prior.shape != (H_w, W_w):
                raise ValueError(
                    f"osm_prior shape {osm_prior.shape} ≠ world_shape "
                    f"{(H_w, W_w)}"
                )
            corrupted = osm_prior.astype(np.float32, copy=True)
            if prior_noise > 0.0:
                flip = self._rng.random((H_w, W_w)) < prior_noise
                corrupted[flip] = 1.0 - corrupted[flip]
            self._map[CHANNEL_OCCUPANCY] = self._downsample_mean(corrupted)

    # -- Properties ------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(N_CHANNELS, H_b, W_b)``."""
        return (N_CHANNELS, *self._belief_shape)

    @property
    def downsample_factor(self) -> int:
        return self._ds

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def map(self) -> np.ndarray:
        """Read-only copy of the full belief tensor."""
        return self._map.copy()

    @property
    def occupancy(self) -> np.ndarray:
        return self._map[CHANNEL_OCCUPANCY].copy()

    @property
    def fire(self) -> np.ndarray:
        return self._map[CHANNEL_FIRE].copy()

    @property
    def traffic(self) -> np.ndarray:
        return self._map[CHANNEL_TRAFFIC].copy()

    @property
    def certainty(self) -> np.ndarray:
        return self._map[CHANNEL_CERTAINTY].copy()

    # -- Mutators --------------------------------------------------------

    def update_observation(
        self,
        observed_mask_world: np.ndarray,
        observed_fire_world: np.ndarray | None = None,
        observed_traffic_world: np.ndarray | None = None,
        observed_occupancy_world: np.ndarray | None = None,
    ) -> None:
        """Refresh belief inside the sensor footprint.

        Parameters
        ----------
        observed_mask_world:
            ``bool[H_w, W_w]``. Cells inside the agent's sensor footprint
            (roughly: cells the LiDAR / camera could resolve this step).
            World blocks where *any* cell is observed will refresh.
        observed_fire_world, observed_traffic_world, observed_occupancy_world:
            Per-channel ground-truth slices for cells inside the
            footprint. Missing channels are left untouched. Must be
            ``[H_w, W_w]``-shaped if provided.
        """
        if observed_mask_world.shape != self._world_shape:
            raise ValueError(
                f"observed_mask_world shape {observed_mask_world.shape} ≠ "
                f"world_shape {self._world_shape}"
            )

        obs_block = self._downsample_any(observed_mask_world)

        if observed_occupancy_world is not None:
            occ_block = self._downsample_mean(
                observed_occupancy_world.astype(np.float32, copy=False)
            )
            self._map[CHANNEL_OCCUPANCY][obs_block] = occ_block[obs_block]

        if observed_fire_world is not None:
            fire_block = self._downsample_mean(
                observed_fire_world.astype(np.float32, copy=False)
            )
            self._map[CHANNEL_FIRE][obs_block] = fire_block[obs_block]

        if observed_traffic_world is not None:
            traf_block = self._downsample_mean(
                observed_traffic_world.astype(np.float32, copy=False)
            )
            self._map[CHANNEL_TRAFFIC][obs_block] = traf_block[obs_block]

        # Refresh certainty inside the footprint to 1; outside is
        # untouched (it'll decay on the next decay() call).
        self._map[CHANNEL_CERTAINTY][obs_block] = 1.0

    def decay(self, dt: float = 1.0) -> None:
        """Apply ``c *= exp(-dt / τ)`` to the certainty channel."""
        self._map[CHANNEL_CERTAINTY] *= np.float32(
            decay_factor(dt, self._tau)
        )

    def reset(self) -> None:
        """Zero all channels (including the OSM-derived occupancy)."""
        self._map.fill(0.0)

    # -- Internal --------------------------------------------------------

    def _downsample_any(self, arr: np.ndarray) -> np.ndarray:
        """Block-reduce a world-resolution mask to belief resolution via
        ``any``-aggregation. Returns ``bool[H_b, W_b]``."""
        H_b, W_b = self._belief_shape
        return arr.reshape(H_b, self._ds, W_b, self._ds).any(axis=(1, 3))

    def _downsample_mean(self, arr: np.ndarray) -> np.ndarray:
        """Block-reduce a world-resolution scalar field to belief
        resolution via mean-aggregation. Returns ``float32[H_b, W_b]``."""
        H_b, W_b = self._belief_shape
        return (
            arr.reshape(H_b, self._ds, W_b, self._ds)
            .mean(axis=(1, 3))
            .astype(np.float32, copy=False)
        )
