"""flare/envs/flare_pomdp.py — FLARE-PO Gymnasium environment.

The Paper 2 environment. Composes the Phase 1 hazards (fire CA, smoke
plume, dynamic NFZ), the Phase 2 sensor models (LiDAR, camera, GPS),
and the Phase 3 belief memory map into a single Gymnasium ``Env``.

Action space (``Discrete(10)``)
-------------------------------
``0..7`` move one cell in one of the eight Moore directions
(``N, NE, E, SE, S, SW, W, NW`` — world frame, +x east, +y north).
``8`` is HOVER (no movement). ``9`` is SCAN360 (no movement, but the
camera reports a 360° sweep instead of its usual sector — a more
expensive belief refresh that the policy can choose to spend).

Observation space (``Dict``) — see :data:`OBS_SPACE` for details.

Reward (v1)
-----------
* ``+1.0`` on reaching the goal.
* ``-1.0`` on death (entering occupied / fire / NFZ cell, or off-grid).
* ``-0.01`` per step (mild urgency).
* ``+0.001 · |Δknown|`` for newly observed belief cells (curiosity bonus).

Phase 4 will replace the constant goal-reward with a calibrated
mission-decay value ``M(t_arrival)``.

Termination
-----------
* ``terminated = True`` on goal or death.
* ``truncated = True`` after ``max_steps`` without termination.

Determinism (Invariant I2)
--------------------------
All randomness flows through ``np.random.Generator`` instances created
in :meth:`reset`. Same ``seed`` ⇒ identical trajectory.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flare.belief.memory_map import (
    CHANNEL_CERTAINTY,
    CHANNEL_FIRE,
    CHANNEL_OCCUPANCY,
    BeliefMap,
)
from flare.core.hazards.fire_deterministic import FireSpreadModel
from flare.core.hazards.fire_stochastic import StochasticFireSpreadModel
from flare.core.hazards.nfz import DynamicNFZ
from flare.core.hazards.smoke import GaussianSmokePlume
from flare.scenarios.synthetic import Scenario, make_synthetic
from flare.sensors.camera import CameraSensor
from flare.sensors.gps import GPSSensor
from flare.sensors.lidar import LiDARSensor

# Action constants ----------------------------------------------------------

# Δ(row, col) per discrete direction. Index aligned with the 0..7 N…NW order.
_ACTION_DELTAS: list[tuple[int, int]] = [
    (-1, 0),   # 0  N
    (-1, 1),   # 1  NE
    (0, 1),    # 2  E
    (1, 1),    # 3  SE
    (1, 0),    # 4  S
    (1, -1),   # 5  SW
    (0, -1),   # 6  W
    (-1, -1),  # 7  NW
]
ACTION_HOVER: int = 8
ACTION_SCAN: int = 9
N_ACTIONS: int = 10


# ---------------------------------------------------------------------------
# Egocentric helpers
# ---------------------------------------------------------------------------


def _world_patch(
    field: np.ndarray,
    agent_x: float,
    agent_y: float,
    size: int,
    default: float = 0.0,
) -> np.ndarray:
    """Extract an axis-aligned ``size × size`` patch centered on the agent.

    Cells beyond the world boundary are filled with ``default``. Output
    dtype matches ``field.dtype``.
    """
    H, W = field.shape
    half = size // 2
    ax = int(round(agent_x))
    ay = int(round(agent_y))
    out = np.full((size, size), default, dtype=field.dtype)

    y0 = max(0, ay - half)
    y1 = min(H, ay + half + 1)
    x0 = max(0, ax - half)
    x1 = min(W, ax + half + 1)
    if y0 >= y1 or x0 >= x1:
        return out
    py0 = (y0 - (ay - half))
    py1 = py0 + (y1 - y0)
    px0 = (x0 - (ax - half))
    px1 = px0 + (x1 - x0)
    out[py0:py1, px0:px1] = field[y0:y1, x0:x1]
    return out


def _project_points_to_ego(
    xs: np.ndarray,
    ys: np.ndarray,
    vals: np.ndarray,
    agent_x: float,
    agent_y: float,
    size: int,
) -> np.ndarray:
    """Rasterize axis-aligned point detections to a ``size × size`` grid.

    Multiple points landing in the same ego cell ⇒ ``maximum`` of values.
    """
    grid = np.zeros((size, size), dtype=np.float32)
    if xs.size == 0:
        return grid
    half = size // 2
    ax = int(round(agent_x))
    ay = int(round(agent_y))
    ego_rows = ys.astype(np.int64) - ay + half
    ego_cols = xs.astype(np.int64) - ax + half
    valid = (
        (ego_rows >= 0) & (ego_rows < size)
        & (ego_cols >= 0) & (ego_cols < size)
    )
    if not np.any(valid):
        return grid
    np.maximum.at(
        grid, (ego_rows[valid], ego_cols[valid]), vals[valid].astype(np.float32)
    )
    return grid


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class FlarePOEnv(gym.Env):
    """POMDP wildfire UAV environment.

    Parameters
    ----------
    scenario:
        :class:`flare.scenarios.synthetic.Scenario`. ``None`` ⇒ a default
        synthetic 60×60 scenario is generated.
    observability:
        Cosmetic in v1 (forwarded to ``info`` so policies can condition
        on it). Future versions will scale sensor degradation.
    wind_speed, wind_direction:
        Forwarded to fire CA, smoke plume.
    max_steps:
        Episode truncation horizon.
    ego_size:
        Side length of the ego-centric observation grids. Default 25.
    belief_shape:
        Belief map size ``(H_b, W_b)``. Must divide the scenario grid.
    n_initial_ignitions:
        Random fire seeds at episode start.
    fire_mode:
        ``"stochastic"`` (default; Beta-Alexandridis) or
        ``"deterministic"`` (Paper 1 CA, for I1 continuity tests).
    render_mode:
        ``None`` (default) or ``"rgb_array"``.
    """

    metadata = {"render_modes": [None, "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        scenario: Scenario | None = None,
        observability: float = 0.6,
        wind_speed: float = 5.0,
        wind_direction: float = 0.0,
        max_steps: int = 200,
        ego_size: int = 25,
        belief_shape: tuple[int, int] = (10, 10),
        n_initial_ignitions: int = 2,
        fire_mode: str = "stochastic",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if not 0.0 <= observability <= 1.0:
            raise ValueError(
                f"observability must be in [0, 1], got {observability}"
            )
        if max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {max_steps}")
        if ego_size < 3:
            raise ValueError(f"ego_size must be ≥ 3, got {ego_size}")
        if fire_mode not in ("stochastic", "deterministic"):
            raise ValueError(
                f"fire_mode must be 'stochastic' or 'deterministic', "
                f"got {fire_mode!r}"
            )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be in {self.metadata['render_modes']}, "
                f"got {render_mode!r}"
            )

        self._scenario = scenario if scenario is not None else make_synthetic()

        H, W = self._scenario.grid_shape
        H_b, W_b = belief_shape
        if H % H_b != 0 or W % W_b != 0:
            raise ValueError(
                f"scenario grid {self._scenario.grid_shape} must divide "
                f"belief_shape {belief_shape}"
            )

        self._observability = float(observability)
        self._wind_speed = float(wind_speed)
        self._wind_direction = float(wind_direction)
        self._max_steps = int(max_steps)
        self._ego_size = int(ego_size)
        self._belief_shape = (int(H_b), int(W_b))
        self._n_initial_ignitions = int(n_initial_ignitions)
        self._fire_mode = fire_mode
        self.render_mode = render_mode

        # Spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "lidar_occupancy": spaces.Box(
                    0.0, 1.0, (ego_size, ego_size), dtype=np.float32
                ),
                "fire_detection": spaces.Box(
                    0.0, 1.0, (ego_size, ego_size), dtype=np.float32
                ),
                "smoke_density": spaces.Box(
                    0.0, 100.0, (ego_size, ego_size), dtype=np.float32
                ),
                "belief_map": spaces.Box(
                    0.0, 1.0, (4, H_b, W_b), dtype=np.float32
                ),
                "pose": spaces.Box(
                    -np.inf, np.inf, (3,), dtype=np.float32
                ),
                "mission_ctx": spaces.Box(
                    -np.inf, np.inf, (5,), dtype=np.float32
                ),
                "goal_vector": spaces.Box(
                    -1.0, 1.0, (3,), dtype=np.float32
                ),
            }
        )

        # Will be created in reset()
        self._fire: FireSpreadModel | StochasticFireSpreadModel | None = None
        self._smoke: GaussianSmokePlume | None = None
        self._nfz: DynamicNFZ | None = None
        self._lidar: LiDARSensor | None = None
        self._camera: CameraSensor | None = None
        self._gps: GPSSensor | None = None
        self._belief: BeliefMap | None = None

        self._agent_x: float = 0.0
        self._agent_y: float = 0.0
        self._heading: float = 0.0
        self._step_count: int = 0
        self._last_distance: float = 0.0
        self._prev_known_count: int = 0

    # -- Properties ------------------------------------------------------

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def agent_pos(self) -> tuple[float, float]:
        return (self._agent_x, self._agent_y)

    @property
    def heading(self) -> float:
        return self._heading

    @property
    def step_count(self) -> int:
        return self._step_count

    # -- Gym API ---------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        # Single Generator drives every component in this episode.
        rng = np.random.Generator(np.random.PCG64(seed))

        sc = self._scenario
        H, W = sc.grid_shape
        cell = sc.cell_size_m

        # Hazards
        fire_kwargs = dict(
            map_shape=(H, W),
            rng=rng,
            n_ignition=self._n_initial_ignitions,
            landuse_map=sc.landuse,
            roads_mask=sc.roads_mask,
            wind_speed=self._wind_speed,
            wind_direction=self._wind_direction,
        )
        if self._fire_mode == "stochastic":
            self._fire = StochasticFireSpreadModel(**fire_kwargs)
        else:
            self._fire = FireSpreadModel(**fire_kwargs)
        self._smoke = GaussianSmokePlume(
            grid_shape=(H, W),
            cell_size_m=cell,
            wind_speed=self._wind_speed,
            wind_direction=self._wind_direction,
        )
        self._nfz = DynamicNFZ(grid_shape=(H, W), rng=rng)

        # Sensors
        self._lidar = LiDARSensor(rng=rng)
        self._camera = CameraSensor(rng=rng)
        self._gps = GPSSensor(rng=rng)

        # Belief — OSM prior is the static occupancy.
        osm_prior = sc.occupancy.astype(np.float32)
        self._belief = BeliefMap(
            world_shape=(H, W),
            belief_shape=self._belief_shape,
            osm_prior=osm_prior,
            tau_decay_steps=50.0,
            rng=rng,
        )

        # Agent
        self._agent_x = float(sc.start[0])
        self._agent_y = float(sc.start[1])
        self._heading = 0.0
        self._step_count = 0
        self._gps.reset(self._agent_x * cell, self._agent_y * cell, 0.0)
        self._last_distance = self._distance_to_goal()
        self._prev_known_count = int((self._belief.certainty > 0.0).sum())

        # First observation pass
        self._smoke.update(self._fire.fire_mask)
        self._update_belief_from_sensors(scan_360=False)

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._fire is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        sc = self._scenario
        H, W = sc.grid_shape

        # 1. Movement
        scan_360 = False
        if action < 8:
            d_row, d_col = _ACTION_DELTAS[action]
            new_x = self._agent_x + d_col
            new_y = self._agent_y + d_row
            # Update heading to match the move direction (world frame).
            self._heading = math.atan2(-float(d_row), float(d_col))
        else:
            new_x, new_y = self._agent_x, self._agent_y
            if action == ACTION_SCAN:
                scan_360 = True

        # 2. Step physics (fire CA → smoke → NFZ)
        self._fire.step()
        self._smoke.set_wind(self._wind_speed, self._wind_direction)
        self._smoke.update(self._fire.fire_mask)
        self._nfz.step()

        # 3. Resolve agent collision before commit
        terminated = False
        reward = 0.0
        info_extra: dict[str, Any] = {}

        cell_x = int(round(new_x))
        cell_y = int(round(new_y))
        oob = not (0 <= cell_x < W and 0 <= cell_y < H)

        if oob:
            terminated = True
            reward = -1.0
            info_extra["death_cause"] = "out_of_bounds"
            # Snap to last valid position so subsequent state queries are safe.
            new_x, new_y = self._agent_x, self._agent_y
        elif sc.occupancy[cell_y, cell_x] > 0:
            terminated = True
            reward = -1.0
            info_extra["death_cause"] = "occupied"
        elif bool(self._fire.fire_mask[cell_y, cell_x]):
            terminated = True
            reward = -1.0
            info_extra["death_cause"] = "fire"
        elif bool(self._nfz.nfz_mask[cell_y, cell_x]):
            terminated = True
            reward = -1.0
            info_extra["death_cause"] = "nfz"

        if not terminated:
            self._agent_x = float(new_x)
            self._agent_y = float(new_y)

        # 4. Check goal
        gx, gy = sc.goal
        at_goal = (
            abs(self._agent_x - gx) <= 0.5
            and abs(self._agent_y - gy) <= 0.5
        )
        if at_goal and not terminated:
            terminated = True
            reward = 1.0
            info_extra["death_cause"] = None
            info_extra["success"] = True

        # 5. Sensors + belief update
        self._update_belief_from_sensors(scan_360=scan_360)

        # 6. Shaping rewards (only when not terminated)
        if not terminated:
            d_now = self._distance_to_goal()
            progress = (self._last_distance - d_now)
            self._last_distance = d_now
            known_count = int((self._belief.certainty > 0.0).sum())
            d_known = max(0, known_count - self._prev_known_count)
            self._prev_known_count = known_count
            reward = 0.01 * progress + 0.001 * d_known - 0.01

        self._step_count += 1
        truncated = self._step_count >= self._max_steps and not terminated

        obs = self._build_obs()
        info = self._build_info()
        info.update(info_extra)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        # Minimal ARGB-ish rendering: stack occupancy + fire + agent.
        sc = self._scenario
        H, W = sc.grid_shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[sc.occupancy > 0] = (90, 90, 90)
        if self._fire is not None:
            img[self._fire.fire_mask] = (220, 60, 0)
            img[self._fire.burned_mask] = (40, 40, 40)
        if self._nfz is not None:
            img[self._nfz.nfz_mask] = (50, 80, 220)
        ax = int(round(self._agent_x))
        ay = int(round(self._agent_y))
        if 0 <= ay < H and 0 <= ax < W:
            img[ay, ax] = (0, 220, 0)
        gx, gy = sc.goal
        if 0 <= gy < H and 0 <= gx < W:
            img[gy, gx] = (220, 220, 0)
        return img

    # -- Internals -------------------------------------------------------

    def _distance_to_goal(self) -> float:
        gx, gy = self._scenario.goal
        return math.hypot(gx - self._agent_x, gy - self._agent_y)

    def _update_belief_from_sensors(self, scan_360: bool) -> None:
        assert self._lidar is not None
        assert self._camera is not None
        assert self._belief is not None
        assert self._fire is not None
        assert self._smoke is not None

        sc = self._scenario
        H, W = sc.grid_shape
        ext = self._smoke.extinction.astype(np.float32)
        occ = sc.occupancy.astype(np.int8)

        # 1. Run sensors (used to generate observations + drive belief refresh).
        self._last_lidar_scan = self._lidar.scan(
            self._agent_x,
            self._agent_y,
            self._heading,
            occ,
            ext,
            sc.cell_size_m,
        )
        # SCAN360 widens the camera HFOV one-shot.
        camera_to_use = self._camera
        if scan_360:
            # A throwaway 360° camera that shares the same RNG / spec; we
            # construct it on the fly to avoid mutating the persistent one.
            camera_to_use = CameraSensor(
                rng=self._camera._rng,  # share Generator state
                hfov_deg=359.0,
                max_range_m=self._camera.max_range_m,
                false_negative_rate=0.05,
            )
        self._last_camera_scan = camera_to_use.scan(
            self._agent_x,
            self._agent_y,
            self._heading,
            occ,
            self._fire.fire_mask,
            ext,
            sc.cell_size_m,
        )

        # 2. Build the world-resolution observation footprint: a circle of
        # radius max(lidar_range, camera_range) around the agent.
        max_range_cells = int(
            math.ceil(
                max(self._lidar.max_range_m, self._camera.max_range_m)
                / sc.cell_size_m
            )
        )
        yy, xx = np.mgrid[:H, :W]
        d2 = (yy - self._agent_y) ** 2 + (xx - self._agent_x) ** 2
        footprint = d2 <= max_range_cells**2

        # 3. Decay first (so the freshly-observed footprint can refresh).
        self._belief.decay(dt=1.0)
        self._belief.update_observation(
            observed_mask_world=footprint,
            observed_fire_world=self._fire.fire_mask,
            observed_traffic_world=self._nfz.nfz_mask,
            observed_occupancy_world=(occ > 0),
        )

    # Sensor-derived ego channels --------------------------------------

    def _ego_lidar_occupancy(self) -> np.ndarray:
        """Rasterize LiDAR hits whose ``hit_type`` indicates a real return
        (occupied or smoke cut-off) to the ego grid."""
        scan = self._last_lidar_scan
        # hit_type 1 = occupied, 2 = smoke cutoff (treat as obstacle)
        keep = scan.valid & ((scan.hit_types == 1) | (scan.hit_types == 2))
        if not np.any(keep):
            return np.zeros(
                (self._ego_size, self._ego_size), dtype=np.float32
            )
        vals = np.ones(int(keep.sum()), dtype=np.float32)
        return _project_points_to_ego(
            scan.hit_x[keep],
            scan.hit_y[keep],
            vals,
            self._agent_x,
            self._agent_y,
            self._ego_size,
        )

    def _ego_fire_detection(self) -> np.ndarray:
        scan = self._last_camera_scan
        if scan.detected_x.size == 0:
            return np.zeros(
                (self._ego_size, self._ego_size), dtype=np.float32
            )
        return _project_points_to_ego(
            scan.detected_x,
            scan.detected_y,
            scan.confidences,
            self._agent_x,
            self._agent_y,
            self._ego_size,
        )

    def _ego_smoke_density(self) -> np.ndarray:
        ext = self._smoke.extinction.astype(np.float32)
        # Multiply by a per-metre→per-cell scale so the value range covered
        # by the Box [0, 100] is comfortable in typical scenarios.
        patch = _world_patch(
            ext, self._agent_x, self._agent_y, self._ego_size, default=0.0
        )
        return np.clip(patch * 100.0, 0.0, 100.0).astype(np.float32)

    # Observation pipeline ---------------------------------------------

    def _build_obs(self) -> dict[str, np.ndarray]:
        sc = self._scenario
        cell = sc.cell_size_m
        # GPS reading for pose
        # Building density at agent position: simple 5×5 average.
        ax = int(round(self._agent_x))
        ay = int(round(self._agent_y))
        H, W = sc.grid_shape
        y0 = max(0, ay - 2); y1 = min(H, ay + 3)
        x0 = max(0, ax - 2); x1 = min(W, ax + 3)
        density = float(sc.occupancy[y0:y1, x0:x1].mean())
        x_meas, y_meas, psi_meas, _ = self._gps.update(
            self._agent_x * cell,
            self._agent_y * cell,
            self._heading,
            dt=1.0,
            building_density=density,
        )
        pose = np.array(
            [x_meas, y_meas, psi_meas], dtype=np.float32
        )

        # Goal vector — (cos θ, sin θ, normalized distance)
        gx, gy = sc.goal
        dx = gx - self._agent_x
        dy = -(gy - self._agent_y)  # world y = -row
        d = math.hypot(dx, dy)
        max_d = float(math.hypot(H, W))
        if d < 1e-9:
            goal_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            goal_vec = np.array(
                [dx / d, dy / d, min(d / max_d, 1.0)],
                dtype=np.float32,
            )

        return {
            "lidar_occupancy": self._ego_lidar_occupancy(),
            "fire_detection": self._ego_fire_detection(),
            "smoke_density": self._ego_smoke_density(),
            "belief_map": self._belief.map.astype(np.float32),
            "pose": pose,
            "mission_ctx": np.zeros(5, dtype=np.float32),  # Phase 4 fills
            "goal_vector": goal_vec,
        }

    def _build_info(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "agent_pos": (self._agent_x, self._agent_y),
            "heading": self._heading,
            "observability": self._observability,
            "fire_mode": self._fire_mode,
        }
