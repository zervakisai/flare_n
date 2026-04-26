"""flare/sensors/camera.py — forward-looking fire-detection camera.

Generic FPV camera looking along the agent's heading. Detects burning
cells that are simultaneously:

* within the horizontal FOV ``± HFOV/2`` of the heading;
* within the clean-air range ``≤ max_range_m``;
* in line-of-sight (no building occlusion, no smoke cut-off).

Smoke degrades visibility through the same Beer–Lambert mechanism used
by the LiDAR ray-cast — so dense smoke between the camera and a fire
will hide the fire even though it is geometrically in view.

Output is a list of detected fire cells with per-cell confidence in
``[0, 1]``. The env wraps these into the ego-centric
``fire_detection`` observation channel (Phase 3).

Confidence model
----------------
Confidence decays with distance to the fire (``conf ∝ exp(-r / r_max)``).
Smoke is implicit in the LOS check (the ray is cut before it reaches
the fire); detections that survive the LOS test get the geometric
confidence above. A small false-negative dropout (5 % per detection)
models thermal-frame jitter.

Citations
---------
- HFOV / range numbers from generic FPV digital cameras (Caddx Ant,
  RunCam Phoenix 2 — datasheets list 130–170° HFOV; we pick 130°).
- Visibility coupling: Jin, T. (1971). "Visibility through fire smoke."
  *Report of the Fire Research Institute of Japan*, No. 33.

Determinism: all randomness through caller-supplied ``np.random.Generator``.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flare.sensors.ray_cast import cast_ray

# === Camera spec constants ================================================

CAMERA_HFOV_DEG: float = 130.0
"""Horizontal field of view (degrees). Source: generic FPV cameras
(Caddx Ant: 165°, RunCam Phoenix 2: 155°). Conservative 130° default."""

CAMERA_MAX_RANGE_M: float = 30.0
"""Clean-air detection range for a glow/flame target (m). Source: empirical
flame-detection range for 2-MP CMOS sensors at 720p."""

CAMERA_FALSE_NEGATIVE_RATE: float = 0.05
"""Per-detection false-negative dropout to model thermal-frame jitter."""

CAMERA_SNR_THRESHOLD: float = 0.05
"""Round-trip transmission cutoff for the LOS ray-cast (matches LiDAR)."""


@dataclass
class CameraScan:
    """A single camera frame's fire detections.

    Length-K parallel arrays where ``K`` ≤ number of fire cells in FOV.
    """

    detected_y: np.ndarray  #: int32[K], cell rows of detected fires
    detected_x: np.ndarray  #: int32[K], cell cols of detected fires
    confidences: np.ndarray  #: float32[K], per-detection confidence ∈ [0, 1]


class CameraSensor:
    """Forward-looking fire-detection camera.

    Parameters
    ----------
    rng:
        Caller-supplied ``np.random.Generator``.
    hfov_deg, max_range_m, false_negative_rate, snr_threshold:
        See module-level constants for defaults and citations.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        hfov_deg: float = CAMERA_HFOV_DEG,
        max_range_m: float = CAMERA_MAX_RANGE_M,
        false_negative_rate: float = CAMERA_FALSE_NEGATIVE_RATE,
        snr_threshold: float = CAMERA_SNR_THRESHOLD,
    ) -> None:
        if not 0.0 < hfov_deg <= 360.0:
            raise ValueError(f"hfov_deg must be in (0, 360], got {hfov_deg}")
        if max_range_m <= 0:
            raise ValueError(f"max_range_m must be > 0, got {max_range_m}")
        if not 0.0 <= false_negative_rate <= 1.0:
            raise ValueError(
                f"false_negative_rate must be in [0, 1], got {false_negative_rate}"
            )
        if not 0.0 < snr_threshold < 1.0:
            raise ValueError(
                f"snr_threshold must be in (0, 1), got {snr_threshold}"
            )

        self._rng = rng
        self._hfov_rad = math.radians(float(hfov_deg))
        self._max_range_m = float(max_range_m)
        self._fn_rate = float(false_negative_rate)
        self._snr_threshold = float(snr_threshold)

    # -- Properties ------------------------------------------------------

    @property
    def hfov_deg(self) -> float:
        return math.degrees(self._hfov_rad)

    @property
    def max_range_m(self) -> float:
        return self._max_range_m

    # -- Scan ------------------------------------------------------------

    def scan(
        self,
        agent_x: float,
        agent_y: float,
        heading: float,
        occupancy: np.ndarray,
        fire_mask: np.ndarray,
        extinction: np.ndarray,
        cell_size_m: float,
    ) -> CameraScan:
        """One camera frame.

        Parameters
        ----------
        agent_x, agent_y:
            Agent position in array space (column, row).
        heading:
            Heading in radians (CCW from +x). The camera looks along
            ``heading`` with ``±hfov/2`` aperture.
        occupancy:
            ``int8[H, W]`` world occupancy (buildings).
        fire_mask:
            ``bool[H, W]``: ``True`` where a cell is currently burning.
        extinction:
            ``float32[H, W]`` smoke ``σ_ext`` field (1/m).
        cell_size_m:
            World cell size in metres.
        """
        if cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be > 0, got {cell_size_m}")
        if fire_mask.shape != occupancy.shape:
            raise ValueError(
                f"fire_mask shape {fire_mask.shape} must equal "
                f"occupancy shape {occupancy.shape}"
            )

        H, W = fire_mask.shape
        max_range_cells = int(math.ceil(self._max_range_m / cell_size_m))
        half_hfov = self._hfov_rad / 2.0

        # All fire cells
        fire_ys, fire_xs = np.where(fire_mask)
        n_fires = fire_ys.size
        if n_fires == 0:
            return _empty_scan()

        # Vector and angle from agent to each fire (world frame: y = -row).
        dx = fire_xs.astype(np.float64) - agent_x
        dy = -(fire_ys.astype(np.float64) - agent_y)
        r_cells = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)

        # Wrap angular delta to (-π, π] before HFOV check.
        delta = ang - heading
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        in_fov = (np.abs(delta) <= half_hfov) & (r_cells <= max_range_cells) & (
            r_cells > 0.0
        )
        if not in_fov.any():
            return _empty_scan()

        cand_ys = fire_ys[in_fov]
        cand_xs = fire_xs[in_fov]
        cand_r = r_cells[in_fov]
        cand_ang = ang[in_fov]
        n_cand = cand_ys.size

        det_y_buf: list[int] = []
        det_x_buf: list[int] = []
        conf_buf: list[float] = []

        # Pre-draw FN coin tosses to keep RNG order stable across builds.
        fn_rolls = self._rng.random(n_cand)

        for k in range(n_cand):
            target_steps = max(1, int(math.ceil(cand_r[k])))
            _, _, _, ht = cast_ray(
                agent_x,
                agent_y,
                float(cand_ang[k]),
                target_steps,
                occupancy,
                extinction,
                cell_size_m,
                self._snr_threshold,
            )
            # ht == 0 means the ray walked the full target_steps without
            # hitting an occupied cell or smoke cut-off → LOS clear.
            if ht != 0:
                continue
            if fn_rolls[k] < self._fn_rate:
                continue
            conf = float(math.exp(-float(cand_r[k]) / max_range_cells))
            det_y_buf.append(int(cand_ys[k]))
            det_x_buf.append(int(cand_xs[k]))
            conf_buf.append(conf)

        return CameraScan(
            detected_y=np.asarray(det_y_buf, dtype=np.int32),
            detected_x=np.asarray(det_x_buf, dtype=np.int32),
            confidences=np.asarray(conf_buf, dtype=np.float32),
        )


def _empty_scan() -> CameraScan:
    return CameraScan(
        detected_y=np.empty(0, dtype=np.int32),
        detected_x=np.empty(0, dtype=np.int32),
        confidences=np.empty(0, dtype=np.float32),
    )
