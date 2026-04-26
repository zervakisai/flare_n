"""flare/sensors/lidar.py — LDRobot LD19 LiDAR sensor model.

Wraps the Numba ray-cast engine in :mod:`flare.sensors.ray_cast` and adds
the noise / dropout characteristics of the LDRobot DTOF LD19:

* 12 m effective range
* 450 rays per sweep (360° / 0.8° angular resolution)
* Range noise ``σ_r = 10 mm + 0.001 · r`` (Gaussian)
* Angular noise ``σ_θ = 2°`` per ray
* False-negative dropout 2 % per beam per sweep

Smoke attenuation is handled inside the ray-cast (Beer–Lambert with
factor 2 for the round-trip pulse). The LiDAR layer just adds noise on
top of the deterministic geometry.

Determinism: all randomness flows through the caller-supplied
``np.random.Generator`` (Invariant I2).

Citations
---------
- LDRobot Technical specifications, *DTOF Single-Line LiDAR LD19* (2022).
- Waveshare wiki / Little Bird Electronics product listing for LD19.
- Beer–Lambert law for active sensors: e.g. McManamon, *LiDAR Technologies
  and Systems*, SPIE Press 2019, §3.2.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flare.sensors.ray_cast import cast_rays

# === LD19 calibration constants ===========================================
# Source: LDRobot DTOF LD19 spec sheet, Waveshare wiki entry.

LIDAR_MAX_RANGE_M: float = 12.0
"""LD19 effective maximum range (m). Source: LDRobot DTOF LD19 spec."""

LIDAR_NUM_RAYS: int = 450
"""360° / 0.8° = 450 rays per sweep. Source: LD19 angular resolution spec."""

LIDAR_RANGE_NOISE_MM_FIXED: float = 10.0
LIDAR_RANGE_NOISE_PROPORTIONAL: float = 0.001
"""σ_r(r) = 10 mm + 0.001 · r. Source: LD19 spec sheet."""

LIDAR_ANGULAR_NOISE_DEG: float = 2.0
"""σ_θ = 2° per beam. Source: LD19 spec sheet."""

LIDAR_FALSE_NEGATIVE_RATE: float = 0.02
"""p_FN ≈ 2 % per beam per sweep under nominal conditions. Calibrated
against PythonRobotics ``lidar_to_grid_map.py`` benchmark scans of
DTOF LiDARs in low-dust environments."""

LIDAR_SNR_THRESHOLD: float = 0.05
"""Round-trip transmission below which the return is considered lost in
smoke. Beer–Lambert with factor 2 for the active pulse."""


@dataclass
class LiDARScan:
    """Single LiDAR sweep.

    All arrays have length ``N = n_rays`` and are aligned ray-by-ray.
    """

    angles: np.ndarray       #: float64[N], cast angles (radians, world-frame)
    distances_m: np.ndarray  #: float32[N], reported range in metres (noisy)
    hit_x: np.ndarray        #: int32[N], terminal column on world grid
    hit_y: np.ndarray        #: int32[N], terminal row on world grid
    hit_types: np.ndarray    #: int8[N], see ``flare.sensors.ray_cast``
    valid: np.ndarray        #: bool[N], False where false-negative dropped


class LiDARSensor:
    """LDRobot LD19 LiDAR sensor.

    Parameters
    ----------
    rng:
        Caller-supplied ``np.random.Generator``.
    max_range_m, n_rays, range_noise_mm_fixed, range_noise_proportional,
    angular_noise_deg, false_negative_rate, snr_threshold:
        See module-level constants for defaults and citations.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        max_range_m: float = LIDAR_MAX_RANGE_M,
        n_rays: int = LIDAR_NUM_RAYS,
        range_noise_mm_fixed: float = LIDAR_RANGE_NOISE_MM_FIXED,
        range_noise_proportional: float = LIDAR_RANGE_NOISE_PROPORTIONAL,
        angular_noise_deg: float = LIDAR_ANGULAR_NOISE_DEG,
        false_negative_rate: float = LIDAR_FALSE_NEGATIVE_RATE,
        snr_threshold: float = LIDAR_SNR_THRESHOLD,
    ) -> None:
        if max_range_m <= 0:
            raise ValueError(f"max_range_m must be > 0, got {max_range_m}")
        if n_rays <= 0:
            raise ValueError(f"n_rays must be > 0, got {n_rays}")
        if not 0.0 <= false_negative_rate <= 1.0:
            raise ValueError(
                f"false_negative_rate must be in [0, 1], got {false_negative_rate}"
            )
        if not 0.0 < snr_threshold < 1.0:
            raise ValueError(
                f"snr_threshold must be in (0, 1), got {snr_threshold}"
            )

        self._rng = rng
        self._max_range_m = float(max_range_m)
        self._n_rays = int(n_rays)
        self._range_noise_mm_fixed = float(range_noise_mm_fixed)
        self._range_noise_prop = float(range_noise_proportional)
        self._angular_noise_rad = math.radians(float(angular_noise_deg))
        self._fn_rate = float(false_negative_rate)
        self._snr_threshold = float(snr_threshold)

        # Pre-computed nominal angles (offsets from heading).
        self._base_offsets = np.linspace(
            0.0, 2.0 * math.pi, self._n_rays, endpoint=False
        )

    # -- Properties ------------------------------------------------------

    @property
    def n_rays(self) -> int:
        return self._n_rays

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
        extinction: np.ndarray,
        cell_size_m: float,
    ) -> LiDARScan:
        """One full LiDAR sweep.

        Parameters
        ----------
        agent_x, agent_y:
            Agent position in array space (column, row).
        heading:
            Heading in radians, CCW from +x (matches
            :mod:`flare.core.wind` TO-direction convention).
        occupancy:
            ``int8[H, W]`` world occupancy (any positive value blocks).
        extinction:
            ``float32[H, W]`` smoke extinction field σ_ext (1/m); usually
            from :class:`flare.core.hazards.smoke.GaussianSmokePlume`.
        cell_size_m:
            World cell size in metres.
        """
        if cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be > 0, got {cell_size_m}")

        # Angular noise per ray
        ang_noise = self._rng.normal(
            0.0, self._angular_noise_rad, self._n_rays
        )
        actual_angles = (
            heading + self._base_offsets + ang_noise
        ).astype(np.float64)

        max_range_cells = int(math.ceil(self._max_range_m / cell_size_m))

        hits_x, hits_y, dist_cells, hit_types = cast_rays(
            agent_x,
            agent_y,
            actual_angles,
            max_range_cells,
            occupancy,
            extinction,
            cell_size_m,
            self._snr_threshold,
        )

        # Convert distance to metres and clip to LiDAR's max range.
        dist_m = np.minimum(
            dist_cells.astype(np.float32) * np.float32(cell_size_m),
            np.float32(self._max_range_m),
        )

        # Range-dependent Gaussian noise
        range_sigma = (
            np.float32(self._range_noise_mm_fixed * 1e-3)
            + np.float32(self._range_noise_prop) * dist_m
        )
        z = self._rng.standard_normal(self._n_rays).astype(np.float32)
        dist_m_noisy = np.maximum(dist_m + z * range_sigma, 0.0).astype(
            np.float32
        )

        # Per-beam false-negative dropout
        valid = self._rng.random(self._n_rays) >= self._fn_rate

        return LiDARScan(
            angles=actual_angles,
            distances_m=dist_m_noisy,
            hit_x=hits_x,
            hit_y=hits_y,
            hit_types=hit_types,
            valid=valid,
        )
