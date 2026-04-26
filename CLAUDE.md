# CLAUDE.md — FLARE-PO: Paper 2 Implementation
## Ο εγκέφαλος του project. Διάβασε ΟΛΟΚΛΗΡΟ πριν γράψεις μία γραμμή κώδικα.

---

## ΤΙ ΧΤΙΖΟΥΜΕ

Paper 2: **"Blind to the Blaze: Mission-Aware Risk-Adaptive Planning under Sensor-Only Partial Observability Resolves the Ranking Inversion in Wildfire UAV Emergency Response"**

Σε μία πρόταση: Ένα POMDP wildfire benchmark όπου ο simulated UAV βλέπει μόνο μέσω modeled sensors (12m LiDAR, camera, GPS ±2m), χτίζει belief εν πτήσει, μαθαίνει adaptive risk coefficient ρ(s,t) conditioned σε calibrated mission physics, και ΛΥΝΕΙ τη ranking inversion του Paper 1 — ανακαλύπτοντας ταυτόχρονα δεύτερη.

**Target venue:** NeurIPS 2026 ή ICRA 2027.

**Timeline:** 4 μήνες μέχρι submission.

**ΑΥΣΤΗΡΑ 2D simulation.** Κανένα real drone. Κανένος δορυφόρος. Κανένα MAVLink. Κανένα hardware deployment. Κανένα εξωτερικό API στο runtime.

---

## ΠΟΙΟΣ ΔΟΥΛΕΥΕΙ

Κωνσταντίνος Ζερβάκης, PhD candidate, Στρατιωτική Σχολή Ευελπίδων.
Supervisor: Αναπλ. Καθηγητής Ηλίας Παναγιωτόπουλος.

---

## LOCAL PATHS

| Τι | Πού |
|----|-----|
| **Working dir (Paper 2)** | `/Users/konstantinos/Dev/Flare_N` |
| **Paper 1 codebase (FLARE)** | `/Users/konstantinos/Dev/planning/uavbench` (git repo, module: `src/flare/`) |
| **Paper drafts** | `/Users/konstantinos/Dev/planning/UAV_v21_final.docx` (latest) |
| **Sensor-paper draft** | `/Users/konstantinos/Dev/planning/FLARE_Sensors_Paper.docx` |

---

## HARDWARE

**Τώρα έχουμε ΜΟΝΟ MacBook M1.** Το RTX 3060 PC θα έρθει αργότερα.
Όλος ο κώδικας ΠΡΕΠΕΙ να τρέχει σε M1. Το heavy training (Phase 5)
θα γίνει στο PC όταν έρθει — αλλά ΟΛΑ τα υπόλοιπα (env, sensors,
belief, missions, tests, MCP server) χτίζονται και τεστάρονται σε M1.

| Πόρος | Specs | Χρήση |
|-------|-------|-------|
| **Dev + Test (NOW)** | MacBook M1, 16GB | Phases 1-4 + 6 (analysis/MCP) |
| Training (LATER) | RTX 3060 12GB, Ryzen 7 5700X | Phase 5 (MARS-RL training) |
| Python | 3.10+ | `python3 -m venv .venv` |
| GPU framework | PyTorch 2.x (**MPS backend** for M1) | Use `device="mps"` not "cuda" |
| RL library | SB3 + sb3-contrib (RecurrentPPO) | CPU training OK for small runs |
| GIS | osmnx, rasterio, geopandas, pyproj | Scenario generation |
| Env standard | Gymnasium 0.29+ | NOT old gym |

**M1-specific rules:**
- `torch.device("mps")` for GPU-accelerated ops on M1 (Metal Performance Shaders)
- Numba works on ARM64 — `@njit` OK
- No CUDA imports — use `device = "mps" if torch.backends.mps.is_available() else "cpu"`
- Small training runs (1M steps, 4 parallel envs) feasible on M1 for debugging
- Full training (10M steps, 32 envs) waits for RTX 3060
- PennyLane `default.qubit` simulator runs fine on M1
- All tests must pass on M1 — no exceptions

---

## SETUP — Run EXACTLY these commands

```bash
# 1. Working directory: /Users/konstantinos/Dev/Flare_N (already exists)
cd /Users/konstantinos/Dev/Flare_N
# git init -b main   # already done

# 2. Python environment
python3 -m venv .venv      # already done
source .venv/bin/activate
pip install --upgrade pip

# 3. Install project + core dependencies (uses pyproject.toml)
pip install -e ".[dev,gis,ml]"

# 4. (Optional) MCP server deps for Phase 6
pip install -e ".[mcp]"

# 5. Verify installation
python3 -c "
import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')
import gymnasium; print(f'Gymnasium {gymnasium.__version__}')
import stable_baselines3; print(f'SB3 {stable_baselines3.__version__}')
import numba; print(f'Numba {numba.__version__}')
import osmnx; print(f'OSMnx {osmnx.__version__}')
"

# 6. Run smoke test
pytest tests/ -q
```

The package structure (created during initial setup):

```
flare-po/
├── flare/{core/hazards,missions,sensors,belief,envs,planners,learning,scenarios,analysis,mcp}
├── configs/scenarios
├── tests
├── scripts
└── checkpoints/mars_rl
```

---

## REFERENCE REPOS — Study these, don't reinvent

USE AS REFERENCE (study patterns, don't copy code):

| Repo | URL | What to learn |
|------|-----|---------------|
| PyroRL (Stanford SISL) | github.com/sisl/PyroRL | Gymnasium wildfire env structure, JOSS-published, grid-based fire spread |
| gym-cellular-automata | github.com/elbecerrasoto/gym-cellular-automata | CA env with Gymnasium API, benchmark/prototype modes |
| PythonRobotics | github.com/AtsushiSakai/PythonRobotics | `lidar_to_grid_map.py` for Bresenham ray-cast + occupancy grid |
| range_libc (MIT) | github.com/kctess5/range_libc | Fast 2D ray casting with Python wrappers, Bresenham + CDDT |
| gym_forestfire | github.com/sahandrez/gym_forestfire | Vectorized fire CA with CNN-based RL (TD3) |
| SB3 RecurrentPPO | sb3-contrib.readthedocs.io | RecurrentPPO examples with LSTM/GRU policies |

**Paper 1 in-tree reference:** `/Users/konstantinos/Dev/planning/uavbench` — port `dynamics/fire_ca.py` to `flare/core/hazards/fire_deterministic.py`; reuse contract test patterns from `tests/contract_test_*.py`.

Key patterns to adopt from PyroRL:
- `pyproject.toml` with `[project.optional-dependencies]` for dev/test
- `gymnasium.register()` in `__init__.py`
- Render mode support (`"human"`, `"rgb_array"`)
- Clean separation: env logic vs rendering vs scenario config

Key patterns to adopt from PythonRobotics:
- Numba-jitted Bresenham: `bresenham((x0,y0), (x1,y1))` returns cell list
- Inverse sensor model for log-odds occupancy update
- Flood-fill for unknown area detection

---

## PRODUCTION PATTERNS — Follow these or waste tokens

### Pattern 1: Every file starts with this template

```python
"""
flare/module/file.py — One-line description.

Part of FLARE-PO Paper 2.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
```

### Pattern 2: Constants with citations

```python
# WRONG — reviewer can't verify
LIDAR_RANGE = 12.0

# RIGHT — traceable to source
LIDAR_MAX_RANGE_M: float = 12.0
"""LD19 effective range. Source: LDRobot DTOF LD19 spec sheet,
Waveshare wiki, Little Bird Electronics product listing."""
```

### Pattern 3: Numba for hot paths

```python
from numba import njit

@njit(cache=True)
def bresenham_ray(x0: int, y0: int, angle: float,
                  max_range: int, grid: np.ndarray) -> tuple:
    """
    Cast single Bresenham ray on occupancy grid.
    Returns (hit_x, hit_y, distance, hit_type).

    MUST be @njit — called 450 times per step.
    Target: <1ms for full 450-ray sweep.
    """
    dx = np.cos(angle)
    dy = np.sin(angle)
    x, y = float(x0), float(y0)
    for step in range(max_range):
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= grid.shape[0] or yi < 0 or yi >= grid.shape[1]:
            return xi, yi, step, 0  # out of bounds
        if grid[xi, yi] > 0:
            return xi, yi, step, int(grid[xi, yi])  # hit
        x += dx
        y += dy
    return int(round(x)), int(round(y)), max_range, 0  # no hit
```

### Pattern 4: Gymnasium env checklist

```python
from gymnasium.utils.env_checker import check_env
env = FlarePOEnv(scenario="penteli", observability=0.6)
check_env(env)  # MUST pass — tests obs/act spaces, reset/step API

assert env.observation_space.contains(env.reset()[0])
assert env.action_space.contains(0)
obs, reward, terminated, truncated, info = env.step(0)
assert env.observation_space.contains(obs)
assert isinstance(reward, float)
assert isinstance(terminated, bool)
assert isinstance(truncated, bool)
assert isinstance(info, dict)
```

### Pattern 5: Seeded randomness (Invariant I2)

```python
# WRONG — global state, not reproducible
import random
random.seed(42)
np.random.seed(42)

# RIGHT — isolated generator, reproducible
class FlarePOEnv(gymnasium.Env):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        # ALL randomness in this episode flows from self._rng
        self.fire.rng = self._rng
        self.sensors.rng = self._rng
```

### Pattern 6: Config-driven experiments

```yaml
# configs/paper2_fast.yaml (M1 debug)
env:
  grid_size: 500
  cell_size_m: 3.0
  observability: 0.6
  fire_mode: stochastic   # or "deterministic" for P1 compatibility
  wind_speed: 5.0
  wind_dir: 0.0

training:
  algorithm: recurrent_ppo
  n_envs: 4               # M1: 4, PC: 32
  total_steps: 1_000_000  # M1: 1M, PC: 10M
  device: auto            # auto-detects mps/cuda/cpu

evaluation:
  seeds: [0, 1, 2]        # M1: 3, PC: 30
  scenarios: [penteli]    # M1: 1, PC: 5
  difficulty: [0.0, 0.6]  # M1: 2, PC: 6
```

### Pattern 7: Test alongside code

```python
# For EVERY new module flare/foo/bar.py, create tests/test_bar.py
# Run tests after EVERY phase:
# pytest tests/ -x --tb=short

# Critical tests that ALWAYS run:
# pytest tests/test_mdp_continuity.py  — Invariant I1
# pytest tests/test_env.py             — Gymnasium compliance
```

---

## COMMON PITFALLS — Read before coding

| Pitfall | Consequence | Prevention |
|---------|-------------|------------|
| Using `gym` instead of `gymnasium` | API mismatch, SB3 crashes | `import gymnasium as gym` ONLY |
| `np.random.seed()` global state | Non-reproducible across runs | Use `np.random.Generator(PCG64(seed))` |
| Fire CA not vectorized | 10× slower training | Use `torch` for batched fire, numba for single |
| Dict obs space with wrong shapes | SB3 training silently fails | `check_env()` after every obs change |
| Missing terminated vs truncated | Wrong episode boundaries | `terminated=death/goal`, `truncated=timeout` |
| CUDA imports on M1 | ImportError | Always `device = "mps" if mps else "cpu"` |
| Belief map wrong dtype | NaN in training | Always `float32`, clip to `[0,1]` |
| Ray-cast in pure Python | 100× too slow | `@njit` mandatory for Bresenham |
| A* on full 500×500 grid | Slow planning | A* on 50×50 downsampled belief map |
| osmnx without internet | Scenario creation fails | Cache scenarios as `.npz` after first download |

---

## ΤΙ ΕΧΟΥΜΕ ΗΔΗ (Paper 1 — DO NOT BREAK)

Paper 1 (FLARE) παρέχει:
- 500×500 OSM-derived grid (3m cells)
- Deterministic Alexandridis CA fire
- 3 coupled hazard layers (fire → collapse → traffic)
- 5 planners: A* (ρ=0), Periodic (ρ=5.0), Aggressive (ρ=0.5), Incremental A* (ρ=2.0), APF (ρ=3.0)
- 3 scenarios: Penteli (insulin), Piraeus (SAR), Downtown (surveillance)
- 3 M(t): quadratic, exponential+fire coupling, linear
- 36 reproducibility contracts, 189 automated tests
- Friedman/Wilcoxon/Cliff's δ statistical pipeline
- Central result: ranking inversion (Friedman χ²(4)=190.8, p<0.001, W=0.57)

**Codebase location:** `/Users/konstantinos/Dev/planning/uavbench`. Paper 1 module path is `src/flare/`; key files for Paper 2 reference:
- `src/flare/dynamics/fire_ca.py` → port to `flare/core/hazards/fire_deterministic.py`
- `src/flare/dynamics/collapse.py` → port to `flare/core/hazards/collapse.py`
- `src/flare/dynamics/traffic.py` → port to `flare/core/hazards/traffic.py`
- `src/flare/planners/{astar,periodic_replan,aggressive_replan,incremental_astar,apf}.py` → port (rename: drop `_replan`/`_astar` suffixes)
- `src/flare/blocking.py` → fold into `flare/core/risk.py` + `flare/core/cost.py`
- `tests/contract_test_determinism.py` → mine pattern for `tests/test_mdp_continuity.py`

**INVARIANT I1:** Στο observability d=0.0 (full obs), το FLARE-PO env
ΠΡΕΠΕΙ να παράγει ΙΔΙΑ αποτελέσματα με το Paper 1 MDP. Αυτό τεστάρεται
σε κάθε commit: `pytest tests/test_mdp_continuity.py`

---

## PROJECT STRUCTURE

```
flare-po/
├── CLAUDE.md                    # THIS FILE
├── pyproject.toml               # pip install -e ".[dev,gis,ml]"
├── README.md
├── LICENSE (MIT)
│
├── flare/
│   ├── __init__.py
│   │
│   ├── core/                    # PAPER 1 FOUNDATION — touch with extreme care
│   │   ├── grid.py              # 500×500 grid world
│   │   ├── hazards/
│   │   │   ├── fire_deterministic.py   # Paper 1 Alexandridis CA
│   │   │   ├── fire_stochastic.py      # NEW: Beta(α,β) spread
│   │   │   ├── collapse.py             # τ_c=80, p_d=0.6
│   │   │   ├── traffic.py              # Dynamic road closures
│   │   │   ├── nfz.py                  # NEW: Dynamic no-fly zones
│   │   │   └── smoke.py                # NEW: Gaussian plume + advection
│   │   ├── risk.py              # R(x) = max(R_f, R_s, R_t, R_b, R_nfz)
│   │   ├── cost.py              # w(x) = 1 + ρ·R(x)
│   │   └── wind.py              # NEW: Anisotropic wind kernel
│   │
│   ├── missions/                # NEW: calibrated M(t) — 8 types
│   │   ├── base.py              # MissionDecay ABC
│   │   ├── insulin.py           # Arrhenius E(t)=exp(-(k(T)·t)^0.5)
│   │   ├── epinephrine.py       # Arrhenius 1st-order
│   │   ├── hemorrhage.py        # Weibull S(t)=exp[-(λt)^k]
│   │   ├── smoke_inhalation.py  # CFK sigmoid ∫C_CO → 40% COHb
│   │   ├── surveillance.py      # exp(-t/τ_heat), τ=2h
│   │   ├── crush.py             # Logistic S(t; t₅₀=48h)
│   │   ├── blood.py             # Piecewise linear FVIII 4h knee
│   │   ├── temperature.py       # T_eff(d) = T_amb + 40·exp(-d/15m)
│   │   └── triage.py            # Multi-POI: U = Σ wᵢ·Sᵢ(τᵢ)
│   │
│   ├── sensors/                 # NEW: all calibrated to real specs
│   │   ├── lidar.py             # LD19: 12m, 450 Bresenham rays
│   │   ├── camera.py            # 130° HFOV, 30m, fire detection
│   │   ├── gps.py               # BN-880: CEP 2.0m, dropout model
│   │   ├── smoke_visibility.py  # Jin V=K/σ_ext, K=3, α_m=8700
│   │   └── ray_cast.py          # Numba Bresenham + occlusion
│   │
│   ├── belief/                  # NEW: belief state management
│   │   ├── occupancy_grid.py    # Per-cell Bayesian log-odds
│   │   ├── certainty_decay.py   # c(t)=c(t_obs)·exp(-Δt/τ)
│   │   ├── memory_map.py        # 4×50×50 global downsampled
│   │   └── particle_filter.py   # Classical baseline (N particles)
│   │
│   ├── envs/                    # Gymnasium environments
│   │   ├── flare_mdp.py         # Paper 1 env (wrapper for continuity)
│   │   └── flare_pomdp.py       # Paper 2: THE MAIN ENV
│   │
│   ├── planners/                # 8 planners
│   │   ├── base.py              # Planner ABC
│   │   ├── astar.py             # P1: ρ=0
│   │   ├── periodic.py          # P1: ρ=5.0
│   │   ├── aggressive.py        # P1: ρ=0.5
│   │   ├── incremental.py       # P1: ρ=2.0
│   │   ├── apf.py               # P1: ρ=3.0
│   │   ├── frontier_astar.py    # NEW: explore before commit
│   │   ├── belief_astar.py      # NEW: certainty-penalized
│   │   └── mars_rl.py           # NEW: THE LEARNED AGENT
│   │
│   ├── learning/                # RL training
│   │   ├── networks.py          # CNN+GRU+FiLM+IQN architecture
│   │   ├── train.py             # RecurrentPPO training loop
│   │   ├── curriculum.py        # Easy→Medium→Hard→Mixed
│   │   └── evaluation.py        # Standardized eval protocol
│   │
│   ├── scenarios/               # Map scenarios
│   │   ├── penteli.py           # P1 (insulin)
│   │   ├── piraeus.py           # P1 (SAR)
│   │   ├── downtown.py          # P1 (surveillance)
│   │   ├── mati_2018.py         # NEW: historical reconstruction
│   │   ├── multi_triage.py      # NEW: multi-casualty
│   │   └── generator.py         # Any-map: osmnx/rasterio → grid
│   │
│   ├── analysis/                # Statistical analysis + figures
│   │   ├── ranking_inversion.py # Friedman/Wilcoxon/Cliff's δ
│   │   ├── phase_diagram.py     # 2D heatmap (obs × hazard)
│   │   ├── pareto.py            # Pareto frontier
│   │   └── figures.py           # Publication-quality matplotlib
│   │
│   └── mcp/                     # MCP server for AI accessibility
│       ├── server.py            # MCP tool definitions
│       └── tools.py             # create_scenario, run_planner, etc.
│
├── configs/                     # YAML experiment configs
│   ├── paper2_full.yaml         # 7,200 episodes config
│   ├── paper2_fast.yaml         # Quick debug config
│   └── scenarios/               # Per-scenario configs
│
├── scripts/
│   ├── run_benchmark.py         # CLI entry point
│   ├── train.py                 # Training entry point
│   └── analyze.py               # Analysis entry point
│
├── tests/
│   ├── test_mdp_continuity.py   # CRITICAL: d=0 = Paper 1
│   ├── test_sensors.py          # Sensor model unit tests
│   ├── test_missions.py         # M(t) calibration tests
│   ├── test_belief.py           # Belief update tests
│   ├── test_fire.py             # Stochastic CA tests
│   └── test_env.py              # Gymnasium compliance
│
└── checkpoints/                 # Pretrained models
    └── mars_rl/
```

---

## IMPLEMENTATION PHASES — ΑΚΟΛΟΥΘΗΣΕ ΑΥΣΤΗΡΑ

**M1 roadmap (τώρα):** Phases 1-4 + Phase 6 (MCP/analysis) = ΟΛΑ σε M1
**PC roadmap (όταν έρθει):** Phase 5 (full training 10M steps)

Phases 1-4 είναι το 70% της δουλειάς — env, sensors, belief, missions.
Αυτά τρέχουν 100% σε M1. Χτίζουμε τώρα, εκπαιδεύουμε μετά.

### PHASE 1: Stochastic Fire + Wind + Smoke (Εβδομάδες 1-2)

**1.1 Stochastic CA Fire** → `flare/core/hazards/fire_stochastic.py`

```python
class StochasticFireCA:
    """
    Probabilistic Alexandridis CA.
    Paper 1 used deterministic spread. Now each cell has
    p_spread ~ Beta(α, β) instead of binary.

    At β→∞: converges to deterministic (Paper 1 behavior).
    This is critical for Invariant I1 (MDP continuity).
    """
    def __init__(self, grid, alpha=2.0, beta=5.0, wind_speed=5.0,
                 wind_dir=0.0, seed=None):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        # All randomness from this single RNG

    def step(self):
        """One CA timestep. Must be vectorizable for 32 parallel envs."""
        # for each burning cell (i,j):
        #   for each neighbor (ni, nj) in Moore neighbourhood:
        #     p_base = self.rng.beta(self.alpha, self.beta)
        #     p_wind = 1 + self.c_w * np.cos(self.wind_dir - angle(i,j,ni,nj))
        #     p_slope = 1 + self.c_s * np.tan(slope(i,j,ni,nj))
        #     p_fuel = fuel_moisture_factor(ni, nj)
        #     p_spread = p_base * p_wind * p_slope * p_fuel
        #     if self.rng.random() < p_spread:
        #       ignite(ni, nj)
```

**CRITICAL:** Use `torch` for batched fire stepping (32 parallel envs on GPU).
Numba `@njit` for single-env CPU fallback.

**1.2 Wind Kernel** → `flare/core/wind.py`

```python
# Anisotropic spread: fire moves faster downwind
# Paper 1 had wind_speed=0 (disabled). Now enabled.
# ρ(θ) = 1 + c_w · cos(θ_wind - θ_ij)
# c_w = 0.5 (moderate) to 2.0 (strong wind)
```

**1.3 Smoke Plume** → `flare/core/hazards/smoke.py`

```python
class SmokePlume:
    """
    Per burning cell: Gaussian plume downwind.
    ρ_smoke(x,y) = Σ_burning Q/(2π·σ_x·σ_y) · exp(...)
    σ_ext(x,y) = α_m · ρ_smoke(x,y)
    α_m = 8700 m²/kg (FDS, mass-specific extinction)

    This field couples to sensor visibility (Phase 2).
    """
```

**1.4 Dynamic NFZ** → `flare/core/hazards/nfz.py`

```python
class DynamicNFZ:
    """
    Paper 1: num_nfz_zones=0 (disabled).
    Paper 2: stochastic NFZ appear/disappear.
    Represents helicopter corridors, firefighting aircraft paths.
    Agent discovers NFZ only when within sensor range.
    """
    def __init__(self, max_zones=3, mean_duration=50, seed=None):
        ...
    def step(self):
        # Poisson arrival, geometric duration
        ...
```

**Tests after Phase 1:**
- `test_fire.py`: stochastic CA produces different outcomes per seed
- `test_fire.py`: at β→∞, stochastic CA matches deterministic (Invariant I1)
- `test_fire.py`: wind kernel produces anisotropic spread
- `test_fire.py`: smoke field non-zero downwind of fire

---

### PHASE 2: Sensor Models (Εβδομάδες 2-3)

**2.1 LiDAR** → `flare/sensors/lidar.py`

```python
# LDRobot DTOF LD19 specs:
LIDAR_MAX_RANGE_M = 12.0        # 12m effective range
LIDAR_NUM_RAYS = 450            # 360° / 0.8° = 450 rays
LIDAR_RANGE_NOISE_MM = 10       # σ_r = 10mm + 0.001·r
LIDAR_ANGULAR_NOISE_DEG = 2.0   # σ_θ = 2°
LIDAR_FALSE_NEGATIVE = 0.02     # p_FN per sweep

def ray_cast(agent_pos, heading, grid, smoke_field):
    """
    Bresenham ray-cast from agent position.
    - Stop on first occupied cell (building occlusion)
    - Smoke reduces effective range:
      r_eff = min(12, ln(SNR_threshold) / (2·σ_ext))
      σ_ext = 8700 · ρ_smoke
    - At σ_ext=0.029: r_eff halved to 6m
    - At σ_ext=0.1: r_eff = 2m
    - At σ_ext>0.5: r_eff < 1m (effectively blind)

    MUST use Numba @njit for performance.
    Target: <1ms for 450 rays on 500×500 grid.
    """
```

**2.2 GPS** → `flare/sensors/gps.py`

```python
# Beitian BN-880 specs:
GPS_CEP_M = 2.0                 # Circular Error Probable
GPS_SIGMA_PER_AXIS = 1.4        # CEP / sqrt(2) ≈ 1.4m
COMPASS_SIGMA_DEG = 1.5         # HMC5883L heading noise

def gps_reading(true_pos, true_heading, building_density, rng):
    """
    - Gaussian noise: σ = 1.4m per axis
    - Heading noise: σ_ψ = 1.5°
    - Urban canyon dropout: p_dropout = sigmoid(building_density - 0.4)
    - During dropout: dead-reckoning drift σ_dr = 0.5m/s
    """
```

**2.3 Camera** → `flare/sensors/camera.py`

```python
# Generic FPV camera specs:
CAMERA_HFOV_DEG = 130.0         # Horizontal FOV
CAMERA_MAX_RANGE_M = 30.0       # Clean air
# Smoke degrades: r_cam = K / σ_ext (Jin, K=3 reflective)
# NOT omnidirectional — sector only (heading ± 65°)
```

**Tests after Phase 2:**
- `test_sensors.py`: LiDAR returns empty beyond 12m
- `test_sensors.py`: buildings block LiDAR rays
- `test_sensors.py`: smoke reduces effective range
- `test_sensors.py`: GPS noise σ ≈ 1.4m empirically
- `test_sensors.py`: camera sector 130° only

---

### PHASE 3: Belief + POMDP Env (Εβδομάδες 3-4)

**3.1 Belief System** → `flare/belief/`

```python
class BeliefMap:
    """
    4-channel 50×50 global downsampled belief:
    Channel 0: static occupancy (buildings) — from OSM prior (5% error)
    Channel 1: fire probability — Bayesian update inside sensor range
    Channel 2: traffic/debris probability — same
    Channel 3: certainty — how recently observed

    Certainty decay (PyroTrack-style):
    c(x,t) = c(x,t_obs) · exp(-(t - t_obs) / τ_fire)

    Fire can spread into "safe" areas while agent isn't looking.
    """
```

**3.2 FLARE-PO Gymnasium Env** → `flare/envs/flare_pomdp.py`

```python
class FlarePOEnv(gymnasium.Env):
    """THE MAIN ENVIRONMENT. This is what gets published."""

    observation_space = Dict({
        "lidar_occupancy":  Box(0, 1, (25, 25)),    # ego-centric LiDAR
        "fire_detection":   Box(0, 1, (25, 25)),    # camera fire detect
        "smoke_density":    Box(0, 100, (25, 25)),  # smoke estimate
        "belief_map":       Box(0, 1, (4, 50, 50)), # global memory
        "pose":             Box(-inf, inf, (3,)),   # x_noisy, y_noisy, ψ
        "mission_ctx":      Box(-inf, inf, (5,)),   # T_rem, λ_eff, task[3]
        "goal_vector":      Box(-1, 1, (3,)),       # cos θ, sin θ, dist
    })

    action_space = Discrete(10)
    # {N, NE, E, SE, S, SW, W, NW, HOVER, SCAN360}

    def __init__(self, scenario, observability=0.6, mission="insulin",
                 wind_speed=5.0, wind_dir=0.0, seed=None):
        """
        observability ∈ [0, 1]:
          0.0 = full obs (MDP, Paper 1 compatible) — MUST pass I1 test
          0.2 = light fog: LiDAR 10m, camera 25m, no GPS noise
          0.4 = moderate: LiDAR 8m, camera 15m, σ_GPS=2m
          0.6 = heavy smoke: LiDAR 5m, camera 8m, σ_GPS=3m
          0.8 = dense: LiDAR 3m, camera 4m, σ_GPS=5m
          1.0 = near-blind: LiDAR 1m, camera 2m, GPS sporadic
        """

    def reset(self, seed=None):
        # 1. Initialize grid from scenario
        # 2. Place fire, set wind
        # 3. Initialize belief from OSM prior (5% corruption)
        # 4. Return initial observation
        ...

    def step(self, action):
        # 1. Execute movement (or HOVER/SCAN)
        # 2. Step stochastic fire CA
        # 3. Step smoke plume
        # 4. Step collapse cascade
        # 5. Step dynamic traffic + NFZ
        # 6. Generate observation via sensor models
        # 7. Update belief
        # 8. Compute reward
        # 9. Check termination
        ...

    # CRITICAL: at observability=0.0, this MUST produce identical
    # trajectories to FlareMDPEnv for same planner + seed.
```

**Tests after Phase 3:**
- `test_mdp_continuity.py`: d=0 matches Paper 1 EXACTLY
- `test_env.py`: Gymnasium API compliance (check_env passes)
- `test_env.py`: observation shapes correct
- `test_env.py`: 500+ steps/second with 32 parallel envs

---

### PHASE 4: Calibrated Missions (Εβδομάδα 4)

**Κάθε M(t) implements:**

```python
class MissionDecay(ABC):
    @abstractmethod
    def value(self, t: float, d_fire: float, T_ambient: float) -> float:
        """Mission value ∈ [0, 1] at time t."""

    @abstractmethod
    def is_failed(self, value: float) -> bool:
        """Clinically/operationally failed?"""

    @property
    @abstractmethod
    def citation(self) -> str:
        """Primary peer-reviewed source."""
```

**Constants (DO NOT CHANGE without citation update):**

```python
# === Insulin (Arrhenius) ===
# Source: Sadrzadeh J.Pharm.Sci 2024; Beals Diabetes Care 2024
INSULIN_EA_KJ = 100.0           # Activation energy kJ/mol
INSULIN_RATE_RATIO = 10.0       # k(37°C) / k(25°C)
INSULIN_USP_CUTOFF = 0.95       # 95% potency = clinical fail
# Form: E(t) = exp(-(k(T_eff)·t)^0.5)

# === Hemorrhage (Weibull) ===
# Source: Sampalis J.Trauma 1993; Kragh Ann.Surg 2009
HEMORRHAGE_LAMBDA = 0.05        # 5% mortality/min (Sampalis)
HEMORRHAGE_WEIBULL_K = 1.3      # Shape parameter
# Form: S(t) = exp[-(λ_eff·t)^k]

# === CO Inhalation (CFK) ===
# Source: NCBI NBK220007; COHb elimination curve
CO_LETHALITY_COHB = 40.0        # % COHb = lethal threshold
# Form: sigmoid on ∫C_CO(s)ds

# === Temperature Coupling (Butler-Cohen) ===
# Source: USFS RP-INT-497 1998; PMC 9641190
TEMP_AMBIENT = 25.0             # °C
TEMP_FIRE_DELTA = 40.0          # °C at flame contact
TEMP_DECAY_DIST = 15.0          # m, exponential decay
# Form: T_eff(d) = T_amb + 40·exp(-d/15)

# === Crush Rescue (Logistic) ===
# Source: PMC 11325850 (Türkiye-Syria 2023, N=377)
CRUSH_T50_HOURS = 48.0          # Median survival time
CRUSH_K = 0.05                  # Steepness parameter
```

---

### PHASE 5: MARS-RL Agent (Εβδομάδες 5-8)

**5.1 Network Architecture** → `flare/learning/networks.py`

```python
class MARSPolicy(nn.Module):
    """
    Mission-Aware Risk-Sensitive RL Policy.

    Flow:
    [LiDAR 25×25 + fire 25×25 + smoke 25×25]
        → CNN (Conv2d×2 + AdaptiveAvgPool) → 1600-d
    [mission_ctx 5-d + goal 3-d + pose 3-d]
        → MLP (Linear×2) → 64-d
    [mission_ctx 5-d]
        → FiLM (Linear → γ, β) → scale/shift merged features
    [merged 1664-d]
        → GRU-256 (recurrent memory for POMDP belief)
    [GRU output 256-d]
        → IQN critic (32 quantile τ values)
        → Actor head → action ∈ Discrete(10)
        → Rho head → ρ̂ ∈ [0, 10] (continuous risk dial)

    NOVEL: ρ̂ as learned output. First time in literature.
    """

    def __init__(self):
        # CNN for spatial inputs
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(5),
            nn.Flatten()  # → 64*25 = 1600
        )
        # MLP for scalar inputs
        self.mlp = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        # FiLM conditioning from mission_ctx
        self.film_gamma = nn.Linear(5, 1664)
        self.film_beta = nn.Linear(5, 1664)
        # GRU for temporal memory
        self.gru = nn.GRU(1664, 256, batch_first=True)
        # Actor: action + ρ̂
        self.action_head = nn.Linear(256, 10)
        self.rho_head = nn.Sequential(
            nn.Linear(256, 1), nn.Sigmoid()  # → [0,1] * 10 = [0,10]
        )
        # IQN critic
        self.quantile_embed = nn.Linear(64, 256)
        self.critic = nn.Linear(256, 1)
```

**5.2 Hierarchical Execution**

```python
# MARS-RL does NOT plan end-to-end.
# RL = "slow brain": decides ρ̂ and whether to SCAN/HOVER
# A* = "fast brain": executes path with w(x) = 1 + ρ̂ · R_belief(x)
#
# This is 10× more sample-efficient than end-to-end RL
# and interpretable: you can plot ρ̂ curve per episode.
```

**5.3 Training** → `flare/learning/train.py`

```python
# Device selection — M1 or RTX 3060
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"       # Mac M1 Metal
elif torch.cuda.is_available():
    DEVICE = "cuda"      # RTX 3060 when available
else:
    DEVICE = "cpu"

# === M1 DEBUG TRAINING (NOW) ===
# 4 parallel envs, 1M steps, ~4-8 hours on M1
# Purpose: verify training pipeline works, policy improves
# NOT for publication — just for debugging
# Config: configs/paper2_fast.yaml

# === FULL TRAINING (WHEN PC ARRIVES) ===
# RecurrentPPO from sb3-contrib
# 32 parallel SubprocVecEnv
# 10M steps, ~36h/seed on RTX 3060, 5 seeds
# Config: configs/paper2_full.yaml

# Reward: M(t_arrival) if success, -1 if dead,
#         -0.01/step, +0.001·|ΔKnown|
# Curriculum: low smoke → medium → high → mixed
# Wandb logging
#
# Generalization test: train Penteli+Piraeus, test Downtown+Mati
```

**5.4 M1 vs PC Training Configs**

```yaml
# configs/paper2_fast.yaml (M1 — debug)
n_envs: 4
total_steps: 1_000_000
device: "mps"
scenarios: ["penteli"]       # single scenario for speed
seeds: [0, 1, 2]            # 3 seeds only
observability: [0.0, 0.6]   # 2 levels only

# configs/paper2_full.yaml (RTX 3060 — publication)
n_envs: 32
total_steps: 10_000_000
device: "cuda"
scenarios: ["penteli", "piraeus", "downtown", "mati", "triage"]
seeds: [0..29]              # 30 seeds
observability: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 6 levels
```

---

### PHASE 6: Experiments + Paper (Εβδομάδες 9-12)

**6.1 Run 7,200 Episodes (NEEDS PC for full run)**

```
FULL (PC — publication):
8 planners × 5 scenarios × 6 difficulty × 30 seeds = 7,200
Classical planners: CPU (~0.1s/episode)
MARS-RL: GPU inference (~0.5s/episode)
Total: ~2-4 hours with 32 parallel envs

MINI (M1 — validation):
8 planners × 2 scenarios × 2 difficulty × 5 seeds = 160
Enough to verify pipeline works end-to-end
Total: ~10 minutes

Output: results.parquet
Columns: [planner, scenario, difficulty, seed, SR, MS, time,
          info_regret, ccfr, scan_rate, pareto_rank,
          memory_mb, ms_per_step]
```

**6.2 Figures to Generate**

1. **Phase diagram** — 2D heatmap (d × hazard) showing which planner ranks #1 in MS
2. **Ranking inversion bars** — SR rank vs MS rank at d=0 and d=0.6
3. **ρ̂(t) trajectory** — one episode showing adaptive risk behavior
4. **SCAN usage vs difficulty** — emergence curve
5. **Calibrated vs P1 M(t) effect** — |δ| comparison
6. **Leave-one-out generalization** — table
7. **Pareto frontier** — SR vs MS per difficulty

---

## 5 INVARIANTS — NEVER BREAK

**I1: MDP Continuity** — At d=0.0, FLARE-PO = Paper 1 FLARE. Same trajectories, same SR, same MS. Tested every commit.

**I2: Deterministic Replay** — Same seed → same trajectory. ALL randomness from `np.random.Generator(seed)`. No global state.

**I3: Gymnasium API** — `check_env(FlarePOEnv())` passes. Compatible with SB3, CleanRL, any Gymnasium framework.

**I4: Separation of Concerns** — Environment never knows planner. Planner never accesses true state. Belief is updated by env wrapper. Mission decay computed from true state, reported as reward.

**I5: No External APIs at Runtime** — OSMnx requires internet at scenario CREATION. After that, scenario is self-contained `.npz` file. No NASA, no EFFIS, no weather API during benchmark.

---

## CODING STANDARDS

- Python 3.10+, type hints on every function
- Black formatter, 88 chars
- isort imports
- Google-style docstrings
- Numba `@njit` for ray-casting and CA stepping
- PyTorch for batched operations
- **Target: 200+ env steps/second on M1** (single env, CPU)
- **Target: 500+ env steps/second on PC** with 32 parallel envs (when available)
- Every module has test file
- Every physics constant has citation in docstring

---

## WHAT NOVEL MEANS (4 claims for Paper 2)

1. **First mission-conditioned RL policy** — FiLM on M(t) params as input
2. **First learned ρ output** — ρ̂ as policy output, not hyperparameter
3. **First calibrated pharmacokinetic/trauma M(t) in UAV benchmark** — 8 types
4. **Double ranking inversion** — original + information-driven (phase diagram)

---

## MCP SERVER (build in Phase 6, parallel to paper)

```python
# flare/mcp/server.py
# 5 tools exposed via Model Context Protocol:

@server.tool("flare_create_scenario")
async def create_scenario(place: str, cell_size: float, ...): ...

@server.tool("flare_run_planner")
async def run_planner(scenario_id: str, planner: str, ...): ...

@server.tool("flare_compare")
async def compare(scenario_id: str, planners: list, ...): ...

@server.tool("flare_analyze_mission")
async def analyze_mission(scenario_id: str, mission: str, ...): ...

@server.tool("flare_train_agent")
async def train_agent(scenario_id: str, algorithm: str, ...): ...
```

Access: `claude mcp add --transport http flare https://mcp.flare-uav.dev`

---

## STATUS TRACKER

- [x] Project bootstrap (de93267)
- [x] Phase 1.1: deterministic + stochastic CA fire (34 tests)
- [x] Phase 1.2: Wind kernel extracted to flare/core/wind.py (Alexandridis + cosine, 36 tests)
- [x] Phase 1.3: Gaussian smoke plume → flare/core/hazards/smoke.py (FFT convolution, α_m=8700, Jin visibility, 22 tests)
- [x] Phase 1.4: Dynamic NFZ → flare/core/hazards/nfz.py (Poisson arrival × geometric duration, 21 tests)
- [x] Phase 2.4: Numba Bresenham ray-cast → flare/sensors/ray_cast.py (cast_ray + cast_rays, smoke-coupled cutoff via Beer–Lambert ×2, 16 tests, 450-ray sweep < 5ms warm)
- [x] Phase 2.1: LiDAR LD19 → flare/sensors/lidar.py (12m, 450 rays, σ_r=10mm+0.001·r, σ_θ=2°, 2% FN, 17 tests)
- [x] Phase 2.3: Camera fire detection → flare/sensors/camera.py (130° HFOV, 30m, sector + range + LOS gating, 21 tests)
- [x] Phase 2.2: GPS BN-880 + HMC5883L → flare/sensors/gps.py (CEP 2m, σ_ψ=1.5°, sigmoid urban-canyon dropout, Wiener DR drift, 19 tests)
- [ ] Phase 3.1: Belief system (occupancy + certainty decay)
- [ ] Phase 3.2: FLARE-PO Gymnasium env
- [ ] Phase 3.3: MDP continuity test PASSES
- [ ] Phase 4.1: Insulin M(t) — Arrhenius
- [ ] Phase 4.2: Hemorrhage M(t) — Weibull
- [ ] Phase 4.3: All 8 M(t) implemented + tested
- [ ] Phase 5.1: MARS-RL network architecture
- [ ] Phase 5.2: RecurrentPPO training pipeline
- [ ] Phase 5.3: Hierarchical execution (RL → A*)
- [ ] Phase 5.4: Training converges (SR > 50% on Penteli)
- [ ] Phase 6.1: Run 7,200 episodes
- [ ] Phase 6.2: Statistical analysis (Friedman/Cliff's δ)
- [ ] Phase 6.3: Generate 7 figures
- [ ] Phase 6.4: Paper draft
- [ ] Phase 6.5: MCP server deployed

---

## WHEN IN DOUBT

1. Does this break Invariant I1? → Don't do it.
2. Does the file go where the tree says? → Check structure.
3. Is the constant cited? → Add docstring with source.
4. Is it fast enough? → Profile. 500 steps/sec minimum.
5. Is there a test? → Write one alongside the code.

When the answer is in this file, don't ask. When it's not, ask.
