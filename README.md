# FLARE-PO

POMDP wildfire UAV benchmark with mission-aware risk-adaptive planning.
Paper 2 implementation — successor to FLARE (Paper 1).

> **Status:** In active development. See [CLAUDE.md](CLAUDE.md) for full specification, architecture, and phase plan.

## What it is

Sensor-only partial observability POMDP where a simulated UAV navigates a 2D
urban wildfire scenario using modeled sensors (LDRobot LD19 LiDAR, 130° HFOV
camera, BN-880 GPS), builds belief in flight, and learns an adaptive risk
coefficient ρ̂(s,t) conditioned on calibrated mission physics (Arrhenius
insulin decay, Weibull hemorrhage survival, CFK CO inhalation, etc.).

Resolves the Paper 1 ranking inversion under sensor uncertainty — and reveals
a second one along the observability axis.

**Strictly 2D simulation. No real drones, no MAVLink, no satellites, no
external APIs at runtime.**

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gis,ml]"
pytest tests/ -q
```

See SETUP section in [CLAUDE.md](CLAUDE.md) for full instructions and
verification commands.

## Hardware roadmap

- **Now (M1 16GB):** Phases 1–4 + Phase 6 (env, sensors, belief, missions,
  analysis, MCP server). Small RL debug runs OK on MPS.
- **Later (RTX 3060 PC):** Phase 5 — full MARS-RL training (10M steps × 5
  seeds × 5 scenarios × 6 difficulties).

## License

MIT
