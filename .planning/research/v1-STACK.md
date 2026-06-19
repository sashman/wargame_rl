# Stack Research: RL Tabletop Wargame — Upcoming Features

**Focus:** Libraries, tools, and techniques for adding combat, terrain/LOS, self-play, and advanced mechanics to the existing Gymnasium + PyTorch Lightning stack.

## Current Stack (Keep As-Is)

The existing stack is well-suited and should not change:

- **Gymnasium 1.x** — standard RL env API, already registered
- **PyTorch + Lightning 2.5+** — training loops, checkpointing, Wandb integration
- **PPO (default) + DQN** — both implemented; PPO preferred for multi-unit problems
- **Transformer (NanoGPT-style)** — handles variable-length entity sequences
- **Pydantic + pydantic-yaml** — config validation, YAML-driven scenarios
- **Wandb** — experiment tracking, video recording
- **UV** — package management, lockfile

**Confidence:** High — no reason to swap any of these.

## Combat Resolution (Shooting & Melee)

### Recommendation: Pure Python/NumPy implementation

No external library needed. The tabletop attack sequence (hit roll → wound roll → save → damage) is straightforward probability math.

- **NumPy** (already a dependency) — batch dice rolls via `rng.integers()`, vectorized comparison for hit/wound/save thresholds
- Implement as domain services under `domain/` following existing DDD patterns
- Combat resolution is deterministic given RNG state — keep it reproducible with seeded generators

**What NOT to use:**
- Game engines (Unity ML-Agents, Godot RL) — massive overhead for what amounts to dice math on a grid. The existing Gymnasium env is the right abstraction level.

**Confidence:** High

## Line of Sight (LOS)

### Recommendation: Bresenham's line algorithm + grid-based occlusion

- **Pure Python/NumPy** — Bresenham's line is ~20 lines of code; no need for external raycasting libraries
- Pre-compute LOS tables when board layout changes (terrain placement), cache during episode
- For blocking terrain: mark cells as opaque, trace ray from shooter to target, check for intersections

**Alternative considered:**
- **shapely** (computational geometry) — overkill for grid-based LOS. Bresenham on a discrete grid is simpler and faster.
- **scipy.ndimage** — could use for flood-fill visibility, but Bresenham is more faithful to the tabletop "draw a line" rule.

**Confidence:** High

## Terrain System

### Recommendation: NumPy terrain grid + enum-based terrain types

- **Terrain as a 2D NumPy array** — `(board_height, board_width)` with integer terrain type codes
- Terrain types as a Python `IntEnum`: `OPEN=0, COVER=1, DIFFICULT=2, BLOCKING=3`
- Include terrain in observation space as an additional tensor channel
- Movement cost modifier: difficult terrain halves speed (check terrain at destination cell)
- Cover bonus: modify save rolls during combat resolution

**Map generation:**
- Start with template-based layouts (JSON/YAML terrain maps in `examples/`)
- Later add procedural generation with `numpy.random` for training variety

**Confidence:** High

## Self-Play Training

### Recommendation: Frozen checkpoint pool + PPO

- **No new libraries needed** — self-play is an training loop pattern, not a library dependency
- Store periodic checkpoints as opponent pool; sample opponent from pool each episode
- Use existing `OpponentPolicy` registry: add a `model` policy type that loads a checkpoint and runs inference
- **PettingZoo** (optional) — if you want a standard multi-agent API. However, the existing single-agent-with-opponent architecture may be simpler to extend (player acts, opponent auto-executes).

**What NOT to use:**
- **OpenSpiel** — designed for perfect/imperfect info games (poker, Go). Overhead for a spatial wargame with custom mechanics.
- **RLlib multi-agent** — heavy framework; the project already has its own Lightning training loop.

**Elo tracking:**
- **Pure Python** — Elo rating is a simple formula (~10 lines). No library needed.
- Log Elo ratings to Wandb as a custom metric.

**Confidence:** High for frozen-pool self-play. Medium for full simultaneous multi-agent (adds complexity).

## Hyperparameter Sweeps

### Recommendation: Wandb Sweeps

- **wandb.sweep** — already using Wandb for tracking; Sweeps integrates natively
- Define sweep config YAML, run `wandb agent` — parallelizes across machines
- Alternative: **Optuna** with Wandb integration if you want more sophisticated search (TPE, pruning). Optuna is lightweight and well-maintained.

**Confidence:** High for Wandb Sweeps. Medium for Optuna (more powerful but another dependency).

## Profiling & Performance

### Recommendation: Keep pyinstrument, add torch.profiler

- **pyinstrument** (already used) — great for Python-level profiling
- **torch.profiler** (built into PyTorch) — GPU kernel profiling, memory analysis
- For larger scenarios (10+ models): profile observation tensor construction and network forward pass; batch where possible

**Confidence:** High

## Summary of Additions

| Need | Recommendation | New Dependency? |
|------|---------------|-----------------|
| Combat resolution | NumPy dice rolls | No |
| Line of sight | Bresenham's algorithm | No |
| Terrain system | NumPy terrain grid + IntEnum | No |
| Self-play | Frozen checkpoint pool | No |
| Multi-agent (optional) | PettingZoo | Optional |
| Hyperparameter sweeps | Wandb Sweeps or Optuna | Optional (Optuna) |
| GPU profiling | torch.profiler | No (built-in) |

**Key insight:** The existing stack handles nearly everything. The upcoming features are primarily domain logic and reward shaping additions, not infrastructure changes. The main architectural decision is whether to adopt PettingZoo for multi-agent or extend the existing single-agent-with-opponent pattern.

---
*Stack research: 2026-04-02*
