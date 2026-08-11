# Game State I/O & Analysis

This document describes the structured game state system — how match data flows in and out of the environment, what formats it takes, and what analysis can be built on top.

## Motivation

The RL training pipeline produces tensor observations optimised for neural networks: normalised floats, one-hot encodings, compressed representations. These are lossy and opaque — useless for evaluating whether an agent's behaviour is tactically sound, rule-compliant, or improving across training runs.

The state I/O layer provides a **parallel output path** that preserves the complete, human- and machine-readable game state. It does not replace or interfere with the RL tensor pipeline; it runs alongside it and is disabled by default (zero overhead during training unless explicitly opted in).

## Architecture

The `envs/state/` package sits at the same dependency level as `renders/` and `reward/`:

```
domain/  →  types/  (only)
                ↑
    ┌───────────┼───────────────────────┐
    │           │                       │
reward/     renders/     state/         env_components/
    │           │           │               │
    └───────────┴───────────┴───────────────┘
                        ↑
                   WargameEnv (facade)
```

`state/` depends only on `BattleView` (via snapshot data) and `types/`. It never imports from `env_components`, `reward`, or `renders`.

## Module Layout

```
wargame_rl/wargame/envs/state/
├── __init__.py         # Public API re-exports
├── snapshot.py         # GameStateSnapshot model, build_snapshot(), validation, SnapshotEncoder
├── restore.py          # The read direction: rebuild clock, models, objectives, combat results
├── events.py           # StateDelta, ResetEvent, StepEvent, compute_delta(), apply_delta()
├── event_log.py        # EventLog: append-only accumulator with anchor snapshots
├── exporter.py         # StateExporter protocol, EventLogExporter
├── replay.py           # ReplayController: seek to any step from an EventLog
├── codecs.py           # MatchCodec protocol, JsonMatchCodec (JSONL), CODEC_REGISTRY
├── narrator.py         # StepNarrator: LLM-readable text summaries
└── analysis.py         # MatchAnalysis model, analyze_match()
```

## Data Flow

### Recording

```
WargameEnv.reset() ──→ to_snapshot() ──→ exporter.on_reset(snapshot)
                                                    │
WargameEnv.step()  ──→ to_snapshot() ──→ exporter.on_step(snapshot)
                                                    │
                                                    ▼
                                         EventLog (in memory)
                                                    │
                                                    ▼
                                         JsonMatchCodec.encode()
                                                    │
                                                    ▼
                                         recordings/xxx_events.jsonl
```

### Replay & Analysis

```
recordings/xxx_events.jsonl
        │
        ▼
JsonMatchCodec.decode()
        │
        ▼
    EventLog
        │
        ├──→ ReplayController.seek(step)  ──→ GameStateSnapshot
        │
        ├──→ ReplayController.iter_snapshots() ──→ [GameStateSnapshot, ...]
        │                                                    │
        │                                                    ▼
        │                                          analyze_match()
        │                                                    │
        │                                                    ▼
        │                                          MatchAnalysis (JSON/text)
        │
        └──→ StepNarrator.narrate(snapshot) ──→ Human-readable text
```

## Key Abstractions

### GameStateSnapshot

> Source: [`wargame_rl/wargame/envs/state/snapshot.py`](../wargame_rl/wargame/envs/state/snapshot.py)

A complete, serialisable Pydantic model of the game at one point in time. This is the "universal interchange format" — everything downstream operates on snapshots.

> **`clock` vs `action_phase`.** The snapshot is taken after `step()` has advanced the clock, so `clock.battle_phase` is the phase that will execute *next*. The actions in the same snapshot were executed in `action_phase`. Always attribute `player_actions` / `player_action_descriptions` to `action_phase`.

#### Schema

```python
class GameStateSnapshot(BaseModel):
    schema_version: str = "2.1"
    step: int
    max_steps: int
    clock: ClockSnapshot
    action_phase: str | None
    n_rounds: int
    board_width: int
    board_height: int
    player_models: list[ModelSnapshot]
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]
    deployment_zone: list[int]
    opponent_deployment_zone: list[int]
    terrain_footprints: list[list[list[float]]] | None = None  # 2.1: outline per ruin
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int
    objective_control: list[str]
    player_actions: list[int] | None
    opponent_actions: list[int] | None
    player_action_descriptions: list[str] | None
    player_combat_results: list[CombatResultSnapshot]
    opponent_combat_results: list[CombatResultSnapshot]
    reward: RewardSnapshot
    is_terminated: bool
    is_truncated: bool
    player_alive_count: int
    opponent_alive_count: int
    player_total_wounds: int
    opponent_total_wounds: int
    mission_type: str
    mission_params: dict[str, int | float | str]
```

#### Sub-models

```python
class ClockSnapshot(BaseModel):
    game_phase: str  # "deployment" | "battle"
    battle_round: int | None
    active_player: str | None
    battle_phase: str | None  # "command" | "movement" | "shooting" | ...


class ModelSnapshot(BaseModel):
    location: list[int]
    previous_location: list[int] | None
    group_id: int
    alive: bool
    current_wounds: int
    max_wounds: int
    toughness: int
    save: int
    advanced_this_turn: bool
    weapons: list[WeaponSnapshot]
    distances_to_objectives: list[float]
    at_objective: list[bool]
    closest_objective_idx: int | None
    closest_objective_distance: float | None


class ObjectiveSnapshot(BaseModel):
    location: list[int]
    radius_size: int
    player_models_in_range: list[int]
    opponent_models_in_range: list[int]


class WeaponSnapshot(BaseModel):
    weapon_range: int
    attacks: int
    ballistic_skill: int
    strength: int
    ap: int
    damage: int


class CombatResultSnapshot(BaseModel):
    attacker_idx: int
    target_idx: int
    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int
    expected_damage: float
    hit_probability: float
    wound_probability: float


class RewardSnapshot(BaseModel):
    total: float | None
    breakdown: dict[str, float]
    phase_name: str
    phase_index: int
```

#### Field summary

| Group | Fields | Purpose |
|-------|--------|---------|
| **Timing** | `step`, `max_steps`, `clock`, `action_phase`, `n_rounds` | Where in the episode and game clock |
| **Board** | `board_width`, `board_height`, `deployment_zone`, `opponent_deployment_zone`, `terrain_footprints` | Static geometry (terrain footprints recorded on the reset + anchors only, not deltas) |
| **Units** | `player_models`, `opponent_models` | Full per-model state inc. weapons, wounds, distances |
| **Objectives** | `objectives`, `objective_control` | Positions, radii, ownership |
| **Actions** | `player_actions`, `opponent_actions`, `player_action_descriptions` | Raw + decoded actions taken |
| **Combat** | `player_combat_results`, `opponent_combat_results` | Hits, wounds, damage with expected-value analytics |
| **Reward** | `reward` | Per-calculator breakdown and active phase |
| **VP** | `player_vp`, `opponent_vp`, `*_vp_delta` | Victory point state and step-wise change |
| **Terminal** | `is_terminated`, `is_truncated` | Episode end flags |
| **Attrition** | `player_alive_count`, `opponent_alive_count`, `*_total_wounds` | Aggregate health |
| **Mission** | `mission_type`, `mission_params` | Active mission and its parameters |

### StateExporter Protocol

```python
class StateExporter(Protocol):
    def on_reset(self, snapshot: GameStateSnapshot) -> None: ...
    def on_step(self, snapshot: GameStateSnapshot) -> None: ...
```

The env holds an optional list of exporters. When non-empty, `to_snapshot()` is called at the end of `reset()` and `step()`, and each exporter is notified. This is identical to how the renderer is wired — a parallel output sink.

### EventLog & Delta Encoding

The EventLog stores an episode as:
- **ResetEvent**: full initial snapshot (anchor)
- **StepEvent**: granular delta (only changed fields) + optional full anchor

Anchors are inserted every N steps (configurable, default 10). This enables efficient random-access: find the nearest anchor, apply forward deltas.

The delta representation (`StateDelta`) captures per-step changes at field level — model positions that moved, VP that changed, combat results, reward. Unchanged fields are `None` and occupy no space.

### Codec Registry

```python
CODEC_REGISTRY: dict[str, type[MatchCodec]] = {
    "json": JsonMatchCodec,  # JSONL (newline-delimited JSON)
}
```

Follows the same registry pattern as reward calculators and opponent policies. New codecs (e.g., MessagePack, Protobuf) can be registered without changing existing code. Each codec implements `encode(EventLog) -> bytes` and `decode(bytes) -> EventLog`.

The default `JsonMatchCodec` uses **JSONL** (newline-delimited JSON, one object per line). The first line is a header with version and anchor_interval; subsequent lines are individual events. This enables streaming writes, crash-safe recording, and easy inspection with standard tools (`head`, `tail`, `wc -l`, `jq`).

### ReplayController

Provides random-access state reconstruction:
- `seek(step)` — reconstruct `GameStateSnapshot` at any recorded step
- `iter_snapshots()` — full ordered list of all states
- `snapshot_range(start, end)` — subset reconstruction

Uses anchor+delta architecture internally for efficiency.

## Analysis Layer

The `analyze_match()` function takes a list of snapshots and produces a `MatchAnalysis` — a structured Pydantic model covering:

| Dimension | Metrics |
|-----------|---------|
| **Movement efficiency** | Objective approach rate, idle rate, edge contact rate, mean distance to objective |
| **Tactical quality** | Group cohesion (inter-model distance), time to first objective, VP/step, objective occupancy (final / peak / drift ratio), target selection optimality |
| **Rule compliance** | Movement violations (teleportation), bounds violations |
| **Degenerate behavior** | Action entropy, oscillation rate, reward stagnation |
| **Composite** | Tactical score (0-100), issue flags |

The analysis output is available as:
- Human-readable text (terminal report)
- JSON (programmatic consumption, AI evaluation)
- Comparison tables (multiple runs side-by-side)

## CLI Tools

| Command | Purpose |
|---------|---------|
| `just record <config>` | Train 1 epoch with event recording, no wandb (quick E2E test) |
| `just record-sim <ckpt> <config> [n] [net]` | Record N episodes from a trained checkpoint, no rendering |
| `just replay <file>` | Narrate a recorded match step-by-step |
| `just replay-summary <file>` | Match metadata overview |
| `just analyze <file>` | Full analysis report (text) |
| `just analyze-json <file>` | Analysis report as JSON |
| `just analyze-compare f1 f2` | Side-by-side metric comparison |

Training and simulation also support `--record-events` for production use. The
`train` recipe takes it as its third *positional* argument, not as a flag:
```bash
just train config.yaml                            # normal training
just train config.yaml '' true                    # + event log
uv run train.py --env-config-path config.yaml --record-events
```

## Extension Points

### Adding a new analysis metric

1. Add the field to `MatchAnalysis` in `analysis.py`
2. Compute it in the relevant `_analyze_*` function (or add a new pass)
3. Optionally integrate it into the composite score and issue detection

### Adding a new codec

1. Implement the `MatchCodec` protocol (encode/decode/content_type)
2. Register in `CODEC_REGISTRY`
3. Available immediately via `build_codec("my_format")`

### Adding a new exporter type

1. Implement the `StateExporter` protocol (on_reset/on_step)
2. Pass instances to `WargameEnv(state_exporters=[...])` or via `create_environment()`

Examples: streaming exporter (WebSocket), database writer, Wandb artifact logger.

### Building custom analysis pipelines

The raw building blocks are fully composable:

```python
from wargame_rl.wargame.envs.state import (
    JsonMatchCodec,
    ReplayController,
    StepNarrator,
    analyze_match,
)

# Load
codec = JsonMatchCodec()
log = codec.decode(Path("recording.jsonl").read_bytes())
ctrl = ReplayController(log)

# Seek to specific moment
snap = ctrl.seek(step=42)

# Narrate for LLM consumption
narrator = StepNarrator()
text = narrator.narrate(snap)

# Full analysis
report = analyze_match(ctrl.iter_snapshots())
print(report.model_dump_json(indent=2))
```

## Design Principles

1. **Zero cost when disabled** — exporters list is empty by default; no `to_snapshot()` call, no overhead.
2. **Parallel to RL pipeline** — state I/O never touches `observation_to_tensor()`, Lightning modules, or checkpoints.
3. **Snapshot is the universal format** — everything operates on `GameStateSnapshot`. New consumers just need the snapshot.
4. **Registry pattern for extension** — codecs, exporters, and analyzers all follow the same pattern used by reward calculators.
5. **Interoperable** — JSONL output readable by any language, LLM, or tool. No framework lock-in. Standard tools (`head`, `tail`, `jq`) work out of the box.
