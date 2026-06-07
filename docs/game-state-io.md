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
├── events.py           # StateDelta, ResetEvent, StepEvent, compute_delta(), apply_delta()
├── event_log.py        # EventLog: append-only accumulator with anchor snapshots
├── exporter.py         # StateExporter protocol, EventLogExporter
├── replay.py           # ReplayController: seek to any step from an EventLog
├── codecs.py           # MatchCodec protocol, JsonMatchCodec, CODEC_REGISTRY
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
                                         recordings/xxx_events.json
```

### Replay & Analysis

```
recordings/xxx_events.json
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

A complete, serialisable Pydantic model of the game at one point in time. Contains:

- Board dimensions, clock state, round/phase
- All player and opponent model states (position, wounds, weapons, objective distances)
- Objective states (control, models in range)
- Actions taken (raw integers + decoded descriptions)
- Combat results with analytical context (expected damage, hit/wound probabilities)
- Reward breakdown (per-calculator contributions, active phase)
- VP state, termination flags

This is the "universal interchange format" — everything downstream operates on snapshots.

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
    "json": JsonMatchCodec,
}
```

Follows the same registry pattern as reward calculators and opponent policies. New codecs (e.g., MessagePack, Protobuf) can be registered without changing existing code. Each codec implements `encode(EventLog) -> bytes` and `decode(bytes) -> EventLog`.

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
| **Tactical quality** | Group cohesion (inter-model distance), time to first objective, VP/step, target selection optimality |
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
| `just record <config>` | Train 1 epoch with event recording (quick E2E test) |
| `just replay <file>` | Narrate a recorded match step-by-step |
| `just replay-summary <file>` | Match metadata overview |
| `just analyze <file>` | Full analysis report (text) |
| `just analyze-json <file>` | Analysis report as JSON |
| `just analyze-compare f1 f2` | Side-by-side metric comparison |

Training and simulation also support `--record-events` for production use:
```bash
just train config.yaml           # normal training
just train config.yaml --record-events  # + event log
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
log = codec.decode(Path("recording.json").read_bytes())
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
5. **Interoperable** — JSON output readable by any language, LLM, or tool. No framework lock-in.
