# Architecture Research: v9.0 — Structured Game State & LLM-Readable Representation

**Focus:** Dependency patterns and architectural constraints for the new state/serialisation layer
**Researched:** 2026-05-23
**Confidence:** HIGH (all findings based on direct codebase analysis)

---

## 1. DDD Architecture Summary (from `docs/ddd-envs.md`)

The environment follows a three-layer DDD structure:

```
┌──────────────────────────────────────────────────────────────────┐
│  WargameEnv (facade)                                            │
│  Implements BattleView protocol · wires layers · Gym API        │
├──────────────────────────────────────────────────────────────────┤
│  env_components/  │  reward/  │  renders/  │  mission/  │ opp/  │
│  (adapters)       │           │            │            │       │
│  Actions,         │ Phases,   │ Human      │ VP         │ Scr.  │
│  observation,     │ calcs,    │ renderer   │ calcs      │ pol.  │
│  distance cache   │ criteria  │            │            │       │
├──────────────────────────────────────────────────────────────────┤
│  domain/                                                         │
│  Battle (aggregate) · entities · value_objects · GameClock       │
│  placement · termination · turn_execution · LOS · shooting       │
├──────────────────────────────────────────────────────────────────┤
│  types/                                                          │
│  Config (Pydantic) · game_timing · observation types · actions   │
└──────────────────────────────────────────────────────────────────┘
```

**Dependency direction (strict, enforced by convention):**
- **domain/** → `types/` only. No imports from `env_components`, `reward`, `renders`.
- **reward/** and **renders/** → `BattleView` protocol + `types/`. No import of `WargameEnv` or `Battle`.
- **env_components/** → domain + types.
- **WargameEnv** → everything (it's the composition root).

**Key insight:** `BattleView` is the architectural seam. It's a read-only Protocol that exposes battle state without mutation capability. Every external consumer (renderers, reward calculators, VP calculators, observation builder) depends on this protocol, not on the env or aggregate directly.

---

## 2. BattleView Consumer Map

### Who reads BattleView today?

| Consumer | File | How it uses BattleView |
|----------|------|----------------------|
| **Renderer** (abstract) | `renders/renderer.py` | `setup(view)`, `render(view)` — reads models, objectives, board dimensions, clock state for visual display |
| **HumanRender** | `renders/human.py` | Accesses `view.player_models`, `view.opponent_models`, `view.objectives`, `view.config`, `view.game_clock_state`, `view.player_vp`, `view.opponent_vp`, `view.deployment_zone`, `view.board_width`/`board_height` |
| **RewardPhaseManager** | `reward/phase_manager.py` | `calculate_reward(view, ctx)`, `check_success(view, ctx)` — reads `view.player_models` for alive filtering |
| **PerModelRewardCalculator** | `reward/calculators/base.py` | `calculate(model_idx, model, view, ctx)` — per-model + view access |
| **GlobalRewardCalculator** | `reward/calculators/base.py` | `calculate(view, ctx)` — full view access |
| **SuccessCriteria** | `reward/criteria/base.py` | `is_successful(view, ctx)` |
| **VPCalculator** | `mission/vp_calculator.py` | `compute_vp(view, scoring_side, round, player_side)` — reads models, objectives, computes distances |
| **Observation builder** | `env_components/observation_builder.py` | `build_observation(view, ...)`, `build_info(view)` — reads all BattleView properties to construct `WargameEnvObservation` |
| **WargameEnv itself** | `envs/wargame.py` | Passes `self` (which implements BattleView) to all of the above |

### What does a "state exporter" look like as a consumer?

A state exporter would be another `BattleView` consumer, exactly like the renderer or observation builder. It reads the view and produces a different output format (JSON, text, events) instead of Pygame surfaces or tensor-ready observations.

**It fits naturally as a peer of renderers and observation builder** — same dependency direction, same protocol contract.

---

## 3. Where Should the State Serialisation Layer Live?

### Recommended: New `state/` package at `envs/state/`

```
wargame_rl/wargame/envs/
├── state/                      # NEW — state export, serialisation, events
│   ├── __init__.py
│   ├── exporter.py             # StateExporter protocol + base
│   ├── snapshot.py             # Full state snapshot model (Pydantic)
│   ├── delta.py                # State delta / change model
│   ├── event.py                # Event types for append-only log
│   ├── event_log.py            # EventLog accumulator
│   ├── formatters/             # Pluggable output formats
│   │   ├── __init__.py
│   │   ├── json_formatter.py   # Default JSON
│   │   ├── text_formatter.py   # LLM-readable text
│   │   └── registry.py         # Formatter registry
│   └── replay.py               # Deterministic replay from event log
├── domain/                     # Unchanged
├── env_components/             # Unchanged
├── reward/                     # Unchanged
├── renders/                    # Unchanged
└── ...
```

**Rationale:**
- **Same dependency level as `renders/` and `reward/`**: depends on `BattleView` + `types/`, not on `WargameEnv`.
- **Does not pollute domain/**: the domain stays rule-focused; serialisation is an adapter concern.
- **Does not pollute env_components/**: env_components is about Gym machinery (actions, observation spaces, distance cache). State export is a different concern.
- **Parallel to renders/**: just as `renders/` produces visual output from BattleView, `state/` produces data output from BattleView.

### Dependency position in the graph

```
domain/  →  types/  (only)
                ↑
    ┌───────────┼───────────────────────┐
    │           │                       │
reward/     renders/     state/ (NEW)   env_components/
    │           │           │               │
    └───────────┴───────────┴───────────────┘
                        ↑
                   WargameEnv (facade)
```

`state/` depends on `BattleView` and `types/` — identical dependency direction to `reward/` and `renders/`.

---

## 4. Registry Pattern Analysis

The codebase uses a consistent registry pattern across four subsystems:

| Subsystem | Registry file | Pattern | Key → Class mapping |
|-----------|--------------|---------|-------------------|
| Reward calculators | `reward/calculators/registry.py` | Module-level `dict[str, type]`, `build_calculator(type_name, weight, params)` | `"closest_objective"` → `ClosestObjectiveCalculator` |
| Success criteria | `reward/criteria/registry.py` | Module-level `dict[str, type]`, `build_criteria(type_name, params)` | `"all_at_objectives"` → `AllAtObjectivesCriteria` |
| Opponent policies | `opponent/registry.py` | Module-level `dict`, `register_policy(name, cls)`, `build_opponent_policy(config, env)` + `_auto_register()` via importlib | `"random"` → `RandomPolicy` |
| VP calculators | `mission/registry.py` | Module-level `dict[str, type]`, `build_vp_calculator(type_name, params)` | `"default"` → `DefaultVPCalculator` |

### Common pattern

```python
REGISTRY: dict[str, type[BaseClass]] = {
    "key": ConcreteClass,
    ...
}

def build_thing(type_name: str, **kwargs) -> BaseClass:
    cls = REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown type '{type_name}'. Available: ...")
    return cls(**kwargs)
```

### Can state exporters/formatters use this pattern?

**Yes, directly.** A formatter registry would follow the exact same shape:

```python
FORMATTER_REGISTRY: dict[str, type[StateFormatter]] = {
    "json": JSONFormatter,
    "text": TextFormatter,
}

def build_formatter(type_name: str, params: dict[str, Any]) -> StateFormatter:
    cls = FORMATTER_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(...)
    return cls(**params)
```

This enables YAML configuration of which formatters are active, just like reward calculators:

```yaml
state_exporters:
  - type: json
    params: { indent: 2, schema_version: 1 }
  - type: text
    params: { include_combat_stats: true }
```

The opponent policy registry's `_auto_register()` pattern (lazy import via importlib) is also reusable if formatters are contributed as plugins.

---

## 5. Backward Compatibility Constraints

### What must NOT break

| Surface | Constraint | Impact on state layer |
|---------|-----------|---------------------|
| **Gym API** (`step()` → obs, reward, done, truncated, info) | Return signature is fixed by Gymnasium 1.x | State export is a *side channel*, not a replacement for the Gym return tuple |
| **`WargameEnvObservation`** dataclass | Shape consumed by `observation_to_tensor()`, then by TransformerNetwork/MLPNetwork | State export produces a *separate* representation; observation pipeline is untouched |
| **`WargameEnvConfig` (Pydantic + YAML)** | Existing YAML configs must keep working with defaults | New config fields must default to no-op (e.g. `state_exporters: []` or `state_exporters: None`) |
| **`BattleView` protocol** | Adding properties is additive (new `@property` methods). Removing/renaming breaks all consumers | State layer can read existing properties. If it needs new data, add to BattleView (additive, non-breaking) |
| **Training pipeline** (`train.py` → Lightning → Agent → env) | The pipeline calls `env.reset()` and `env.step()` only. It never sees raw state. | State export hooks are invisible to the training pipeline unless explicitly enabled |
| **`observation_to_tensor()` tensor pipeline** | Fixed 5-tensor output: game_features, objectives, player_models, opponent_models, action_mask | State I/O is a parallel pathway. The tensor pipeline is the RL-facing contract; state export is the human/LLM-facing contract |
| **Checkpoint format** | Lightning checkpoints contain model weights + optimizer state, not env state | State export doesn't affect checkpoints |

### Safe extension points

1. **New optional field on `WargameEnvConfig`** (e.g. `state_exporters: list[StateExporterConfig] | None = None`) — existing YAML untouched because default is `None`.
2. **New properties on `BattleView`** — additive protocol extension. Existing implementors (`WargameEnv`) just need the new `@property`.
3. **New package `envs/state/`** — no existing code references it, so adding it has zero impact.
4. **Hook methods on `WargameEnv`** (e.g. `_notify_exporters()`) — private, internal wiring only.

---

## 6. RL Pipeline vs State Export: Parallel Paths

### Current RL observation flow

```
WargameEnv.step()
    → build_observation(view, cache, registry)
        → WargameEnvObservation (dataclass)
            → observation_to_tensor(obs, device)
                → 5 torch.Tensors
                    → TransformerNetwork.forward(...)
```

**Key characteristics:**
- `observation_to_tensor()` in `model/common/observation.py` consumes `WargameEnvObservation` (a flat dataclass with model/objective lists, action masks, game state scalars).
- It produces **normalised numeric tensors** — positions in [-1,1], one-hot groups, normalised wounds, etc.
- This is a lossy, RL-optimised representation. It discards entity names, weapon descriptions, deployment zone semantics, round context, etc.

### Proposed state export flow (separate path)

```
WargameEnv.step()
    → [existing] build_observation(view) → WargameEnvObservation → tensors → RL
    → [NEW]      export_state(view)      → GameStateSnapshot → JSON/text → LLM/API
```

**State export does NOT replace or modify the observation pipeline.** They serve different consumers:
- Tensor pipeline → RL networks (numeric, compressed, GPU-bound)
- State export → LLMs, APIs, replay systems, debugging (human-readable, complete, CPU-bound)

The `observation_to_tensor()` function and Lightning modules remain completely untouched.

---

## 7. State Export Hook Attachment Points

### Where in the env lifecycle should export happen?

| Hook point | When | What to export | Use case |
|-----------|------|---------------|----------|
| **After `reset()`** | Episode start, before first action | Full snapshot (initial board state) | Episode start event, replay anchor |
| **After `step()`** | Each agent action resolved | Delta (action taken, new positions, damage, VP changes) + optional full snapshot | Step-by-step logging, streaming API |
| **On demand** | External call | Full snapshot at current moment | LLM query ("what's the current state?"), API polling |
| **At episode end** | Terminal step detected | Terminal snapshot + episode summary | Replay packaging, training analytics |

### Recommended hook design

```python
class StateExporter(Protocol):
    """Receives BattleView snapshots at lifecycle points."""
    def on_reset(self, view: BattleView, episode_id: str) -> None: ...
    def on_step(self, view: BattleView, action: WargameEnvAction,
                reward: float, terminated: bool) -> None: ...
    def get_snapshot(self, view: BattleView) -> GameStateSnapshot: ...
```

**Attachment:** `WargameEnv` holds an optional list of exporters (like it holds an optional renderer). At `reset()` and `step()`, after all domain logic completes, it notifies registered exporters. This mirrors how `self.renderer.render(self)` is called at the end of `reset()` and in `render()`.

### Env integration (minimal wiring)

In `wargame.py`:
- `__init__`: accept optional `state_exporters: list[StateExporter]` (default empty)
- End of `reset()`: `for exp in self._state_exporters: exp.on_reset(self, episode_id)`
- End of `step()`: `for exp in self._state_exporters: exp.on_step(self, action, reward, terminated)`

This is 5-10 lines of wiring in the facade. No domain changes. No observation pipeline changes.

---

## 8. What Data is Available from BattleView?

Everything needed for a complete game state snapshot is already on the protocol:

| BattleView property | Type | Content |
|-------------------|------|---------|
| `board_width`, `board_height` | `int` | Board dimensions |
| `config` | `WargameEnvConfig` | Full configuration (models, objectives, phases, rules) |
| `player_models` | `list[WargameModel]` | All player units: location, stats (wounds), group, alive status |
| `opponent_models` | `list[WargameModel]` | All opponent units (same shape) |
| `objectives` | `list[WargameObjective]` | Objective locations + radii |
| `deployment_zone`, `opponent_deployment_zone` | `np.ndarray` | Zone bounds |
| `current_turn` | `int` | Step counter |
| `game_clock_state` | `GameState` | Round, phase, active player, game phase |
| `n_rounds` | `int` | Total rounds |
| `player_vp`, `opponent_vp` | `int` | Cumulative VP |
| `player_vp_delta`, `opponent_vp_delta` | `int` | Per-step VP change |
| `last_reward` | `float \| None` | Most recent reward |
| `has_line_of_sight_between_cells(...)` | method | LOS query |

**What's NOT on BattleView but may be needed for full state export:**
- **Action that was just taken** — available in `step()` as a parameter, can be passed to exporters
- **Reward breakdown** — available as `phase_manager.last_reward_breakdown` on the env
- **Episode ID** — not currently tracked; would need a simple counter or UUID on reset
- **Shooting results** — available as `_last_player_shooting_results` / `_last_opponent_shooting_results` on the env (private)
- **Previous model locations** — available as `model.previous_location` on each `WargameModel`

For shooting results and reward breakdown, the exporter can either:
1. Accept them as extra parameters in `on_step()`, or
2. Get them from a small `StepResult` carrier alongside `BattleView`

---

## 9. Existing Extension Patterns and Dev Guides

### docs/ file inventory

| File | Covers |
|------|--------|
| `docs/ddd-envs.md` | **Main extension guide.** Covers adding entities, value objects, placement rules, termination, reward/criteria, rendering, actions/phases. Clear "where to add X" patterns. |
| `docs/reward-phases.md` | Reward calculator/criteria registration, YAML configuration, phase config schema |
| `docs/opponent-policies.md` | Opponent policy registration, YAML config, observation impact table |
| `docs/missions-and-vp.md` | VP calculator system, mission config |
| `docs/shooting.md` | Shooting rules, domain design |
| `docs/movement.md` | Movement system |
| `docs/tabletop-rules-reference.md` | Full rules reference |
| `docs/goals-and-roadmap.md` | Historical roadmap (v0-era) |
| `docs/multi-run-training.md` | Parallel training configuration |

**No dedicated "how to add a new subsystem" guide exists** — `ddd-envs.md` covers adding to existing subsystems (entities, reward, rendering) but doesn't cover adding an entirely new subsystem like `state/`. The pattern to follow is clear though: mirror how `renders/` was set up (protocol + concrete implementations + optional wiring in the facade).

---

## 10. Snapshot Model Shape (Recommended)

Based on what BattleView exposes, a canonical snapshot would look like:

```python
@dataclass(frozen=True)
class GameStateSnapshot:
    schema_version: int
    episode_id: str
    step: int

    # Board
    board_width: int
    board_height: int

    # Timing
    game_phase: str
    battle_round: int | None
    active_player: str | None
    battle_phase: str | None
    n_rounds: int

    # Entities
    player_models: list[ModelSnapshot]
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]

    # Scoring
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int

    # Zones
    deployment_zone: tuple[int, int, int, int]
    opponent_deployment_zone: tuple[int, int, int, int]
```

All fields are derived from BattleView properties with no additional domain logic needed.

---

## 11. Summary of Architectural Constraints

1. **State export is a BattleView consumer** — same architectural role as renderers and reward calculators. It reads the view, it doesn't mutate it.

2. **Lives in `envs/state/`** — parallel to `envs/renders/`, same dependency direction (→ BattleView, → types/).

3. **Registry pattern for formatters** — identical to reward/criteria/mission/opponent registries. Enables YAML-driven configuration.

4. **Zero impact on RL pipeline** — `observation_to_tensor()`, Lightning modules, `train.py`, and checkpoints are completely untouched. State export is a parallel output path.

5. **Zero impact on existing YAML configs** — new config fields default to disabled (`None` or empty list).

6. **BattleView already exposes enough** — the protocol has all the data needed for a complete game state snapshot. Minor additions (shooting results, reward breakdown) can be passed as step-level context.

7. **Hooks attach after `reset()` and `step()`** — same pattern as the renderer. The facade notifies exporters after domain logic completes.

8. **The event log is an exporter concern** — `EventLog` accumulates events from `on_reset` / `on_step` calls. Replay reads the log. Neither touches domain or env internals.

---

## 12. Risks and Open Questions

| Risk/Question | Impact | Mitigation |
|--------------|--------|-----------|
| **Performance overhead of serialisation on every step during training** | Could slow training if JSON serialisation is heavy | Make exporters optional and disabled by default. Only activate for inference/debugging/API serving. |
| **Schema versioning strategy** | Breaking changes to snapshot format affect downstream LLM prompts and APIs | Include `schema_version` field from day one. Maintain a version changelog. |
| **Delta encoding complexity** | Computing minimal diffs between states could be complex | Start with full snapshots. Add delta optimisation in a later phase once the snapshot model is stable. |
| **Event ordering in multi-phase turns** | Opponent turn events happen between player `step()` calls | The exporter sees the result after opponent execution. Sub-step events (opponent shooting results) need a richer hook or event granularity. |
| **Thread safety for streaming consumers** | If state is streamed over WebSocket while training runs | Training is single-threaded. Streaming consumers (v8.0 web viewer) can consume from a thread-safe queue filled by the exporter. |
