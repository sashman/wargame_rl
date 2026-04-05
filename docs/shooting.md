# Shooting & Target Selection

## Action Space Extension

Shooting is an additional slice in the union action space managed by `ActionRegistry`. When opponent models exist and `n_shoot_targets > 0`, the handler registers a `"shooting"` slice after the movement slice:

| Slice | Indices | Valid phases |
|-------|---------|--------------|
| `stay` | `0` | All phases |
| `movement` | `1 .. N×S` | Movement phase only |
| `shooting` | `N×S+1 .. N×S+T` | Shooting phase only |

Where `T = number_of_opponent_models`. With the defaults (`n_movement_angles=16`, `n_speed_bins=6`) and 4 opponents, the total action space is **101** (1 stay + 96 movement + 4 shooting targets).

The shooting slice is conditionally registered — configs without opponents produce the same action space as before (no slice, no index growth).

## Target Encoding

Each shooting action index maps to a specific opponent model slot:

```
target_idx = action - shooting_slice.start
```

Target index K corresponds to opponent model K in the observation tensor. This positional alignment is a critical invariant: the transformer learns the correspondence between "opponent features at observation position K" and "action index K in the shooting slice."

| Action | Meaning |
|--------|---------|
| `0` | Stay (pass shooting, valid in all phases) |
| `shooting_slice.start + 0` | Shoot at opponent model 0 |
| `shooting_slice.start + 1` | Shoot at opponent model 1 |
| ... | ... |
| `shooting_slice.start + T-1` | Shoot at opponent model T-1 |

## Phase-Gated Masking

Each `env.step()` corresponds to one battle phase. The `GameClock` advances through the phase sequence (command → movement → shooting → charge → fight), and `skip_phases` controls which phases the agent steps through. By default, shooting is skipped.

During the **movement phase**, only stay and movement actions are valid. During the **shooting phase**, only stay and shooting actions are valid. The registry's `valid_phases` on each slice handles this automatically.

### Enabling the Shooting Phase

Shooting requires explicit YAML opt-in. Remove `shooting` from `skip_phases`:

```yaml
skip_phases:
  - command
  - charge
  - fight
```

With shooting enabled, each game turn produces two agent decisions: one movement step and one shooting step.

## Shooting Mask Computation

During the shooting phase, a per-model target validity mask is overlaid on the base phase mask. A target K is valid for player model M if **all** of:

1. **Model M is alive** — dead models get `STAY_ACTION` only
2. **Opponent K is alive** — dead targets are masked out
3. **In range** — Euclidean distance from M to K ≤ max weapon range of M
4. **Line of sight** — `has_line_of_sight` from M's cell to K's cell returns True

The overlay is computed by `compute_shooting_masks()` (a pure function in `env_components/shooting_masks.py`) and applied via bitwise AND on the shooting slice of the base mask.

If no targets are valid for a model, only `STAY_ACTION` remains — the model passes its shooting.

### Range Calculation

Range uses Euclidean distance on the grid, consistent with the `DistanceCache`. A model's effective range is the **maximum** across all its weapons — a target is "in range" if any weapon can reach it.

```python
max_range = max(w.range for w in model_config.weapons)
```

Models with no weapons (`weapons: []`) have max range 0 and cannot shoot anyone.

### Line of Sight

LOS uses the Bresenham ray-tracing service from `domain/los.py`. The same `has_line_of_sight_between_cells` method on `BattleView` is used for mask computation and human rendering. See `docs/tabletop-rules-reference.md` for LOS semantics (interior-cell-only blocking, no model occlusion).

## Weapon Configuration

Weapon profiles are configured per model via `ModelConfig.weapons`:

```yaml
models:
  - x: 3
    y: 1
    weapons:
      - range: 12
      - range: 24
```

`WeaponProfile` currently has only a `range` field (integer, grid cells). Phase 5 adds resolution stats (`attacks`, `ballistic_skill`, `strength`, `ap`, `damage`).

| Field | Type | Description |
|-------|------|-------------|
| `range` | `int` (> 0) | Maximum range in grid cells |

`ModelConfig.weapons` defaults to an empty list. Existing configs without weapon definitions are unaffected — models simply cannot shoot.

## Action Dispatch

`ActionHandler.apply()` is phase-aware. When an action falls in the shooting slice, it is treated as a **no-op** — the model's location is not changed. Shooting resolution (hit → wound → save → damage) is Phase 5; Phase 4 only handles target selection and masking.

```
if action in shooting_slice range:
    continue  # no-op, Phase 5 adds resolution
else:
    apply movement displacement
```

## Observation Context

During a shooting step the agent observes:

- `battle_phase_index` indicating the current phase is shooting
- Opponent model features (position, alive flag, wound status) at fixed observation slots
- `action_mask` with only shooting targets and stay valid, filtered by LOS/range/alive

The transformer attends over opponent tokens and selects a target index. The positional alignment between observation slot K and action index K allows implicit pointer-style learning.

## Future Extensions

### Multi-Weapon Targeting

Models with multiple weapons (up to 7) will independently assign targets per weapon via **sub-steps within the shooting phase** — one step per weapon firing opportunity. The action space stays `Discrete(n_targets + stay)`, the transformer architecture is unchanged, just more steps per turn. Models with 1 weapon get 1 step, models with 5 get 5.

### Pointer-Network Attention

If the transformer struggles with the implicit observation-to-action index mapping, a pointer-network style cross-attention mechanism can produce shooting logits directly from attention scores between the acting model's token and opponent tokens. This is a network architecture change, not an action space change.

### Precomputed Probability Matrices

Attacker × defender expected damage tables (hit chance, wound chance, expected value) computed from weapon profiles and target stats. Dual purpose: observation feature for the transformer (perfect information, mirroring real player capability) and explainability tool. Requires full weapon profiles from Phase 5.
