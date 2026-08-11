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
2. **M did not advance this turn** — `advanced_this_turn` gate. Dormant: nothing in the env ever sets the flag True (only `load_state` restores it from a snapshot), so today this never masks anything. See [rules/09-movement-phase.md](rules/09-movement-phase.md#advance-move) for the rule it will enforce.
3. **M is not locked in engagement** — masked out entirely if the nearest enemy is within `engagement_range` (config, authored in inches and resolved into board units by `domain/rules_quantities.py`; defaults to 1, against the rules' 2 — see `docs/rules/implementation-status.md`)
4. **Opponent K is alive** — dead targets are masked out
5. **In range** — Euclidean distance from M to K ≤ max weapon range of M
6. **Line of sight** — `has_line_of_sight` from M's cell to K's cell returns True

The overlay is computed by `compute_shooting_masks()` (a pure function in `env_components/shooting_masks.py`) and applied via bitwise AND on the shooting slice of the base mask.

If no targets are valid for a model, only `STAY_ACTION` remains — the model passes its shooting.

### Both sides are masked

The player's overlay is applied in `build_observation` (`env_components/observation_builder.py`); the opponent's is applied in `WargameEnv._opponent_action_mask`, with the sides swapped. `compute_shooting_masks` is positional despite its `player_`/`opponent_` parameter names, so the same function serves both.

The opponent's overlay is built **only for policies that declare `shoots = True`** (see [opponent-policies.md](opponent-policies.md#policies-that-shoot)). It costs up to `n_opponent × n_player` line-of-sight walks per shooting phase, which is not worth paying for the movement-only policies that most configs run.

This masking is the *only* enforcement of shooting legality. Resolution re-checks nothing beyond the action falling inside the shooting slice (`ActionHandler.decode_shooting_targets`) and then attacker-alive, target-alive and the attacker carrying at least one weapon (`domain.shooting.resolve_shooting_phase`) — it trusts the mask for both sides.

### Range Calculation

Range uses Euclidean distance on the grid, consistent with the `DistanceCache`. A model's effective range is the **maximum** across all its weapons — a target is "in range" if any weapon can reach it.

```python
max_range = max(w.range for w in model_config.weapons)
```

Models with no weapons (`weapons: []`) have max range 0 and cannot shoot anyone.

Note the asymmetry with resolution: masking uses the *longest*-ranged weapon, while `resolve_shooting_phase` always fires `weapons[0]`. For a multi-weapon model that makes targets between the first weapon's range and the longest weapon's range selectable but resolved with the wrong profile. Single-weapon models — every current config — are unaffected. Multi-weapon targeting is listed under Future Extensions below.

### Line of Sight

LOS uses the sampled-ray service in `domain/los.py`. The shooting mask takes `BattleView.line_of_sight_matrix`, which traces every candidate pair in one vectorised pass; the renderer uses the single-pair `has_line_of_sight_between_points`. See [terrain.md](terrain.md) for LOS semantics (interior-sample-only blocking, no model occlusion).

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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `range` | `int` (> 0) | *required* | Maximum range in grid cells |
| `attacks` | `int` (> 0) | `2` | Hit rolls per shooting action |
| `ballistic_skill` | `int` (2–6) | `3` | D6 roll needed to hit (3 means 3+) |
| `strength` | `int` (> 0) | `4` | Compared against target toughness for the wound roll |
| `ap` | `int` (≥ 0) | `1` | Armour penetration; worsens the target's save by this much |
| `damage` | `int` (> 0) | `1` | Wounds inflicted per failed save |

The defending side's stats come from `ModelConfig`: `toughness` (default `3`), `save` (default `4`, where `7` means no armour), and `max_wounds` (default `1`).

`ModelConfig.weapons` defaults to an empty list. Existing configs without weapon definitions are unaffected — models simply cannot shoot.

## Action Dispatch

`WargameEnv._apply_player_action` branches on the current phase (and `_apply_opponent_action` does the same for the opponent):

```
if phase is shooting:
    _resolve_shooting_action(...)   # hit → wound → save → damage
else:
    ActionHandler.apply(...)        # movement displacement
```

`ActionHandler.apply()` is itself phase-aware and displaces models only in the movement phase, so a scripted policy that bypasses the mask cannot move outside it. Shooting-slice actions reaching `apply()` are no-ops.

## Shooting Resolution

`_resolve_shooting_action` is a two-line composition: `ActionHandler.decode_shooting_targets` reads the action tuple into `(attacker_idx, target_idx)` pairs, then `domain.shooting.resolve_shooting_phase` filters and resolves them. The decode is action-space knowledge and the filtering is a rule, which is why they sit either side of the domain boundary.

`resolve_shooting_phase` runs the full attack sequence per firing model via `resolve_shooting(weapon, defender, rng)` in the same module:

1. **Hit** — roll `attacks` D6; a roll ≥ `ballistic_skill` hits, unmodified 6 always hits, unmodified 1 always misses.
2. **Wound** — one D6 per hit against `wound_roll_threshold(strength, toughness)` (same 1-always-fails / 6-always-succeeds rule).
3. **Save** — one D6 per wound against `save + ap`; failures become unsaved.
4. **Damage** — `unsaved × damage` wounds applied to the target; a model at 0 wounds is dead.

Rolls use `self._combat_rng`, seeded from `np_random` at each `reset()`, so a seeded episode resolves identically. Each shot is recorded as a `PairedShootingResult` (attacker index, target index, `ShootingResult`, and whether this shot made the kill) and exposed for the step via `env.last_player_shooting_results` / `env.last_opponent_shooting_results` (both on `BattleView`, so the v2 renderer draws the damaging ones as tracers). `domain/shooting.py` also provides `expected_damage(weapon, defender)`, a closed-form expectation with no dice.

## Observation Context

During a shooting step the agent observes:

- `battle_phase_index` indicating the current phase is shooting
- Opponent model features (position, alive flag, wound status, and the seven combat stats: weapon attacks/BS/strength/AP/damage plus toughness and save) at fixed observation slots
- `action_mask` with only shooting targets and stay valid, filtered by LOS/range/alive

The transformer attends over opponent tokens and selects a target index. The positional alignment between observation slot K and action index K allows implicit pointer-style learning.

## Future Extensions

### Multi-Weapon Targeting

Models with multiple weapons (up to 7) will independently assign targets per weapon via **sub-steps within the shooting phase** — one step per weapon firing opportunity. The action space stays `Discrete(n_targets + stay)`, the transformer architecture is unchanged, just more steps per turn. Models with 1 weapon get 1 step, models with 5 get 5.

### Pointer-Network Attention

If the transformer struggles with the implicit observation-to-action index mapping, a pointer-network style cross-attention mechanism can produce shooting logits directly from attention scores between the acting model's token and opponent tokens. This is a network architecture change, not an action space change.

### Precomputed Probability Matrices

Attacker × defender expected damage tables (hit chance, wound chance, expected value) computed from weapon profiles and target stats. Dual purpose: observation feature for the transformer (perfect information, mirroring real player capability) and explainability tool. The per-pair primitive already exists as `expected_damage` in `domain/shooting.py`; what is missing is materialising it as a matrix in the observation.
