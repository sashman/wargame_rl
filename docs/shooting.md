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

**A weapon names an enemy *unit*, never a model** ([rules/04](rules/04-making-attacks.md#2-select-targets)). Each shooting action index is one enemy unit:

```
target_group = action - shooting_slice.start
```

**The action index *is* the `group_id`.** That is why the slice is sized by `unit_count()` rather than by a model count or by `max_groups`: it must be wide enough for the highest id in play, and a config whose `ModelConfig`s name their own groups can have fewer units than the count-based split would produce.

| Action | Meaning |
|--------|---------|
| `0` | Stay (pass shooting, valid in all phases) |
| `shooting_slice.start + 0` | Shoot at enemy unit 0 |
| `shooting_slice.start + 1` | Shoot at enemy unit 1 |
| ... | ... |
| `shooting_slice.start + U-1` | Shoot at enemy unit U-1 |

On a 25v25 board that is **5 actions, not 25**.

The network keeps opponent tokens per *model* — position, wounds and stats are per model — and pools them into unit tokens only at the shooting head (`TransformerNetwork._pool_opponents_into_units`). **That pooling excludes the dead**: otherwise a destroyed model's latent still leaks into its unit's token, and mutating a corpse moves live logits, defeating the key-padding mask at the very last step.

## Allocation: the defender picks who bleeds

An attack is aimed at a unit; which model takes it is the defender's choice, preferring one that has already lost Wounds ([rules/05](rules/05-attack-sequence.md#4-inflict-damage)). `domain/shooting.py:_allocate_target` is that rule.

**An attack is discarded only when the whole target unit is destroyed** — *"excess attacks against a wiped-out unit are lost"*. Measured at **3.6%** of declared shots. It was **36-40%** while a weapon named a model and a shot at an already-dead one silently evaporated, which is what a squad concentrating fire did to most of its own volley.

Attacking units resolve one at a time in group order, which is the rules' own sequencing (*"shoots with their units one at a time"*). That also makes deferred removal a no-op at `max_wounds: 1`: a destroyed model stops being allocatable the moment it dies, and the next attacking unit sees the board either way. It would become observable with multi-wound models.

Allocation *groups* are not modelled — they split a unit by CHARACTER and by distinct (W, Sv, InSv), and this project has one profile per army and no characters, so every unit is a single group.

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

During the shooting phase, a per-model **unit**-target validity mask is overlaid on the base phase mask. Enemy unit U is a valid target for player model M if **all** of:

1. **Model M is alive** — dead models get `STAY_ACTION` only
2. **M did not advance this turn** — `advanced_this_turn` gate. **Live since 2026-08-22**, when the advance move landed: a unit that takes an action in the `advance` slice forfeits its shooting for the turn, which is correct here because no weapon has the ability that would permit firing after an advance. The flag is set for **every model in the advancing unit** (a move type is per unit, so advancing one model and shooting with its squadmates is not available) and cleared per turn by `WargameModel.begin_turn`. ⚠ It only fires in scenarios with `n_advance_speed_bins > 0`; that defaults to **0**, so on every config shipped before that date this still masks nothing. See [rules/09-movement-phase.md](rules/09-movement-phase.md#advance-move).
3. **M's UNIT is not locked in engagement** — masked out entirely if any living enemy is within `engagement_range` of **any model of M's unit** (casualties are excluded first; a corpse engages nobody, and a model with no living enemy left is not engaged) (config, authored in inches and resolved into board units by `domain/rules_quantities.py`; defaults to 1, against the rules' 2 — see `docs/rules/implementation-status.md`). ⚠ **This was per-MODEL until 2026-08-25**, so one model of a five-model unit could walk into contact — making the enemy unit unshootable by everybody — while its four squadmates kept firing. `_engaged_shooters` reduces over the unit, as `engaged_units` has always done on the target side of the same function.
4. **Unit U has a living model** — a unit is a legal target while any model in it survives, so killing one model closes nothing
5. **Some model of U is in range** — Euclidean distance from M ≤ max weapon range of M
6. **Some model of U is visible** to M

**Conditions 5 and 6 are checked independently and need not be satisfied by the same model** — *"it is enough that some model in the target unit is visible and some model in it is in range"*. Reducing a per-model "visible AND in range" mask over the unit would quietly keep them coupled and reject legal targets.

The overlay is computed by `compute_unit_shooting_masks()` (a pure function in `env_components/shooting_masks.py`) and applied via bitwise AND on the shooting slice of the base mask. Sight is still gated to keep the batch small, but at *unit* granularity: a pair is traced when the target's unit has some model in range, since the visible model need not be the reachable one.

If no targets are valid for a model, only `STAY_ACTION` remains — the model passes its shooting.

### Both sides are masked

The player's overlay is applied in `build_observation` (`env_components/observation_builder.py`); the opponent's is applied in `WargameEnv._opponent_action_mask`, with the sides swapped. `compute_unit_shooting_masks` is positional despite its `player_`/`opponent_` parameter names, so the same function serves both.

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

Rolls use `self._combat_rng`, seeded from `np_random` at each `reset()`, so a seeded episode resolves identically. Each shot is recorded as a `PairedShootingResult` (attacker index, target index, `ShootingResult`, whether this shot made the kill, and whether the target had cover) and exposed for the step via `env.last_player_shooting_results` / `env.last_opponent_shooting_results` (both on `BattleView`, so the v2 renderer draws the damaging ones as tracers).

### Expected damage

`domain/shooting.py` provides `expected_damage(weapon, defender, *, in_cover=False)`, a closed-form expectation with no dice, and `hit_probability(ballistic_skill, *, in_cover=False)`, the hit-roll term on its own. Both go through `ranged_skill`, which is what `resolve_shooting` applies to the dice — the two paths cannot state different rules.

It is the no-abilities case of the general result in [expected-damage.md](expected-damage.md) — `attacks × p_hit × p_wound × p_fail_save × damage`, with no critical-hit abilities, rerolls or `Shrug` term. That document gives the cell edits each ability contributes, so extending the closed form as abilities land is a table lookup rather than a rederivation. In its taxonomy cover is a **Modifier**: it shifts which faces pass the hit gate, and leaves the gate's shape alone.

**Cover is a parameter, not an assumption.** A target in cover worsens the attack's Ranged Skill by 1 ([rules/13](rules/13-terrain.md#cover)), and the closed form read the weapon's skill straight off the profile until 2026-08-16, so every expectation quoted beside a shot into terrain described a target standing in the open. `hit_probability` also carries the two bounds the naive `(7 − RS) / 6` drops: an unmodified 1 always fails and an unmodified 6 always hits, so RS 6 in cover resolves at 7 and still lands 1 in 6 rather than being reported as impossible. Recordings carry `in_cover` per shot from schema 2.4, so `expected_damage`, `hit_probability` and `wound_probability` on a `CombatResultSnapshot` are computed under the rules the dice were rolled under; the narrator marks those shots `in cover`.

Two things it still does not model, both by construction: it returns **wounds, not casualties** (damage does not spill between models, and it knows nothing about allocation, so `damage > 1` against one-wound models overstates a volley), and there is no invulnerable save because [InSv is absent everywhere](rules/implementation-status.md).

`expected_damage_matrix` deliberately passes **no** cover, so the observation block is the open-ground expectation for every pair. Cover is a fact about two positions rather than two stat lines: folding it in would collapse the per-profile memoisation the matrix exists for, and it would change the network's input — a scenario change to be measured, not a correction to apply silently.

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

Attacker × defender expected damage tables (hit chance, wound chance, expected value) computed from weapon profiles and target stats. Dual purpose: observation feature for the transformer (perfect information, mirroring real player capability) and explainability tool. **The stat-line half of this shipped** — `expected_damage_matrix` is in the per-model block (see [model/CLAUDE.md](../wargame_rl/wargame/model/CLAUDE.md)). What is still open is the *positional* half: an entry that changes as the pair moves, which today means cover. It is one number per model pair rather than per profile pair, so it costs a real computation per step and changes what the network sees — screen it like a shaping term rather than shipping it as a fix.
