# Phase 5: Shooting Resolution - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Shooting actions resolve damage through the tabletop attack sequence: **hit roll → wound roll → save → damage**, applying wounds to the target model via the existing `take_damage` entry point. Weapon profiles gain full combat stats (attacks, BS, strength, AP, damage) and models gain defensive stats (toughness, save). The agent's observation grows to include combat-relevant features — raw weapon/defense stats and precomputed expected damage per (attacker, target) pair — so the policy can make informed targeting decisions.

Phase 5 does **not** add multi-weapon sub-steps (all weapons fire at one target per action), weapon keywords (Blast, Melta, Lethal Hits, etc.), invulnerable saves, cover bonuses, or morale effects from casualties. It implements the core resolution pipeline and the observation features needed for combat learning.

</domain>

<decisions>
## Implementation Decisions

### Resolution Mechanics
- **D-01:** Shooting uses **stochastic D6 rolls** matching tabletop rules. Hit roll (D6 ≥ BS), wound roll (S vs T table), saving throw (D6 + AP ≥ Sv). Unmodified 6 always succeeds, unmodified 1 always fails, per tabletop convention.
- **D-02:** Use a **dedicated `numpy.random.Generator`** seeded per episode for all combat rolls. This keeps resolution stochastic in training but deterministic for testing and replay when a fixed seed is provided.
- **D-03:** Damage from failed saves is applied via the existing `take_damage(amount)` on `WargameModel`. Excess damage from a single attack is **lost** (does not carry over to other models), matching tabletop rules.

### Wound Roll Table
- **D-04:** The wound roll threshold follows the standard tabletop comparison:

  | Comparison | Required Roll |
  |-----------|--------------|
  | S ≥ 2×T | 2+ |
  | S > T | 3+ |
  | S = T | 4+ |
  | S < T | 5+ |
  | S ≤ T/2 | 6+ |

### Weapon Profile Stats
- **D-05:** Extend `WeaponProfile` with resolution fields: `attacks: int` (number of hit rolls per shooting action), `ballistic_skill: int` (required D6 roll to hit, e.g. 3 means 3+), `strength: int` (for wound roll comparison), `ap: int` (armour penetration modifier, positive integer representing save penalty, e.g. 1 means -1 AP), `damage: int` (wounds inflicted per failed save).
- **D-06:** Default weapon profile calibrated for ~50% chance of dealing at least 1 wound per shooting action: `attacks=2, ballistic_skill=3, strength=4, ap=1, damage=1`. Per-attack probability: 0.667 × 0.667 × 0.667 = 29.6%; P(≥1 wound from 2 attacks) = 50.4%.
- **D-07:** Existing `range` field on `WeaponProfile` is unchanged. All resolution fields should have defaults matching D-06 so configs that only specify `range` (Phase 4 style) get the standard combat profile.

### Model Defensive Stats
- **D-08:** Add `toughness: int` and `save: int` to `ModelConfig`. `toughness` is the wound roll comparison stat; `save` is the base armour save value (e.g. 4 means 4+, lower is better).
- **D-09:** Defaults: `toughness=3, save=4`. Combined with D-06 weapon defaults, this gives a symmetric ~50% wound-per-action baseline for both sides.
- **D-10:** No invulnerable save in Phase 5. The save step uses only `Sv + AP` comparison. Invulnerable saves are deferred to v4.0 (weapon keywords milestone).
- **D-11:** Toughness and save are added to `WargameModel.stats` dict (alongside `max_wounds`, `current_wounds`) and wired through `battle_factory` from `ModelConfig`.

### Observation Features (OBS-02)
- **D-12:** Per player model: add normalized weapon stats to the observation tensor — `attacks`, `ballistic_skill`, `strength`, `ap`, `damage` from the model's weapon profile (use max/first weapon if multiple). Per opponent model: add normalized `toughness` and `save`.
- **D-13:** Add **precomputed expected damage** per (attacker, target) pair as an additional observation feature. Computed from weapon profile vs target defensive stats using the probability tables (P(hit) × P(wound) × P(fail save) × damage × attacks). This gives the agent explicit efficiency information — the same calculation a human player does mentally.
- **D-14:** The ~50% default baseline (D-06, D-09) means the expected damage feature starts at a clean, interpretable value for default configs. Deviations from 50% are immediately meaningful to the policy.

### Advance/Engagement Restrictions
- **D-15:** Add per-model tracking flag `advanced_this_turn: bool` (default `False`) to `WargameModel`. The shooting mask checks this flag — models that advanced cannot shoot. Since advance movement doesn't exist until v3.0, the flag is always `False` and never restricts shooting.
- **D-16:** Add engagement range check to the shooting mask — models within engagement range of an enemy cannot shoot. Since engagement range mechanics don't exist until v3.0, this check always passes (no model is ever "in engagement range"). When v3.0 defines engagement range, the mask automatically restricts.
- **D-17:** Both flags are reset per turn (not per phase) by the domain layer, matching tabletop semantics where "advanced this turn" persists across all phases of a single player turn.

### Claude's Discretion
- Whether shooting resolution lives as a pure domain function in `domain/shooting.py` or as part of `env_components/`
- Exact normalization scheme for weapon/defense observation features (match existing patterns in `observation.py`)
- Whether expected damage is precomputed at episode init and cached, or recomputed each step (wounds reduce targets, eliminations change valid pairs)
- Internal structure of the D6 roll functions (vectorized numpy vs per-attack loop)
- How `advanced_this_turn` reset is wired into the turn execution pipeline
- Exact engagement range threshold value to stub (e.g. 0 or 1 grid cell) — the check exists but never triggers

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` — SHOT-01, SHOT-02, SHOT-04, SHOT-05, SHOT-06, OBS-02
- `.planning/ROADMAP.md` — Phase 5 goal, success criteria, dependencies (Phase 4)

### Prior phase context
- `.planning/phases/01-wounds-elimination/01-CONTEXT.md` — `take_damage(amount)` as sole wound reduction entry point (D-01, D-02), flag-based elimination, fixed model indices (D-03, D-04)
- `.planning/phases/02-alive-aware-observation/02-CONTEXT.md` — Alive/wound features in observation tensor (D-01, D-05), tensor pipeline extension patterns, checkpoint compatibility (D-09)
- `.planning/phases/04-shooting-action-space/04-CONTEXT.md` — Shooting slice in ActionRegistry (D-03), target index K = opponent slot K positional alignment invariant (D-04), stateless dispatch with no-op (D-11, D-12), WeaponProfile has only range (D-06)

### Tabletop rules
- `docs/tabletop-rules-reference.md` — Attack Sequence section (hit roll, wound roll table, saving throw, damage), Shooting section (eligible units, engagement restriction), Models & Datasheets (stat definitions: BS, S, T, Sv, W, AP, D)

### Architecture and domain patterns
- `docs/ddd-envs.md` — DDD structure, BattleView protocol, domain layer purity, extension points
- `.planning/codebase/CONVENTIONS.md` — Naming, registry pattern, config field defaults, type hints

### Implementation touchpoints
- `wargame_rl/wargame/envs/types/config.py` — `WeaponProfile` (extend), `ModelConfig` (extend with T, Sv)
- `wargame_rl/wargame/envs/env_components/actions.py` — `ActionHandler.apply` shooting no-op branch (replace with resolution call)
- `wargame_rl/wargame/envs/domain/entities.py` — `WargameModel`, `take_damage`, `stats` dict, `is_alive`
- `wargame_rl/wargame/envs/domain/battle_factory.py` — `_build_models` stats dict construction (add T, Sv)
- `wargame_rl/wargame/envs/wargame.py` — `WargameEnv.step`, `_apply_player_action`, `_apply_opponent_action`, seeded RNG wiring
- `wargame_rl/wargame/envs/domain/turn_execution.py` — Phase advancement, opponent action application
- `wargame_rl/wargame/envs/env_components/observation_builder.py` — `build_observation`, shooting mask overlay, observation assembly
- `wargame_rl/wargame/envs/env_components/shooting_masks.py` — `compute_shooting_masks` (extend with advance/engagement checks)
- `wargame_rl/wargame/envs/types/model_observation.py` — `WargameModelObservation` (extend with weapon/defense fields)
- `wargame_rl/wargame/model/common/observation.py` — `_models_to_features`, `feature_dim`, tensor pipeline (extend with combat features)
- `wargame_rl/wargame/envs/reward/step_context.py` — `StepContext` (extend with combat outcome fields for Phase 6 reward calculators)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ActionHandler.apply` already recognizes shooting-slice indices and no-ops them — replace the `continue` with a call into resolution logic
- `take_damage(amount)` on `WargameModel` — single entry point for wound reduction, clamped at 0
- `compute_shooting_masks` in `shooting_masks.py` — extend with `advanced_this_turn` and engagement range checks
- `WargameModel.stats` dict — extend with `toughness` and `save` alongside existing `max_wounds` / `current_wounds`
- `_models_to_features` in `observation.py` — central place to widen per-model features for combat stats
- Dedicated `numpy.random.Generator` pattern — `WargameEnv` can create one at reset with the episode seed

### Established Patterns
- Config fields default to no-op values for backward compatibility — weapon resolution fields need defaults matching D-06
- Domain layer is pure (no Gym imports); resolution logic belongs in domain
- Per-model features are horizontal `hstack` of normalized blocks — weapon/defense stats extend this
- `StepContext` is the extensible data carrier for reward calculators — add combat outcome fields here

### Integration Points
- `ActionHandler.apply` during `BattlePhase.shooting` — decode target index from shooting slice, call resolution, apply `take_damage`
- `_apply_opponent_action` mirrors player action — opponent shooting uses the same resolution path
- `battle_factory._build_models` — wire `toughness` and `save` from `ModelConfig` into `stats` dict
- Observation tensor `feature_dim` grows when weapon/defense features are added — network architecture adapts automatically via `observation_space`

</code_context>

<specifics>
## Specific Ideas

- The ~50% wound-per-action baseline (D-06, D-09) is deliberate: default configs produce a clean 50% expected outcome, making the precomputed expected damage observation feature immediately interpretable. Configs that deviate from these defaults create asymmetric matchups the agent must learn to exploit.
- Phase 4 deferred "precomputed probability matrices" (04-CONTEXT.md) is partially fulfilled by D-13 (expected damage in observation). Full attacker × defender matrices for explainability/UI remain deferred.
- Multi-weapon sub-steps (04-CONTEXT.md deferred) are explicitly not in Phase 5. "All weapons fire at one target" semantic continues from Phase 4.

</specifics>

<deferred>
## Deferred Ideas

### Multi-Weapon Sub-Steps (Phase 4 deferred, still deferred)
- Models with multiple weapons independently assign targets per weapon via sub-steps within shooting phase. Not Phase 5 — continues "all weapons fire at one target" semantic.

### Invulnerable Saves (v4.0)
- Some models have invulnerable saves that ignore AP. Deferred to weapon keywords milestone (v4.0). Phase 5 uses Sv + AP only.

### Full Probability Matrices (v9.0 / explainability)
- Complete attacker × defender expected damage tables for UI/explainability. Phase 5 adds per-pair expected damage to observations; full matrix visualization is future work.

### Pointer-Network Attention (Phase 4 deferred, still deferred)
- Cross-attention between acting model tokens and opponent tokens for shooting logits. Network architecture enhancement if shooting learning plateaus.

### Cover Saves (v2.0)
- Terrain-based cover bonus (+1 to armour saves) during shooting resolution. Terrain is v2.0; Phase 5 has no cover interaction.

</deferred>

---

*Phase: 05-shooting-resolution*
*Context gathered: 2026-04-06*
