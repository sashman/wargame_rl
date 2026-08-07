# Implementation status

What the environment implements against the [rules reference](README.md), and where it
knowingly departs.

Status values:

- **implemented** — the environment follows the rule.
- **partial** — some of the rule is in place; the missing part is named.
- **divergent** — the environment does something *different* on purpose. Not a bug.
- **absent** — nothing in the environment corresponds to the rule.

Every **divergent** row is a deliberate simplification, and saying so is the point: this
table is the roadmap for the next implementation phase.

Updated 2026-08-07 for continuous positions and model bases — see
[the report](../../reports/2026-08-07-continuous-space-and-model-bases.md). Rows not
touched by that work still describe the environment accurately.

## Board and timing

| Rule | Status | Owner / note |
|---|---|---|
| [Distances in inches](README.md#conventions) | implemented | `domain/scale.py` defines `inches_per_unit` (default 1.0) and `domain/rules_quantities.py` resolves every rules distance into board units once at construction. Coordinates are in units; rules distances are authored in inches. |
| [Board 44" × 60"](15-missions-and-scoring.md#setting-up-a-battle) | **divergent** | `board_width` / `board_height` are in units, default `50 × 50`; the shipped scenarios use `60 × 44`. At the default scale that is 60" × 44" — the right area, landscape rather than portrait. |
| [Five battle rounds](07-battle-round.md) | **divergent** | `number_of_battle_rounds` defaults to **100**. Episodes are sized for training, not for the tabletop. |
| [Round → turn → five phases](07-battle-round.md#player-turns) | implemented | `envs/types/game_timing.py` (`BattlePhase`), `envs/domain/game_clock.py`. |
| [Phase order](07-battle-round.md#player-turns) | implemented | `BATTLE_PHASE_ORDER`. |
| [Setup step order](15-missions-and-scoring.md#setting-up-a-battle) | partial | `SetupPhase` enumerates the steps; `game_clock.skip_setup` runs straight past them. |
| [Start/end triggers, rules sequencing](01-core-concepts.md#resolving-simultaneous-rules) | absent | Only one hook exists — `on_before_advance`, fired on every phase boundary by `envs/domain/turn_execution.py`. |
| [Battle continues after a wipe](15-missions-and-scoring.md#ending-the-battle) | implemented | `terminate_on_player_elimination` defaults to `False`. |

## Units and models

| Rule | Status | Owner / note |
|---|---|---|
| [Unit contains models](01-core-concepts.md#armies-units-and-models) | **divergent** | There is no unit entity. `WargameModel` (`envs/domain/entities.py`) is an individual; `group_id` is the closest thing to a unit and carries no rules of its own. |
| [Move (M)](02-unit-profiles.md#model-characteristics) | **divergent** | Not per-model. One global `max_move_speed` in the config. |
| [Toughness, Save, Wounds](02-unit-profiles.md#model-characteristics) | implemented | `ModelConfig` → `WargameModel.stats` (`toughness`, `save`, `max_wounds`, `current_wounds`) via `envs/domain/battle_factory.py`. |
| [Invulnerable save (InSv)](02-unit-profiles.md#model-characteristics) | absent | No field on `ModelConfig`; `resolve_shooting` checks one save only. |
| [Resolve (Rv)](02-unit-profiles.md#model-characteristics) | absent | — |
| [Control Value (CV)](02-unit-profiles.md#model-characteristics) | **divergent** | Control is a headcount of alive models in range; CV is implicitly 1 for everyone. Range is now measured from the base **edge** (`distance_cache.py`, `model_obj_norms_offset`). |
| [Weapon profile](02-unit-profiles.md#weapon-characteristics) | partial | `WeaponProfile` carries `range`, `attacks`, `ballistic_skill`, `strength`, `ap`, `damage`. No `melee_skill`, no multiple profiles. |
| [Modifier order and clamps](02-unit-profiles.md#modifiers) | absent | Nothing modifies a characteristic at runtime, so nothing needs clamping. Pydantic bounds on `ModelConfig`/`WeaponProfile` are validation, not the rules' clamps. |
| [Random characteristics](02-unit-profiles.md#random-characteristics) | absent | Every characteristic is a fixed integer. |
| [Healing and reviving](02-unit-profiles.md#wounds) | absent | `take_damage` only ever decreases. |
| [Keywords](02-unit-profiles.md#keywords) | absent | No keyword system, so every keyword-gated rule below is unreachable. |

## Moving

| Rule | Status | Owner / note |
|---|---|---|
| [Move as a distance budget](03-moving.md#making-a-move) | partial | Polar encoding — one (angle × speed) displacement per model per step, now applied **exactly** rather than rounded to a lattice. A model never travels further than its Move, including when collision response redirects it. |
| [Cannot cross the board edge](03-moving.md#making-a-move) | implemented | Positions are clipped to the board. |
| [Cannot move through enemy models](03-moving.md#making-a-move) | implemented | `domain/movement.py` sweeps the base and stops it short of the first enemy base on the path. Friendly bases may be passed through but not ended on, per the rules. |
| [Coherency (2" / 9")](03-moving.md#coherency) | **divergent** | `group_max_distance` now derives from the 9" bound, but it only shapes placement and the `group_cohesion` reward. The 2" nearest-neighbour rule has no consumer, nothing enforces coherency, and nothing destroys models for breaking it. |
| [Engagement range 2"](03-moving.md#engagement) | partial | Derived from the rules and measured **base to base** (`shooting_masks.py`, `base_separation`). Horizontal only — the board has no height, so the 5" vertical term has no analogue. Still used only to gate shooting. |
| [Remain stationary](09-movement-phase.md#remain-stationary) | implemented | The `"stay"` action slice, `STAY_ACTION = 0`. |
| [Normal move](09-movement-phase.md#normal-move) | implemented | The `"movement"` action slice. |
| [Advance move](09-movement-phase.md#advance-move) | absent | `WargameModel.advanced_this_turn` exists and is read by the shooting mask, but nothing ever sets it. Dead until the advance move lands. |
| [Fall-back move](09-movement-phase.md#fall-back-move) | absent | — |
| [Set up / deployment zones](03-moving.md#setting-up) | partial | `envs/domain/placement.py` places models randomly inside a zone; no alternating deployment, no *wholly within* check. |

## Attacks

| Rule | Status | Owner / note |
|---|---|---|
| [Select weapons](04-making-attacks.md#1-select-weapons) | **divergent** | `wargame.py:_resolve_shooting_action` fires `weapons[0]` only. A model with two weapons uses one. |
| [Select targets — visible, in range, unengaged](04-making-attacks.md#2-select-targets) | implemented | `env_components/shooting_masks.py:compute_shooting_masks`. |
| [Identical attacks / pooling](04-making-attacks.md#identical-attacks) | absent | Each shot is resolved on its own. |
| [Hit rolls, critical hit on 6, unmodified 1 fails](05-attack-sequence.md#1-hit-rolls) | implemented | `envs/domain/shooting.py:resolve_shooting`. |
| [Wound ladder](05-attack-sequence.md#2-wound-rolls) | implemented | `envs/domain/shooting.py:wound_roll_threshold`. Asserted by `tests/test_rules_constants.py`. |
| [Highest Toughness in a mixed unit](05-attack-sequence.md#mixed-toughness) | absent | Targets are individual models. |
| [Allocation groups and order](05-attack-sequence.md#3-save-rolls) | absent | Damage is applied to the targeted model directly. |
| [Save vs AP](05-attack-sequence.md#4-inflict-damage) | implemented | `save + ap`, with an unmodified 1 always failing. |
| [Excess damage is lost](05-attack-sequence.md#4-inflict-damage) | implemented | `WargameModel.take_damage` clamps at 0. |
| [Destruction resolves after the attacking unit finishes](05-attack-sequence.md#suffering-damage-and-being-destroyed) | **divergent** | Kills apply immediately, recorded by `PairedShootingResult.killed`. |
| [Piercing damage](06-visibility-and-damage.md#piercing-damage) | absent | — |
| [Backfire rolls](06-visibility-and-damage.md#backfire-rolls) | absent | — |

## Shooting phase

| Rule | Status | Owner / note |
|---|---|---|
| [Normal shooting](10-shooting-phase.md#normal-shooting) | implemented | The `"shooting"` action slice; see `docs/shooting.md`. |
| [Advance blocks normal shooting](10-shooting-phase.md#normal-shooting) | partial | The mask reads `advanced_this_turn`, which is never set — so the gate is dormant, not wrong. |
| [Run-and-gun / sidearm / indirect shooting](10-shooting-phase.md) | absent | Only one shooting type exists. |
| [Engaged units cannot shoot](10-shooting-phase.md) | implemented | `compute_shooting_masks` rejects an attacker within `engagement_range` of any opponent. |
| [Engaged large models can be shot at](10-shooting-phase.md#shooting-at-engaged-large-models) | absent | No `MONSTER`/`VEHICLE` distinction. |

## Charge and fight

| Rule | Status | Owner / note |
|---|---|---|
| [Charge phase](11-charge-phase.md) | absent | The phase exists in `BattlePhase` and is a stub — only `"stay"` is legal. It is in `skip_phases` by default. |
| [Fight phase](12-fight-phase.md) | absent | Same — a stub in `skip_phases`. |
| [Strikes First](16-ability-reference.md#strikes-first) | absent | — |
| [Pile in / consolidate](12-fight-phase.md) | absent | — |

## Terrain and visibility

| Rule | Status | Owner / note |
|---|---|---|
| Terrain **walls** (structures inside a piece) | absent | A footprint is an outline only. The L- and U-shaped walls that break sight *within* a ruin are not modelled; a concave footprint is not a stand-in for one. |
| [Line of sight](06-visibility-and-damage.md#visibility) | implemented | `domain/visibility.py` samples the segment between two models and tests every blocker vectorised. Three rays — centre to centre and the two outer tangents — give the full *hidden* / *visible* / *fully visible* split. |
| [Terrain categories](13-terrain.md#terrain-categories) | **divergent** | One category. `Footprint` (`envs/domain/terrain.py`) is a polygon outline that blocks line of sight. |
| [Terrain and movement](13-terrain.md#terrain-and-movement) | absent | Movement ignores terrain completely — models pass through footprints freely. |
| [Solid: see out of and into a feature](13-terrain.md#solid) | partial | `Terrain.blocking_footprints_for_endpoints` excludes any footprint containing either endpoint, which reproduces the see-out and see-into behaviour in two dimensions. No height, so the 3" threshold has no analogue. |
| [Obscuring](13-terrain.md#obscuring) | **divergent** | Achieved by the same footprint blocking, keyed off the feature rather than an enclosing terrain area. |
| [Cover (−1 RS)](13-terrain.md#cover) | implemented | A target that is visible but not *fully* visible has cover, and `resolve_shooting` worsens the attack's hit target by 1. Applies to the player's shooting; the opponent's resolution path does not yet check it. |
| [Hidden and detection range](13-terrain.md#hidden) | absent | — |
| [Elevated fire](16-ability-reference.md#elevated-fire) | absent | The board has no height. |

## Objectives and scoring

| Rule | Status | Owner / note |
|---|---|---|
| [Objective markers, within 3"](14-objectives.md#what-an-objective-is) | implemented | `objective_radius_size` is authored in inches and defaults to the rules value of 3". |
| [Terrain objectives](14-objectives.md#what-an-objective-is) | implemented | An objective may be given a `polygon`: the area itself is the objective, and a model is in range while its base overlaps it. Marker objectives remain the default. |
| [Level of control](14-objectives.md#level-of-control) | implemented | `env_components/distance_cache.py:objective_ownership_from_norms_offset` — strictly greater count controls, ties are uncontrolled. |
| [Control re-evaluated at the end of every phase](14-objectives.md#level-of-control) | **divergent** | Evaluated only when VP are scored, on leaving the command phase (`wargame.py:_on_before_advance`). |
| [Suppression zeroes Control Value](01-core-concepts.md#suppression) | absent | No suppression. |
| [Secured objectives](14-objectives.md#secured-objectives) | absent | — |
| [VP caps](15-missions-and-scoring.md#caps) | **divergent** | `DefaultVPCalculator` uses `vp_per_objective=5`, `cap_per_turn=15`, `min_round=2`. The per-round cap matches; there is no total cap. |
| [Most VP wins, ties draw](15-missions-and-scoring.md#determining-the-victor) | partial | VP are tracked for both sides; success is defined by the configured criteria (`player_vp_min`, `player_ahead_on_vp`) rather than a plain comparison. |

## Abilities

| Rule | Status | Owner / note |
|---|---|---|
| [Every weapon ability](16-ability-reference.md#weapon-abilities) | absent | `WeaponProfile` has no ability field. |
| [Every unit ability](16-ability-reference.md#unit-abilities) | absent | `ModelConfig` has no ability field. |

Adding either means a new field on the config, a place for it in the observation tensor,
and a branch in `resolve_shooting` — see `wargame_rl/wargame/model/CLAUDE.md` for the
tensor pipeline, and the checklist in `CLAUDE.md` under *Adding New Entity Types*.
