# Domain-Driven Design in wargame/envs

This document describes the DDD-inspired structure of the wargame environment and how to extend the application without breaking the boundaries.

## Motivation

The environment was refactored so that:

1. **Domain logic lives in one place.** Battle state, placement rules, turn/phase rules, and termination conditions are core to "a wargame battle." Keeping them under `domain/` makes the rules explicit and testable without starting a full Gymnasium env.

2. **The env is a thin facade.** `WargameEnv` orchestrates reset/step and delegates to the domain. It does not own the rules; it wires config to the domain and exposes Gym observation/action spaces. New behaviour is added by extending the domain or the adapters (observation builder, action handler), not by piling logic into the env.

3. **Consumers depend on a read-only view.** Renderers and reward calculators need battle state (models, objectives, clock, config) but must not mutate it. `BattleView` is a protocol that describes exactly what they can see. The env implements it; in the future a standalone `Battle` or a replay could implement it too. This keeps dependencies pointing inward: domain and view are stable, adapters depend on them.

4. **Extension points are clear.** New entity types, new placement rules, new termination conditions, or new reward/rendering behaviour each have a known place. The doc below explains where and how.

## Structure

```
wargame_rl/wargame/envs/
├── domain/                    # The battle: one bounded context, shaped as sub-domains
│   ├── battle.py              # Aggregate root: models, objectives, zones, dimensions
│   ├── battle_view.py         # Protocol: read-only battle state (names result types below)
│   ├── battle_factory.py      # Builds Battle from config
│   ├── kernel/                # Shared kernel every sub-domain imports, importing none
│   │   ├── entities.py        #   WargameModel, WargameObjective
│   │   ├── value_objects.py   #   Position + POSITION_DTYPE, BoardDimensions, DeploymentZone
│   │   ├── scale.py           #   Scale: rules inches <-> board units
│   │   ├── rules_constants.py #   Universal rules values, in inches (mirrors docs/rules/constants.yaml)
│   │   ├── rules_quantities.py#   RulesQuantities: rules distances resolved into units once
│   │   └── dice.py            #   DiceSource port, DiceCall, RollerAdapter
│   ├── battlefield/           # Terrain, sight, layouts, deployment
│   │   ├── terrain.py         #   Footprint, Terrain (LOS-blocking geometry)
│   │   ├── los.py             #   Sampled ray vs padded polygon blockers (vectorised)
│   │   ├── sight.py           #   "Can A see B?": los + terrain + blocking_mask
│   │   ├── map_layout.py      #   MapLayout: one fixed layout, drawn from a pool
│   │   ├── terrain_placement.py # generate_terrain: random per-episode layouts
│   │   └── placement.py       #   place_for_episode, placement helpers
│   ├── sequencing/            # The order of play
│   │   ├── game_clock.py      #   Turn/phase/round logic
│   │   ├── termination.py     #   is_battle_over, check_max_turns_reached
│   │   └── activation.py      #   PhaseActivation: units act one at a time; the declarations
│   ├── movement/              # Moves, engagement, coherency, and the referees of a unit's move
│   │   ├── moves.py           #   resolve_move, back_off_to_unengaged
│   │   ├── engagement.py      #   engagement_matrix, engaged_with_any
│   │   ├── coherency.py       #   The unit coherency predicate: chain, spread, connectivity
│   │   ├── coherency_enforcement.py # The end-of-move revert, attrition, coherency_after_unit_move
│   │   ├── unit_moves.py      #   unit_moved, revert_unit, unit_is_coherent, touched_enemy_units
│   │   └── fall_back.py       #   fall_back_stands
│   ├── attacks/               # What shooting and melee share (05-attack-sequence.md)
│   │   ├── stats.py           #   AttackStats, WeaponStats, MeleeStats, DefenderStats, ShootingResult
│   │   ├── sequence.py        #   wound_roll_threshold, ranged_skill, hit_probability, resolve_attack
│   │   ├── allocation.py      #   allocate_target: the defender's choice
│   │   └── expectation.py     #   expected_attack_damage
│   ├── shooting/              # 10-shooting-phase.md
│   │   ├── targets.py         #   Target legality: visible, in range, unengaged (unit masks)
│   │   ├── cover.py           #   unit_cover_mask: the one cover implementation
│   │   ├── resolve.py         #   resolve_shooting, resolve_shooting_phase, PairedShootingResult
│   │   └── expectation.py     #   expected_damage, expected_damage_matrix
│   └── melee/                 # 11-charge-phase.md, 12-fight-phase.md
│       ├── charge.py          #   charge_stands
│       ├── pile_in.py         #   pile_in, agent_move_is_legal, short_move_stands
│       ├── fight.py           #   resolve_fight, resolve_fight_step, fight_dragged_in_units
│       ├── fight_sequence.py  #   FightSequence: the fight step, resumable; fight_one_model
│       └── consolidate.py     #   consolidate_objective, ConsolidationMode, the mode referees
├── board/                     # Board-wide reads: sampling grid, next-turn threat
│                              #   field, unit matchups. A LEAF -- domain and types only
├── env_components/            # Adapters: actions, observation, distances
├── map_pool.py                # Loads map files into MapLayouts, draws one per episode
├── opponent/                  # Opponent policies + registry
├── mission/                   # VP calculators + registry
├── baseline/                  # Scripted reference policies + evaluation
├── debug/                     # Hand-stepping a live match: undo stack, session loop,
│                           #   opponent overrides, reproducing a recorded episode
├── state/                     # Snapshots, event log, replay, narration, analysis
├── reward/                    # Reward phases, calculators, criteria (use BattleView)
├── renders/                   # Pygame etc. (use BattleView)
├── types/                     # Shared kernel: config, observation/info types,
│                              #   game timing, and geometry (see below)
├── turn_execution.py          # run_until_player_phase, run_after_player_action (phase facade)
└── wargame.py                 # WargameEnv: facade that implements BattleView
```

- **Domain** does not import from `env_components`, `reward`, or `renders`. It may use `types/` (config, game timing). Inside it the arrows point one way: `kernel/` imports nothing of the domain; `battlefield/`, `sequencing/`, `movement/` and `attacks/` import only the kernel; `shooting/` may import `attacks/`, `battlefield/` (sight, for cover) and `movement/`; `melee/` may import `attacks/`, `movement/` and `shooting/`; the aggregate and `battle_view` sit above all of them. `tests/test_per_model_layering.py` pins every arrow.
- **Env** and **env_components** create and use the domain (Battle, factory, placement, clock, termination, turn execution).
- **Reward** and **renders** depend only on `BattleView` (and types); they receive a view in `calculate_reward` / `check_success` / `setup` / `render`.
- **debug** drives a live env, so like `baseline/` it may import `WargameEnv`. This is what splits the debug mode across two packages: its presenter reads a `BattleView` and therefore lives under `renders/`, while the undo stack and session loop hold the env and live here.

## Key concepts

### Battle (aggregate root)

`Battle` holds the current battle state: board dimensions, player models, opponent models, objectives, deployment zones, terrain, and victory points. All mutations to that state go through the aggregate (e.g. placement, VP accrual, or the action handler applying moves to the models held by the battle). The env holds a `_battle` created by `battle_factory.from_config(config)` and delegates reset placement to `place_for_episode(_battle, config, rng)`.

### BattleView (protocol)

`BattleView` is a read-only interface: board size, config, metadata, player/opponent models, objectives, deployment zones, terrain, current turn, last reward (with its per-calculator breakdown and the episode's running total), the last phase's shooting results for each side, game clock state, n_rounds, player/opponent VP and VP deltas, `player_max_ranges` / `opponent_max_ranges`, plus the sight seam — `line_of_sight_matrix` and its single-pair convenience `has_line_of_sight_between_points`. `WargameEnv` implements it so that reward calculators, success criteria, and renderers can take `view: BattleView` instead of the full env. That keeps their contract minimal and makes it easy to test or reuse them with another view implementation (e.g. a replay or a headless battle).

### Domain services

- **battle_factory.from_config**: builds a `Battle` and its entities (models, objectives, zones, terrain) from `WargameEnvConfig`. Module-level functions, not a class.
- **place_for_episode**: places player models, objectives, and opponent models for a new episode (fixed or random from config).
- **GameClock**: advance setup/battle phases, rounds, turns; `is_game_over`.
- **termination**: `is_battle_over(clock, current_turn, max_turns, success_flag, all_eliminated=False)`.
- **los / terrain**: sampled-ray LOS against padded outlines, vectorised over segments and shapes; `Terrain` supplies those outlines and the arrays the trace consumes.
- **scale / rules_quantities**: `Scale` is the only definition of how many rules inches one board unit spans. `resolve_rules_quantities(config)` converts every rules distance into units **once, at construction**, and the env exposes the result on `BattleView.rules_quantities` — runtime reads plain floats and never divides. `rules_constants.py` holds the rules' own values in inches and mirrors `docs/rules/constants.yaml`, pinned by `tests/test_rules_constants.py`. Note the constants are the *rules'* values, not the env's: where a scenario deviates (engagement range is 1", the rules say 2") the config states it and `implementation-status.md` records it.
- **sight**: `line_of_sight_matrix` composes the two — the sampled ray, the terrain outlines that contain neither endpoint (the see-out rule), and the static `blocking_mask`. Kept apart from `los.py` so the geometry primitive stays free of game rules, and so the question "can A see B?" has one home. The batch form is the real entry point: the single-pair `has_line_of_sight_between_points` in a loop is a measured 3x regression.
- **shooting**: `resolve_shooting(weapon, defender, rng, *, in_cover=False)` and `expected_damage(weapon, defender, *, in_cover=False)`, plus `wound_roll_threshold`, `ranged_skill` and `hit_probability` — the last two are the shared hit-roll rule, so the dice and the closed form cannot state it differently. `resolve_shooting_phase` resolves a whole phase from already-decoded `(attacker_idx, target_idx)` shots — decoding the action tuple is the adapter's job (`ActionHandler.decode_shooting_targets`), because the action-space slice lives in `env_components/`.
- **turn_execution**: `run_until_player_phase`, `run_after_player_action` (skip phases, run opponent turn, advance clock).

The env calls these; it does not reimplement their logic.

## Extending the application

### Adding a new entity type

1. **Define the entity** in `domain/kernel/entities.py` (or a new file under `domain/` if you prefer). Follow the same pattern as `WargameModel` / `WargameObjective`: attributes, `reset_for_episode` if it has episode state, and optionally a `to_space()` for the Gym observation space if the env needs it.
2. **Add config** in `types/config/` — `entities.py` for a per-entity model, `terrain.py` for terrain, `battle.py` for turn order / opponent / mission, `coherency.py` for the coherency rule, `env.py` for a scenario-level field. Keep new fields optional or default so existing YAML stays valid. `__init__.py` re-exports everything, so importers use `types.config` either way.
3. **Wire the factory**: in `domain/battle_factory.py`, create instances from config and attach them to the `Battle` (e.g. new list + property). If the aggregate must expose them for observation or rules, add them to `Battle` and to `BattleView`.
4. **Observation**: if the new entity appears in the Gym observation, extend the observation types in `types/`, then in `env_components/observation_builder.py` add the mapping from `view` to that part of the observation (using `BattleView` so the builder stays view-based).
5. **Backward compatibility**: if something used to live at envs root, keep a thin re-export from there that imports from `domain`.

### Adding a new value object

Add a frozen dataclass (or Pydantic model) in `domain/kernel/value_objects.py`. Use it inside the aggregate or in domain services (e.g. placement, factory). If the env or config layer needs to expose it as a tuple/array for compatibility, add a small adapter (e.g. `as_array()`) on the value object or in the facade.

### Adding or changing placement rules

Placement is in `domain/battlefield/placement.py`. To add a new strategy (e.g. by scenario name), extend `place_for_episode` or add a helper that it calls, using `Battle` and config only. The env continues to call `place_for_episode(_battle, config, rng)` after `_battle.reset_for_episode()` and clock reset. Do not put placement logic in `wargame.py`.

Terrain follows the same pattern, and there are three mutually exclusive modes. A fixed `terrain` list is built once by the factory. With `random_terrain`, `place_for_episode` calls `generate_terrain` from `domain/battlefield/terrain_placement.py` and installs the result via `Battle.set_terrain`. With `map_pool`, the *env* draws a `MapLayout` and passes it in as `place_for_episode(..., layout=...)`, which installs its terrain and — when the map carries them — its objectives via `Battle.set_objectives`. Loading map files is the env layer's job (`envs/map_pool.py`) precisely so `domain/` never touches the filesystem: it only ever sees a `MapLayout`, which is a domain value. LOS resolves terrain through the aggregate on every query, so a replacement takes effect immediately with no cache to invalidate.

### Adding termination conditions

Termination is in `domain/sequencing/termination.py`. `is_battle_over` currently combines turn limit, clock completion, side elimination (`all_eliminated`), and a `success_flag` the env computes from the active reward phase's configured success criteria. To add another condition, extend `is_battle_over` (or a helper it calls) with an extra parameter or a small domain service that the env can call. Keep the env to a single call that decides “is the episode over?” so step() stays simple.

### Adding or changing reward / success criteria

Reward calculators and success criteria already take `view: BattleView` and (where needed) `StepContext`. To add a new calculator or criterion:

1. Implement the interface in `reward/calculators/` or `reward/criteria/` (see existing base classes). Use only `view` and `ctx`; do not take the full env.
2. If your logic needs something not on `BattleView` (e.g. a new entity list), add that property to the `BattleView` protocol and to `WargameEnv`, then use it in the calculator/criterion.
3. Register the new class in the reward registry and document it in [reward-phases.md](reward-phases.md) (tables and file layout) so YAML can reference it by type key.

### Adding or changing rendering

Renderers take `view: BattleView` in `setup(view)` and `render(view)`. To add a new renderer or a new panel, implement the renderer interface and use only `view` and the types it exposes (models, objectives, board dimensions, clock state, etc.). If you need a new piece of state for rendering, add it to `BattleView` and to the env (and to the domain if it is part of battle state).

### Adding a new action or phase

Action space and phase-aware masking live in `env_components/actions.py` (ActionRegistry, ActionHandler). The domain exposes turn/phase via `GameClock` and `BattleView.game_clock_state`. To add a new phase or action type, extend the action registry and the handler; keep phase and turn rules in the domain (game_clock, turn_execution) and use them when building masks or applying actions.

## Dependency direction

- **Domain** → types (config, game timing, geometry). Domain does not import env_components, reward, or renders.
- **Env** → domain, board, env_components, reward, renders, types. The env is the only place that ties them together.
- **Reward / Renders / Board** → `BattleView`, types. They do not import the env class or the aggregate; they receive a view.
- **Renders may import Board; Board must never import Renders, `env_components`, `reward` or `wargame.py`.** `board/` is a
  leaf, and `tests/test_board_layer_is_a_leaf.py` pins it with an AST walk. The arrow exists because the sampling grid has
  to be shared: the threat overlays and the threat field must sample the same cells or they disagree about the same ground
  while both look right. ⚠ Only `renders/v2/control.py` may take that import — it calls itself "the one place v2 touches
  the domain", and the board layer is a domain read like any other. The same test enforces that too.

⚠ **The Reward/Renders bullet is aspirational for `renders/`, and knowingly so.** `renders/v2/control.py` already imports
`env_components/distance_cache.py`. That predates the rule as written; it is recorded here rather than left silently false,
because a reader who trusts the bullet will draw the boundary in the wrong place.

This keeps the domain stable and testable in isolation, and makes it clear where to add new behaviour (domain vs adapters vs env wiring).

### `types/` is the shared kernel, not just a bag of DTOs

The name undersells it. `types/` is the one package everything else may depend
on, and it holds value objects with real behaviour, not only data shapes.

**Geometry belongs in `types/`, as `types/geometry.py`.** This is forced rather
than chosen: `config` has to *validate* shapes — that a polygon is well formed,
that a terrain piece is not thinner than the line-of-sight sample step — so it
needs the geometry type, and `types/` cannot import `domain/` without inverting
the direction above. Putting `Polygon` in `domain/` and importing it from config
would invert it; a prototype tried exactly that and had to move the module.

The alternative considered was a standalone dependency-free `geometry/` package
that both `types/` and `domain/` import. It states the intent more clearly and
costs one directory. It was not taken because `types/` already plays this role —
`game_timing` exports `BattlePhase` and phase-ordering constants that the domain
depends on for correctness, not merely for typing — so a second shared-kernel
package would split the same concept in two. Revisit if `types/` grows a
dependency that geometry should not inherit.

Positions are the counter-example worth knowing: `Position`, `POSITION_DTYPE`
and the constructors live in `domain/kernel/value_objects.py`, not in `types/`, because
nothing in `config` builds one. Config carries plain `x`/`y` integers and
placement turns them into positions. Keep it that way — the whole point of the
single dtype declaration is that it has one home.

## Two application contexts over one domain

Since 2026-09-08 there are **two facades** over `domain/`, and they share the
rules and nothing about the step:

| | `envs/wargame.py` (phase facade) | `envs/per_model/` (per-model facade) |
|---|---|---|
| one `step()` is | one side's whole phase: every model's action at once | one **decision**: open a unit with a declaration, act with one model, name a charge target, or close the turn |
| observation | `WargameEnvObservation` (fixed-width, per-model blocks) | `PerModelObservation`: a `DecisionPoint` stating the next legal decision, plus what the turn revealed |
| dice | three private generators | a `DiceSource` port (`domain/kernel/dice.py`); `SeededDice` reproduces the phase facade's streams |
| scripts | `BaselinePolicy.select_action` per phase | the same policy through `ScriptedSeat`, planned once per phase and replayed in the engine's resolution order |
| reward (stage 1) | per step | the same `StepContext`, settled per **window** — one of our stepped phases — at the point the phase facade's step would have ended |

What is **shared**: `Battle`, `GameClock`, placement, the `ActionHandler`'s
decode and legality helpers (never `apply`), `resolve_move` /
`back_off_to_unengaged`, `resolve_shooting_phase`, the fight scheduler's
rules, every reward calculator, `BattleView`, the mission calculators and the
scripted policies. **Nothing existing was edited** to add the second facade:
the rules it needed that the domain did not yet state landed as new domain
modules — `domain/sequencing/activation.py` (the unit lock and the declaration value
objects), `domain/melee/fight_sequence.py` (the alternating-activation scheduler as
a resumable sequence, with `fight_one_model`), `domain/movement/unit_moves.py`, `movement/fall_back.py`, `melee/charge.py` (the
unit-close judgements and the compulsory consolidation mode) and
`domain/kernel/dice.py` (the port). `tests/test_per_model_layering.py` pins that
those import only `domain/` and `types/`, that `per_model/` never imports
`wargame.py`, and that nothing outside `per_model/` imports it.

The bridge is the contract between them: `tests/test_per_model_bridge.py`
plays a scripted policy through both facades on the same layout and dice and
asserts positions, wounds, VP, reward and the combat dice position agree
**bit for bit** at every point the phase facade's step would have returned —
up to the first **divergence from the phase facade**. The per-model step honours orderings the
whole-army step has no room for: a unit selects its targets after an earlier
unit's casualties are removed, and has its cover judged against the members
it faces; attrition culls every unit on the board, not the active side's; a
battle continues after a wipe; the fight step hands the sequence over while
the other player still has a Strikes First unit; each consolidation mode is
refereed by its own rule and the Engaging drag-in fires at the unit's close.
The facade records the first time each makes a difference
(`PerModelEnv.divergences`, a list of `FacadeDivergence`), and the bridge
requires identity up to the first boundary settled after one — on 4 of 15
cases the two games never part. On a melee scenario it also stops at the
first charge that stands, after which the per-model facade plays the rules
chapter rather than the phase facade (both seats pile in and consolidate; a
striker chooses its target). The phase facade's side of each divergence is
tracked as an issue (#314 to #319) and stays as it is until it lands there.

Every artefact the per-model facade emits carries `facade: "per_model"`
(`PerModelProvenance`); an untagged artefact is the phase facade's, and
`require_per_model` refuses it. The observation the network will train on,
the PPO loop, and the evaluation tooling are stages 2–4 of GitHub issue #283
and do not exist yet.
