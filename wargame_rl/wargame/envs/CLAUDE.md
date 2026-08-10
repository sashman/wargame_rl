# Gymnasium Env Patterns

Applies to everything under `wargame_rl/wargame/envs/`.

## Structure

- `WargameEnv` extends `gymnasium.Env`; typed obs/action/info/config (all Pydantic)
- Config: `WargameEnvConfig` (pydantic-yaml) from `configs/`
- Actions: polar (angle, speed) pairs per model via `ActionHandler(config, n_models=..., n_shoot_targets=...)`; a `shooting` slice is registered only when `n_shoot_targets > 0`
  - `ActionHandler.best_action_toward(dx, dy, max_step_length=None)` for scripted policies; never hardcode action counts — use `ActionHandler.n_actions`
- `DistanceCache` pre-computes model-objective distances each step; `min_distances_to_same_group` is vectorised and its `.min()` is a *selection*, so a `where(..., inf).min(axis=1)` rewrite stays bit-identical
- **A per-model calculator is called once per model, so anything it computes that does not depend on `model_idx` must be memoised.** Two calculators were ~80% of a 25v25 `env.step()` for exactly this reason. Key the memo on `ctx` identity — a fresh `StepContext` is built every step and held by the env — and give **each cached quantity its own key field**: sharing one key across two quantities computed at different points in a step freezes the later one at its first value. `objective_hold` and `group_cohesion` are the reference patterns; `just measure-throughput` shows the per-calculator split
- Reward: `RewardPhaseManager` (phased reward only); battle-over logic in `domain/termination.py` (`env_components/termination.py` is a re-export plus `get_termination(distance_cache)`)

## Game timing and phases

- `skip_phases`: list of battle phases to auto-advance (default: all non-movement). `max_turns = n_rounds × (5 - len(skip_phases))`; default is 1 step per round (movement only). Set `skip_phases: []` for full per-phase stepping (5 steps per round).
- After the player's turn, the opponent's full turn is auto-executed before the next observation. Turn order: `turn_order` (player / opponent / random).

## Terrain and cover measurement

- `terrain` (fixed list) and `random_terrain` (regenerated each reset) are mutually exclusive. Generation lives in `domain/terrain_placement.py`, called from `place_for_episode`
- **`random_terrain.count` is fixed while size and position vary** — `observations_to_tensor_batch` stacks terrain, so a varying piece count cannot be collated. `mirror: true` keeps the two fixed deployment zones on equal ground
- **Fixed real layouts live outside the config**, in `configs/evaluation/maps/` as `TerrainMapConfig` files (a name plus footprints). `just measure-maps` swaps one onto a scenario for final evaluation, clearing `random_terrain` as it does — leaving the generator on would regenerate a layout at reset and silently discard the map
- **Piece count is what makes cover available, not coverage.** `just measure-terrain <config>` reports *cells hidden from a squad* — the only number that matters, because exposure is "at least one enemy sees me". Batch 1/2's 7 pieces of 5-7 scored 0.058; 29 pieces of 3-7 scores 0.198 at similar coverage. Many small pieces beat few large ones. Tune a profile there, not after a training run
- The packing validator bounds the **expected** footprint, not the worst case — bounding worst-case-square rejects exactly the wall-shaped specs it should allow. `_MAX_LAYOUT_ATTEMPTS` is the real backstop
- `track_exposure: true` accumulates `env.exposure_rate`, `env.terrain_proximity` and `env.firepower_ratio` (`env_components/exposure.py`), all from one `compute_threat_counts` scan per shooting phase — it returns a `ThreatCounts` namedtuple with both threat counts and both "has a target" masks. All are `None`, never `0.0`, when unmeasured — 0.0 would read as "never exposed"
- **Prefer `firepower_ratio` to `exposure_rate`.** Exposure counts only our side, so it falls both when a policy manoeuvres and when it hides; the ratio of shooters separates them. LOS is exactly symmetric (`wargame.py` sorts endpoints to guarantee it) but symmetry is *pairwise* and does not equalise the counts — which is what makes cover worth using at all
- **Count shooters, not targets.** Because LOS is symmetric, an exposed model is exactly a model that can fire, so "enemies we can see" is *their* shooter count. The metric's first version had this backwards and scored `random` top of the table; see [docs/metrics.md](../../../docs/metrics.md) § Cover metrics
- `exposure_rate` deliberately ignores engagement-range gating, and averages over *alive* models so casualties lower it on their own. See [docs/metrics.md](../../../docs/metrics.md) § Cover metrics before comparing it across configs
- **`observe_objective_control` puts per-objective control state on the objective token** (player count, opponent count, radius; 2 → 5 dims, default off). VP is scored on `player_count > opponent_count` per objective, but an objective otherwise reaches the network as *nothing but a location*. Two reward levers keyed on those counts — `closest_objective_v2`'s overstack penalty and `objective_hold`'s `surplus_value` (since removed) — each halved objective occupancy on both seeds, a penalty and a discount failing identically — neither was attributable without this input, so the policy could only experience either as "objectives pay less". The lever that finally worked, `objective_hold.crowding_exponent`, keys on the same counts and is configured alongside this flag for that reason. Widening the objective token is safe (no `_alive_feature_index` trap); turning it on changes the embedding shape, so old checkpoints fail loudly
- **There is no line-of-sight input in the observation.** `observe_threat_count` added one and measured null across both seeds of batch 3, so it was removed. A per-model threat *count* says how many guns bear on a model but not from where, which cannot support "step two cells left and the wall covers me". A directional encoding is untested
- **A unit is not a group.** `unit_id` is a *rules* concept — a model ignores others in its own unit, and in its target's, when tracing sight. `group_id` drives cohesion rewards, spawn clustering, baseline squad assignment and the observation one-hot, and is read by 18 modules. Unset, `unit_id` makes each model **its own unit**; it deliberately does *not* fall back to `group_id`, which defaults to 0 for everyone and would turn a default config into one 25-model unit, switching the sight rule off without saying so
- **Models occlude, and partial cover is a real state.** Three rays per pair — the centre line and two parallel to it, offset by the wider of the two bases. All clear is visible, none is hidden, anything between is **cover**, which worsens the attack's Ranged Skill by 1. The offsets are parallel and symmetric in the pair rather than tangents to the target: the literal tangent construction is directional, and `firepower_ratio` / `exposure_rate` are built on sight being exactly symmetric. **All of it is a no-op at `base_radius: 0`** — no disc occludes and the three rays coincide — which is why every pre-base result still reproduces
- **Sight is a sampled ray, and the seam is a matrix.** `domain/los.py` is the geometry primitive (segments × samples × blockers in one vectorised pass); `domain/sight.py` composes it with the terrain rules. `BattleView.line_of_sight_matrix(origins, targets, candidates)` is the real entry point and `has_line_of_sight_between_points` is a single-pair convenience — **calling the single-pair form in a loop is the shape that measured a 3x regression**, so the shooting mask and the exposure scan take the matrix and pass a `candidates` mask so range gating rules pairs out before they are traced. `los_sample_step` (config, inches, default 0.25) is the resolution guarantee: a blocker thinner than it can fall between two samples and leak sight. Samples sit at **absolute** distances along each segment, so a pair's answer never depends on what else was in the batch — deriving them from the batch's longest segment is cheaper and made splitting a query change its result
- `reset(options={"combat_seed": n})` seeds the dice independently of the layout. The derived draw still happens either way, or the placement stream would shift and the two would stay entangled. `just measure-noise-floor` uses it; on 25v25 the dice contribute *more* spread than the scenario does
- `objective_min_separation` / `objective_terrain_clearance` constrain random objective placement (rejection sampling in `objective_placement`, best-effort). Both default to `None` = the historical draw, which overlaps discs in 25% of episodes and puts 11% of objectives inside a ruin. **Enabling either changes the scenario distribution — runs with and without are not comparable**

## Placement (`ModelConfig`/`ObjectiveConfig` in YAML)

- No key → full auto · Key without x/y → random + config attrs · Key with x/y → fixed
- Properties: `has_fixed_model_positions` / `has_fixed_objective_positions` / `has_fixed_opponent_positions`

## Opponents

- Reuse `WargameModel`/`ModelConfig`; YAML keys: `number_of_opponent_models`, `opponent_models`, `turn_order`, `opponent_policy`
- `TurnOrder`: `player`/`opponent`/`random`
- Policies in `envs/opponent/`, registry: `RandomPolicy("random")`, `ScriptedAdvanceToObjectivePolicy("scripted_advance_to_objective")`, `ScriptedAdvanceAndShootPolicy("scripted_advance_and_shoot")`
- New policies: one class per behaviour, `Scripted` prefix, register as `"scripted_<name>"`; implement `select_action(opponent_models, env, action_mask=None)`
- Read env state through the public properties — `opponent_action_handler`, `player_action_handler`, `last_player_shooting_results`, `last_opponent_shooting_results`, `terrain` — not the private attributes
- A policy that fires must set `shoots = True`; only then does the env overlay range/LOS/engagement validity on its mask (`_opponent_action_mask`). Nothing downstream re-checks legality — see [docs/shooting.md](../../../docs/shooting.md)
- `scripted_advance_and_shoot` and `random` both shoot back (`shoots = True`); `scripted_advance_to_objective` does not. Switching a config to a shooting opponent invalidates every baseline and agent score measured on that config
- `number_of_opponent_models=0` (default) → no policy, env unchanged
- **`scripted_advance_to_objective` parks permanently.** It returns `STAY` unconditionally once inside an objective radius, so measured total opponent displacement is exactly 0.0 from round 9 onward, and its 0.3 centroid blend leaves **at least one objective with zero opponents in every episode**. Its final allocation is a deterministic function of initial positions, so it is predictable at reset. Exploiting that is *less* effective than it sounds — see `contest_and_spread` in `baseline/`, which does exactly this and still loses to `squad_march_shoot`

## Rendering

- `"human"` (Pygame) / `"rgb_array"` (video); renderer injected via constructor
- Player: blue/green circles · Opponents: red/warm triangles
- Use a single FPS cap for human rendering; avoid phase-conditional throttle — it breaks when default stepping changes (e.g. skip_phases)

## Adding Features

1. Config → `types/config/` (`entities.py` per-entity, `terrain.py` terrain, `battle.py` turn order/opponent/mission, `env.py` scenario-level). `__init__.py` re-exports, so importers still use `types.config`. **Every config model sets `model_config = ConfigDict(extra="forbid")`** — a new one must too, and `tests/test_config_rejects_unknown_keys.py` enumerates them so it fails if you forget. Pydantic's default *ignores* an unknown key, which turns a typo into a silent no-op: `objective_radius_sze=99` used to give you radius 1 and no warning, i.e. an arm not measuring what its config claims. Strictness stops at the model boundary — `params` dicts on calculators and policies stay free-form, since each registry entry defines its own
2. Logic → `env_components/`
3. Obs → `env_observation.py`/`env_info.py` + obs builder
4. Tensor → `model/common/observation.py`
5. Network → `TransformerNetwork` in `model/net.py`
6. Reward if signals change · Renderer if visual · Tests + backward compat

## Design

- **The board is continuous.** Locations are real points, built through `position()` / `zero_position()` from `domain/value_objects.py` — never a literal dtype. `POSITION_DTYPE` is the single declaration, which is what made the move off the grid one edit rather than a hunt; a missed site would truncate silently with no exception and no failing test. **Arithmetic widens too** — anything added to or clipped against a position must carry `POSITION_DTYPE`
- **A model has a base.** `base_radius` (config, inches) resolves through `RulesQuantities` and is stored on `WargameModel`. It is read off the model rather than passed to `compute_distances`, because there are 17 call sites and a forgotten argument would silently score one of them under dimensionless rules. It shortens the objective distance (`norms_offset`), separates models at placement, insets the board clamp, and widens engagement to base-to-base. Radius 0.0 is the default and reproduces the point-model behaviour every pre-continuous result was measured under
- **A footprint is a continuous rectangle, and configs author cells.** `Footprint.from_cell_rect` is the single boundary that converts: `(5,5,5,5)` names one cell and becomes `[5,6]x[5,6]`. Read literally as a continuous rect it would have zero area. The same off-by-one lurks in mirroring — reflect about `width`, not `width - 1`. Nothing fails when you get this wrong; the terrain just quietly gets smaller
- Pre-compute expensive ops at `__init__` (numpy vectorized)
- **Rules distances are authored in inches; coordinates are in units.** `inches_per_unit` (default 1.0) is the mapping, `domain/scale.py` owns the conversion, and `resolve_rules_quantities(config)` resolves every rules distance into units **once at construction** — read them off `BattleView.rules_quantities`, never divide at runtime. `domain/rules_constants.py` holds the *rules'* values (mirrored from `docs/rules/constants.yaml`, pinned by a test); where the env deliberately differs, the config states it and `docs/rules/implementation-status.md` records it. Do not "fix" a divergence by wiring in a rules constant — every shipped baseline was measured under the env's value
- **`RulesQuantities` holds only what is read through it.** Add a field with the code that consumes it, never ahead of it: a config field that is settable but inert is the same failure this project hit with reward levers the agent could not observe
- Track rendering state (e.g. `previous_location`) in same change
- New entities mirror existing patterns; always backward compatible
- Follow [docs/ddd-envs.md](../../../docs/ddd-envs.md): keep domain logic in `domain/`, use `BattleView` for read-only state, and preserve dependency direction (domain → types only; reward/renders → BattleView)
