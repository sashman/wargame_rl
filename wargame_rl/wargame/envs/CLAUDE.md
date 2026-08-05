# Gymnasium Env Patterns

Applies to everything under `wargame_rl/wargame/envs/`.

## Structure

- `WargameEnv` extends `gymnasium.Env`; typed obs/action/info/config (all Pydantic)
- Config: `WargameEnvConfig` (pydantic-yaml) from `examples/env_config/`
- Actions: polar (angle, speed) pairs per model via `ActionHandler(config, n_models=..., n_shoot_targets=...)`; a `shooting` slice is registered only when `n_shoot_targets > 0`
  - `ActionHandler.best_action_toward(dx, dy, max_step_length=None)` for scripted policies; never hardcode action counts — use `ActionHandler.n_actions`
- `DistanceCache` pre-computes model-objective distances each step
- Reward: `RewardPhaseManager` (phased reward only); battle-over logic in `domain/termination.py` (`env_components/termination.py` is a re-export plus `get_termination(distance_cache)`)

## Game timing and phases

- `skip_phases`: list of battle phases to auto-advance (default: all non-movement). `max_turns = n_rounds × (5 - len(skip_phases))`; default is 1 step per round (movement only). Set `skip_phases: []` for full per-phase stepping (5 steps per round).
- After the player's turn, the opponent's full turn is auto-executed before the next observation. Turn order: `turn_order` (player / opponent / random).

## Terrain and cover measurement

- `terrain` (fixed list) and `random_terrain` (regenerated each reset) are mutually exclusive. Generation lives in `domain/terrain_placement.py`, called from `place_for_episode`
- **`random_terrain.count` is fixed while size and position vary** — `observations_to_tensor_batch` stacks terrain, so a varying piece count cannot be collated. `mirror: true` keeps the two fixed deployment zones on equal ground
- `track_exposure: true` accumulates `env.exposure_rate` and `env.terrain_proximity` (`env_components/exposure.py`). Both are `None`, never `0.0`, when unmeasured — 0.0 would read as "never exposed". Costs one extra shooting-mask build per shooting phase (~4% of step time on 25v25)
- `exposure_rate` deliberately ignores engagement-range gating, and averages over *alive* models so casualties lower it on their own. See [docs/metrics.md](../../../docs/metrics.md) § Cover metrics before comparing it across configs
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

## Rendering

- `"human"` (Pygame) / `"rgb_array"` (video); renderer injected via constructor
- Player: blue/green circles · Opponents: red/warm triangles
- Use a single FPS cap for human rendering; avoid phase-conditional throttle — it breaks when default stepping changes (e.g. skip_phases)

## Adding Features

1. Config → `WargameEnvConfig` in `types/`
2. Logic → `env_components/`
3. Obs → `env_observation.py`/`env_info.py` + obs builder
4. Tensor → `model/common/observation.py`
5. Networks → both `MLPNetwork` and `TransformerNetwork` in `model/net.py`
6. Reward if signals change · Renderer if visual · Tests + backward compat

## Design

- Integer locations; pre-compute expensive ops at `__init__` (numpy vectorized)
- Track rendering state (e.g. `previous_location`) in same change
- New entities mirror existing patterns; always backward compatible
- Follow [docs/ddd-envs.md](../../../docs/ddd-envs.md): keep domain logic in `domain/`, use `BattleView` for read-only state, and preserve dependency direction (domain → types only; reward/renders → BattleView)
