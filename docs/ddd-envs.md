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
├── domain/                    # Domain layer (no Gym, no env_components)
│   ├── battle.py              # Aggregate root: models, objectives, zones, dimensions
│   ├── battle_view.py         # Protocol: read-only battle state
│   ├── battle_factory.py      # Builds Battle from config
│   ├── entities.py            # WargameModel, WargameObjective
│   ├── value_objects.py       # BoardDimensions, DeploymentZone
│   ├── game_clock.py          # Turn/phase/round logic
│   ├── placement.py           # place_for_episode, placement helpers
│   ├── termination.py         # is_battle_over, check_max_turns_reached
│   └── turn_execution.py      # run_until_player_phase, run_after_player_action
├── env_components/            # Adapters: actions, observation, distance cache, re-exports
├── reward/                    # Reward phases, calculators, criteria (use BattleView)
├── renders/                   # Pygame etc. (use BattleView)
├── types/                     # Config, observation/info types, game timing types
└── wargame.py                 # WargameEnv: facade that implements BattleView
```

- **Domain** does not import from `env_components`, `reward`, or `renders`. It may use `types/` (config, game timing).
- **Env** and **env_components** create and use the domain (Battle, factory, placement, clock, termination, turn execution).
- **Reward** and **renders** depend only on `BattleView` (and types); they receive a view in `calculate_reward` / `check_success` / `setup` / `render`.

## Key concepts

### Battle (aggregate root)

`Battle` holds the current battle state: board dimensions, player models, opponent models, objectives, and deployment zones. All mutations to that state go through the aggregate (e.g. placement, or the action handler applying moves to the models held by the battle). The env holds a `_battle` created by `BattleFactory.from_config(config)` and delegates reset placement to `place_for_episode(_battle, config, rng)`.

### BattleView (protocol)

`BattleView` is a read-only interface: board size, config, player/opponent models, objectives, deployment zones, current turn, last reward, game clock state, n_rounds. `WargameEnv` implements it so that reward calculators, success criteria, and renderers can take `view: BattleView` instead of the full env. That keeps their contract minimal and makes it easy to test or reuse them with another view implementation (e.g. a replay or a headless battle).

### Domain services

- **BattleFactory**: builds a `Battle` and its entities from `WargameEnvConfig`.
- **place_for_episode**: places player models, objectives, and opponent models for a new episode (fixed or random from config).
- **GameClock**: advance setup/battle phases, rounds, turns; `is_game_over`.
- **termination**: `is_battle_over(clock, current_turn, max_turns, max_turns_override, all_models_at_objectives_flag)`.
- **turn_execution**: `run_until_player_phase`, `run_after_player_action` (skip phases, run opponent turn, advance clock).

The env calls these; it does not reimplement their logic.

## Extending the application

### Adding a new entity type

1. **Define the entity** in `domain/entities.py` (or a new file under `domain/` if you prefer). Follow the same pattern as `WargameModel` / `WargameObjective`: attributes, `reset_for_episode` if it has episode state, and optionally a `to_space()` for the Gym observation space if the env needs it.
2. **Add config** in `types/config.py` (e.g. number of X, list of X configs, deployment). Keep new fields optional or default so existing YAML stays valid.
3. **Wire the factory**: in `domain/battle_factory.py`, create instances from config and attach them to the `Battle` (e.g. new list + property). If the aggregate must expose them for observation or rules, add them to `Battle` and to `BattleView`.
4. **Observation**: if the new entity appears in the Gym observation, extend the observation types in `types/`, then in `env_components/observation_builder.py` add the mapping from `view` to that part of the observation (using `BattleView` so the builder stays view-based).
5. **Backward compatibility**: if something used to live at envs root, keep a thin re-export from there that imports from `domain`.

### Adding a new value object

Add a frozen dataclass (or Pydantic model) in `domain/value_objects.py`. Use it inside the aggregate or in domain services (e.g. placement, factory). If the env or config layer needs to expose it as a tuple/array for compatibility, add a small adapter (e.g. `as_array()`) on the value object or in the facade.

### Adding or changing placement rules

Placement is in `domain/placement.py`. To add a new strategy (e.g. by scenario name), extend `place_for_episode` or add a helper that it calls, using `Battle` and config only. The env continues to call `place_for_episode(_battle, config, rng)` after `_battle.reset_for_episode()` and clock reset. Do not put placement logic in `wargame.py`.

### Adding termination conditions

Termination is in `domain/termination.py`. `is_battle_over` currently combines turn limit, clock completion, and “all models at objectives.” To add another condition (e.g. “all opponents eliminated”), extend `is_battle_over` (or a helper it calls) with an extra parameter or a small domain service that the env can call. Keep the env to a single call that decides “is the episode over?” so step() stays simple.

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

- **Domain** → types (config, game timing). Domain does not import env_components, reward, or renders.
- **Env** → domain, env_components, reward, renders, types. The env is the only place that ties them together.
- **Reward / Renders** → `BattleView`, types. They do not import the env class or the aggregate; they receive a view.

This keeps the domain stable and testable in isolation, and makes it clear where to add new behaviour (domain vs adapters vs env wiring).
