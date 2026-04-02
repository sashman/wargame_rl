# Architecture

**Analysis Date:** 2026-04-02

## Pattern Overview

**Overall:** Layered application with a **Gymnasium** environment facade, **domain-driven** battle rules under `envs/domain/`, **registry-based** plug-ins for reward and opponents, and **PyTorch Lightning** modules that orchestrate **DQN** or **PPO** training against the same env.

**Key Characteristics:**

- **Thin env, fat domain:** `WargameEnv` wires config, Gym spaces, adapters, and delegates rules to `domain/` services (`placement`, `game_clock`, `turn_execution`, `termination`).
- **Read-only view for side effects:** Reward calculators, success criteria, and renderers depend on `BattleView` (implemented by `WargameEnv`), not on mutable aggregate internals they should not touch.
- **Config-to-runtime:** YAML parses to `WargameEnvConfig` (`wargame_rl/wargame/envs/types/config.py`); factories build battles, VP calculators, opponent policies, and reward phase managers from that config.
- **Shared RL spine:** `wargame_rl/wargame/model/common/` holds observation tensors, dataset, Lightning base, env factory; algorithm folders add agents and Lightning modules; `net.py` defines `RL_Network` and concrete architectures.

## Layers

**CLI / scripts (application entry):**

- Purpose: Parse args, load YAML env config, construct env and Lightning trainer.
- Location: `train.py`, `simulate.py`, `main.py` (repo root).
- Contains: Typer commands, algorithm selection (PPO/DQN), network type selection.
- Depends on: `wargame_rl.wargame.envs.types`, `wargame_rl.wargame.model.common.factory`, algorithm Lightning modules, `net.py`.
- Used by: Operators and CI (`just` recipes).

**Training / RL model layer:**

- Purpose: Collect experience, optimize policies (DQN replay + Q-network; PPO actor-critic + GAE), log to Wandb, checkpoint.
- Location: `wargame_rl/wargame/model/` — `common/`, `dqn/`, `ppo/`, `net.py`.
- Contains: `WargameLightningBase`, `DQNLightning`, `PPOLightning`, `Agent` (per algo), `ReplayBuffer`, `RLDataset`, `observation_to_tensor`, callbacks.
- Depends on: `WargameEnv`, observation/action types from `wargame_rl/wargame/envs/types`, `wargame_rl/wargame/types.py` (`Experience`, `ExperienceBatch`).
- Used by: `train.py`, tests under `tests/`.

**Environment facade (Gymnasium):**

- Purpose: Implement `gym.Env` contract (`reset`, `step`, `render`), own `_battle` aggregate reference, expose `observation_space` / `action_space`, orchestrate turn flow and reward.
- Location: `wargame_rl/wargame/envs/wargame.py`.
- Contains: `WargameEnv`, `BattleView` property surface, wiring to `ActionHandler`, `RewardPhaseManager`, mission VP hook, opponent policy.
- Depends on: `domain/`, `env_components/`, `reward/`, `renders/`, `mission/`, `opponent/`, `types/`.
- Used by: Model layer, `create_environment` in `wargame_rl/wargame/model/common/factory.py`, `simulate.py`, Gym registration in `wargame_rl/__init__.py`.

**Domain (battle rules, no Gym):**

- Purpose: Mutable battle state (`Battle`), placement, clock, turn sequencing after player acts, termination predicate.
- Location: `wargame_rl/wargame/envs/domain/`.
- Contains: `battle.py`, `battle_factory.py`, `battle_view.py` (protocol), `entities.py`, `value_objects.py`, `game_clock.py`, `placement.py`, `termination.py`, `turn_execution.py`.
- Depends on: `wargame_rl/wargame/envs/types/` (config, timing). Does **not** import `reward/`, `renders/`, or `env_components/`.
- Used by: `wargame.py` and tests that exercise rules without full Gym stack.

**Env adapters (Gym-specific glue):**

- Purpose: Map domain/env state to observations, info dicts, discrete polar actions, distance caches; env-local termination helpers if any.
- Location: `wargame_rl/wargame/envs/env_components/` — e.g. `actions.py`, `observation_builder.py`, `distance_cache.py`, `placement.py`, `termination.py`, `game_clock.py` (re-exports/wrappers as applicable).
- Depends on: `WargameEnv` or `BattleView` patterns, `types/`.
- Used by: `wargame.py`.

**Reward subsystem:**

- Purpose: Curriculum phases, per-step reward aggregation, success checks for eval.
- Location: `wargame_rl/wargame/envs/reward/` — `phase_manager.py`, `phase.py`, `step_context.py`, `calculators/`, `criteria/`, registries.
- Contains: Calculators and criteria registered by string keys for YAML; all consume `BattleView` + `StepContext` where applicable.
- Depends on: `BattleView`, `types/`. Does not import the concrete env class for core logic.
- Used by: `wargame.py` (`phase_manager.calculate_reward`, success in Lightning eval via `env.phase_manager`).

**Rendering:**

- Purpose: Human / rgb_array visualization via renderer protocol.
- Location: `wargame_rl/wargame/envs/renders/` — `renderer.py`, `human.py`.
- Depends on: `BattleView`-compatible object passed as `setup`/`render` argument (env implements the view).
- Used by: `WargameEnv.reset` / `render`, training record callback when configured.

**Opponent policies:**

- Purpose: Non-player model actions during opponent turns.
- Location: `wargame_rl/wargame/envs/opponent/` — `policy.py`, `registry.py`, concrete policies (e.g. `scripted_advance_to_objective_policy.py`, `random_policy.py`).
- Depends on: Env reference for masking/state where policies need it (built via `build_opponent_policy`).
- Used by: `wargame.py` during `run_until_player_phase` / `run_after_player_action`.

**Mission / VP:**

- Purpose: Mission-type scoring (VP calculator) keyed off config.
- Location: `wargame_rl/wargame/envs/mission/` — `registry.py`, `vp_calculator.py`.
- Used by: `WargameEnv._on_before_advance` when clock advances.

**Types & thin entity modules:**

- Purpose: Pydantic config, Gym observation/info/action datatypes, timing enums; lightweight `WargameModel` / `WargameObjective` helpers co-located with env package.
- Location: `wargame_rl/wargame/envs/types/`, `wargame_rl/wargame/envs/wargame_model.py`, `wargame_rl/wargame/envs/wargame_objective.py`, `wargame_rl/wargame/types.py`.

**Plotting (auxiliary):**

- Location: `wargame_rl/plotting/training.py` — not on the hot training path unless invoked separately.

## Data Flow

**Training (high level):**

1. `train.py` loads `WargameEnvConfig` from YAML via `parse_yaml_raw_as` and calls `create_environment` (`wargame_rl/wargame/model/common/factory.py`).
2. Constructs `RL_Network` subclass (`TransformerNetwork` / `MLPNetwork` from `wargame_rl/wargame/model/net.py`) and either `DQNLightning` or `PPOLightning`.
3. Lightning `Trainer` runs `training_step` / `validation_step`; agents roll out `env.reset` / `env.step` using `observation_to_tensor` (`wargame_rl/wargame/model/common/observation.py`) and action masks.
4. Checkpoints and optional Wandb logging via `wargame_rl/wargame/model/common/wandb.py` and callbacks (`checkpoint_callback.py`, `record_episode_callback.py`, `env_config_callback.py`).

**Environment step (`WargameEnv.step` in `wargame_rl/wargame/envs/wargame.py`):**

1. Reset VP deltas on `_battle`; apply player action via `ActionHandler` (`env_components/actions.py`).
2. Increment `current_turn`; `run_after_player_action` (`domain/turn_execution.py`) advances clock and runs opponent policy when needed; `_on_before_advance` may apply mission VP via `build_vp_calculator` output.
3. `compute_distances` builds `DistanceCache` (optionally model–model for reward).
4. `is_battle_over` (`domain/termination.py`) combines clock, turn budget, and optional “all at objectives” success flag from phase manager config.
5. Build `StepContext`; `RewardPhaseManager.calculate_reward(self, ctx)` aggregates registered calculators; observation from `build_observation`; info from `build_info`.

**Inference (`simulate.py`):**

1. Load checkpoint; rebuild env with same pattern as training factory; `Agent` + `RL_Network.from_checkpoint` drives `step` with greedy or scripted exploration as implemented in DQN agent.

**State Management:**

- **Battle state:** Owned by `Battle` inside `WargameEnv._battle`; lists like `wargame_models` / `objectives` alias aggregate fields.
- **RL training state:** Lightning module + PyTorch parameters; replay buffer for DQN in `wargame_rl/wargame/model/dqn/experience_replay.py`.
- **Episode-local:** `GameClock` state, `current_turn`, `last_step_context`, reward breakdown accumulators on the env.

## Key Abstractions

**Battle:**

- Purpose: Aggregate root for board, models, objectives, zones, VP tallies.
- Examples: `wargame_rl/wargame/envs/domain/battle.py`, construction in `wargame_rl/wargame/envs/domain/battle_factory.py`.
- Pattern: Factory builds from config; `place_for_episode` mutates for each reset.

**BattleView:**

- Purpose: Protocol for read-only access used by reward and renderers.
- Examples: `wargame_rl/wargame/envs/domain/battle_view.py`; `WargameEnv` implements via properties (`player_models`, `game_clock_state`, VP fields, etc.).

**RL_Network:**

- Purpose: Shared interface for policy/value heads, env-shaped construction, checkpoint loading.
- Examples: `wargame_rl/wargame/model/net.py` (`MLPNetwork`, `TransformerNetwork`).

**BaseAgent / WargameLightningBase:**

- Purpose: Episode rollouts and shared eval hooks (including success via `phase_manager.check_success`).
- Examples: `wargame_rl/wargame/model/common/agent_base.py`, `wargame_rl/wargame/model/common/lightning_base.py`.

**Registries:**

- Purpose: String-keyed constructors for YAML-extensible subsystems.
- Examples: `wargame_rl/wargame/envs/reward/calculators/registry.py`, `wargame_rl/wargame/envs/reward/criteria/registry.py`, `wargame_rl/wargame/envs/opponent/registry.py`, `wargame_rl/wargame/envs/mission/registry.py`.

**Renderer protocol:**

- Examples: `wargame_rl/wargame/envs/renders/renderer.py`, `HumanRender` in `human.py`.

## Entry Points

**Training CLI:**

- Location: `train.py`
- Triggers: `uv run` / `just train` invoking Typer `train` command.
- Responsibilities: Load env YAML, select algorithm and network, build Lightning module and `Trainer`, optional multi-run suffixes (see `docs/multi-run-training.md`).

**Simulation CLI:**

- Location: `simulate.py`
- Triggers: `just simulate-latest` or direct Typer invocation.
- Responsibilities: Load checkpoint, instantiate env and DQN `Agent`, run episodes with optional human render.

**Legacy / demo main:**

- Location: `main.py`
- Triggers: `python main.py` with flags.
- Responsibilities: Optional env smoke test via `wargame_rl/wargame/envs/interactive_demo.py`.

**Gymnasium registration:**

- Location: `wargame_rl/__init__.py`
- Registers `gymnasium_env/Wargame-v0` → `wargame_rl.wargame.envs.wargame:WargameEnv`.

## Error Handling

**Strategy:** Fail fast at CLI and config boundaries; use explicit exceptions for missing files and checkpoint shape mismatches; domain and env avoid silent swallowing of errors.

**Patterns:**

- `train.py` / `simulate.py`: `FileNotFoundError` for missing YAML or checkpoint paths.
- `simulate.py`: Catches `RuntimeError` with “size mismatch” messaging when checkpoint does not match env observation layout.
- Lightning and PyTorch: Standard training exceptions propagate; agents use typed env contracts (`WargameEnvAction`, `WargameEnvObservation`).

## Cross-Cutting Concerns

**Logging:** Loguru in `main.py` setup; standard `logging` in `simulate.py`; Lightning/Wandb for experiment metrics.

**Validation:** Pydantic models for `WargameEnvConfig` and training configs (`wargame_rl/wargame/model/dqn/config.py`, `wargame_rl/wargame/model/ppo/config.py`, `wargame_rl/wargame/model/common/config.py`); strict mypy on the package per `pyproject.toml`.

**Authentication:** Not applicable (local ML project; Wandb uses its own API key via environment, not read here).

---

*Architecture analysis: 2026-04-02*
