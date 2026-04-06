# Architecture

**Analysis Date:** 2026-04-06

## Pattern Overview

**Overall:** Layered application with a **Gymnasium** environment facade, **domain-driven** battle rules under `envs/domain/`, **registry-based** plug-ins for reward calculators, success criteria, opponent policies, and mission VP scoring, and **PyTorch Lightning** modules that orchestrate **DQN** or **PPO** training against the same env.

**Key Characteristics:**

- **Thin env, fat domain:** `WargameEnv` (`wargame_rl/wargame/envs/wargame.py`) wires config, Gym spaces, adapter components, and delegates all battle rules to stateless `domain/` services (`placement`, `game_clock`, `turn_execution`, `termination`, `los`). The env class is ~380 lines of orchestration; the domain layer is ~700 lines of rules.
- **Read-only view for side effects:** Reward calculators, success criteria, renderers, and VP calculators depend on `BattleView` (`wargame_rl/wargame/envs/domain/battle_view.py`), a `Protocol` that `WargameEnv` implements via properties. This prevents side-effect code from mutating the mutable `Battle` aggregate.
- **Config-to-runtime materialisation:** YAML is parsed to `WargameEnvConfig` (`wargame_rl/wargame/envs/types/config.py`) at startup. Factories (`battle_factory`, `RewardPhaseManager.from_configs`, `build_vp_calculator`, `build_opponent_policy`) fully materialise live objects before the first `env.reset()`. Runtime is therefore simple: all lookup and construction has already happened.
- **Registry pattern for YAML-extensible subsystems:** Reward calculators (`wargame_rl/wargame/envs/reward/calculators/registry.py`), success criteria (`wargame_rl/wargame/envs/reward/criteria/registry.py`), opponent policies (`wargame_rl/wargame/envs/opponent/registry.py`), and mission VP calculators (`wargame_rl/wargame/envs/mission/registry.py`) use string-keyed registries so new implementations are added without touching env code.
- **Shared RL spine:** `wargame_rl/wargame/model/common/` holds observation tensor conversion, `BaseAgent`, `WargameLightningBase`, env factory, dataset, device helpers, Wandb integration, and callbacks. Algorithm folders (`dqn/`, `ppo/`) add agents and Lightning modules; `net.py` defines `RL_Network` and concrete architectures (Transformer, MLP).

## Layers

**CLI / scripts (application entry):**

- Purpose: Parse CLI args, load YAML env config, construct env and Lightning trainer, select algorithm and network.
- Location: `train.py`, `simulate.py`, `main.py` (repo root).
- Contains: Typer commands, `AlgorithmType` enum, `NetworkType` enum, checkpoint resume/warm-start logic, Wandb initialisation, Trainer callbacks assembly.
- Depends on: `wargame_rl.wargame.envs.types.WargameEnvConfig`, `wargame_rl.wargame.model.common.factory.create_environment`, algorithm Lightning modules (`DQNLightning`, `PPOLightning`), `net.py`.
- Used by: Operators and CI (`just` recipes).

**Training / RL model layer:**

- Purpose: Collect experience, optimise policies (DQN: replay + Q-network, PPO: actor-critic + GAE), log metrics to Wandb, checkpoint.
- Location: `wargame_rl/wargame/model/` — `common/`, `dqn/`, `ppo/`, `net.py`.
- Contains:
  - Base: `WargameLightningBase` (`common/lightning_base.py`), `BaseAgent` (`common/agent_base.py`), `observation_to_tensor` / `observations_to_tensor_batch` (`common/observation.py`), `RLDataset` (`common/dataset.py`), `create_environment` (`common/factory.py`), callbacks (`checkpoint_callback.py`, `record_episode_callback.py`, `env_config_callback.py`), Wandb helpers (`common/wandb.py`), device helpers (`common/device.py`).
  - DQN: `DQNLightning` (`dqn/lightning.py`), DQN `Agent` (`dqn/agent.py`), `ReplayBuffer` (`dqn/experience_replay.py`), `DQNConfig` / `DQNTrainingConfig` (`dqn/config.py`), `Block` / `LayerNorm` (`dqn/layers.py`).
  - PPO: `PPOLightning` (`ppo/lightning.py`), PPO `Agent` (`ppo/agent.py`), `PPOModel` (`ppo/networks.py`), `PPO_Transformer` (`ppo/ppo.py`), `PPOConfig` / `PPOTrainingConfig` (`ppo/config.py`).
  - Networks: `RL_Network` (ABC), `MLPNetwork`, `TransformerNetwork` (`net.py`).
- Depends on: `WargameEnv`, observation/action types from `wargame_rl/wargame/envs/types/`, `wargame_rl/wargame/types.py` (`Experience`, `ExperienceBatch`).
- Used by: `train.py`, `simulate.py`, tests.

**Environment facade (Gymnasium):**

- Purpose: Implement `gym.Env` contract (`reset`, `step`, `render`), own `_battle` aggregate, expose `observation_space` / `action_space`, orchestrate turn flow, reward calculation, opponent execution, and mission VP scoring.
- Location: `wargame_rl/wargame/envs/wargame.py`.
- Contains: `WargameEnv` class (~380 lines) implementing `BattleView` protocol, holding references to `ActionHandler`, `RewardPhaseManager`, `OpponentPolicy`, `VPCalculator`, `GameClock`, and the `Battle` aggregate.
- Depends on: `domain/`, `env_components/`, `reward/`, `renders/`, `mission/`, `opponent/`, `types/`.
- Used by: Model layer via `create_environment` (`wargame_rl/wargame/model/common/factory.py`), `simulate.py`, Gym registration in `wargame_rl/__init__.py`.

**Domain (battle rules, no Gym):**

- Purpose: Pure battle domain — mutable `Battle` aggregate, entity definitions, value objects, placement, clock, turn sequencing, termination predicate, line-of-sight.
- Location: `wargame_rl/wargame/envs/domain/`.
- Contains:
  - `battle.py` — `Battle` aggregate root: board dimensions, player/opponent model lists, objectives, deployment zones, VP tallies, per-episode reset.
  - `battle_factory.py` — `from_config()` builds a `Battle` from `WargameEnvConfig`; also `create_wargame_models()`, `create_objectives()`, `create_opponent_models()`.
  - `battle_view.py` — `BattleView` `Protocol` (read-only view of battle state).
  - `entities.py` — `WargameModel` (unit with location, stats, wounds, group, distance cache), `WargameObjective` (capture target with location and radius), `alive_mask_for()`.
  - `value_objects.py` — `BoardDimensions` and `DeploymentZone` frozen dataclasses.
  - `game_clock.py` — `GameClock`: tracks setup stages → battle rounds → player turns → phases; phase advancement, round rollover, game-over detection.
  - `placement.py` — `place_for_episode()`: group-aware random or fixed placement within deployment zones for player models, opponent models, and objectives.
  - `termination.py` — `is_battle_over()`: combines elimination, turn limit, clock completion, and all-at-objectives.
  - `turn_execution.py` — `run_until_player_phase()` and `run_after_player_action()`: advance clock, skip excluded phases, auto-execute opponent turns.
  - `los.py` — `has_line_of_sight()` and `iter_los_cells()`: Bresenham grid ray with injectable blocking predicate.
- Depends on: `wargame_rl/wargame/envs/types/` (config, game_timing). Does **not** import `reward/`, `renders/`, or `env_components/`.
- Used by: `wargame.py` and tests that exercise rules without the full Gym stack.

**Env adapters (Gym-specific glue):**

- Purpose: Bridge Gym spaces/tensors to domain state — actions, observations, distances, shooting masks.
- Location: `wargame_rl/wargame/envs/env_components/`.
- Contains:
  - `actions.py` — `ActionHandler` (polar-coordinate movement, phase-aware action masking via `ActionRegistry` with registered slices: stay, movement, shooting), `ActionSlice`, `ActionRegistry`.
  - `observation_builder.py` — `build_observation()` and `build_info()`: construct `WargameEnvObservation` / `WargameEnvInfo` from `BattleView`, including distance updates, action masks, phase-aware shooting mask overlay.
  - `distance_cache.py` — `DistanceCache` (model→objective and optional model→model norms), `compute_distances()`, `objective_ownership_from_norms_offset()`.
  - `shooting_masks.py` — `compute_shooting_masks()`: per-model boolean validity against opponent targets (alive, in range, LOS).
  - `placement.py`, `termination.py`, `game_clock.py` — thin re-exports / wrappers where applicable.
- Depends on: `BattleView`, `types/`, `domain/entities`.
- Used by: `wargame.py`.

**Reward subsystem:**

- Purpose: Curriculum phases with pluggable per-model and global reward calculators and success criteria.
- Location: `wargame_rl/wargame/envs/reward/`.
- Contains:
  - `phase_manager.py` — `RewardPhaseManager`: manages ordered `RewardPhase` list, current phase index, `calculate_reward()` (per-model averaged + globals + terminal bonuses), `check_success()`, `try_advance()` (epoch-gated phase progression with consecutive-success-rate thresholds).
  - `phase.py` — `RewardPhaseConfig`, `RewardCalculatorConfig`, `SuccessCriteriaConfig` (Pydantic models for YAML serialisation).
  - `step_context.py` — `StepContext` dataclass: extensible data carrier passed to all calculators/criteria each step (distance_cache, turn, board size, termination flag, round, battle_phase).
  - `calculators/base.py` — `PerModelRewardCalculator` (ABC, per-model scalar), `GlobalRewardCalculator` (ABC, once per step).
  - `calculators/closest_objective.py` — `ClosestObjectiveCalculator`: distance-shaping reward with non-improvement penalty and optional best-distance bonus.
  - `calculators/group_cohesion.py` — `GroupCohesionCalculator`: negative reward when models exceed `max_distance` from nearest same-group model (requires model→model distances).
  - `calculators/vp_gain.py` — `VPGainCalculator`: global reward proportional to `player_vp_delta / cap_per_turn`.
  - `calculators/registry.py` — `CALCULATOR_REGISTRY` dict, `build_calculator()`.
  - `criteria/base.py` — `SuccessCriteria` ABC with `is_successful()` and optional `vp_threshold_for_terminal_bonus()`.
  - `criteria/all_at_objectives.py`, `criteria/all_models_grouped.py`, `criteria/player_vp_min.py` — concrete criteria.
  - `criteria/registry.py` — `build_criteria()`.
  - `types/model_rewards.py` — `ModelRewards` for per-model reward history tracking.
- Depends on: `BattleView`, `types/`, `domain/entities`. Does not import the concrete env class.
- Used by: `wargame.py` (`phase_manager.calculate_reward`), Lightning eval via `env.phase_manager.check_success`.

**Rendering:**

- Purpose: Human / rgb_array visualisation via a renderer protocol.
- Location: `wargame_rl/wargame/envs/renders/`.
- Contains: `renderer.py` (`Renderer` ABC with `setup`, `render`, `close`), `human.py` (`HumanRender` using Pygame).
- Depends on: `BattleView` — passed as argument to `setup()` and `render()`.
- Used by: `WargameEnv.reset()` / `render()`, `RecordEpisodeCallback`.

**Opponent policies:**

- Purpose: Non-player model actions during opponent turns.
- Location: `wargame_rl/wargame/envs/opponent/`.
- Contains: `policy.py` (`OpponentPolicy` ABC with `select_action()`), `registry.py` (`build_opponent_policy()`, `_auto_register()` imports built-in modules), `random_policy.py`, `scripted_advance_to_objective_policy.py`.
- Depends on: `WargameEnv` reference (for action space and state queries); masking via `ActionHandler.registry`.
- Used by: `wargame.py` during `_apply_opponent_action()`.

**Mission / VP:**

- Purpose: Mission-type scoring: compute VP at end of command phase from round 2.
- Location: `wargame_rl/wargame/envs/mission/`.
- Contains: `vp_calculator.py` (`VPCalculator` ABC, `DefaultVPCalculator` — VP per controlled objective capped per turn, `NoneVPCalculator` — disabled), `registry.py` (`VP_CALCULATOR_REGISTRY`, `build_vp_calculator()`).
- Depends on: `BattleView`, `domain/entities`, `env_components/distance_cache`.
- Used by: `WargameEnv._on_before_advance()` when clock advances past command phase.

**Types & thin entity modules:**

- Purpose: Pydantic config schemas, Gym observation/info/action datatypes, timing enums, lightweight re-exported entity classes.
- Location:
  - `wargame_rl/wargame/envs/types/` — `config.py` (`WargameEnvConfig`, `ModelConfig`, `ObjectiveConfig`, `OpponentPolicyConfig`, `MissionConfig`, `TurnOrder`, `WeaponProfile`), `env_observation.py` (`WargameEnvObservation`), `model_observation.py` (`WargameModelObservation`), `objective_observation.py` (`WargameEnvObjectiveObservation`), `env_info.py` (`WargameEnvInfo`), `env_action.py` (`WargameEnvAction`), `game_timing.py` (`SetupPhase`, `BattlePhase`, `GamePhase`, `PlayerSide`, `GameState`, `BATTLE_PHASE_ORDER`, `NON_MOVEMENT_PHASES`).
  - `wargame_rl/wargame/envs/wargame_model.py` — re-export of `WargameModel`.
  - `wargame_rl/wargame/envs/wargame_objective.py` — re-export of `WargameObjective`.
  - `wargame_rl/wargame/types.py` — `Experience`, `ExperienceBatch` (RL batch types).

**Plotting (auxiliary):**

- Location: `wargame_rl/plotting/training.py` — offline training visualisation, not on the hot training path.

## Data Flow

**Training (high level):**

1. `train.py` loads `WargameEnvConfig` from YAML via `parse_yaml_raw_as` and calls `create_environment()` (`wargame_rl/wargame/model/common/factory.py`).
2. `WargameEnv.__init__` materialises all runtime objects: `Battle` via `battle_factory.from_config()`, `RewardPhaseManager.from_configs()`, `build_vp_calculator()`, `ActionHandler` (player and opponent), `build_opponent_policy()`, `GameClock`.
3. Constructs `RL_Network` subclass (`TransformerNetwork.from_env` or `MLPNetwork.from_env` from `wargame_rl/wargame/model/net.py`) and either `DQNLightning` or `PPOLightning`.
4. Lightning `Trainer.fit()` runs epochs. PPO: each `training_step` collects `n_steps` rollout transitions (single or parallel envs), computes GAE returns, then runs `n_epochs` of minibatch PPO clipped surrogate updates. DQN: replay buffer stores `Experience` tuples, sampled in minibatches for Q-learning.
5. `WargameLightningBase.on_train_epoch_end()` runs evaluation episodes (`_evaluate_episodes`) using `BaseAgent.run_episode()`, checks success via `phase_manager.check_success()`, and calls `try_advance()` for curriculum phase progression.
6. Checkpoints saved via `get_checkpoint_callback()`; optional Wandb logging via `init_wandb()` and `get_logger()`; optional episode recording via `RecordEpisodeCallback`.

**Environment step (`WargameEnv.step`):**

1. `_battle.reset_vp_deltas()` — clear per-step VP deltas.
2. `_apply_player_action(action)` — `ActionHandler.apply()` decodes polar actions and mutates model locations (or no-ops for shooting/dead models).
3. `current_turn += 1`.
4. `run_after_player_action()` — `GameClock.advance_phase()`, then `run_until_player_phase()` which: skips excluded phases (e.g. command, charge, fight when `skip_phases` is configured), auto-executes opponent turns via `_apply_opponent_action()`, fires `_on_before_advance()` VP scoring hook on each clock advance.
5. `compute_distances()` builds `DistanceCache` (model→objective norms, optionally model→model for group cohesion reward).
6. `is_battle_over()` checks: all player/opponent eliminated, turn budget, clock complete, or all-at-objectives (gated by `phase_manager.terminate_on_success`).
7. Build `StepContext` with distance cache, turn info, board dims, termination flag, round, phase.
8. `phase_manager.calculate_reward(self, ctx)` — iterates per-model calculators (weighted, summed per model, averaged across alive models), adds global calculators, adds conditional terminal bonuses (success bonus scaled by remaining turns, VP bonus gated by criteria threshold).
9. `build_observation()` and `build_info()` produce `WargameEnvObservation` / `WargameEnvInfo` from `BattleView`.
10. Return `(observation, reward, terminated, truncated=False, info)`.

**Environment reset (`WargameEnv.reset`):**

1. Reset counters, VP, reward state.
2. `_battle.reset_for_episode()` — clear VP tallies and reset all model wounds/state.
3. `_resolve_player_side()` — set `PlayerSide` based on `TurnOrder` config.
4. `GameClock.reset()` + `skip_setup()` — jump from setup to battle round 1.
5. `place_for_episode()` — fixed or random placement within deployment zones.
6. `run_until_player_phase()` — if opponent goes first, auto-execute their turn.
7. Build initial observation + info, optionally `renderer.setup()` + `render()`.

**Observation tensor pipeline:**

1. `build_observation()` (`env_components/observation_builder.py`) produces `WargameEnvObservation` with: normalised model locations, distances-to-objectives, group IDs, alive flags, wounds, action masks (phase-aware with shooting LOS overlay), battle round/phase, VP.
2. `observation_to_tensor()` (`model/common/observation.py`) converts to 5 tensors: `game_features (6,)`, `objectives (n_obj, 2)`, `player_models (n_models, feature_dim)`, `opponent_models (n_opp, feature_dim)`, `action_mask (n_models, n_actions)`.
3. Feature dim per model = `2 (loc) + n_objectives×2 (dists) + max_groups (group one-hot) + 1 (closest same-group dist) + 3 (alive, wound_ratio, max_wounds_norm)`.
4. `TransformerNetwork.forward()` embeds each tensor type to `embedding_size`, concatenates as token sequence `[game, objectives, player_models, (opponent_models)]`, runs through NanoGPT-style transformer blocks, then applies policy head (per-model logits from model tokens) or value head (scalar from game token).

**Inference (`simulate.py`):**

1. Load checkpoint; rebuild env with same factory; `Agent` + `RL_Network.from_checkpoint()` drives `step` with greedy action selection.

**State Management:**

- **Battle state:** Owned by `Battle` inside `WargameEnv._battle`; `wargame_models`, `objectives`, `opponent_models` are alias properties on the env pointing to aggregate lists. VP tallies live on `Battle` with per-step deltas.
- **RL training state:** Lightning module + PyTorch parameters; replay buffer for DQN in `wargame_rl/wargame/model/dqn/experience_replay.py`; PPO rollout buffer is ephemeral per `training_step`.
- **Episode-local:** `GameClock` state, `current_turn`, `last_step_context`, `last_reward_breakdown`, `episode_reward_breakdown` accumulators on the env.
- **Reward phase state:** `RewardPhaseManager._current_idx`, `_epoch_entered`, `_consecutive_epochs_above_threshold` — persist across episodes within a training run.

## Key Abstractions

**Battle:**

- Purpose: Aggregate root for board dimensions, player models, opponent models, objectives, deployment zones, VP tallies.
- Examples: `wargame_rl/wargame/envs/domain/battle.py`, construction in `wargame_rl/wargame/envs/domain/battle_factory.py`.
- Pattern: Factory builds from `WargameEnvConfig`; `reset_for_episode()` clears VP and model state; `place_for_episode()` mutates locations per episode.

**BattleView:**

- Purpose: `Protocol` for read-only access to battle state, consumed by reward calculators, success criteria, renderers, and VP calculators.
- Examples: `wargame_rl/wargame/envs/domain/battle_view.py`; `WargameEnv` implements via properties (`player_models`, `opponent_models`, `game_clock_state`, VP fields, `config`, `board_width`, etc.).
- Pattern: Dependency inversion — side-effect code depends on the protocol, not on the mutable aggregate or the full Gym env.

**RL_Network:**

- Purpose: Shared `nn.Module` interface for all network architectures — `forward(xs)`, `from_env()` factory, `from_checkpoint()` loader, `policy_from_env()` / `value_from_env()` convenience methods.
- Examples: `wargame_rl/wargame/model/net.py` — `TransformerNetwork` (NanoGPT-style, default, actively developed), `MLPNetwork` (legacy).
- Pattern: `TransformerNetwork` has `encode_state()` → `policy_from_encoded()` / `value_from_encoded()` separation enabling shared-backbone PPO.

**PPOModel:**

- Purpose: Combines policy and value `RL_Network` instances, provides `get_action()` (sampling with log-probs) and `evaluate_actions()` (for PPO ratio computation). Optional shared-backbone via `share_backbone_with()`.
- Examples: `wargame_rl/wargame/model/ppo/networks.py`, `wargame_rl/wargame/model/ppo/ppo.py` (`PPO_Transformer`).

**BaseAgent / WargameLightningBase:**

- Purpose: `BaseAgent` (`common/agent_base.py`) runs episode rollouts (`play_step`, `run_episode`, `run_episode_with_experiences`), tracks reward breakdowns. `WargameLightningBase` (`common/lightning_base.py`) adds shared eval (`_evaluate_episodes`), reward phase advancement (`_advance_reward_phase`), and `on_train_epoch_end` hook.
- Pattern: Algorithm-specific agents and Lightning modules inherit and specialise.

**ActionHandler / ActionRegistry:**

- Purpose: `ActionRegistry` (`env_components/actions.py`) partitions the flat action space into named contiguous `ActionSlice`s (stay, movement, shooting) each valid in specific `BattlePhase`s. `ActionHandler` pre-computes polar displacement tables and applies actions to models.
- Pattern: Phase-aware action masking produced by `get_model_action_masks()`, dead-model restriction to STAY_ACTION, shooting mask overlay from `shooting_masks.py`.

**StepContext:**

- Purpose: Extensible data carrier assembled each step and passed to all reward calculators and success criteria. Contains `DistanceCache`, turn/round/phase, board dimensions, termination flag.
- Examples: `wargame_rl/wargame/envs/reward/step_context.py`.
- Pattern: New mechanics add fields here rather than changing calculator signatures.

**Registries:**

- Purpose: String-keyed constructors enabling YAML-configurable plug-in subsystems.
- Examples:
  - `wargame_rl/wargame/envs/reward/calculators/registry.py` — `CALCULATOR_REGISTRY`: `closest_objective`, `group_cohesion`, `vp_gain`.
  - `wargame_rl/wargame/envs/reward/criteria/registry.py` — `build_criteria()`: `all_at_objectives`, `all_models_grouped`, `player_vp_min`.
  - `wargame_rl/wargame/envs/opponent/registry.py` — `build_opponent_policy()`: `random`, `scripted_advance_to_objective`.
  - `wargame_rl/wargame/envs/mission/registry.py` — `VP_CALCULATOR_REGISTRY`: `default`, `none`.
- Pattern: Each registry maps a string key to a class; a `build_*()` function instantiates with kwargs from config params.

**GameClock:**

- Purpose: Full tabletop timing engine — setup stages, battle rounds, player turns, phase progression. Tracks total steps.
- Examples: `wargame_rl/wargame/envs/domain/game_clock.py`.
- Pattern: State machine with `advance_phase()`, `advance_to_next_player_turn()`, `advance_to_next_round()`, `is_game_over`. Env's `_on_before_advance` hook fires VP scoring at phase transitions.

**Renderer protocol:**

- Purpose: Pluggable rendering backend.
- Examples: `wargame_rl/wargame/envs/renders/renderer.py` (`Renderer` ABC), `human.py` (`HumanRender` — Pygame).

## Entry Points

**Training CLI:**

- Location: `train.py`
- Triggers: `uv run python train.py` / `just train <config.yaml> [algorithm] [network]`.
- Responsibilities: Load env YAML, select algorithm (PPO default, DQN), select network (Transformer default, MLP), build Lightning module and `Trainer`, handle Wandb init, checkpoint resume/warm-start, multi-run suffixes.

**Simulation CLI:**

- Location: `simulate.py`
- Triggers: `just simulate-latest [network_type]` or direct Typer invocation.
- Responsibilities: Load checkpoint, instantiate env and `Agent`, run episodes with optional human render.

**Legacy / demo main:**

- Location: `main.py`
- Triggers: `python main.py` with flags.
- Responsibilities: Optional env smoke test via `wargame_rl/wargame/envs/interactive_demo.py`.

**Gymnasium registration:**

- Location: `wargame_rl/__init__.py`
- Registers `gymnasium_env/Wargame-v0` → `wargame_rl.wargame.envs.wargame:WargameEnv`.

## Error Handling

**Strategy:** Fail fast at config and startup boundaries; validate exhaustively via Pydantic validators and domain `__post_init__`; let training exceptions propagate through Lightning.

**Patterns:**

- **Config validation:** `WargameEnvConfig` has `@model_validator` for entity counts, coordinate consistency, blocking mask shape, opponent policy requirement. `BoardDimensions` and `DeploymentZone` validate in `__post_init__`. `GameClock` validates `n_rounds >= 1`.
- **Registry lookups:** `build_calculator()`, `build_criteria()`, `build_opponent_policy()`, `build_vp_calculator()` all raise `ValueError` with available keys when type is unknown.
- **CLI:** `train.py` / `simulate.py` raise `FileNotFoundError` for missing YAML or checkpoint paths. `simulate.py` catches `RuntimeError` with "size mismatch" messaging for checkpoint/env observation layout incompatibility. Mutually exclusive checkpoint modes validated upfront.
- **Domain:** `GameClockError` for invalid state transitions (advance during setup, advance after game over). Placement raises `RuntimeError` after `_MAX_PLACEMENT_RETRIES` exhausted.
- **General:** No silent exception swallowing; agents use typed env contracts (`WargameEnvAction`, `WargameEnvObservation`).

## Cross-Cutting Concerns

**Logging:** Loguru in `main.py` setup and `RewardPhaseManager` phase transitions; standard `logging` in `simulate.py`; Lightning/Wandb for experiment metrics and training loss. Reward component breakdowns logged per-epoch by `PPOLightning`/`DQNLightning`.

**Validation:** Pydantic models for all configs: `WargameEnvConfig` and sub-models (`wargame_rl/wargame/envs/types/config.py`), `RewardPhaseConfig` (`wargame_rl/wargame/envs/reward/phase.py`), `DQNConfig` / `DQNTrainingConfig` (`wargame_rl/wargame/model/dqn/config.py`), `PPOConfig` / `PPOTrainingConfig` (`wargame_rl/wargame/model/ppo/config.py`), `TransformerConfig` (`wargame_rl/wargame/model/common/config.py`). Strict mypy via `pyproject.toml`.

**Observation contract:** Feature dimension is computed from env config at tensor-pipeline time: `2 + n_objectives×2 + max_groups + 1 + 3`. Networks call `from_env()` which resets the env once to derive sizes, so any observation shape change is automatically picked up by new network instances.

**Authentication:** Not applicable (local ML project; Wandb uses its own API key via environment).

---

*Architecture analysis: 2026-04-06*
