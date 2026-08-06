# Testing

Applies to everything under `tests/`. General testing philosophy lives in the root `CLAUDE.md`
("Good Enough" Testing); this file covers the project-specific setup.

## Setup

- Pytest + shared fixtures in `conftest.py`; run via `just test`; coverage → `coverage.xml`
- Fixtures: `n_steps`, `env`, `experiences`, `replay_buffer`, `policy_net` (parametrized MLP+Transformer), `dqn_mlp_net`, `dqn_transformer_net`, `ppo_net`, `ppo_transformer_net`

## Rules

- No `@lru_cache` on fixtures — use `scope="module"`/`scope="session"` instead
- Use `RL_Network` (not `MLPNetwork`) when accepting parametrized `policy_net`
- Prefer integration tests; test through public APIs (add properties rather than accessing `_private` attrs)
- Type-annotate fixtures and test functions
- `wargame_rl/wargame/envs/interactive_demo.py` is NOT a test
- When env default config changes stepping (e.g. `skip_phases`), tests that rely on per-phase or all-phase behaviour must set config explicitly (e.g. `skip_phases=[]` in a shared `_make_env` or in the test)

## Test Files

**Env & clock** — `test_env` (reset/step) · `test_clock_integration` (phases, skip_phases, rounds) · `test_game_clock` (clock unit) · `test_turn_execution_fairness` · `test_fixed_placement` (config/placement) · `test_rollout_env_lifecycle` · `test_objective_control_distance_cache`

**Rules & combat** — `test_shooting_action` · `test_shooting_resolution` · `test_shooting_opponent` · `test_shooting_baseline` · `test_action_masking` · `test_los` · `test_wounds` · `test_killing_reward` · `test_terrain` · `test_terrain_observation` · `test_terrain_render` · `test_random_terrain` · `test_exposure_metric` (exposure, terrain proximity, `firepower_ratio` — including that a concentrated force outguns what it exposes, the case the metric's first version got backwards)

**Reward & mission** — `test_reward_phases` · `test_new_per_model_calculators` · `test_per_model_credit` · `test_player_ahead_on_vp_criteria` · `test_mission_vp` · `test_killing_reward` · `test_models_lost_reward` (the loss penalty must be global — per-model is identically zero on one-wound models) · `test_curriculum_configs` (every 25v25 config must keep `vp_gain` + a per-model calculator; also pins the batch-2 and batch-3 arm factorials)

**Model & training** — `test_dqn` (networks/loss/training) · `test_ppo` · `test_agent` (actions/episodes) · `test_state` (obs/batch tensors) · `test_transformer_shooting_policy` (shooting head, dead-token masking) · `test_batched_eval` (lockstep eval waves) · `test_training_diagnostics` · `test_train_resume` · `test_train_run_name` · `test_artifact_retention` (the Wandb artifact-backlog sweep; the ordering is pinned because cloud deletion is unrecoverable) · `test_simulate` · `test_z_e2e_training`

**Baselines & opponents** — `test_baselines` (scripted reference policies) · `test_opponents` (opponent system)

**State I/O** — `test_snapshot` · `test_state_injection` · `test_event_stream` · `test_narrator`

**Cross-cutting** — `test_integration` (backward compat) · `test_v9_milestone_validation` · `test_imports` · `test_interactive_demo`

## New Features

Cover: config validation · env integration · obs/info · tensor shapes · DQN forward · backward compat

Run `just validate` before pushing.
