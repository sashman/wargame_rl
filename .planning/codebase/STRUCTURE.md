# Codebase Structure

**Analysis Date:** 2026-04-02

## Directory Layout

```
wargame_rl/                          # Repository root
├── wargame_rl/                      # Installable Python package
│   ├── __init__.py                  # Gymnasium env registration
│   ├── plotting/
│   │   └── training.py
│   └── wargame/
│       ├── types.py                 # Experience, ExperienceBatch (RL batch types)
│       ├── envs/
│       │   ├── wargame.py           # WargameEnv (Gym facade)
│       │   ├── wargame_model.py
│       │   ├── wargame_objective.py
│       │   ├── interactive_demo.py
│       │   ├── domain/              # Battle rules (no Gym)
│       │   ├── env_components/      # Actions, observation, distances, etc.
│       │   ├── reward/              # Phases, calculators, criteria, registries
│       │   ├── renders/             # Renderer protocol + human implementation
│       │   ├── opponent/            # Policies + registry
│       │   ├── mission/             # VP / mission registry
│       │   └── types/               # Config, observation, info, actions, timing
│       └── model/
│           ├── net.py               # RL_Network, MLPNetwork, TransformerNetwork
│           ├── common/              # Factory, observation tensors, Lightning base, callbacks
│           ├── dqn/                 # DQN agent, Lightning, replay, config
│           └── ppo/                 # PPO agent, Lightning, networks, config
├── examples/env_config/             # Scenario YAML (incl. set_deployments/, with_opponents/)
├── tests/                           # Pytest suite (conftest.py for fixtures)
├── docs/                            # Design docs (ddd-envs, reward-phases, etc.)
├── train.py                         # Training Typer CLI
├── simulate.py                      # Inference Typer CLI
├── main.py                          # Legacy entry / env demo
├── pyproject.toml                   # Project metadata, deps, mypy
├── Justfile                         # Task runner recipes
└── .github/workflows/               # CI
```

Generated or local artifacts (typically gitignored): `checkpoints/`, `wandb/`, `.venv/`.

## Directory Purposes

**`wargame_rl/wargame/envs/domain/`:**

- Purpose: Pure battle domain — aggregate, factory, clock, placement, termination, turn pipeline.
- Contains: Python modules only; no Gym imports in domain layer per `docs/ddd-envs.md`.
- Key files: `battle.py`, `battle_view.py`, `battle_factory.py`, `turn_execution.py`, `game_clock.py`, `placement.py`, `termination.py`.

**`wargame_rl/wargame/envs/env_components/`:**

- Purpose: Bridge Gym spaces and tensors to domain state — actions, observations, caches.
- Key files: `actions.py`, `observation_builder.py`, `distance_cache.py`.

**`wargame_rl/wargame/envs/reward/`:**

- Purpose: Curriculum phases and pluggable reward/success logic.
- Key files: `phase_manager.py`, `step_context.py`, `calculators/registry.py`, `criteria/registry.py`.

**`wargame_rl/wargame/model/common/`:**

- Purpose: Algorithm-agnostic training utilities and env construction.
- Key files: `factory.py`, `observation.py`, `lightning_base.py`, `agent_base.py`, `dataset.py`, `device.py`, `wandb.py`, `checkpoint_callback.py`, `env_config_callback.py`, `record_episode_callback.py`.

**`wargame_rl/wargame/model/dqn/` and `ppo/`:**

- Purpose: Algorithm-specific agents, Lightning modules, configs, and (DQN) replay; PPO adds `networks.py`, `ppo.py`.

**`examples/env_config/`:**

- Purpose: Authoritative YAML scenarios for training and tests; subfolders group deployment layouts and opponent setups.

**`tests/`:**

- Purpose: Unit and integration tests mirroring env, reward, opponents, training smoke tests.

## Key File Locations

**Entry Points:**

- `train.py`: Primary training CLI.
- `simulate.py`: Checkpoint rollout CLI.
- `main.py`: Demos / legacy flags.
- `wargame_rl/__init__.py`: `register()` for Gymnasium id `gymnasium_env/Wargame-v0`.

**Configuration:**

- `pyproject.toml`: Dependencies and mypy strict settings.
- `examples/env_config/*.yaml`: Env scenarios loaded into `WargameEnvConfig` (`wargame_rl/wargame/envs/types/config.py`).
- `wargame_rl/wargame/model/dqn/config.py`: `DQNConfig`, `DQNTrainingConfig`, `TrainingConfig`, `NetworkType`.
- `wargame_rl/wargame/model/ppo/config.py`: `PPOConfig`, `PPOTrainingConfig`.

**Core Logic:**

- `wargame_rl/wargame/envs/wargame.py`: Gym env orchestration.
- `wargame_rl/wargame/model/net.py`: Neural network factories tied to `WargameEnv` spaces.

**Testing:**

- `tests/conftest.py`: Shared fixtures.
- `tests/test_env.py`, `tests/test_reward_phases.py`, `tests/test_ppo.py`, `tests/test_dqn.py`, `tests/test_integration.py`, `tests/test_z_e2e_training.py`: Representative coverage areas.

## Naming Conventions

**Files:**

- Snake_case module names: `observation_builder.py`, `phase_manager.py`, `battle_factory.py`.
- Test files: `test_<area>.py` under `tests/`.

**Directories:**

- Lowercase with underscores: `env_components/`, `reward/calculators/`.

**Symbols (prescriptive for new code):**

- Env config and observation types: `Wargame*` prefix where established (`WargameEnvConfig`, `WargameEnvObservation`).
- Scripted opponent behaviours: `Scripted*` class prefix (e.g. `ScriptedAdvanceToObjectivePolicy` in `wargame_rl/wargame/envs/opponent/scripted_advance_to_objective_policy.py`).
- Registry-discoverable plugins: implement base classes in `reward/calculators/base.py` or `reward/criteria/base.py` and register in the adjacent `registry.py`.

## Where to Add New Code

**New env domain rule (placement, termination, clock):**

- Primary code: `wargame_rl/wargame/envs/domain/` (extend the relevant module; keep imports inward per `docs/ddd-envs.md`).
- Tests: `tests/test_*.py` targeting behaviour without full Gym if possible, plus env integration if needed.

**New reward calculator or success criterion:**

- Implementation: `wargame_rl/wargame/envs/reward/calculators/` or `reward/criteria/`.
- Registration: same subtree `registry.py`; document in `docs/reward-phases.md`.
- Tests: `tests/test_reward_phases.py` or focused new test module.

**New opponent policy:**

- Implementation: `wargame_rl/wargame/envs/opponent/` + register in `wargame_rl/wargame/envs/opponent/registry.py`.
- Docs: `docs/opponent-policies.md` when behaviour is user-visible.

**Observation or action space change:**

- Types: `wargame_rl/wargame/envs/types/`.
- Builder / handler: `wargame_rl/wargame/envs/env_components/observation_builder.py`, `actions.py`.
- Tensor path: `wargame_rl/wargame/model/common/observation.py`.
- Networks: `wargame_rl/wargame/model/net.py` (shape-dependent layers).
- Renderer: `wargame_rl/wargame/envs/renders/` if new visible state.

**New RL algorithm or training feature:**

- Prefer new package under `wargame_rl/wargame/model/<algo>/` mirroring `dqn/` and `ppo/`, reusing `common/lightning_base.py` and `factory.py` where possible.
- Wire CLI in `train.py`.

**New CLI or script:**

- Root-level Typer scripts alongside `train.py` / `simulate.py`, or extend those files; add `just` recipe in `Justfile` when user-facing.

## Special Directories

**`.planning/codebase/`:**

- Purpose: GSD mapper outputs (this file, `ARCHITECTURE.md`, etc.) for planning and execution agents.
- Generated: By mapping workflow.
- Committed: Typically yes, if the team tracks planning artifacts.

**`checkpoints/`:**

- Purpose: Lightning checkpoints and sibling `env_config.yaml` copies from training.
- Generated: Yes (training).
- Committed: No (usually gitignored).

**`.cursor/`:**

- Purpose: Editor rules and skills; not runtime dependencies of the package.

---

*Structure analysis: 2026-04-02*
