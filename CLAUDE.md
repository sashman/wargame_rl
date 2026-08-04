# Wargame RL

Reinforcement learning project that trains agents (DQN, PPO) to play tabletop wargames on a discrete grid. Agents control multiple models (units) using polar-coordinate movement to capture objectives.

## Tech Stack

- **Python 3.13** — UV package manager (`uv.lock`)
- **Gymnasium 1.x** — RL environment (`WargameEnv`)
- **PyTorch + PyTorch Lightning** — DQN and PPO training
- **Wandb** — experiment tracking & video recording
- **Pydantic + pydantic-yaml** — config & type models
- **Typer** — CLI (`train.py`, `simulate.py`)
- **Loguru** — logging
- **Pygame** — human rendering

## Development Tooling

- **Just** — command runner (see `Justfile`)
- **Ruff** — linter & formatter (line length 88, double quotes)
- **Mypy** — strict type checking (`disallow_untyped_defs`, `no_implicit_optional`)
- **isort** — import sorting (Black profile)
- **autoflake** — removes unused imports
- **Pytest** — testing
- **Pre-commit** — hooks for all of the above

## Project Layout

```
wargame_rl/
├── wargame_rl/                    # Main package
│   ├── __init__.py                # Registers Gymnasium env
│   └── wargame/
│       ├── envs/                  # Gymnasium environment, reward, rendering
│       │   ├── wargame.py         # WargameEnv — facade, implements BattleView
│       │   ├── domain/            # Battle aggregate, BattleView, clock, placement,
│       │   │                      #   termination, LOS, shooting, terrain, turn execution
│       │   ├── env_components/    # Adapters: actions, distance cache, observation builder
│       │   ├── baseline/          # Scripted baseline policies + registry + evaluate
│       │   ├── reward/            # Phase manager, calculators, criteria
│       │   ├── mission/           # VP calculators + registry
│       │   ├── opponent/          # Opponent policies + registry
│       │   ├── state/             # Snapshots, event log, replay, narrator, analysis
│       │   ├── types/             # Config, observations, actions, info
│       │   └── renders/           # Pygame renderer
│       ├── model/                 # RL algorithms
│       │   ├── net.py             # RL_Network base, MLPNetwork, TransformerNetwork
│       │   ├── common/            # Shared: lightning_base (eval + baselines + phase
│       │   │                      #   advancement), factory, observation, dataset, callbacks
│       │   ├── dqn/               # DQN: agent, lightning module, replay buffer, config
│       │   └── ppo/               # PPO: actor-critic, lightning module, agent, config
│       └── types.py               # Experience, ExperienceBatch
├── examples/env_config/           # YAML environment configurations
├── tests/                         # Pytest suite with conftest.py fixtures
├── docs/                          # Design docs (movement, reward phases, missions-and-vp,
│                                  #   roadmap, rules, metrics, shooting, terrain)
├── reports/                       # Experiment findings, kept for retrospection
├── scripts/                       # Run-inspection tooling (run_summary, measure_phase_gates,
│                                  #   measure_baselines, measure_checkpoint)
├── train.py                       # Training entry point (Typer CLI)
├── simulate.py                    # Inference/simulation entry point
├── replay_events.py               # Replay / narrate a match event log
├── analyze_events.py              # Analyse / compare match event logs
└── main.py                        # Legacy entry (env test with random actions)
```

## Key Commands

| Task | Command |
|---|---|
| Setup | `just setup` |
| Sync deps | `just dev-sync` |
| Format | `just format` |
| Lint | `just lint` |
| Test | `just test` |
| Full validation | `just validate` |
| Train (PPO, default) | `just train <config.yaml>` |
| Train (DQN) | `just train <config.yaml> dqn` |
| Train multiple configs in parallel | `just train-multi config1.yaml config2.yaml` |
| Ship (branch → commit → push → PR) | `just ship <branch> "<message>"` |
| Simulate latest | `just simulate-latest` |
| Simulate / record a checkpoint | `just simulate <ckpt> <config.yaml>` · `just record-sim <ckpt> <config.yaml>` |
| Test env (random) | `just test-env` |
| Record a match event log | `just record <config.yaml>` |
| Replay / narrate a log | `just replay <file>` · `just replay-summary <file>` |
| Analyse a log | `just analyze <file>` · `just analyze-compare <files...>` |
| Inspect a Wandb run | `just run-summary <run_id> [bucket]` |
| Measure reward-phase gates | `just measure-phase-gates <ckpt> <config.yaml> [n_episodes]` |
| Scripted baselines (floor + bar) | `just measure-baselines <config.yaml> [n_episodes] [record] [seed_base]` |
| Score a checkpoint (baseline-comparable) | `just measure-checkpoint <ckpt> <config.yaml> [n_episodes] [record]` |
| Profile | `just profile <config.yaml> [model] [max_epochs]` |
| Clean | `just clean` |

## Key Components

### Environment

- `WargameEnv` — Gymnasium env with configurable board, models, objectives
- **Polar movement** — actions encoded as (angle × speed) per model
- **Reward phases** — curriculum learning with phased reward configs
- **VP reward and success** — `vp_gain` calculator, `player_vp_min` success criteria, optional terminal VP bonus; observation includes `player_vp_delta` for step-wise VP signal
- **Deployment zones** — configurable spawn areas for player and opponent
- **Group cohesion** — optional penalty for unit separation
- **DDD layering** — `domain/` owns the rules (Battle aggregate, clock, placement, termination, LOS, shooting); `wargame.py` is a facade; reward/renders depend only on the `BattleView` protocol. See [docs/ddd-envs.md](docs/ddd-envs.md)

### Game State I/O (`envs/state/`)

Snapshot/event pipeline for recording and inspecting matches — `GameStateSnapshot`, event-log deltas, `StateExporter` (wired into `step()`), replay, narration, and `analyze_match` metrics. Driven by `replay_events.py` / `analyze_events.py` and the `record` · `replay` · `analyze` · `analyze-compare` recipes. See [docs/game-state-io.md](docs/game-state-io.md)

### RL Algorithms

- **DQN** — epsilon-greedy agent, replay buffer, DQN Lightning module
- **PPO** — actor-critic with GAE, clipped surrogate objective, PPO Lightning module

### Networks

- **TransformerNetwork** — NanoGPT-style transformer (default, actively developed)
- **MLPNetwork** — simple MLP (legacy, will be dropped)

### Configuration

- Environment configs live in `examples/env_config/`
- Algorithm configs: `DQNConfig`, `PPOConfig` in respective `config.py` files
- Training config: `DQNTrainingConfig` (`model/dqn/config.py`) · `PPOTrainingConfig` (`model/ppo/config.py`)

### Directory-scoped guidance

Detailed patterns live next to the code they govern — read them when working in these areas:

- `wargame_rl/wargame/envs/CLAUDE.md` — Gymnasium env, phases, placement, opponents, rendering
- `wargame_rl/wargame/model/CLAUDE.md` — networks, DQN/PPO, observation tensor pipeline
- `tests/CLAUDE.md` — fixtures, test file map, per-feature coverage checklist

---

## Interacting with the User

- Keep responses brief and to the point
- Ask clarifying questions in ambiguous problems

## Coding Practice

- Before saying tasks are finished, ALWAYS run `just format` and `just lint`. Fix any errors.
- Run `just test` after large changes
- Run `just test` after adding any tests
- ALWAYS run `just format && just lint` (or `just validate`) on files that have changed; use Justfile recipes rather than `uv run` directly
- ALWAYS add type hinting for inputs and outputs
- Pass dependencies in; don't construct them inside classes
- NEVER include unimportable resources
- Public facing methods should have docstrings
- Follow the package dependency flow (see [docs/ddd-envs.md § Dependency direction](docs/ddd-envs.md#dependency-direction)); layers above must not depend on layers below.
- Follow KISS
- Prefer complexity at startup, keep runtime simple
- Prefer validation at initialisation / construction, keep runtime simple
- Comments should focus on WHY it is implemented that way
- Docstrings should explain WHAT is happening
- When suggesting or writing PR titles, use conventional commits (`feat:`, `fix:`, etc.) then a space and a lower case letter (CI expects this)
- Go from a config to an execution context before usage

## Coding Style

- Prefer simplicity over cleverness — write the simplest solution that works
- Avoid unnecessary abstractions, metaclasses, or design patterns unless clearly justified
- Use descriptive variable and function names; avoid abbreviations
- Keep functions small and focused (single responsibility, ~30–40 lines max)
- Explicit is better than implicit: no hidden side effects, be explicit about return values
- Use Python type hints for all public functions; prefer built-in generics (`list[str]`) over `typing.List`
- Raise meaningful exceptions; never swallow exceptions silently

## Python Conventions

- All functions typed (mypy strict); `from __future__ import annotations`
- Modern syntax: `str | None`, `list[int]`, `dict[str, Any]`
- Imports: isort Black profile (stdlib → third-party → local); absolute (`from wargame_rl.wargame...`)
- Classes `PascalCase` · functions/vars `snake_case` · constants `UPPER_SNAKE_CASE` · private `_leading_underscore`
- Ruff: 88 chars, 4-space indent, double quotes
- Pydantic for structured data/config · `loguru` for logging · `numpy` typed arrays for perf

## Naming & Design

- Prefer general, future-proof names over narrow ones (e.g. `models` not `model_placements`, `ModelConfig` not `ModelPlacement`)
- When adding per-entity configuration, make positional fields optional so attributes (stats, group, etc.) can be specified independently of placement
- Avoid hardcoded magic numbers in factory methods — push defaults into Pydantic models with `Field(default=...)`
- Scripted behaviours: name classes descriptively with a `Scripted` prefix (e.g. `ScriptedAdvanceToObjectivePolicy`, not `ScriptedPolicy`)
- Use registry pattern with string identifiers for YAML-configurable subsystems (opponent policies, reward calculators/criteria)

## Adding New Entity Types

- Mirror existing entity patterns: reuse the same model class (`WargameModel`), same config schema (`ModelConfig`), same placement logic
- Parameterize shared infrastructure (e.g. `ActionHandler(n_models=...)`) rather than duplicating it
- Always default new config fields to the no-op value so existing YAML configs keep working (e.g. `number_of_opponent_models=0`)
- Full checklist: config types → env state → observation types → observation builder → tensor pipeline → DQN networks → renderer → tests → backward compat tests
- When adding config that changes step semantics or episode length, update docs (`docs/reward-phases.md`, `docs/tabletop-rules-reference.md`, `docs/opponent-policies.md`, `docs/goals-and-roadmap.md`) and any tests that assume steps-per-round or phase order
- When adding new reward calculators or success criteria, register them and document in `docs/reward-phases.md` (tables and file layout)
- When changing the environment, domain, reward, or rendering, follow [docs/ddd-envs.md](docs/ddd-envs.md): keep domain logic in `domain/`, use `BattleView` for read-only state, and preserve dependency direction (domain → types only; reward/renders → BattleView)

## Testing ("Good Enough" Testing)

- Test the happy path and critical edge cases — not every possible permutation
- Focus tests on logic and behavior, not implementation details
- Prioritize tests that would catch real bugs over exhaustive coverage
- If a bug is found, add a regression test before fixing it
- Avoid mocking: ONLY mock when absolutely necessary (e.g., external APIs, paid services)
- Prefer real dependencies and integration tests over unit tests with mocks
- Avoid lots of tests, use parameterization and hypothesis testing
- Tests should be deterministic: no randomness without a fixed seed
- Use Arrange–Act–Assert structure

## Package Management

- Use `uv add [--dev] <pkg>` to add new packages; run `just dev-sync` after pulling lock changes; never manually edit `uv.lock`
- **Always start from the Justfile.** Before running any project operation (format, lint, test, train, validate, sync, simulate, etc.), check `Justfile` for an existing recipe and use `just <recipe>` — never invoke `uv run` (or other tool wrappers) directly when a recipe exists
- If no recipe exists for what you need, prefer adding one to the Justfile over running ad-hoc `uv run` commands; only fall back to `uv run` when a one-off is clearly not worth a recipe
- Discover recipes with `just` / `just --list` when unsure of the name or arguments

## Training Runs

- Default algorithm is **PPO**; do not default to DQN unless explicitly asked
- Default network type is **transformer**; do not default to MLP unless explicitly asked
- Start training: `just train <env_config.yaml>` · DQN: `just train <env_config.yaml> dqn` · DQN + network: `just train <env_config.yaml> dqn transformer`
- **Multiple configs in parallel**: `just train-multi config1.yaml config2.yaml` runs one training per config concurrently (PPO + transformer); each run gets a unique `--run-suffix` and shared `--wandb-group` so they appear grouped in the Wandb UI
- Env configs live in `examples/env_config/` — copy an existing one to create new scenarios
- Training logs to Wandb automatically; checkpoints saved to `checkpoints/`. Reward phase index and phase advancement are logged (`reward_phase`, `phase_advanced_at_epoch`) so curriculum runs show phase transitions in the dashboard
- **Reading run metrics:** see [docs/metrics.md](docs/metrics.md) for what each Wandb key means and the procedure for evaluating a run. Several metrics are means-of-means or change definition silently — check the reading rules there before drawing conclusions from `success_rate`, `terminal_success_bonus`, or any `reward/components/*` value
- **Always quote a result against a baseline:** `just measure-baselines <env_config> [n] record` gives the floor (`random`, 0.00) and the bar (`squad_march_shoot`, **1.00** on 25v25). A `success_rate` with no floor and no ceiling is how a policy scoring 17% against an 80% heuristic was read as progress. Note the bar is the *shooting* baseline: the movement-only ones cap at 0.78, which is the ceiling of a policy class the agent is not in
- **Training does not log the bar.** `eval/baseline_*` covers only `random` and `squad_march` (`BASELINE_POLICIES` in `model/common/lightning_base.py`), so the auto-logged reference is the movement-only 0.78 — beating it is not beating `squad_march_shoot`. Run `just measure-baselines` for the real ceiling
- **The 1.00 bar is an artefact of an opponent that never fires.** Every 25v25 config uses `scripted_advance_to_objective`, which does not shoot, and no config uses the `scripted_advance_and_shoot` policy — against it `squad_march_shoot` falls to 0.60 and `squad_march` 0.80 → 0.24. Switching a config's opponent invalidates every baseline and agent score measured on it — re-measure both. See [docs/opponent-policies.md](docs/opponent-policies.md)
- **Read the traces, not just the aggregates:** `record` also writes reference traces to `recordings/`, so `just analyze-compare <agent> <baseline>` puts them side by side. Only `vp_per_step` ranks policy quality — occupancy saturates for anything competent, and `idle_rate`, `objective_approach_rate` and `tactical_score` are structurally misleading here. See [docs/metrics.md](docs/metrics.md) § Trace metrics
- **Training configs:** `examples/env_config/25v25_single_phase.yaml` (control) and `25v25_curriculum.yaml` (two rungs). They share a scenario and a final phase, so comparing them isolates the curriculum. Every phase must keep `vp_gain` and at least one per-model calculator — `tests/test_curriculum_configs.py` enforces both
- **Past experiments:** [reports/](reports/README.md) records findings from previous runs, including refuted hypotheses. **Start with [the correction](reports/2026-08-04-correction-what-was-actually-broken.md)** — it retracts most pre-2026-08-04 conclusions, including the earlier claims that `gamma` 0.99 and `ent_coef` 0.01 were refuted (they were measured under a training loop that never applied the reward being tuned)
- **Inspecting a run:** `just run-summary <run_id> [bucket]` for rolling means (single-epoch `success_rate` is an `n_episodes`-sample binomial — never read a point value); `just measure-phase-gates <ckpt> <env_config> 40` for per-phase criteria rates and the whole `min_fraction` curve
- Key CLI options: `--record-during-training`, `--max-epochs`, `--render-mode`, `--algorithm`, `--no-wandb`, `--run-suffix`, `--wandb-group`
- Profile a run: `just profile <config.yaml> [model] [max_epochs]` generates `profile.html`
- Simulate latest checkpoint: `just simulate-latest [network_type]` · Clean up: `just clean` removes `checkpoints/` and `wandb/`

## Git Workflow

- Always verify the current branch before committing (especially after a PR merge)
- Create feature branches for all changes; avoid committing directly to `main`
- Branch naming: `feature/<topic>`, `fix/<topic>`, `refactor/<topic>`
- Commit messages: imperative mood, concise summary (e.g. "Add reward shaping for distance")
- If pre-commit hooks reject a commit, fix the issues and make a new commit — no `--amend`, no `--no-verify`
- After pushing a new feature branch, always create a PR using `gh pr create`
- Run `just validate` (format + lint + test) before pushing; `just format && just lint` for quick iteration
- **Shipping:** always create a new branch from up-to-date `main` — never reuse an existing feature branch for a new PR. Checkout `main`, pull latest, then branch. Never push directly on an in-progress branch from another workflow. The `/ship` skill (`.claude/skills/ship/`) automates this via `just ship`

## CUDA Environment

- Do NOT preemptively disable CUDA — only set `CUDA_VISIBLE_DEVICES=""` when training actually fails with CUDA errors
- By default, let PyTorch use the GPU
