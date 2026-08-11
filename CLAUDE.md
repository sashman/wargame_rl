# Wargame RL

Reinforcement learning project that trains agents (PPO) to play tabletop wargames on a discrete grid. Agents control multiple models (units) using polar-coordinate movement to capture objectives.

## Tech Stack

- **Python 3.13** — UV package manager (`uv.lock`)
- **Gymnasium 1.x** — RL environment (`WargameEnv`)
- **PyTorch + PyTorch Lightning** — PPO training
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
│       │   ├── net.py             # RL_Network base, TransformerNetwork
│       │   ├── common/            # Shared: lightning_base (eval + baselines + phase
│       │   │                      #   advancement), factory, observation, layers, callbacks
│       │   └── ppo/               # PPO: actor-critic, lightning module, agent, config
│       └── types.py               # Experience
├── configs/                       # Env configs, tiered by what breaks if edited
│   ├── golden/                    #   backs a published number
│   ├── experiments/               #   arms; deleted once answered
│   ├── evaluation/maps/           #   the real table layouts
│   └── dev/                       #   fixtures and demos
├── tests/                         # Pytest suite with conftest.py fixtures
├── docs/                          # Design docs (movement, reward phases, missions-and-vp,
│                                  #   roadmap, metrics, shooting, terrain,
│                                  #   training-throughput)
│   └── rules/                     # Rules specification + constants.yaml + gap map
├── reports/                       # Experiment findings, kept for retrospection
├── scripts/                       # Run-inspection tooling (run_summary, measure_phase_gates,
│                                  #   measure_baselines, measure_checkpoint, measure_terrain,
│                                  #   measure_noise_floor, measure_objective_split,
│                                  #   measure_throughput)
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
| Train | `just train <config.yaml> [max_epochs]` |
| Train multiple configs in parallel | `just train-multi config1.yaml config2.yaml` |
| Train an arm (config × training flags) | `just train-arm <max_epochs> <n_seeds> <group> <tag> <flags> <configs...>` |
| Train one seed with flags (parallelisable) | `just train-seed-flags <max_epochs> <seed> <group> <tag> <flags> <configs...>` |
| Ship (branch → commit → push → PR) | `just ship <branch> "<message>"` |
| Simulate latest | `just simulate-latest` |
| Simulate / record a checkpoint | `just simulate <ckpt> <config.yaml>` · `just record-sim <ckpt> <config.yaml>` |
| Test env (random) | `just test-env` |
| Watch a scripted policy play (no checkpoint) | `just play [config.yaml] [policy] [theme]` |
| Record a match event log | `just record <config.yaml>` |
| Replay / narrate a log | `just replay <file>` · `just replay-summary <file>` |
| Replay a log visually (window or MP4) | `just replay-render <file> [out.mp4]` |
| Analyse a log | `just analyze <file>` · `just analyze-compare <files...>` |
| Inspect a Wandb run | `just run-summary <run_id> [bucket]` |
| Measure reward-phase gates | `just measure-phase-gates <ckpt> <config.yaml> [n_episodes]` |
| Scripted baselines (floor + bar) | `just measure-baselines <config.yaml> [n_episodes] [record] [seed_base]` |
| Score a checkpoint (baseline-comparable) | `just measure-checkpoint <ckpt> <config.yaml> [n_episodes] [record] [distinct]` |
| Score on the real table layouts | `just measure-maps <policy\|ckpt> <config.yaml> [n_episodes] [maps_dir]` |
| Why an objective was not held | `just measure-objective-split <policy\|ckpt> <config.yaml> [n_episodes] [distinct]` |
| Dice-vs-scenario noise floor | `just measure-noise-floor <config.yaml> [n_layouts] [n_combat_seeds] [policy]` |
| Terrain-profile statistics | `just measure-terrain <config.yaml> [n_layouts]` |
| Where epoch time goes | `just measure-throughput <config.yaml> [n_steps] [engaged]` |
| Profile | `just profile <config.yaml> [max_epochs]` |
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
- **Rules specification** — [docs/rules/](docs/rules/README.md) is the game's rules authority: a self-contained spec written for this project, with `constants.yaml` (every number, in inches) and [implementation-status.md](docs/rules/implementation-status.md) (per-rule: implemented / partial / divergent / absent). Before implementing a mechanic, read its chapter and its gap-map row. `tests/test_no_ip_references.py` keeps the repo free of references to the commercial product the rules derive from — the spec names no product, publisher, edition or faction, and neither should anything else

### Game State I/O (`envs/state/`)

Snapshot/event pipeline for recording and inspecting matches — `GameStateSnapshot`, event-log deltas, `StateExporter` (wired into `step()`), replay, narration, and `analyze_match` metrics. Driven by `replay_events.py` / `analyze_events.py` and the `record` · `replay` · `analyze` · `analyze-compare` recipes. See [docs/game-state-io.md](docs/game-state-io.md)

### RL Algorithm

- **PPO** — actor-critic with GAE, clipped surrogate objective, PPO Lightning module

### Networks

- **TransformerNetwork** — NanoGPT-style transformer, the only network. DQN and
  `MLPNetwork` were removed once neither had been trained in months; `git log --
  wargame_rl/wargame/model/dqn` restores them

### Configuration

- Environment configs live in `configs/` — see [configs/README.md](configs/README.md) for the tiering
- Algorithm config: `PPOConfig` (`model/ppo/config.py`)
- Training config: `PPOTrainingConfig` (`model/ppo/config.py`)

### Directory-scoped guidance

Detailed patterns live next to the code they govern — read them when working in these areas:

- `wargame_rl/wargame/envs/CLAUDE.md` — Gymnasium env, phases, placement, opponents, rendering
- `wargame_rl/wargame/model/CLAUDE.md` — networks, PPO, observation tensor pipeline
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
- Full checklist: config types → env state → observation types → observation builder → tensor pipeline → networks → renderer → tests → backward compat tests
- When adding config that changes step semantics or episode length, update docs (`docs/reward-phases.md`, `docs/rules/implementation-status.md`, `docs/opponent-policies.md`, `docs/goals-and-roadmap.md`) and any tests that assume steps-per-round or phase order
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

> ### ⚠ Every number below this line predates continuous space and is void
>
> The board stopped being a chessboard on 2026-08-10. Positions are real points,
> a move covers exactly the distance its speed bin says (a "speed 1" diagonal
> used to travel 1.41), the vector to the objective is no longer truncated to
> whole units, sight is a sampled ray rather than a Bresenham walk, and models
> can carry a base radius. **This changes the dynamics, so every baseline, every
> `vp_margin` and the `v2.0` milestone (+30.8 / +27.4 against a bar of +17.0) was
> measured in a different environment.** The *qualitative* lessons still hold —
> read `held` not `on_obj`, prefer `vp_margin` to win rate, quote against a
> baseline on identical layouts, check the agent can observe what a reward keys
> on — and every specific figure needs re-measuring. The two golden gates were
> regenerated deliberately for this change. Baseline re-measurement is WP-12 of
> phase 03; until it lands, treat quoted numbers as history, not as targets.

- PPO on a `TransformerNetwork` is the only thing that trains — there is no algorithm or network to choose. `just train` and `train.py` used to take `--algorithm` and `--network-type`; both are gone, so `just train <config> 800` now means 800 *epochs*
- Start training: `just train <env_config.yaml>` · with an epoch cap: `just train <env_config.yaml> 800`
- **Multiple configs in parallel**: `just train-multi config1.yaml config2.yaml` runs one training per config concurrently (PPO + transformer); each run gets a unique `--run-suffix` and shared `--wandb-group` so they appear grouped in the Wandb UI
- Env configs live in `configs/`, tiered by what breaks if you edit them ([configs/README.md](configs/README.md)). Copy a `golden/` config into `experiments/` to make an arm — never edit a golden config to try something
- Training logs to Wandb automatically; checkpoints saved to `checkpoints/`. Reward phase index and phase advancement are logged (`reward_phase`, `phase_advanced_at_epoch`) so curriculum runs show phase transitions in the dashboard
- **Checkpoints are not uploaded to Wandb** (`log_model=False` in `model/common/wandb.py`). Nothing in the repo ever read a model artifact back — `simulate`, `record-sim`, `measure-checkpoint`, `measure-phase-gates`, `--resume-ckpt-path` and `--warm-start-ckpt-path` all take a local path under `checkpoints/` — while each run uploaded ~591 MB (4 × 148 MB), which filled the storage quota. Metrics, history and videos still log normally. **`checkpoints/` is now the only copy of the weights, so `just clean` is destructive**
- **Reading run metrics:** see [docs/metrics.md](docs/metrics.md) for what each Wandb key means and the procedure for evaluating a run. Several metrics are means-of-means or change definition silently — check the reading rules there before drawing conclusions from `success_rate`, `terminal_success_bonus`, or any `reward/components/*` value
- **Always quote a result against a baseline:** `just measure-baselines <env_config> [n] record` gives the floor (`random`, 0.00) and the bar (`squad_march_shoot`, **1.00** on 25v25). A `success_rate` with no floor and no ceiling is how a policy scoring 17% against an 80% heuristic was read as progress. Note the bar is the *shooting* baseline: the movement-only ones cap at 0.78, which is the ceiling of a policy class the agent is not in
- **The bar is a distribution over layout sets, never a single number — always pass `seed_base`.** `squad_march_shoot` on the *same* config scores 0.45 (seeds 10000-10019, n=20), 0.53 (10000-10029, n=30) and **0.77** (700000-700029, n=30). A 32-point swing on a deterministic scripted policy, purely from which maps you draw. This dwarfs the ~7pp seed-noise limit below and it is how batch 3 concluded the agent had cleared a bar it was 10pp beneath. **Score agent and baseline on identical layouts or the comparison is meaningless:** `just measure-checkpoint <ckpt> <config> 30` uses seeds 700000+, so pair it with `just measure-baselines <config> 30 "" 700000`. Training's own `eval/baseline_*` uses 20 episodes at seeds 10000+ while `eval/win_rate` uses 10 at seeds 500000+ — those two are *not* comparable to each other either
- **Training logs the bar.** `eval/baseline_*` covers `random`, `squad_march` and `squad_march_shoot` (`BASELINE_POLICIES` in `model/common/lightning_base.py`). Read `eval/baseline_squad_march_shoot_win_rate`, not the movement-only one — beating 0.78 is not beating 1.00. `just measure-baselines` adds the middle rungs
- **The 1.00 bar is an artefact of an opponent that never fires.** The original 25v25 configs use `scripted_advance_to_objective`, which does not shoot — against `scripted_advance_and_shoot` on the same fixed terrain, `squad_march_shoot` falls to 0.60 and `squad_march` 0.80 → 0.24. The cover-experiment configs all use the shooting opponent. Switching a config's opponent invalidates every baseline and agent score measured on it — re-measure both. See [docs/opponent-policies.md](docs/opponent-policies.md)
- **Config inventory for this scenario:** `configs/golden/25v25_shooting_opponent.yaml` (the configuration that beats the bar) and `configs/golden/25v25_cover_control.yaml` (the control it was developed against). Every experiment arm around them was deleted once its question was answered — batch 1/2's, batch 3's `cover_reason`, and batch 4's eight `25v25_beat_*` arms. `git log -- configs/` restores any of them; `git checkout batch-1-2-configs -- configs/` for the oldest. Both surviving configs regenerate terrain every episode and set `track_exposure`, which adds `eval/exposure_rate`, `eval/terrain_proximity`, `eval/firepower_ratio` and `eval/fraction_alive`. `exposure_rate` averages over *alive* models, so casualties lower it on their own — read [docs/metrics.md](docs/metrics.md) § Cover metrics before comparing it across configs
- **The agent does not use terrain for cover — it manages range.** Established by deleting all terrain (exposure moved 0.116 → 0.120) and by doubling weapon range so distance stops working (win collapsed to 6.8%). Don't re-derive this; see [the report](reports/2026-08-05-stochastic-terrain-and-cover.md). Note the report's **correction**: arm F was confounded, because that terrain profile left only 5.8% of the board hidden — cover was not an available alternative there either
- **Batch 3 answered the cover question and the answer is still no** — `25v25_cover_{control,reason}.yaml`, originally a 2x2 over (`observe_threat_count` × the `models_lost` reward), two seeds each. With 19.8% of the board hidden, a per-model LOS input and priced losses, exposure stayed at 0.092–0.110 across every arm. **Don't re-run this experiment**; see [the report](reports/2026-08-06-cover-signal-reason-geometry.md). What it did find: `models_lost` is worth **+7 vp_margin** with non-overlapping seeds, `observe_threat_count` was **null** (and has since been removed, along with its two arms), and the penalty made the agent lose *more* models — the opposite of the mechanism it was added for. Batch-3 numbers are not comparable to batch 1 or 2. **Two of these claims were corrected on 2026-08-06 — read the corrections in the report before reusing any of it:** the `models_lost` +7 is window-dependent and reverses on held-out layouts (its sign is unestablished), and the "all four arms clear the bar" line is wrong. The bar was previously quoted here as **0.45**; that figure is the in-run 20-episode baseline, not what `measure-baselines` returns, and it was compared against arms scored on different layouts
- **`eval/firepower_ratio` replaced `eval/firepower_advantage` on 2026-08-06; the two are not comparable.** The old count difference was wrong twice over — a difference is dominated by how much engagement happens, and since LOS is symmetric, "enemies we can see" is *their* shooter count, not ours. It scored `random` (0% win) top of the table. The ratio counts shooters on each side and puts `random` last at 0.23. It measures the *firefight*, not policy quality: the bar wins 0.56 at a ratio of 0.49. Read it beside `vp_margin`
- **The agent now beats the shooting bar: +30.8 (s1) and +27.4 (s2) vp_margin against `squad_march_shoot`'s +17.0** (n=100, identical layouts, epoch 1000, **`--no-tf32` — the default again since 2026-08-09**). Re-measured on 2026-08-09 and reproduced *bit-identically* from the original weights. Two prior phrasings of this line were loose: **+28.4 is the mean of the two seeds, not a figure either seed scored**, and it came from checkpoints at epochs 970 and 692 rather than 1000 (the old `last.ckpt` bug) — that selection turned out to be worth almost nothing here, since honest epoch-1000 scores +29.1 on average, slightly *higher*. **Under TF32 the same config scores +21.2/+19.9**, so quote this result only against runs sharing the flag. The lever is `objective_hold`'s **`crowding_exponent`** — a point pays a fixed pot split between its occupants instead of paying every occupant the same wage. **`configs/golden/25v25_shooting_opponent.yaml` is that configuration** (a=1.0, weight 1.25) and is the config to train on this scenario; the a=0.5 arm only reached the bar and was deleted with the rest of the screen. See [the report](reports/2026-08-08-paying-the-pot-beats-the-bar.md). Note this scenario is effectively a **two-objective mission** — both policies concede the third point in nearly every episode (the opponent stacks ~13 there, flipping it costs 14), so `held` is bounded near 2
- **Don't reach for "positive rewards beat penalties" — that hypothesis was tested and refuted here.** `surplus_value` *is* the positive version of the overstack penalty (it pays surplus models less, never below zero, which was the whole reason to expect a different result) and it failed identically: occupancy 0.784 → 0.284 against the penalty's 0.925 → 0.520. The winning arm still carries two negative terms, and `crowding_exponent` also *reduces* a crowded model's pay (0.25 → 0.096), so sign does not separate the winner from the losers — total income does. Fair residue: a penalty is *more likely* to destroy total income by accident, so run the check. Magnitude matters more than sign (`group_cohesion` at -0.2 inverted the baseline ranking; at -0.05 it is in the winning config), and potential-based shaping is safe by construction
- **Per-model rewards are necessary and nowhere near sufficient — the number must vary across the choice the model is making.** Flat `objective_hold` *is* per-model, yet the thirteenth model on a point earned the same 0.25/step as the first, so no model ever had a private reason to leave. Per-model is not per-model-*differentiated*; only the second produces a gradient. Ask what the model's presence actually changes and price that (the *difference rewards* idea — the approximation need not be principled to work). Two counterweights: global terms are broadcast whole and largely absorbed by the value baseline, but some quantities genuinely cannot be per-model — `models_lost` must be global, because `phase_manager` iterates *alive* models and at `max_wounds: 1` a per-model loss penalty is identically zero. See [docs/reward-phases.md](docs/reward-phases.md) § Choosing calculators
- **The crowding result is confound-controlled, and raising the objective weight *alone* is catastrophic.** The arm holding weight 1.25 with the exponent back to 0.0 (config since deleted) scores **−40.4 vp_margin** against the control's +3.25 and `share`'s +28.4, the worst trained arm ever measured here. At fixed weight the exponent alone is worth **68 vp**. The failure mode is textbook: **20.2 of 20.8 survivors end on one objective**, `on_obj` 0.955, `alive` 0.83 (barely fighting), second objective abandoned 0.6 to 10.4 — flat pay at 1.25/step integrates to ~50 an episode against this config's ~10-per-term budget, so standing still out-earns everything. Note the exponent both prices crowding *and* auto-regulates magnitude (dividing by ~10 occupants returns `share` to ~0.125/step); no experiment here separates the two
- **An anti-concentration lever must redistribute reward, not destroy it.** Three levers were aimed at the same over-stacking defect. `closest_objective_v2.overstack_penalty_per_extra` (occupancy 0.925 → 0.520) and `objective_hold.surplus_value` (0.784 → 0.284) both *lower* total objective income, so the policy experiences either as "objectives pay less" and does fewer of them. `crowding_exponent` at a=1 conserves the pot — `k` models on one point earn it once, `k/2` on each of two earn it twice — so spreading strictly *raises* income. Before training a shaping term, ask whether the behaviour it wants pays **more in total** than the behaviour it replaces; if not, it is a tax on the whole activity
- **`last.ckpt` was not the last epoch until 2026-08-08.** `get_checkpoint_callback` put `monitor` and `save_last=True` on one `ModelCheckpoint`, so `last.ckpt` was only rewritten on epochs entering the top-k — it held *the last epoch that improved*. One 1000-epoch batch's four files held epochs **970, 692, 948, 998**, and the 692 run's epoch-1000 weights were never written. **Every score in this repo labelled "at N epochs" before that date is really "at whatever epoch that run last improved by its own training reward"**, a spread worth ~13 vp_margin — and not a common rule across arms, since each arm's training reward is a different function. Now split into two callbacks. **A second bug hid inside that fix until 2026-08-09:** the unmonitored `ModelCheckpoint(save_top_k=0, save_last=True)` writes `last.ckpt` only at `on_train_end`, so completed runs were correct but a **killed run left no `last.ckpt` at all** — and runs here are routinely killed, leaving only the top-k files and reinstating the same selection bias. `PeriodicLastCheckpoint` now writes every 25 epochs, at `on_train_end`, and on `KeyboardInterrupt`, staging and renaming so a kill mid-write cannot truncate it. The old tests asserted callback *configuration* and stayed green throughout; `tests/test_checkpoint_callback.py` is now behavioural — it runs a real `fit` and checks the file is on disk while training is still running
- **`held` ranks policies but is not the mechanism — a flat `held` is not a null result.** It is an *end-state* snapshot while VP accrues every round, so a policy can hold more during the episode and end level. `share_soft` gained **+19.3 vp_margin with `held` unchanged**, and `share` beats the bar by 11.4 vp while holding *fewer* objectives at the end (1.59 v 1.64). Read `held` to rank; read `vp_margin` to decide
- **Split `held` before shaping a reward against it.** `just measure-objective-split <policy|ckpt> <config>` reports per-objective `(player, opponent)` counts at episode end plus a **redistribution ceiling** — what the same survivors would hold if surplus models moved to the cheapest lost point. It is deliberately optimistic (no travel time, no return fire), so a ceiling near current `held` *rules re-allocation out*; a large one does not rule it in. On the batch-3 scenario the trained agent parks **12.9 of its 15.8 survivors on a point defended by 0.25 opponents** and loses the second 4.2 to 2.7 — ceiling 2.06 against the bar's 1.88, so allocation alone would clear the bar. Note both policies concede the third objective (the opponent stacks ~12 there, flipping it costs 13), so `held` is bounded near 2 and this is effectively a two-objective mission
- **`objectives_held` (`held`) is the metric that ranks policies, not `on_obj`.** Mean count of objectives controlled under VP's own strict count rule. Ordered by `held`, vp_margin is perfectly monotonic across every scripted and learned policy measured; ordered by `on_obj` it is not, because `on_obj` is a fraction of alive models on *any* objective and cannot tell 15 models on one point from 5 each on three. Three experimental rounds were aimed at an `on_obj` deficit that was mostly n=30 noise (0.925 vs 1.000 at n=30; 0.945 vs 0.960 at n=100) while the real 15 VP gap sat in `held` (1.42 vs the bar's 1.64) and was never measured. See [docs/metrics.md](docs/metrics.md)
- **Before training a reward lever, check the agent can observe what it keys on.** A desk check that costs seconds and has already burned ~10 GPU-hours. The overstack penalty and `objective_hold.surplus_value` are mechanically opposite levers that both halved objective occupancy, because both key on per-objective model counts the agent could not see — an objective reached the network as nothing but an `(x, y)` location. An unattributable reward is experienced only as "this pays less", so the policy does less of it. Ask: *if two states differ only in what this term keys on, do they differ in the observation?* If not, add the input first. See [docs/reward-phases.md](docs/reward-phases.md) § Design rules
- **Score with enough episodes to resolve the effect.** `measure-checkpoint` and `measure-baselines` now default to **n=100**, not 30. Per-episode `vp_margin` sd is ~45–50 on 25v25, so n=30 gives a standard error of ~8–9 — larger than most arm differences ever measured here (4–10 vp). Scoring costs minutes against a training run's hours; it was the cheap half being under-sampled. Training eval likewise runs 30 episodes in the seeded recipes rather than PPO's default 10
- **Screen at ~300 epochs, quote effect sizes at 1000+.** Measured on batch 4: epochs 0–300 move `vp_margin` −76 → −2, but 300–1000 add another **+8**, which is the same size as the arm differences — so an early cut is comparable to the signal, and 1000 epochs is if anything too few (the control is still climbing at 950). But the *ordering* separates early: the losing arm was already clearly behind by epoch 200–299. A 300-epoch screen is 3.3x cheaper and would have caught batch 4's failures in ~2h rather than 7. Treat a *marginal* 300-epoch result as "run it longer", not "rejected"
- **Training is deterministic given seed + config + code.** Two independently trained runs at the same seed reproduced bit-identical eval metrics on all ten fields (different checkpoint checksums; greedy evaluation collapses low-order weight differences). **Never retrain a control that already exists at the same epoch budget** — two of batch 4 round 1's eight runs bought nothing
- **Don't query the Wandb API while runs are training.** Four concurrent runs segfaulted simultaneously (signal 11) after `ConnectionResetError` in the wandb service client, minutes after two `wandb.Api()` queries. Causation is unproven — the same queries ran harmlessly during earlier rounds — but the shared wandb service is the only thing that explains four independent processes dying within 20 seconds, and mid-run reads are available from the local `wandb/run-*/files/output.log` instead
- **Win rate cannot resolve differences under ~7pp on these configs.** Measured within-arm seed spread across batch 3 was 6.0–7.3pp on win rate while `vp_margin` separated cleanly — prefer `vp_margin` for arm-to-arm comparisons, and never read a single-seed win-rate gap as an effect
- **Terrain: count dominates size.** `just measure-terrain` reports *cells hidden from a squad*, the only figure that matters, since exposure is "at least one enemy sees me". Many small pieces beat few large ones at equal coverage. Tune a profile there, in seconds, rather than after a training run
- **The dice contribute more outcome spread than the scenario does.** `just measure-noise-floor` holds layouts fixed and varies only `reset(options={"combat_seed": ...})`: on the batch-3 control, `squad_march_shoot` has a vp_margin sd of 50.6 within a layout against 45.0 between layouts. Run **two seeds per arm** before reading any difference smaller than ~10pp
- **Read the traces, not just the aggregates:** `record` also writes reference traces to `recordings/`, so `just analyze-compare <agent> <baseline>` puts them side by side. Only `vp_per_step` ranks policy quality — occupancy saturates for anything competent, and `idle_rate`, `objective_approach_rate` and `tactical_score` are structurally misleading here. See [docs/metrics.md](docs/metrics.md) § Trace metrics
- **Training configs:** `configs/golden/25v25_single_phase.yaml` (control) and `25v25_curriculum.yaml` (two rungs). They share a scenario and a final phase, so comparing them isolates the curriculum. Every phase must keep `vp_gain` and at least one per-model calculator — `tests/test_curriculum_configs.py` enforces both. **`25v25_shooting_opponent.yaml` is a different scenario and is not comparable to either** — it faces `scripted_advance_and_shoot` on regenerated terrain with `objective_min_separation`, where those two face the non-shooting `scripted_advance_to_objective` on fixed terrain. `crowding_exponent` has only ever been measured on the shooting scenario, so do not port it into the other two without measuring there
- **Past experiments:** [reports/](reports/README.md) records findings from previous runs, including refuted hypotheses. **Start with [the correction](reports/2026-08-04-correction-what-was-actually-broken.md)** — it retracts most pre-2026-08-04 conclusions, including the earlier claims that `gamma` 0.99 and `ent_coef` 0.01 were refuted (they were measured under a training loop that never applied the reward being tuned)
- **Inspecting a run:** `just run-summary <run_id> [bucket]` for rolling means (single-epoch `success_rate` is an `n_episodes`-sample binomial — never read a point value); `just measure-phase-gates <ckpt> <env_config> 40` for per-phase criteria rates and the whole `min_fraction` curve
- Key CLI options: `--record-during-training`, `--max-epochs`, `--render-mode`, `--no-wandb`, `--run-suffix`, `--wandb-group`, `--n-eval-episodes`, `--seed`, `--tf32`, `--precision`, `--eval-every-n-epochs`, `--lr`, `--max-grad-norm`
- **Evaluation is ~22% of a real epoch and is not counted in `perf/epoch_s`.** At the `--n-eval-episodes 30` every seeded recipe passes, it is 1200 env steps against the rollout's 2048. `--eval-every-n-epochs 4` cuts total wall-clock ~16% (measured: 89.0 s → 74.6 s over 8 epochs, with training time and the final score unchanged — the last epoch always evaluates). **Single-phase configs only**: it raises on a curriculum config, because `try_advance` counts *consecutive* epochs above threshold, so a coarser cadence changes which epoch a phase advances on and therefore what the run trains
- Profile a run: `just profile <config.yaml> [max_epochs]` generates `profile.html` (`--no-wandb`, capped at 5 epochs by default)
- **Training speed was an environment problem; on the 4090 it is now split evenly with the update.** See [docs/training-throughput.md](docs/training-throughput.md). `just measure-throughput <config>` gives the per-section and per-reward-calculator split of `env.step()`; every run also logs `perf/rollout_s`, `perf/update_s`, `perf/eval_s`, `perf/env_steps_per_s` and `perf/update_ms_per_minibatch`, so a slowdown shows up beside the reward curves. Two calculators were ~80% of a 25v25 step because each recomputed a model-independent quantity once per model; memoising them plus three smaller repeats took the rollout step from 11.34 ms to 2.26 ms (23.2 s → 4.6 s of env time per epoch). Any change to the reward pipeline must keep `tests/test_reward_golden.py` **bit-identical** — it pins per-step reward, per-model reward, breakdown, VP and positions, and is verified to catch a one-ULP change
- **TF32 is off by default because it costs ~8.5 vp_margin.** It shipped *on* on 2026-08-08 and was reverted on 2026-08-09, when it was first measured against a trained result rather than a benchmark: on `25v25_shooting_opponent.yaml` at epoch 1000, n=100 identical layouts, **s1 +30.8 → +21.2 and s2 +27.4 → +19.9** — the difference between beating the bar by 12.1 and by 3.6. The `--no-tf32` control reproduced the pre-TF32 run **bit-identically** (222/222 tensors, max abs diff 0.0, both seeds), which both proves TF32 is the whole effect and re-confirms that everything else in the window — env hot-path memoisation, DQN removal, the config restructure — changed training by nothing. The old entry here claimed the effect was "far below the ~7pp win-rate / ~10 vp resolution limits"; that was inferred from the mantissa drop (24→11 bits) and the 1.34x update speedup, never measured, and it is wrong. **Win rate really would have missed it** (0.705 → 0.65, inside the ~7pp limit) while `vp_margin` separated on both seeds — the "prefer `vp_margin`" rule paying out. The speed was also oversold: 1.34x on the *update* is **17.8% of an epoch** (12.95 → 10.99 s/epoch). Pass `--tf32` for smoke, profiling and throughput runs only. Treat any precision or numerics setting as a **reward-affecting change** and screen it like a shaping term. See [the report](reports/2026-08-09-tf32-costs-eight-vp.md). "Training is deterministic given seed + config + code" holds only *within* one setting of this flag. **`--precision bf16-mixed` is another 1.8x on the update and is opt-in because only its speed has been measured** — A/B it over two seeds before trusting a run under it. `torch.compile` measures 3.26x stacked and is deliberately not wired: it prefixes every `state_dict` key with `_orig_mod.`, and `_apply_warm_start_weights` uses `strict=False`, so such a checkpoint would load as *nothing at all* and score a random network as a trained one
- Simulate latest checkpoint: `just simulate-latest` · Clean up: `just clean` removes `checkpoints/` and `wandb/`

## Git Workflow

- Always verify the current branch before committing (especially after a PR merge)
- Create feature branches for all changes; avoid committing directly to `main`
- Branch naming: `feature/<topic>`, `fix/<topic>`, `refactor/<topic>`
- Commit messages: imperative mood, concise summary (e.g. "Add reward shaping for distance")
- If pre-commit hooks reject a commit, fix the issues and make a new commit — no `--amend`, no `--no-verify`
- After pushing a new feature branch, always create a PR using `gh pr create`
- Run `just validate` (format + lint + test) before pushing; `just format && just lint` for quick iteration
- **Shipping:** always create a new branch from up-to-date `main` — never reuse an existing feature branch for a new PR. Checkout `main`, pull latest, then branch. Never push directly on an in-progress branch from another workflow. The `/ship` skill (`.claude/skills/ship/`) automates this via `just ship`
- **Docs-drift check:** a `PostToolUse` hook (`.claude/settings.json` → `.claude/hooks/docs_check.py`) fires after `gh pr create` and `just ship`. It diffs the branch against `main` and names the live docs that cite the changed paths, symbols, recipes or config fields. Fix mechanical drift (renamed symbol, changed default, missing table row) directly; only *suggest* anything asserting behaviour. It is silent when nothing is implicated, and never fails a ship. `reports/` and `.planning/` are exempt — they record what was believed at the time. Run it by hand with `python3 .claude/hooks/docs_check.py --dry-run [<base>..<head>]`

## CUDA Environment

- Do NOT preemptively disable CUDA — only set `CUDA_VISIBLE_DEVICES=""` when training actually fails with CUDA errors
- By default, let PyTorch use the GPU
