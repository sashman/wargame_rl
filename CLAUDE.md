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
│       │   ├── domain/            # The battle as sub-domains: kernel, battlefield,
│       │   │                      #   sequencing, movement, attacks, shooting, melee;
│       │   │                      #   the aggregate and BattleView at the root
│       │   ├── board/             # Board-wide reads (leaf): sampling grid, the
│       │   │                      #   next-turn threat field, unit matchups
│       │   ├── env_components/    # Adapters: actions, distance cache, observation builder
│       │   ├── per_model/         # The per-model facade: one DECISION per step over the
│       │   │                      #   same domain; tokens (the set observation), the
│       │   │                      #   re-timed reward, the batched evaluation in waves
│       │   │                      #   and the recording at two cadences (#283 stages 1-4)
│       │   ├── evaluation/        # The evaluation kernel both facades share: EvalResult,
│       │   │                      #   the end-of-episode readouts (held / on_obj / alive)
│       │   ├── map_pool.py        # Draws a real table per episode from a pool of maps
│       │   ├── baseline/          # Scripted baseline policies + registry + evaluate
│       │   ├── debug/             # Hand-stepping a live match: undo stack, session loop
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
│       │   ├── ppo/               # PPO: actor-critic, lightning module, agent, config
│       │   ├── per_model/         # The set network over the per-model facade: collate,
│       │   │                      #   encoder + selector + heads + value, SetAgent (#285);
│       │   │                      #   ppo (the loop over decision steps), evaluate,
│       │   │                      #   checkpoint (#286)
│       │   └── opponent/          # A checkpoint seated on the opponent side
│       ├── rating/                # Elo: margin score, Bradley-Terry fit, schedule,
│       │                          #   arena, ledger, table
│       ├── selectors.py           # One resolver for both facades: a name or .ckpt -> selector,
│       │                          #   a name / random / set_network / .pt -> chooser
│       ├── scoring.py             # Score or record a spec on the facade that owns it
│       └── types.py               # Experience
├── configs/                       # Env configs, tiered by what breaks if edited
│   ├── golden/                    #   backs a published number
│   ├── experiments/               #   arms; deleted once answered
│   ├── evaluation/maps/           #   the real table layouts
│   └── dev/                       #   fixtures and demos
├── tests/                         # Pytest suite with conftest.py fixtures
├── docs/                          # Design docs (movement, reward phases, missions-and-vp,
│                                  #   roadmap, metrics, shooting, expected-damage,
│                                  #   terrain, training-throughput, play-doctrine, elo,
│                                  #   self-play)
│   └── rules/                     # Rules specification + constants.yaml + gap map
├── reports/                       # Experiment findings, kept for retrospection
├── ratings/                       # Rating ledgers, one per scenario fingerprint
├── scripts/                       # Run-inspection tooling (fetch_map_layouts,
│                                  #   run_summary, measure_phase_gates,
│                                  #   measure_baselines, measure_checkpoint, measure_terrain,
│                                  #   measure_noise_floor, measure_objective_split,
│                                  #   measure_income_share, measure_maps,
│                                  #   behaviour_clone, measure_seat_parity,
│                                  #   measure_matchups, measure_threat_field,
│                                  #   measure_elo, elo_table,
│                                  #   measure_throughput)
├── train.py                       # Training entry point (Typer CLI)
├── train_per_model.py             # PPO over per-model decision steps (Typer CLI, #286)
├── simulate.py                    # Inference/simulation entry point
├── debug.py                       # Step a live match by hand, and rewind it
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
| Train a self-play arm and its control for one seed | `just train-self-play-screen [max_epochs] [seed] [group] [env_config] [anchor]` |
| Strip checkpoints for a release (verifies bit-identity, writes SHA256SUMS) | `just prepare-release <out_dir> <label> <ckpt>...` |
| Ship (branch → commit → push → PR) | `just ship <branch> "<message>" [<issue>]` — an issue number appends `Closes #N` |
| Simulate latest | `just simulate-latest` |
| Simulate / record a checkpoint | `just simulate <ckpt> <config.yaml> [overlays]` · `just record-sim <ckpt> <config.yaml>` |
| Regenerate the eval tables from the layout API | `just fetch-maps [owner] [maps_dir]` |
| Record the README's GIFs (exact colours, median of N) | `just record-gifs <policy\|ckpt> <config> [tables]` |
| Test env (random) | `just test-env` |
| Watch a scripted policy play (no checkpoint) | `just play [config.yaml] [policy] [theme] [overlays]` |
| Train the set network over the per-model facade (budget in ROUNDS) | `just train-per-model <config.yaml> [rounds] [flags]` |
| Watch the per-model facade play, one frame per decision | `just play-per-model [config.yaml] [policy\|random\|set_network\|run/last.pt] [theme] [overlays] [cadence]` |
| Record the per-model facade to an MP4 | `just record-per-model [config.yaml] [policy\|random\|set_network] [out.mp4] [cadence]` |
| Record the per-model facade to an EVENT LOG (phase or decision cadence) | `just record-per-model-events <config.yaml> [policy\|random\|set_network\|run/last.pt] [cadence] [seed] [out]` |
| Where a per-model DECISION's time goes, in rounds | `just measure-throughput-per-model <config.yaml> [rounds]` |
| Step a match by hand and rewind it | `just debug [config.yaml] [policy\|ckpt] [theme] [overlays]` |
| Recreate a recorded match exactly and step it | `just debug-recording <file> [policy\|ckpt] [theme] [overlays]` |
| Record a match event log | `just record <config.yaml>` |
| Replay / narrate a log | `just replay <file>` · `just replay-summary <file>` |
| Replay a log visually (window or MP4) | `just replay-render <file> [out.mp4] [theme] [overlays]` — tabletop by default |
| Analyse a log | `just analyze <file>` · `just analyze-compare <files...>` |
| Inspect a Wandb run | `just run-summary <run_id> [bucket]` |
| Measure reward-phase gates | `just measure-phase-gates <ckpt> <config.yaml> [n_episodes]` |
| Scripted baselines (floor + bar) | `just measure-baselines <config.yaml> [n_episodes] [record] [seed_base] [key=value...]` |
| Score a checkpoint (baseline-comparable; a `.ckpt` or a per-model `.pt`) | `just measure-checkpoint <ckpt> <config.yaml> [n_episodes] [record] [decode_topk] [key=value...]` |
| Score on the real table layouts (a name, `.ckpt` or `.pt`) | `just measure-maps <policy\|ckpt> <config.yaml> [n_episodes] [maps_dir] [decode_topk] [key=value...]` |
| Why an objective was not held | `just measure-objective-split <policy\|ckpt> <config.yaml> [n_episodes]` |
| What a policy buys with the advance move, and what it pays | `just measure-advance-use <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| How often the VP cap binds, and what it discards | `just measure-vp-cap <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| What holding a point earns against what it costs | `just measure-hold-hazard <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| How often a policy is in unit coherency | `just measure-coherency <policy\|ckpt> <config.yaml> [n_episodes]` |
| Which calculator pays, and how much is global | `just measure-income-share <policy\|ckpt> <config.yaml> [n_episodes]` |
| Which of our units beats which of theirs, pre-game | `just measure-matchups <config.yaml>` |
| Where it is dangerous to stand NEXT turn, and a policy's exposure | `just measure-threat-field <policy\|ckpt> <config.yaml> [n] [maps_dir] [decode_topk]` |
| Clone a scripted policy into the network (warm-start checkpoint) | `just behaviour-clone <policy> <config.yaml> [n_episodes] [epochs] [out]` |
| Two policies on identical layouts, paired per episode (names, `.ckpt` or `.pt`) | `just measure-paired <policy\|ckpt> <policy\|ckpt> <config.yaml> [n_episodes] [seed_base] [key=value...]` |
| Dice-vs-scenario noise floor | `just measure-noise-floor <config.yaml> [n_layouts] [n_combat_seeds] [policy] [key=value...]` |
| Are the two seats the same game (the rating precondition) | `just measure-seat-parity <config.yaml> [policy] [n_layouts]` |
| Rate policies against each other on one scale | `just measure-elo <config.yaml> [n_layouts] <entrant...>` |
| Fit and print the rating table from legs already played | `just elo-table <config.yaml>` |
| Terrain-profile statistics | `just measure-terrain <config.yaml> [n_layouts]` |
| Where epoch time goes | `just measure-throughput <config.yaml> [n_steps] [engaged]` |
| Profile | `just profile <config.yaml> [max_epochs]` |
| Clean | `just clean` |

## Key Components

### Environment

- `WargameEnv` — Gymnasium env with configurable board, models, objectives.
  Polar movement (angle × speed per model), phased reward curriculum, `vp_gain`
  and `player_vp_min` for VP, deployment zones, optional group cohesion.
- **DDD layering** — `domain/` owns the rules as one bounded context in
  sub-domains (`kernel/`, `battlefield/`, `sequencing/`, `movement/`, `attacks/`,
  `shooting/`, `melee/`, the `Battle` aggregate and `BattleView` at the root);
  `wargame.py` is a facade; reward and renders depend only on `BattleView`.
  See [docs/ddd-envs.md](docs/ddd-envs.md).
- **Rules specification** — [docs/rules/](docs/rules/README.md) is the rules
  authority: the spec, `constants.yaml` (every number, in inches) and
  [implementation-status.md](docs/rules/implementation-status.md), the per-rule
  gap map. Read a mechanic's chapter and gap-map row before implementing it.
  `tests/test_no_ip_references.py` keeps the repo free of references to the
  commercial product the rules derive from.
- **Play doctrine** — [docs/play-doctrine.md](docs/play-doctrine.md) is how the
  game is *won*: 43 numbered claims, each with its extension point. It is a store
  of hypotheses, never evidence. ⚠ **DO NOT EDIT IT** — it is an immutable
  external reference. Results go in
  [docs/play-doctrine-findings.md](docs/play-doctrine-findings.md), one additive
  entry per claim priced; where the record disagrees with an entry, the record wins.
- **Melee is off by default** (`melee.enabled`). Off it registers no slice and
  draws no dice, so every golden config and fixture is bit-identical. On it steps
  the charge phase; the charge is declared by the unit's leader in the command
  phase and binds the unit, the fall back is refereed, and the charge roll is an
  observation column. ⚠ Turning it on voids every baseline and agent score on
  that config, and the melee configs step `command` (`max_turns` 60 → 80).
  See [docs/melee.md](docs/melee.md).
- **Threat field** — `envs/board/` is a leaf package of board-wide reads.
  ⚠ Threat is a **next-turn** quantity (the opponent moves before it shoots): the
  `[R]` overlay reads false-safe for anyone choosing where to end a turn (18.7%
  of the board called clear that the next-turn field does not). Cover is not
  applied, which biases the field against objectives. `just measure-threat-field`.
- **Unit matchups** — `just measure-matchups` reads the two armies' stat lines
  before a model has moved; it is a reduction of the per-model expected-damage
  observation, so the table and the number the network sees cannot disagree.
  Range never enters the damage scalar. On a config where both armies share one
  profile it is 1×1 and says nothing.

### Game State I/O (`envs/state/`)

Snapshot/event pipeline for recording and inspecting matches — snapshots,
event-log deltas, `StateExporter` (wired into `step()`), replay, narration,
`analyze_match`. Driven by `replay_events.py` / `analyze_events.py` and the
`record` · `replay` · `analyze` · `analyze-compare` recipes.
See [docs/game-state-io.md](docs/game-state-io.md).

### Ratings (`rating/`)

Bradley-Terry fit putting scripted baselines and learned checkpoints on one
scale, with deployment-zone, first-turn and player-seat advantages as fitted
terms, a bootstrap over layouts, and an append-only ledger in `ratings/` keyed by
a scenario fingerprint. Recipes: `measure-seat-parity` · `measure-elo` ·
`elo-table`. See [docs/elo.md](docs/elo.md) · [docs/self-play.md](docs/self-play.md).

- ⚠ **A training run's own `eval/elo` and `self_play/learner_elo` are NOT on this
  scale.** The first is a monotone transform of `eval/vp_margin`; the second is a
  ladder against the run's own pool. Never put one in a ledger.
- ⚠ **The two seats are not the same game.** On `25v25_shooting_opponent` one
  policy from both seats loses from the player seat by −24.6 ± 9.4 vp;
  `25v25_maps_two_mode` passes the gate (+6.5 ± 6.1). `h_seat` needs three
  entrants or a self-pairing, and the gate is advisory. **Run it at n ≥ 100**
  (at n=30 it is a coin flip). On a map-pool config the gate lumps seat and side
  of table and appends nothing. See
  [the report](reports/2026-08-19-the-two-seats-are-not-the-same-game.md).
- ⚠ **`--pfsp-mode hard` (the default) and `even` drew uniformly** until the
  rating table was filled — nothing has been run under them; `uniform` is the
  pre-registered control.

### RL Algorithm and Networks

- **PPO** — actor-critic with GAE and the clipped surrogate, as a Lightning module.
- **TransformerNetwork** — NanoGPT-style, the only network. DQN and `MLPNetwork`
  were removed (`git log -- wargame_rl/wargame/model/dqn` restores them).

### Configuration

- Environment configs live in `configs/` — see [configs/README.md](configs/README.md) for the tiering.
- Algorithm config: `PPOConfig`; training config: `PPOTrainingConfig` (`model/ppo/config.py`).

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

PPO on a `TransformerNetwork` is the only thing that trains — there is no
algorithm or network to choose, and `just train <config> 800` means 800 *epochs*.
The rules below were each paid for; the evidence is in the linked report, and a
session that wants to re-litigate one reads the report first.

### The board

`configs/evaluation/maps/` is generated by `just fetch-maps` from the public
layout API: 45 tables, 16 terrain pieces each, 5 or 6 objectives, their own
deployment zones ([report](reports/2026-08-20-the-tables-are-generated-now.md)).

- ⚠ **The API is the source for terrain and zones only.** Objectives come from
  `scripts/objective_markers.json` (the API disagrees with the layout cards on 6
  of 45 tables). An objective is a RUIN, and a tie designates both, so a table
  carries 5 or 6. Pinned in `tests/test_map_objective_counts.py`.
- ⚠ **Do not "tidy" `objective_budget` to 5 or `terrain_budget` from 16** —
  either changes the tensor width and orphans every checkpoint.
- ⚠ **`long_edges` puts the armies 20" apart** against 24–40 elsewhere; at a
  12" weapon range that is a different game on 6 of 45 tables.
- **Zones are polygons**, not the `deployment_zone` rectangle; 34 of 45 are
  triangles, staircases and arcs. See `envs/CLAUDE.md`.
- **The pool has a ~6 vp resolution floor**: the variance is *across* tables and
  only 45 exist.
- **The bar on all 45, n=30, seeds 700000+:** `random` −222.5 ·
  `squad_march_take` +5.9 · `squad_march_shoot` −5.9 · `squad_march_deny` +5.4.
  `shoot` is the *weakest* script here — **name the policy, never say "the bar"**.
- **The opponent is worth ~120 vp**: `squad_march_take` scores +126.2 against
  `scripted_advance_and_shoot` and +5.9 against itself, same tables. Nothing
  trains on `25v25_maps_coherency` any more — it is `25v25_maps_two_mode` against
  an opponent ~120 vp weaker (`tests/test_map_config_pairs.py` pins this).
- **The other four goldens** generate their own terrain. `squad_march_shoot`,
  n=100, seeds 700000+: `25v25_shooting_opponent` +13.3, `25v25_cover_control`
  +15.9, `25v25_single_phase` / `25v25_curriculum` +70.3 (identical, a shared
  scenario). ⚠ **A bar of 1.00 is an artefact of an opponent that never fires**;
  switching a config's opponent voids every number on it.

### Where the agent stands

Reissued 2026-08-24 at `f741e14`; six seeds of `configs/golden/25v25_maps_two_mode.yaml`
(`ent_coef` 0.003, 300 epochs), held-out nine, n=30, K=3 verified, refereed eval
configs, scripts re-measured per opponent:

| opponent | agent | best script | gap | t | sign |
|---|---|---|---|---|---|
| `squad_march_deny` | **+20.0** | −6.1 (`take`) | **+26.1** | 3.51 | 7/9 |
| `squad_march_take` | +19.4 | +6.5 (`deny`) | +13.0 | 1.44 | 7/9 |
| `squad_march_shoot` | +33.2 | +27.7 (`deny`) | +5.5 | 0.58 | 3/9 |
| `contest_and_spread` | +16.7 | **+30.5** (`take`) | −13.8 | −1.61 | 4/9 |
| `advance_and_shoot` | +61.4 | **+135.6** (`take`) | **−74.3** | −6.98 | 0/9 |

- **The agent clears the best script significantly on one opponent of five.**
  Coherency 0.937–0.954 against the scripts' 0.863–0.911 everywhere.
- **Its defence is excellent and its offence is capped**: offence is flat at
  −42 to −71 on every row, so its lead is whatever denial happens to be worth
  and it does worse, relative to a script, the weaker the opponent. `held` is
  1.9–2.1 against every opponent. The agent plays the same game regardless.
- ⚠ **Absolute score measures the OPPONENT, not the agent.** Only the same-row
  comparison means anything.
- ⚠ **ALWAYS stamp a revision on a quoted table and state which config it was
  trained on.** Four of five rows moved between two reissues; bisecting a
  staleness claim costs a minute per point.

### The per-model curriculum (#340) — the ladder so far

One rung at a time, one axis of difficulty per rung, the scripted bar
measured first, the pass mark pre-registered, the whole-army trainer as the
pipeline control on the A and C rungs, every run on Wandb under
`curriculum-<id>`. Configs in `configs/experiments/curriculum/`, one PR per
rung stacked on #339.

| rung | axis | verdict | per-model | whole-army control | report |
|---|---|---|---|---|---|
| **A0** three lone models, one objective | — (plumbing) | **PASS with a defect**, 3/3 | success 1.000 / 1.000 / 0.950, turns 4.93 / 4.96 / 5.11 (script 4.93), rounds-to-pass 29k / 52k / 41k | success 1.000 ×3, turns 6.21 / 6.67 / 6.07, rounds-to-pass 16k / 12k / 18k | [2026-09-15](reports/2026-09-15-curriculum-a0-passes.md) |
| **A1** the same three as one squad | squads | **FAIL as pre-registered, 2/3** at 3× the control's rounds | success 1.000 / **0.790** / 1.000, turns 5.01 / 6.10 / 4.90 (script 4.96), rounds-to-pass 53k / never / 32k; coherency greedy 0.76–0.90, **sampled 0.23–0.62** | success 1.000 ×3, turns 6.20 / 7.43 / 6.10, rounds-to-pass 14k / 20k / 18k | [2026-09-15](reports/2026-09-15-curriculum-a1-fails-on-one-seed.md) |
| **A1x** the same runs resumed to 122,880 rounds | budget | **PASS 3/3** — the budget rule was the defect | success 1.000 ×3, turns 4.93 / 4.88 / 4.93; s2 passed on the first evaluation after the resume; coherency greedy 0.97–0.98, sampled 0.70–0.81 | (not re-run) | [2026-09-15](reports/2026-09-15-curriculum-a1x-passes.md) |
| **A2** four squads, one objective | bodies | **FAIL as pre-registered — solved by 10k rounds, then unlearned** at `ent_coef` 0.03 | at budget 0.370 / 0.880 / 0.980; **at 10,240 rounds 1.000 ×3, 4.02 turns (script 4.55), coherency 1.000**; drifted from ~25k; displacement entropy stuck at ~3.3 nats, clip fraction 0.26–0.37 | success 1.000 ×3, turns 5.95–6.01, rounds-to-pass 2k / 2k / 20k, held 100% for 60 epochs | [2026-09-15](reports/2026-09-15-curriculum-a2-solved-then-unlearned.md) |
| **A2b** the same at `ent_coef` 0.003 | entropy | **PASS with drift, 3/3** — the entropy bonus was the cause; 0.003 on every per-model arm from here | success 1.000 / 1.000 / 0.990, turns 4.48 / 4.19 / 4.11; displacement entropy 0.3 nats, clip fraction 0.12–0.15; 7–10 transient in-run dips per seed, all recovered, none after 63k; coherency greedy 0.74–0.79 | (not re-run) | [2026-09-15](reports/2026-09-15-curriculum-a2b-passes-with-drift.md) |
| **A3** four squads, four objectives (spread) | points | **FAIL at the cap, PASS 3/3 resumed to the control's 245k (A3x)** | at 122,880: 0.700 / 0.800 / 0.770, held 3.5–3.7; **at 245,760: 0.960 / 0.960 / 0.970, held 3.94–3.96, turns 5.19–5.33 (script 5.28)**; coherency greedy 0.38–0.52 | 0.850 / 0.980 / 0.930 at 60 epochs, **0.950 / 0.950 / 0.960 at 120** (245k rounds), turns 6.2–6.3 | [A3](reports/2026-09-15-curriculum-a3-behind-the-control-at-the-cap.md) · [A3x](reports/2026-09-15-curriculum-a3x-passes.md) |
| **A4** four squads, three objectives (a spare) | points | **FAIL at the cap, PASS 3/3 resumed to 2× the cap (A4x)** | at 122,880: 0.930 / 0.930 / 0.970, held 2.91–2.97 of 3; **at 245,760: 0.980 / 1.000 / 1.000, held 2.96 / 3.00 / 3.00, turns 4.72–4.75 (script 4.70)**, `on_obj` 0.65–0.73; explained variance 0.42–0.54 → 0.56–0.59; coherency greedy 0.43–0.53 → 0.52–0.70 | 0.980 / 0.980 / 0.990 at 60 epochs, passed in-run at epochs 44–58 (92k–121k rounds), turns 6.1–6.2 (script 4.70) | [A4](reports/2026-09-15-curriculum-a4-one-se-short-at-the-cap.md) · [A4x](reports/2026-09-15-curriculum-a4x-passes.md) |
| **A3 speed screen** three one-change arms on A3, read at the cap against A3's own runs (0.70 / 0.80 / 0.77) | why ~200k rounds | **S1 hold NULL** (0.94 / 0.81 / 0.90) · **S2 matching HARMFUL** (0.83 / 0.69 / 0.61) · **S3 warm-start from A2b AHEAD 3/3** (0.91 / 0.94 / 0.96, in-run 80% at 11k–22k rounds v scratch's 91k–never) | — | (A3's) | [S1](reports/2026-09-15-curriculum-a3-speed-s1-hold.md) · [S2](reports/2026-09-15-curriculum-a3-speed-s2-match.md) · [S3](reports/2026-09-15-curriculum-a3-speed-s3-warm.md) |
| **A5** eight squads, six objectives | bodies + points | **control NULL (scenario) on the letter, and the clause is wrong; per-model arm FAIL 0/3 (A5b), a different failure** | 0.260 / 0.180 / 0.160 at 245,760, never a rolling 50% in-run; `held` 4.1–4.6 of 6 with **8–10 of 24 bodies on points**, empty point different each episode, max stack 3.0–3.5 (script 5.8), turns 9.5–9.7 of 10 (script 6.77), coherency 0.10 — under-arrival, not stacking; clip fraction 0.37–0.42 | 0.890 / 0.820 / 0.600 at 60, **0.800 / 0.940 / 0.980 at 120**: one point left empty with 22 of 24 bodies on points — the whole-army trainer's allocation failure, on a rung the script solves at 1.000 | [A5](reports/2026-09-15-curriculum-a5-control-cannot-allocate.md) · [A5b](reports/2026-09-16-curriculum-a5b-fails-differently.md) |
| **A5, second pass** ten arms and two half-steps | credit · start · bodies · points, one at a time | **PASS on the letter, on the START axis — A5i 0.960 / 0.980 / 0.960; nothing learned from reward** | actor credit: A5c a PPO change (returns 13×), A5d **0.050 / 0.070 / 0.010**, A5e + hold term 0.170 / 0.060 / 0.060; the bar cloned from 1,200 games **0.930**, from 2,000 **0.960** (one failure event: two squads on one point); anchored PPO from the clones HOLDS in the clone's band on twenty-four reads (coef 10 and 1) and 0.1 loses it; **A5i** (from the 2,000-game clone, coef 10 / 0.03) 0.970 ×3 at 40,960, **0.960 / 0.980 / 0.960 at 122,880**, held 5.96–5.98, turns 6.81–7.04 (clone 6.97, bar 6.77), no in-run dip, sampled = greedy, IMPROVES not shown; half-steps at the cap: bodies ×2 **0.960 / 0.930 / 0.860** a turn slow, points +2 **0.030 / 0.060 / 0.330** with A5's census | (A5's) | [2026-09-19](reports/2026-09-19-curriculum-a5-passes-on-the-start-axis.md) · preregs [A5c](reports/2026-09-17-curriculum-A5c-preregistration.md) · [A5e/A5f](reports/2026-09-18-curriculum-A5e-A5f-preregistration.md) · [half-steps](reports/2026-09-18-curriculum-A5-half-steps-preregistration.md) |
| **A5-points, discount** the half-step (six squads, five points) from scratch at `--gamma 0.99` — the ladder's first optimiser-side arm | horizon | **FAIL as pre-registered — the horizon is not the wall** | 0.070 / 0.180 / 0.100 at 122,880 (the original at gamma 0.9: 0.030 / 0.060 / 0.330; at matched rounds the arm is behind on every seed at 40,960 and 81,920 — the original's 80k row is 0.220 / 0.220 / 0.110, two seeds collapsing after it), held 3.04–3.42 of 5, turns 9.65–9.87 (bar 6.71), five or six of eighteen bodies on points from turn 5, a point empty in 46–83% of episodes on every seed, sampled 6–7 vp worse than greedy; explained variance **0.78–0.85** against the original's 0.63–0.73 on the same in-run curve | (none — a diagnostic arm on a half-step) | [2026-09-19](reports/2026-09-19-curriculum-a5-points-discount-the-horizon-is-not-the-wall.md) · [prereg](reports/2026-09-19-curriculum-A5-points-discount-preregistration.md) |
| **A5-points, reward shape** two one-change arms on the half-step from scratch: R1 the success bonus paid in full however late (`terminal_bonus_speed_scaling: false`), R2 R1 plus 1.0 per point held at the clock (`terminal_objective_bonus: 5.0`) | terminal pay | **FAIL both, behind the original at every read — the lump breaks the critic, partial credit buys the near column** | R1 0.000 / 0.030 / 0.010 (held 0.90 / 2.89 / 2.52), R2 0.020 / 0.030 / 0.000 (held 2.41 / 2.93 / 2.38) at 122,880 against the original's 0.030 / 0.060 / 0.330; at 81,920 the arms 0.00–0.02 against the original's 0.22 / 0.22 / 0.11; explained variance **−0.5 to +0.2 on five of six seeds** for the whole run (the original 0.63–0.73, the discount arm 0.78–0.85), in-run success 0.2–2.5% in the last quarter (the original 9–20%); R2's far column empty in 90–100% of episodes at 81,920 on every seed; R1 s1 the sharpest walk-off on the ladder (ten bodies on points at turn 3, two at the end; sampled holds 3.20 against greedy 0.90); first arms recorded in-run, thirteen MP4s per run on Wandb | (none — a diagnostic arm on a half-step) | [2026-09-19](reports/2026-09-19-curriculum-a5-points-reward-shape-the-lump-breaks-the-critic.md) · [prereg](reports/2026-09-19-curriculum-A5-points-reward-shape-preregistration.md) |
| **A5-points, control** the whole-army trainer on the half-step, 60 epochs then the once-only extension to 120 | budget (the pipeline control) | **FAIL on the letter by two seeds — the best read this half-step has had from any trainer; the wall is the per-model trainer's** | (none — this is the control) | **0.980 / 0.620 / 0.870 at 60**, **0.910 / 0.810 / 0.950 at 120** (s1 drifted 0.98 → 0.91), held 4.91 / 4.80 / 4.95, 93–97% of bodies on objectives, turns 7.3–7.7 (bar 6.71) — against 0.00–0.33 from every per-model setting from scratch at the same rounds | [2026-09-20](reports/2026-09-20-curriculum-a5-points-whose-wall-it-is.md) · [prereg](reports/2026-09-19-curriculum-A5-points-control-and-warm-start-preregistration.md) |
| **A5-points, warm start** the per-model trainer from A4x (the three-objective pass) seed-for-seed, 122,880 then the same extension to 245,760 | start (transfer) | **FAIL on the letter; ahead of the original on 3/3 at 2× the cap; the best per-model read on the half-step** | 0.030 / 0.120 / 0.180 at 20,480 (where the original ENDS), 0.150 / 0.110 / 0.250 at 40,960, 0.100 / 0.250 / 0.520 at 81,920, **0.460 / 0.260 / 0.350 at 122,880**, **0.340 / 0.170 / 0.600 at 245,760** (held 3.59 / 3.52 / 4.29; s3 8 of 18 bodies on objectives by turn 7, 8.1 turns); no walk-off on any seed at the end, sampled = greedy within ±6 vp; two seeds fell back over the extension, s3 climbing (in-run 40 → 58%) | (0.910 / 0.810 / 0.950 at 120 epochs) | [2026-09-20](reports/2026-09-20-curriculum-a5-points-whose-wall-it-is.md) |
| **A5-points, backward start** the per-model trainer from scratch with episodes beginning at four squads on four objectives (`--backward-start 4`, share 0.75, advance 0.8 over eight rollouts), the level to walk down to deployment | start (a curriculum over the start state) | **FAIL — the level never left four: given four objectives the policy walks off them** | from deployment **0.000 / 0.030 / 0.030** at 122,880 (0.000 / 0.070 / 0.000 at 81,920), held 2.08 / 3.02 / 2.07, behind the original at every read; the level's own success 0.25–0.35 by quarter against the 0.8 bar; level-4 census at the end (n=30): success 0.53 / 0.33 / 0.50, placed squads still on objectives **2.70 / 1.97 / 3.13 of 4** (the bar 3.9, success 1.00), free squads arriving 0.83 / 1.00 / 0.63; critic healthy (EV 0.55–0.68) | (0.910 / 0.810 / 0.950 at 120 epochs) | [2026-09-20](reports/2026-09-20-curriculum-a5-points-whose-wall-it-is.md) · [prereg](reports/2026-09-19-curriculum-A5-points-backward-start-preregistration.md) |
| **A5-points, rounds per update** the per-model trainer from scratch at the whole-army trainer's 2,048 rounds per update (`--rollout-rounds 512 --num-rollout-envs 4`; ~200 episodes per update against 128 rounds' ~13) | update regime | **FAIL — the regime is not the wall; the bigger batch was worse** | **0.000 / 0.010 / 0.000** at 122,880 (0.000 ×3 at 40,960, 0.000 / 0.010 / 0.000 at 81,920), held 0.12 / 1.49 / 2.04, behind the original (0.030 / 0.060 / 0.330) at every read; the walk-off at its largest (s1: 11.3 of 18 bodies on objectives after turn 3, 0.1 at the end, in formation); the walk-off probe: a body on an objective stays 0.00–0.02 and leaves 0.65–0.81 (the bar stays 0.47), paid the same either way to within 0.005; displacement head diffuse (2.7–4.0 nats), greedy 5–25 vp below sampled | (0.910 / 0.810 / 0.950 at 120 epochs) | [prereg + amendment](reports/2026-09-20-curriculum-A5-points-rounds-per-update-preregistration.md) · [addendum](reports/2026-09-20-curriculum-a5-points-whose-wall-it-is.md) |
| **A5-points, staying** two one-change arms from scratch: `objective_stay`, a per-decision pot for a body that ENDS its step inside an objective, paid to the mover on its own step and split by occupants (S weight 0.5, S-low 0.15) | reward (staying) | **FAIL both — the signal is present at two weights and the policy never keeps an objective; the sixth reward-side lever closed on the half-step** | S **0.210 / 0.190 / 0.120**, S-low **0.230 / 0.110 / 0.110** at 122,880 (the original 0.030 / 0.060 / 0.330: S ahead by 2.8–4.1 SE on two seeds, behind by 3.7 on the third; S-low ahead on one), behind the original on two seeds at 40,960 and 81,920; held 3.05–3.43; the walk-off probe: a body on an objective is paid +0.013 to +0.018 for ending inside against −0.003 for leaving (S; a third of that on S-low), stands still on 0.00 of its decisions there on every seed and leaves on 0.48–0.67 (the original 0.60–0.81, the bar 0.33); the count on objectives rises turn 3 → end on every S seed and the stack stays under 3.5 (the pot works; the aggregate holds by churn); a 3.3× weight change moved nothing; EV 0.66–0.75, greedy ahead of sampled | (0.910 / 0.810 / 0.950 at 120 epochs) | [prereg + amendment](reports/2026-09-20-curriculum-A5-points-staying-preregistration.md) · [addendum 2](reports/2026-09-20-curriculum-a5-points-whose-wall-it-is.md) · design [#384](https://github.com/sashman/wargame_rl/issues/384) |
| **A5-points, commitment Stage 0 (CM1)** the commitment layer (#384) with no head: a per-unit objective assignment written by the environment at deployment (greedy, sticky, retired only when redundant or dead), shown to every member as a marked-target relation and claimant counts, travel and staying keyed to it, the scripts emitting theirs; tested up the ladder — A0 / A1 / A2 plumbing at 40,960, A3 at the cap, the half-step as the gate | commitment (visible, unlearned) | **plumbing PASS ×3; A3 NULL on the letter, ahead on two seeds; CM1 EXECUTION as pre-registered — the members never read the pointer, and the keying alone is a seventh reward-side setting with the staying arms' shape** | A0 / A1 / A2 1.000 ×9 at 40,960 (follow-through 0.99–1.00); A3 **0.820 / 0.850 / 0.930** at 122,880 against A3's own 0.700 / 0.800 / 0.770 (s1 +2.0 SE, s2 +0.9, s3 +3.3; held 3.81–3.92 v 3.49–3.65; every objective empty in ≤ 10% of episodes; leave on the assigned objective 0.40–0.49); CM1 **0.150 / 0.280 / 0.190** against the original's 0.030 / 0.060 / 0.330 (ahead 3.0 and 4.3 SE on two seeds, behind 2.3 on the third), held 3.16–3.35, turns 9.5–9.8 (bar 6.71), persist 0.87–0.88 (bar 0.94), follow 0.76–0.79, **leave 0.50–0.58** (bar 0.01), a body on its objective stands still on 0.00–0.01 of its decisions and is paid −0.004 to 0.000 to leave (A3: −0.004 to −0.011); flag ablation (BLANK / MISDIRECT at play) moves success by ≤ 0.05 on every seed of both rungs; EV 0.69–0.73, greedy 1.6–5.0 vp above sampled, sampled holds more (3.5–3.7) | (0.910 / 0.810 / 0.950 at 120 epochs) | [prereg + amendments](reports/2026-09-21-curriculum-commitment-stage0-preregistration.md) · design [#384](https://github.com/sashman/wargame_rl/issues/384) · arm [#387](https://github.com/sashman/wargame_rl/issues/387) |
| **LR1, the legibility rung** the A3 shape with the environment's assignment ROTATED (every squad assigned the objective greedy gave the next squad) and success `all_units_on_commitment` (each squad on ITS assigned objective); a policy that walks to the nearest covers the points and fails; bar `squad_march_committed` 1.000, plain `take` 0.000; the flag ablation on every read | commitment (legibility) | **NULL on the letter (a 3/3 conjunction); READ on two seeds — the first per-model policies that condition on a commitment** | **0.710 / 0.950 / 0.080** at 122,880 (0.05 / 0.29 / 0.03 at 40,960; 0.38 / 0.84 / 0.03 at 81,920); ablation trained → blank → misdirect → nearest: s2 **0.95 → 0.00 → 0.00 → 0.00**, s1 **0.71 → 0.00 → 0.01 → 0.00**, s3 0.08 → 0.10 → 0.06 → 0.04; held under BLANK falls 3.94 → 1.94 (s2), 3.48 → 2.35 (s1): the members go where the flag points; s2 holds four in 6.11 turns (bar 5.78), persist 1.00, follow 0.97; s3 covers 3.3 with the wrong squads, EV 0.89; sampled = greedy | (none) | [prereg + amendment](reports/2026-09-21-curriculum-commitment-revision-preregistration.md) · arm [#395](https://github.com/sashman/wargame_rl/issues/395) |
| **A5-points, commitment R1 (CM1-R1)** the half-step under the ARRIVED-KEEPS retirement rule (a squad keeps its commitment once it has had a member inside; only a latecomer to an objective another squad holds is re-assigned — Stage 0 re-assigned a walker from a shared objective for free) | commitment (retirement) | **NULL as pre-registered — the price of leaving is not the walk-off; the seventh setting closed on the half-step** | **0.340 / 0.100 / 0.060** at 122,880 against Stage 0's CM1 0.150 / 0.280 / 0.190 (s1 +3.4 SE, s2 −3.4, s3 −2.9); leave on the assigned objective **0.56 / 0.60 / 0.60** (Stage 0 0.58 / 0.56 / 0.50; STAYS needed < 0.33, MOVES < 0.50); persist 0.91–0.92 (was 0.87–0.88), follow 0.73–0.74, held 3.90 / 2.91 / 2.95; a body on its objective stands still on 0.01–0.02 and leaves on 0.54–0.65, paid −0.004 to +0.001 to leave — unchanged; ablation flat (s1 a 2 SE drift); EV 0.69–0.81 | (0.910 / 0.810 / 0.950 at 120 epochs) | [prereg + amendment](reports/2026-09-21-curriculum-commitment-revision-preregistration.md) · arm [#394](https://github.com/sashman/wargame_rl/issues/394) |
| **A3, plan-only rung (PL1 / PL2)** the commitment head trained with SCRIPTED members (`--members squad_march_committed`: the head draws the commitments, the bar's soldiers walk them; only the head and the planning value learn) on A3's shape, PL1 under the broadcast planning credit as shipped, PL2 under the per-unit counterfactual (`--planning-credit counterfactual`, B6); read on the plan-only row of `just measure-plan` (the head plans, the script executes) | planning (the head alone) | **AHEAD ×6 on the letter, PLANS on none — and the PLANS clause was the defect: with execution held at the bar's both heads plan a covering assignment the bar's members complete at 1.000 on 6/6 seeds, faster than the bar; the counterfactual credit adds nothing** | PL1 plan-only **1.000 / 1.000 / 1.000** at 122,880 (turns 5.09 / 5.14 / 5.10, bar 5.28), distinct per turn 0.85 / 0.84 / 0.90 and **0.97 / 0.94 / 0.93 on the last turn**, claimants per claimed objective 1.16 / 1.19 / 1.09, persist 0.77–0.88; PL2 **1.000 / 1.000 / 1.000** (5.00 / 5.22 / 5.01), distinct 0.87 / 0.77 / 0.88 · 0.97 / 0.93 / 0.94, claim 1.14 / 1.27 / 1.12 — paired PL2 − PL1 zero on every column; 1.000 ×6 from 40,960 on; CM3's heads on the same readout 0.530 / 0.000 / 0.130; plan-only ablation 1.00 in every column with no-claimants +0.3–0.8 turns (the head reads the counts); planning EV 0.28–0.36 (PL1) v **0.05–0.07** (PL2); the first launch VOID (KEEP on an empty slot let the head opt out and the script's fallback walked; fixed, KEEP illegal on an empty slot) | (CM3: 0.03 / 0.00 / 0.00 jointly trained) | [prereg + amendments](reports/2026-09-23-curriculum-commitment-plan-preregistration.md) · [#402](https://github.com/sashman/wargame_rl/issues/402) · [#403](https://github.com/sashman/wargame_rl/issues/403) · [#404](https://github.com/sashman/wargame_rl/issues/404) |
| **A3, the join (CM4 / FH1)** Step 4 of the plan question: CM4, the commitment head AND the members learning together from scratch on A3's shape (PL1's recipe, KEEP illegal on an empty slot); FH1, the members learning under PL1's finished head seed for seed (`--frozen-planner`: the planner draws every commitment greedy, its weights fixed), read on the split row of `just measure-plan` beside the executor alone | the join (both halves) | **NULL ×3 on both arms — the head plans at 0.72 under learning members, the members lag their own plan by 0.35–0.72 and reach 0.37–0.63 under a frozen good planner, BEHIND A3 with no plan at all; and a frozen planner is not a fixed plan — investigated: the head SEARCHES rather than plans (first-plan coverage 0.09–0.42, 0.52–0.78 with re-commits forbidden) and the members walk to the nearest objective and hover (mark-following 0.5–0.7 where the commitment is not the nearest, the bar 0.96)** | CM4 as trained **0.000 / 0.370 / 0.320** at 122,880 (held 2.01 / 2.94 / 2.96; CM3 0.03 / 0.00 / 0.00; A3 scratch 0.700 / 0.800 / 0.770; `a3_cm` 0.820 / 0.850 / 0.930), the SAME heads plan-only **0.720 / 0.680 / 0.720** (distinct on the last turn 0.88 / 0.72 / 0.84; from 0.01 / 0.17 / 0.16 at 40,960), leave 0.68–0.77 (the bar's members under the same plan 0.01), STEERS on s2 / s3 (blank → 0.00; no-claimants → 0.00 / 0.00 / 0.06), s1 reads nothing and stacks (max 4.0); planning return 0.24–0.53 (PL1 2.6), commitment entropy 0.39–1.00 nats. FH1 split **0.370 / 0.610 / 0.630** (held 3.21 / 3.59 / 3.60, leave 0.20–0.48; 0.26 / 0.60 / 0.60 at 81,920, two seeds flat over the last third), the executor ALONE with its untrained head **0.400 / 0.480 / 0.720** (ahead of the split row on two seeds: the walk is the geometry, not the plan); PL1's weights cover **0.59–0.71** of the objectives on the last turn with these members against 0.93–0.97 with the bar's; executor EV 0.95, displacement entropy 1.2–1.5 nats | (none — CM3 and `a3_cm` are the comparators) | [prereg, amendment 6](reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-6) · [#405](https://github.com/sashman/wargame_rl/issues/405) · [#408](https://github.com/sashman/wargame_rl/issues/408) |
| **A3, the join at two other sizes (CW1 / CH1)** CM4's recipe with the whole-army trunk (`--n-layers 8 --embedding-size 256`, ~5M parameters, CW1) and with two-layer policy heads on the shipped trunk (`--head-layers 2`, CH1), Sash's "we don't have enough parameters" as two one-flag arms | size (the trunk; the readouts) | **NULL ×3 on both, NOT WIDER, NOT DEEPER — behind the shipped 1.23M network on every seed on both rows; the head's plan under learning members went BACKWARDS at both sizes** | CW1 as trained **0.000 / 0.000 / 0.210** at 122,880 (CM4 0.000 / 0.370 / 0.320), plan-only **0.000 ×3** (CM4 0.72 / 0.68 / 0.72; CW1's own plan-only 0.40 / 0.01 / 0.00 at 40,960 and 0.01 / 0.31 / 0.25 at 81,920 — unlearned), commitment entropy 0.40 / 0.94 / 0.68 nats, planning EV 0.00–0.27, ablation flat on two seeds; CH1 as trained **0.020 / 0.000 / 0.000**, plan-only **0.090 / 0.050 / 0.000** (from 0.27 / 0.20 / 0.02 at 40,960), commitment entropy 0.91–1.02 nats, ablation flat on every seed, max stack 3.5–5.0, leave 0.59–0.76; the members stand still on 0.00 of their decisions on an objective at every size | (CM4 at matched rounds is the comparator) | [prereg, amendment 7](reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-7) · [#409](https://github.com/sashman/wargame_rl/issues/409) · [#410](https://github.com/sashman/wargame_rl/issues/410) |
| **A3, the join on the two-stream reward (FH2 / CM5)** Sash's two constraints as a reward: a member's only income is progress on, and presence at, its unit's committed objective (`fallback_to_nearest: false`), progress as the FRACTION of the distance at commit so completing any commitment pays the same (`normalize_to_commit_distance`), the staying term keyed to the commitment and capped per commitment (`objective_stay` 0.5, `cap_per_commitment`); the planner's stream unchanged (the outcome only); `a3_head_r.yaml`, CM4's recipe; FH2 = the members under PL1's frozen head, CM5 = head and members from scratch | reward (the members' stream) | **FAIL on every criterion on both arms — the normalised progress under a churning head is a heavy-tailed reward that broke the members' critic; the stay term worked** | FH2 split **0.39 / 0.29 / 0.40** at 122,880 (FH1 0.37 / 0.61 / 0.63), executor alone 0.35 / 0.26 / 0.24, mark-following 0.61 / 0.57 / 0.58 (mark ≥ 0.90; FH1 0.56–0.72, bar 0.96), leave-wrong 0.44–0.63 (≥ 0.80), arrival at the committed objective 0.58–0.62 (≥ 0.85), pay differential +0.02 to +0.03 per step (≥ +0.15), executor EV **0.06–0.55** (FH1 0.84–0.96); CM5 as trained **0.29 / 0.00 / 0.22** (CM4 0.00 / 0.37 / 0.32; NULL ×3, ahead on one), plan-only **0.16 / 0.00 / 0.27** (CM4 0.72 / 0.68 / 0.72), first-plan 0.04 / 0.00 / 0.35, re-commits before arrival 0.85–0.98, ablation flat ×3, members' EV **0.07–0.37** (CM4 ~0.75); the desk check passed on the bar (progress 2.0/12 to four decimals, corr with distance −0.03) and could not see it: under a head that re-commits mid-walk 10–32% of spans re-anchor under 8 in from the target and one step pays a whole commitment (+0.17) or −0.33 to −0.71 (the bar: p99 +0.07, min 0) | (FH1 / CM4 at matched rounds, the same probe on their checkpoints) | [prereg, amendments 9–10](reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-9) · [#411](https://github.com/sashman/wargame_rl/issues/411) · [#412](https://github.com/sashman/wargame_rl/issues/412) · [the document](https://claude.ai/artifact/UZ7Qwumzg9Hj8FJmfwsTtM) |
| **A3, the join on the FLOORED two-stream reward (FH2b / CM5b)** the same stream with the normalised progress anchor floored at twelve inches (`normalize_min_distance`, `a3_head_rf.yaml`), so a re-commit can never make a step pay more than a normal step; FH2b = the members under PL1's frozen head, CM5b = head and members from scratch; the desk check run on CM4's own churning checkpoints before launch | reward (the members' stream, the anchor) | **FAIL on the letter on both arms — and the floor restored the critic and the members now FOLLOW the plan; the planner is the failing half** | FH2b split **0.84 / 0.79 / 0.46** at 122,880 (FH1 0.37 / 0.61 / 0.63, FH2 0.39 / 0.29 / 0.40; success ≥ 0.85 on 2/3 missed by a hundredth and six), arrival at the committed objective where it is not the nearest **0.84 / 0.88 / 0.71** (FH1 0.66 / 0.50 / 0.65), mark-following 0.75 / 0.72 / 0.60 (mark 0.90; the bar 0.96), leave-wrong 0.41–0.65, pay differential +0.03 to +0.09, executor EV **0.92–0.94** (FH2 0.06–0.55); CM5b as trained **0.00 / 0.00 / 0.17** (CM4 0.00 / 0.37 / 0.32; NULL ×3), plan-only **0.00 / 0.01 / 0.82** — two heads hard stacks to the end (first-plan 0.00, max stack 4.8 / 8.3, "no squad committed to the empty objective" in 92–100% of failures), s3's head a covering plan in the last quarter; CM5b's members arrive at the committed objective on **0.86 / 0.94 / 0.64** (at the nearest 0.16 / 0.43 / 0.43), mark 0.86 / 0.61 / 0.75, pay differential **+0.19 / +0.13 / +0.14** (the bar +0.20; CM5 −0.01 to +0.03), members' EV 0.67–0.85, s3 STEERS | (FH2 / CM5 and FH1 / CM4 at matched rounds, the same probe on their checkpoints) | [prereg, amendments 11–12](reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-11) · [#413](https://github.com/sashman/wargame_rl/issues/413) · [#414](https://github.com/sashman/wargame_rl/issues/414) · [the document](https://claude.ai/artifact/UZ7Qwumzg9Hj8FJmfwsTtM) |
| **A3, the join with a plan-shape term and the rotated members (CM6 / FH3)** CM6 = CM5b plus `commitment_coverage` 0.3 on the planner's stream (the share of objectives some unit is committed to, paid at every close and never to a member; `a3_head_rf_pc.yaml`); FH3 = the members trained under the rotated environment writer on the floored stream (`a3_legible_rf.yaml`), read on the rotated config and at play under PL1's head | reward (the planner's stream) · start (the members' plan distribution) | **CM6 NULL / AHEAD / AHEAD — the plan-shape term makes the head plan first time and the join reaches 0.88 / 0.94 on two seeds, the best joint reads on the ladder; FH3 FAIL on the letter with one seed of three the best executor on the ladder (0.98 under PL1's head)** | CM6 as trained **0.00 / 0.88 / 0.94** at 122,880 (CM4 0.00 / 0.37 / 0.32; CM5b 0.00 / 0.00 / 0.17; A3 scratch 0.70 / 0.80 / 0.77), first-plan coverage **0.96 / 0.98 / 0.66** (CM5b 0.00 / 0.00 / 0.09; SPREADS met at 81,920), plan-only 0.86 / 0.58 / 0.94, s3's members arrive at a non-nearest committed objective 0.93 and STEER (blank → 0.00), s1's walk to the nearest (0.34) under a perfect plan; planning return 1.5–2.2, planning EV 0.72–0.75, members' EV 0.81–0.92. FH3 rotated arrival **0.94 / 0.38 / 0.50** (at the nearest 0.01 / 0.95 / 0.90; LR1 0.89 / 0.96 / 0.46; FH2b at play 0.49 / 0.53 / 0.28), under PL1's head split **0.98 / 0.70 / 0.50** (FH2b 0.84 / 0.79 / 0.46), s1 in 5.54 turns (bar 5.28) with pay differential +0.16 (bar +0.20) | (CM5b / CM4 and LR1 / FH2b at matched rounds) | [prereg, amendments 13–14](reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-13) · [#415](https://github.com/sashman/wargame_rl/issues/415) · [#416](https://github.com/sashman/wargame_rl/issues/416) |
| **A3, the join from a plan-reader (CM7)** CM6's recipe (`a3_head_rf_pc.yaml`) with the network warm-started from FH3 s1's plan-reading executor, the planner from its initialisation | start (the members' warm start) | **AHEAD / PASS / AHEAD — the first PASS of a joint arm on the ladder; the join works when the soldiers already read the plan** | as trained **0.92 / 0.99 / 0.80** at 122,880 (CM6 0.00 / 0.88 / 0.94; CM4 0.00 / 0.37 / 0.32; A3 scratch 0.70 / 0.80 / 0.77), plan-only 1.000 ×3, first-plan coverage 0.93 / 0.90 / 0.97, the members arrive at a non-nearest committed objective on **0.97 / 0.98 / 0.96** of squads and STEER on every seed (blank → 0.00), held 3.92 / 3.99 / 3.74 in 6.9–7.6 turns (bar 5.28); SURVIVES met at 40,960 (rotated arrival 0.84 / 0.92 / 0.88; s1 dipped to 0.62 at 81,920 under a churning planner and recovered to 0.86); members' EV 0.91–0.97, planning EV 0.54–0.82, commitment entropy 0.20–0.47 | (CM6 at matched rounds; the three seeds share one warm start) | [prereg, amendments 15–16](reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-15) · [#417](https://github.com/sashman/wargame_rl/issues/417) |
| **CM3, the commitment head on A3's shape** Stage 1 of #384: the policy writes each squad's ground commitment at its first open of the turn (a sampled pointer over the objective tokens plus KEEP), the close's outcome terms paid to that decision on a planning stream with a semi-Markov return per squad and its own value head, the members paid the travel potential keyed to it alone | commitment (a learned head) | **NULL / NULL / NULL, STEERS 0 — the head commits, keeps and STACKS; far behind A3 from scratch** | **0.030 / 0.000 / 0.000** at 122,880 (A3's own 0.700 / 0.800 / 0.770; `a3_cm` 0.820 / 0.850 / 0.930; 0.11 / 0.07 / 0.00 at 81,920); persist 0.53 / 0.85 / 0.91, claimants **1.64 / 2.56 / 2.05 per claimed objective** (max 4), complete 0.43 / 0.95 / 0.18, follow 0.74–0.87, held 2.17 / 2.64 / 1.72, one objective empty in 90–100% of episodes; flag ablation flat; planning EV 0.23–0.33, commitment entropy 0.9–1.0 nats, member EV 0.75–0.85; greedy 2–5 vp above sampled | (none) | [prereg + amendment](reports/2026-09-22-curriculum-commitment-stage1-preregistration.md) · arm [#398](https://github.com/sashman/wargame_rl/issues/398) |
| **EX1, the half-step under the plan-weighted execution** the environment's plan (greedy, arrived-keeps) with `commitments.execution: plan`: a soldier is paid approach × the travel term until its squad first arrives at its committed objective, then hold × the staying term (weight 1.0), keyed to that objective and nothing else — leaving in hold mode pays exactly zero; the outcome terms on the planning stream, which no soldier sees | reward (the plan as the only payer) | **NULL — keeping improves and does not add up; arrival is unfinished** | **0.100 / 0.030 / 0.000** at 122,880 against CM1-R1's 0.340 / 0.100 / 0.060 (behind on one seed, level on two); leave on the committed objective 0.38 / 0.60 / 0.52 (bar 0.14; CM1-R1 0.56–0.60), in hold mode 0.33 / 0.52 / 0.50 with leaving paid **+0.000** and staying +0.024–0.031; the hold declaration chosen on 0.00 / 0.00 / 0.29 of hold-mode openings (bar 0.67), standing still 0.00–0.04; complete 0.41–0.56, one or two objectives empty in every census episode on two seeds; member EV 0.95–0.96 | (0.910 / 0.810 / 0.950 at 120 epochs) | [prereg + amendment](reports/2026-09-22-curriculum-commitment-execution-preregistration.md) · arm [#400](https://github.com/sashman/wargame_rl/issues/400) |
| **EX1 on A3's shape** the same reward on `a3_cm`'s plan — the check that paying the hold does not cost the arrival | reward (the same) | **COSTS — half the leaving, two turns slower, a third of the success** | **0.420 / 0.270 / 0.660** against `a3_cm`'s 0.820 / 0.850 / 0.930 (below by 6–9 SE on 3/3); turns **7.88 / 7.90 / 7.93** against 6.02 / 5.65 / 5.59; leave 0.18 / 0.15 / 0.22 against 0.41 / 0.49 / 0.40 (in hold mode 0.07–0.12, leaving paid +0.000); held 3.37 / 3.10 / 3.62; s3 reads the marking (0.66 → 0.03 under BLANK); member EV 0.95–0.97 | (none) | the same · arm [#401](https://github.com/sashman/wargame_rl/issues/401) |
| **T1** A5 warm-started from A4x, against A5b from scratch | start (transfer) | **FAIL as pre-registered, 0/3 — and the transfer is real** | 0.450 / 0.290 / 0.200 at 245,760 (A5b 0.260 / 0.180 / 0.160), `held` 4.9 / 4.3 / 4.5; no seed reaches a rolling 95% in-run (peaks 64 / 51 / 64 v 48 / 43 / 30), so the halving bound is missed on every seed; at 20,480 rounds already where A5b ends at 245,760, then flat; panel normal where A5b's is red | (A5's: 0.800 / 0.940 / 0.980 at 120) | [2026-09-17](reports/2026-09-17-curriculum-t1-transfer-does-not-rescue-a5.md) |
| **C1** A3 plus one enemy unit standing on a point, nobody shooting | enemy | **FAIL as pre-registered, by one seed on a plateau** — warm-started from A3x | 0.990 / **0.930** / 0.990 at 245,760, turns 5.01–5.37 (script 4.98), held 2.96–3.15 of 3; at 80% in-run by 2.6k rounds on every seed; the short point is never the enemy's; coherency greedy 0.50–0.63 | 0.920 / 0.960 / 0.970 at 60 epochs, **0.980 / 0.990 / 0.990 at 120**, turns 6.8–7.1, 95% in-run at epochs 64 / 75 / 65 | [2026-09-16](reports/2026-09-16-curriculum-c1-two-of-three.md) |
| **C2** C1 with the enemy squad firing (`hold_and_shoot`, range 12), ours unarmed | guns (theirs) | **FAIL as pre-registered by one seed at one hundredth under — the first rung where the per-model arm is AHEAD of the control** — warm-started from C1 | **0.890** / 0.920 / 0.950 at 245,760, turns 9.8–10.2 phase-clock (script 9.60), `alive` 0.83–0.86 (bar 0.842), held 2.87–3.00; 80% in-run by 2.6k on every seed, 95% by 8k on two; one drift dip (s1, 121k, recovered); coherency greedy 0.56–0.66, sampled 0.38–0.42 | 0.710 / 0.690 / 0.760 at 60 epochs, **0.790 / 0.690 / 0.780 at 120 — fails its own criterion 3/3**, turns 14.2–14.8 of 16 (arrives in round seven of eight), 90% in-run never | [2026-09-17](reports/2026-09-17-curriculum-c2-one-seed-one-hundredth-short.md) |
| **C3** C2 with ONE of our squads armed and outranging three tough, lethal blockers; the escort's plan is shoot first, then walk | guns (ours) | **FAIL as pre-registered on both clauses — the arm found the DASH, the run with no head start found the plan** — warm-started from C2, a from-scratch companion beside it | **0.660 / 0.920 / 0.830** at 245,760, kill-before-arrival **0.67 / 0.71 / 0.62** (mark 0.90): a last body dashed onto the blockers' point the instant the other three are held, `terminate_on_success` ending the game before the blockers fire, the blockers alive in 70–84%, alive 0.62–0.74 (bar 0.963), drifting after the first pass on every seed; companion **0.740 / 0.740 / 0.090** with ordering **1.00 / 0.97 / 0.95**; bar `scripted_escort` 0.990, plain `take` 0.590 | 0.790 / 0.560 / 0.870 at 60, **0.870 / 0.830 / 0.840 at 120 — fails 3/3**, a round or more behind the escort | [2026-09-17](reports/2026-09-17-curriculum-c3-the-dash-not-the-plan.md) · C3b #372 in flight |
| **C3b** C3 with success read on the FINAL board (`terminate_on_success: false`) | the same, no instant win | **FAIL as pre-registered, every seed of every trainer — the arm learned to shoot first and then stopped** | from C2 **0.200 / 0.130 / 0.230**, ordering 0.86 / 0.77 / 0.85; blockers wiped 75–79% (C3: 16–30%), fires first 77–86%; holds 2.4–2.7 points from round five to the end, stationary 47%, the cleared point and the far point empty in half the episodes; explained variance 0.20–0.25; from scratch 0.02 / 0.02 / 0.06 with a decaying board; bar escort 0.990, plain `take` 0.280 | **0.280 / 0.000 / 0.450 at 120 epochs** — fails harder than on any rung | [2026-09-17](reports/2026-09-17-curriculum-c3b-nobody-holds-the-point.md) |
| **D1** the escort cloned into the set network (300 games × 40 epochs, two fit seeds), scored on C3b | start (supervised) | **FAIL on the letter — and the clone holds the plan** | success **0.830 / 0.850** (bound 0.96), kill before arrival **0.99 / 1.00**, blockers wiped 98%, alive 0.90, held 3.79; joint match 0.44 — the escort's unit-opening order is not in the observation (chance held-out); per head declaration 0.93, unit 1.00, displacement 0.61 held-out against 0.95 on the training episodes (over-fit; D1b runs 4× the games) | (reward on the same scenario: arm 0.20, scratch 0.02–0.06, control 0.28) | [2026-09-17](reports/2026-09-17-curriculum-d1-the-clone-holds-the-plan.md) |
| **D1b** the same from 1,200 games | start (supervised), 4× the data | **PASS on the rung's criterion, FAIL on the fidelity bound (retired)** | success **0.960 / 0.960** (bound 0.96), kill before arrival **1.00 / 1.00**, alive 0.95, held 3.93, blockers wiped 98%; displacement match 0.61 → 0.72 held-out (0.94 on the training games), the opening order at chance held-out; the two clones within a thousandth | (reward on the same scenario: 0.20 / 0.02–0.06 / 0.28) | [2026-09-17](reports/2026-09-17-curriculum-d1b-four-times-the-games.md) |
| **D2** PPO from the escort clone on C3b, four arms: plain · critic fitted · KL anchor · both | start (reward from the plan) | **plain DESTROYS, critic DESTROYS, anchor HOLDS, both holds a hair worse — nothing IMPROVES** | plain **0.028 / 0.183 / 0.028** (gone by 2.6k rounds); critic fitted (EV 0.74) **0.056 / 0.006 / 0.028**; anchor (coef 10, target 0.03) **0.944 / 0.956 / 0.956**, paired vp −1.4 to +0.7 v the clone's 0.950, half the episodes identical, EV 0.66 from cold, drift 0.87 on the displacement head (clone 0.94); both 0.939 / 0.906 / 0.911. The destroyed arms keep the order (fire first 87–97%) and lose the walk | — | [2026-09-17](reports/2026-09-17-curriculum-d2-the-anchor-not-the-critic.md) |
| **D3** anchored PPO from a clone of a WEAKER teacher (plain `take` on C3b, 0.361; the escort 0.978) | start (does the anchor let reward improve?) | **HOLDS on both anchors, IMPROVES on neither — the D-route is imitation only** | D3a (10 / 0.03) 0.339 / 0.356 / 0.333, D3b (1 / 0.10) 0.328 / 0.344 / 0.356, paired against the clone −0.006 to −0.033 (t −0.2 to −1.4), 17–22 of 180 episodes differing; kill before arrival 0.78–0.84, blockers wiped 0.63–0.67, alive 0.52–0.54 — the clone's row on every column at every read from 20k to the end; sampled = greedy | — | [2026-09-19](reports/2026-09-19-curriculum-d3-the-anchor-is-a-brake-with-no-engine.md) |
| **E1** A5's army with guns on both sides, a mirror `take` opponent, flat board | bodies + guns (the join) | **FAIL as pre-registered, all nine runs — the control learns the guns and loses the formation; the per-model arm learns neither; A5i's walk does not survive contact** | from scratch vp **−158 / −152 / −160** v the script's +20.2 (paired n=180), held 0.76 / 0.71 / 1.24 of the bar's 2.93, on_obj 0.10–0.20, coherent 0.33–0.58, alive 0.23–0.52 (bar 0.27); from A5i **−68 / −180 / −159**, held 1.26 / 0.43 / 0.58, the walk overwritten within 10k rounds (stationary 0.42–0.78), 3–4 bodies on points at turn 5 against A5i's seventeen; sampled 13–32 vp worse than greedy on five of six | 120 epochs: **+73.4 ± 10.3 / +45.2 ± 10.5** over the script on s2 / s3 by killing, −15.6 ± 9.2 on s1; held 2.47 / 3.05 / 1.98, coherent 0.86 / 0.78 / 0.77 — fails the three-clause letter on every seed | [2026-09-19](reports/2026-09-19-curriculum-e1-the-walk-does-not-survive-contact.md) · [prereg](reports/2026-09-17-curriculum-E1-preregistration.md) |

- **The per-model pipeline learns**, at 128 rounds per update, on the same
  code that sat at the floor at 8–32. It reaches the script's speed where
  the whole-army control is a round slower, and needs **2–4× the control's
  rounds** to get there. One rung, three models, no formation: a note, not
  a result.
- ⚠ **Pre-register a speed bound against the whole-army control, not
  against the script.** "The script's slowest episode + 1 round" was
  missed 3/3 by a control that learned A0 in under ten epochs, and by E1's
  control before it (6.9–7.0 against 5.55). Twice is a pattern: a converged
  PPO policy on this reward is about a round slower than a straight-line
  walk. Either calibrate the bound on the control's own final read or make
  turns a readout beside the success criterion that decides.
- ⚠ **The health panel's ratio line is recalibrated at 128 rounds per
  update: 1.6–1.9 is the regime's normal, and it does not track failure.**
  `train/ratio_p99` sat at 1.65–1.79 on all three A0 seeds (all passed) and
  at 1.83 / 1.68 / 1.70 on A1's pass / **fail** / pass — the failing seed
  read the lowest. A0 recorded it as a defect and pre-registered the test;
  A1 ran it. Read the tail beside `clip_fraction` (0.16–0.24 here) and call
  it a fault only when it moves *with* a failure.
- ⚠ **The per-model arm needs 2–4× the whole-army control's rounds on the
  A rungs, so its budget is 6× the control's slowest rounds-to-pass, not
  3×.** A0: 29k / 52k / 41k against the control's 12k–18k. A1: 53k / ~62k /
  32k against 14k–20k. At 3× A1's third seed read 0.790 on a rising
  plateau and the rung was called FAIL as pre-registered; the same runs
  resumed to 6× pass 3/3 at the script's speed (A1x). The floor is 6× from
  A2 on, cap 122,880, and a rung that fails at the cap is a real fail.
- ⚠ **The per-model arm can solve a rung and then UNLEARN it, and only
  a read at the END of the budget sees that.** A2: every seed at success
  1.000, half a round faster than the script, coherency 1.000 at 10k
  rounds; 0.37 / 0.88 / 0.98 with coherency 0.08–0.36 at 123k. The
  critic never wavered (explained variance 0.88–0.90); the displacement
  head's entropy plateaued at ~3.3 of 4.57 nats from 12k on and the clip
  fraction rose to 0.26–0.37 — once every episode succeeds, the entropy
  bonus at `ent_coef` 0.03 is what the update is left optimising, and a
  three-head policy pays it three times. Pre-register "pass at the end
  AND no in-run dip after the first pass"; read `clip_fraction` beside
  `ratio_p99` (the tail did not move with this failure, the fraction
  did); and score the periodic checkpoints when a final read fails, so a
  drift and a never-learned read as the two different things they are.
  **A2b confirmed the cause: at `ent_coef` 0.003 the same seeds read
  1.000 / 1.000 / 0.990 at the end, the head at 0.3 nats, the clip
  fraction halved** — with transient in-run dips that recover, none past
  63k rounds. Every per-model arm runs at `--ent-coef 0.003` from A3 on;
  the whole-army control's passes at 0.03 stand.
- ⚠ **On the spread rung the per-model arm trails the whole-army control
  at equal rounds — the first time on the ladder.** A3 at 123k rounds:
  per-model 0.70 / 0.80 / 0.77 with 3.5–3.7 of four points held, control
  0.85 / 0.98 / 0.93 (and 0.95–0.96 at 245k). The bodies arrive; one
  point stays short. Explained variance fell to 0.34–0.45 from 0.70–0.90
  on the rungs below: a success that is a conjunction over points is the
  first thing per-step credit has struggled to value. **Given the
  control's 245k rounds it passes 3/3 (0.96 / 0.96 / 0.97, held
  3.94–3.96) at the script's speed, matching the control's 0.95–0.96 at
  the same rounds** — both trainers take ~200k rounds to learn a
  conjunction over four points. The cap rule is symmetric from A4 on:
  when the control needs its once-only extension, the arm under test gets
  the same rounds before the rung is called — **and when 6× the
  control's slowest pass exceeds the cap, the per-model budget is 2× the
  cap**, read once (A4x: the control passed at 92k–121k of 123k, the
  per-model arm read 0.93 / 0.93 / 0.97 at 123k and **0.98 / 1.00 / 1.00
  at 245k**, at the script's speed where the control is a round and a
  half behind).
- ⚠ **The whole-army control gates the BUDGET, the script gates the
  SCENARIO, and a control that fails is a finding, not a null.** A3's
  control was still climbing at the 60-epoch cap (0.85 / 0.98 / 0.93,
  passing 3/3 at 120); A5's control fails 2 of 3 even at 120 epochs, by
  leaving one of six points empty with 22 of 24 bodies on points — the
  allocation failure the record attributes to the whole-army trainer,
  on a rung the script solves every time. Both pre-registrations called
  that NULL (scenario) on the letter, and both letters were wrong: a
  NULL clause that fires when the control cannot do what the script can
  measures the control. From A5b on: a rising control is extended once
  to 120 epochs; NULL (scenario) needs the script to fail its own
  criterion; a control that fails while the per-model arm passes is
  reported as exactly that.
- ⚠ **A TRANSFER BOUND WRITTEN AS "HALF OF SCRATCH" IS UNSATISFIABLE
  WHEN SCRATCH NEVER PASSES.** T1 (A5 from A4x against A5b from scratch)
  reads 0.45 / 0.29 / 0.20 against 0.26 / 0.18 / 0.16 — ahead on every
  seed, at 20k rounds already where scratch ends at 245k — and FAILS on
  the letter because no run of either kind reaches a rolling 95%, so
  there is no rounds-to-pass to halve. The rung measured transfer onto
  a failure. Pre-register a transfer rung with a readout that exists
  when the comparator fails (success at matched rounds, the in-run peak,
  the census at 20k) beside the rounds-to-pass bound, and read the S3
  screen (A3 from A2b, 4× faster where scratch eventually passes) as the
  clean measurement of the lever. #340's consequence is applied: later
  rungs train from scratch unless the start is the rung's own axis; the
  C rungs' warm-start rule stands on S3 and C1, not on T1.
- ⚠ **ON THE PER-MODEL FACADE, A PER-MODEL STATE TERM IS A GLOBAL TERM.**
  `PerStepReward._pay_close` pays every per-model state calculator
  (`objective_hold`, `group_cohesion`, …) at the turn close as the MEAN
  over alive models — one scalar on the close step — so the model that
  moved onto the empty point and the one that stayed on the crowded one
  are paid the same. The only per-decision credit this facade pays is an
  action-class term on the step, to the model that acted. Measured on the
  A3 speed screen: `objective_hold` at `crowding_exponent` 1.0, the
  phase facade's measured-good lever, read NULL (0.94 / 0.81 / 0.90
  against scratch's 0.70 / 0.80 / 0.77, one seed flat) and the walk-off
  it was aimed at was still there at 20k rounds on two seeds. Do not
  nominate a per-model state term as a per-model lever here until the
  retimer pays state terms to the model that produced them (a build, not
  an arm; the retimer is `envs/per_model/reward_timing.py`). **Built
  2026-09-17: `--credit actor`** pays every payment over the model count
  (a constant), so the actor's action term stays where the mean put it,
  each state term's per-model value lands on that model's own step of the
  turn, and the common payments shrink by the army size (`Credit`,
  `StepPayment.credits`, `_land_credits`; the default `mean` is untouched).
  ⚠ The first cut paid the actor undivided and the unnormalised value loss
  swamped the clipped gradient (returns 13×, pre-clip norm 5×, behind A5b
  at forty thousand rounds) — a reward-scale change is a PPO change here. A5d — A5b
  re-run with the scaled credit, A5c the unscaled control beside it — is
  its first arm; every per-model number before it was paid under the mean.
- ⚠ **A WARM START DOES NOT SURVIVE A CHANGE OF GAME, AND THE PER-MODEL
  TRAINER HAS NOT LEARNED TO SHOOT FROM REWARD ON ANY RUNG.** E1 (A5's
  army with rifles on both sides against a mirror script): the
  per-model arm from scratch reads 150–160 vp behind the script with a
  fifth of its points; the companion warm-started from A5i — the
  ladder's best walk, twenty of twenty-four bodies on six points by
  turn 7 — had that walk overwritten within 10,240 rounds (stationary
  share 0.5–0.9 from the first in-run evaluations) and ended 88–180 vp
  behind with nothing learned in its place. The whole-army control on
  the same rung learned to kill (+45 / +73 vp over the script on two
  seeds, the ladder's largest margins) and lost formation and, on one
  seed, points — failing the three-clause letter on every seed while
  beating the bar on two. **The C rungs' warm-start rule was measured
  on rungs that added an enemy who does not shoot back and does not
  extend to guns on both sides**: on this trainer the walk is unlearned
  before the shooting is learned. Before another E rung, the per-model
  trainer needs a guns-only rung (the walk given and held, the shooting
  to learn), and D3 says the anchor that would hold the walk may add
  nothing. Read sampled beside greedy on a failing per-model row —
  13–32 vp apart here says diffuse, not converged; A5i's agreed within
  1.2.
- ⚠ **THE D-ROUTE ON THE PER-MODEL FACADE IS IMITATION ONLY: ANCHORED
  PPO HOLDS A CLONE AND NEVER LIFTS IT, EVEN WITH TWO THIRDS OF THE
  RUNG TO GAIN.** D2c, A5f and A5i held clones that were already at
  their teacher's level, which left "nothing to gain" as a reading. D3
  closed it: plain `take` cloned on C3b (0.361; the escort 0.978, the
  gap one ordered plan the same reward has half-taught from scratch)
  and anchored at D2c's setting and at A5g's reads 0.33–0.36 on every
  seed at 122,880, the by-phase census the clone's on every column,
  17–22 episodes in 180 differing from the clone at all. Four
  coefficients across three rungs: 10 and 1 hold, 0.1 loses the clone
  slowly, 0 destroys it in 2,560 rounds — there is no setting at which
  reward both keeps a plan and adds to it. **A rung passed from a
  clone is passed by the clone**; read every start-axis row that way,
  do not run more anchor coefficients, and do not test "can reward
  improve a start" with another clone — three seeds off one clone read
  as one policy. What reward learns on this trainer it learns from
  scratch, and the record says what that is (a walk at the bar's speed,
  an engagement rule, half of an ordered plan) and is not (a
  conjunction over five or six points, the second half of the plan).
- ⚠ **A5'S WALL IS THE NUMBER OF POINTS TO COVER AT ONCE, AND THE RUNG
  WAS PASSED BY IMITATION, NOT REWARD.** The second pass on A5 closed
  the credit hypothesis (built as `--credit actor`, run three ways:
  moves nothing; its undivided first cut was a PPO change — returns
  13×, the value loss owning the gradient — so read a reward-scale
  change as a PPO change before reading its arm) and stepped A4's two
  axes one at a time: doubling the army at three points costs a turn
  and one seed (0.96 / 0.93 / 0.86), adding two points to A4's army
  costs everything (**0.03 / 0.06 / 0.33**, A5's census). At the cap
  the spread rungs read 0.70–0.80 at four points, 0.03–0.33 at five,
  ~0 at six. What passed the rung: the bar cloned from 2,000 games
  (0.960; from 1,200, 0.930 — every failure is two squads taking one
  point, a tie the teacher breaks by its own squad order, not in the
  observation) and held by anchored PPO on A5's reward at **0.960 /
  0.980 / 0.960** (A5i). Anchored PPO holds a clone at every
  coefficient that holds (10 and 1; 0.1 loses it slowly) and lifts
  none — twenty-four reads from the 1,200-game clone in its band,
  A5i ahead of its clone by a tenth of a turn on two seeds. The
  ladder row carries both halves. Census a clone's FAILURES by event,
  not its match; three seeds off one clone are that clone's band, not
  seed variance; and never call a pass from the in-run curve — it runs
  5–10 points above the held-out read on every arm here. The lever
  that lifts a clone, and the lever that makes a conjunction over five
  points learnable from reward, are both unnamed.
- ⚠ **THE HORIZON IS NOT THE WALL ON THE SPREAD RUNGS, AND A BETTER
  CRITIC IS NOT A BETTER POLICY.** The D3 addendum measured that a late
  payoff reaches the first decisions at a quarter to a third of its
  value under `gamma` 0.9 per close, and the ladder's first
  optimiser-side arm tested it where it should bite: A5-points from
  scratch at 0.99, the bar's 4.38 of episode reward worth 4.19 at the
  first decision instead of 2.79. It reads 0.07 / 0.18 / 0.10 against
  the original's 0.03 / 0.06 / 0.33, behind it at matched rounds at
  40,960 and 81,920, the same census on every column, and the critic's
  explained variance up 0.15 on every seed (0.63–0.73 → 0.78–0.85) with
  nothing in the policy moved — the value function fits the longer
  return and the policy gradient it feeds still cannot find the fifth
  point, as the whole-army critic probe said in August. Do not run
  another discount, an entropy coefficient (the heads sit at 1.9–2.1
  nats) or a longer budget (flat from 80k on both arms) on this
  half-step; the reward-shape arm (a flat terminal bonus; coverage
  paid per flag to its unique holder) is next. And **read the
  comparator at every checkpoint the arm is read at**: the original's
  s1 and s2 were at 0.22 at 80k and 0.03 / 0.06 at the end, so a
  seed-for-seed clause against its final row reads "ahead on s2" where
  matched rounds read "behind on every seed".
- ⚠ **A TERMINAL LUMP IS A CRITIC PROBLEM BEFORE IT IS A POLICY PROBLEM,
  PARTIAL CREDIT BUYS THE PARTIAL BEHAVIOUR, AND THREE ARMS ON THE
  HALF-STEP NOW READ ONE POLICY.** The reward-shape arms on A5-points
  paid the success bonus in full however late (R1; the remaining-rounds
  scale had paid 1.0 of 5.0 for a first success in round nine) and added
  1.0 per point held at the clock (R2). Both read 0.00–0.03 at 122,880,
  behind the original (0.03 / 0.06 / 0.33) at 40,960, 81,920 and the end,
  with explained variance **−0.5 to +0.2 on five of six seeds for the
  whole run** against the original's 0.63–0.73: a 5.0 jump on a rare
  success, or a 3.0–5.0 jump at the clock keyed to the final board, is a
  value target this critic cannot fit, and the advantages PPO trains on
  became noise — the remaining-rounds scale was also what kept the
  terminal payment small beside the dense stream. R2 at two thirds of
  the budget had the far column empty in 90–100% of episodes on every
  seed while the near column was held: a per-part payment on a
  conjunction pays the easy parts and leaves the hard one as unpaid as
  before. Read the critic's explained variance at 20k rounds on any arm
  that raises a terminal payment; a red panel there is the answer. With
  the discount (critic better), the terminal shape (critic worse) and
  the credit (A5d / A5e) all leaving the same census, the spread wall on
  this trainer is not the reward's timing, size, shape or attribution —
  **do not run a fourth reward arm on this half-step**; what is left is
  directed exploration or an observation that names the empty point.
  And gate a final read on the trainer EXITING, never on `last.pt`
  existing (it exists from the first checkpoint; a chain gated on it
  read the arms at ~85k and called it final).
- ⚠ **ON THE FIVE-OBJECTIVE SPREAD THE WALL IS THE PER-MODEL TRAINER'S,
  THE ONE PER-MODEL LEVER THAT MOVES IT IS A WARM START FROM THE
  OBJECTIVE RUNG BELOW, AND THE POLICY IT TRAINS DOES NOT KEEP AN
  OBJECTIVE IT IS STANDING ON.** Same config, same rounds: the whole-army
  control reads 0.98 / 0.62 / 0.87 at 60 epochs and 0.91 / 0.81 / 0.95
  at 120 with 4.8–4.95 objectives held, where five per-model settings
  from scratch read 0.00–0.33 with a third of the bodies on objectives —
  #283's first clear negative row on the A rungs. The per-model trainer
  warm-started from A4x reads 0.46 / 0.26 / 0.35 at the cap and 0.34 /
  0.17 / 0.60 at twice it, ahead of the original (0.03 / 0.06 / 0.33) on
  every seed there with no walk-off — T1's flat transfer did not repeat
  one objective down, and one seed of three carries the result. The
  backward start (episodes beginning with four squads on four
  objectives, `--backward-start`) never stepped its level down because
  with four objectives GIVEN the policy keeps 2.0–3.1 of them to the end
  (the bar 3.9) and reaches the fifth 63–100% of the time: the walk-off
  every census on this half-step records is a policy that leaves an
  objective it holds, not one that fails to arrive, and its critic was
  healthy. Aim the next per-model arm on this half-step at STAYING (a
  per-model payment for a body standing on an objective was null on A3
  under the mean and on A5 under actor credit, so name a different
  mechanism), read a transfer arm at matched rounds and at the extended
  budget, and calibrate a start curriculum's advance bar on the
  policy's first read at the top level, not on the bar's. **The update regime is
  ruled out too** (2,048 rounds per update, the whole-army trainer's ~200
  episodes: 0.00 / 0.01 / 0.00, worse on two seeds), and so is
  perception (the objective token carries our count on it; every body
  reads its offset to every objective). **What is measured and shared by
  every per-model arm on the half-step is that a body standing on an
  objective is paid the same for staying as for leaving, to within a few
  thousandths, and stays on 0–2% of its decisions** (the bar 47%); the
  walk-off probe (`drafts/walkoff_probe.py` in the session drafts) is
  the read to take on any per-model policy before naming its wall.
  **And paying that body on its own step for ending inside (`objective_stay`,
  a pot split by occupants, at 0.5 and at 0.15) does not make it keep the
  objective**: the leave share moved 0.60–0.81 → 0.48–0.67, the stationary
  action was never learned, success 0.12–0.23 against the original's
  0.03–0.33, and a 3.3× weight change moved nothing — the sixth reward-side
  lever closed on the half-step. **Do not run a seventh.** The per-decision
  signal is there and the policy does not resolve it into a plan it holds;
  what is missing is a commitment a body can carry across steps and a
  channel by which one squad's intent reaches another — the commitment
  layer (#384: a persistent, visible per-unit pointer to a board token,
  progress paid against it) is the next arm, and a readout named for a
  mechanism must be the number the term keys on (the term paid for ENDING
  inside; the readout counted STANDING STILL, and read zero on a policy
  that moved within the objective).
- ⚠ **A COMMITMENT THE POLICY CANNOT SEE IS A REWARD TERM, AND ON THE
  HALF-STEP THAT CLASS IS CLOSED.** The commitment layer's Stage 0 (#384,
  no head) wrote a sticky per-unit objective assignment at deployment,
  marked it on every member's relations and keyed travel and staying to
  it. Up the ladder it is plumbing-clean (1.000 ×9 on A0–A2), lifts A3 on
  two seeds (0.82 / 0.85 / 0.93 against 0.70 / 0.80 / 0.77, the empty
  column nearly gone) and reads EXECUTION on the half-step (0.15 / 0.28 /
  0.19 against 0.03 / 0.06 / 0.33, leave share 0.50–0.58, a body on its
  assigned objective never standing still) — and a play-time ablation that
  blanks or misdirects the flag moves success by hundredths on BOTH rungs:
  after 122,880 rounds from scratch the members do not read a relation
  column whose only meaning is what the reward pays against. What moved
  was the keying: a travel target fixed at deployment instead of
  re-derived every step. Run the flag ablation (`drafts/
  commit_ablation_probe.py`) on any arm that adds an observation the
  reward keys on, before calling it a plan. Two defects for the revision:
  the retirement rule re-assigns a squad that walks off a SHARED objective
  for free (the leaving step pays −0.002 to 0.000 on the half-step against
  −0.004 to −0.011 on A3, where no objective is shared), and Stage 1's
  pointer head presupposes members that condition on the commitment —
  test that on a legibility rung (the assignment deliberately not the
  nearest objective, success keyed to it) before building the head. Score
  a pre-layer comparator on the PLAIN config: the context embedding is one
  `Linear` shared by every token type, so a layer-on config perturbs an
  old checkpoint. And resolve a read chain's run directories from the
  config STEM (`per-model-<stem>-<ts>-s<seed><suffix>`), dry-listed before
  arming — two chains today matched nothing and one scored only the bar.
- ⚠ **THE SET NETWORK READS A COMMITMENT WHEN READING IT IS THE ONLY WAY
  TO WIN, AND ONE SEED IN THREE FINDS THE GEOMETRY FIRST AND STAYS THERE.**
  The legibility rung (A3's shape, each squad assigned an objective that is
  never its greedy pick, success only when every squad is on ITS objective)
  reads 0.71 / 0.95 / 0.08 from scratch at the cap, and on the two seeds
  that pass, blanking the flag or pointing it at the nearest objective
  takes success to 0.00 and held from 3.9 to 1.9 — the first per-model
  policies on the ladder shown to condition on a commitment. So Stage 0's
  "the members never read the pointer" is a fact about rungs where the
  assignment and the geometry agree, not about the network: a relation
  column is learnable when the reward makes it load-bearing. The third
  seed covers the column with the wrong squads and its critic fits that
  well (explained variance 0.89): pre-register a legibility criterion per
  seed, not as a 3/3 conjunction, and read the ablation table before
  success. And the retirement hole was not the walk-off: with a squad
  keeping its commitment through a walk-off, the half-step reads 0.34 /
  0.10 / 0.06 with the leave share at 0.56–0.60 and the leaving step still
  paid ~0 — the travel potential re-anchors on a switch for free whether
  or not the assignment moves. Seven settings now share that census;
  **read Stage 1 on the legibility shape, not on the half-step.** Two
  defects found by measuring the bar first: the env's settle context
  carried no commitment, so a commitment-keyed success criterion was
  always false, and a scripted NAME was scored on the phase facade, which
  has no commitment state — every spec on a writer config now plays the
  per-model facade, and a scripted seat never overwrites the environment's
  assignment. A bar that reads 0.000 on a rung it should pass is a defect
  in the plumbing before it is a fact about the rung.
- ⚠ **A LEARNED COMMITMENT HEAD PAID THE ARMY'S OUTCOME LEARNS TO COMMIT AND
  TO KEEP, AND STACKS — AND A STACKED PLAN COSTS MORE THAN NO PLAN.** Stage 1
  of #384 on A3's shape (the head draws KEEP or an objective at each squad's
  first open; the close's coverage and success bonus paid to that decision
  on a semi-Markov planning stream; the members paid the travel potential
  keyed to it) reads 0.03 / 0.00 / 0.00 at the cap where A3 from scratch
  reads 0.70–0.80 and the environment's greedy assignment 0.82–0.93. The
  head keeps its choices (persistence 0.85–0.91), the members follow and
  complete them (s2: 0.95), and 1.6–2.6 squads claim each claimed
  objective with one always empty. Every squad's decision is paid the same
  broadcast close, so piling onto a covered objective and taking the empty
  one earn identical credit; the planning critic (explained variance
  0.23–0.33) cannot carry the difference on its own. Read the claimants
  readout before success on any commitment-head arm; on a head arm the
  ORDER is good fixed plan > re-derived plan > learned plan that stacks,
  and the keying that made the fixed plan pay is what makes the stacked
  plan cost. The next lever is B6: a per-unit COUNTERFACTUAL on the
  planning stream (the outcome without the squad's bodies), never a switch
  price (D4 stands) and not more rounds (flat from 81,920).
- ⚠ **A SOLDIER PAID BY THE PLAN ALONE KEEPS BETTER AND ARRIVES LATE: A
  POTENTIAL CARRIES NO URGENCY, AND WITH THE OUTCOME ON A STREAM NOBODY
  RECEIVES NOTHING SAYS WHEN.** The execution phase (`commitments.execution:
  plan`: approach × travel until the squad first arrives at its committed
  objective, then hold × staying, nothing else, leaving in hold mode paid
  exactly +0.000 — verified with a mode-aware probe on the scoring rule's
  own inside test) halves the leave share on A3 (0.40–0.49 → 0.15–0.22;
  0.07–0.12 in hold mode) and moves it on the half-step (0.56–0.60 →
  0.38–0.60) — and A3 falls from 0.82–0.93 to 0.27–0.66 with turns 5.6–6.0
  → 7.9, the half-step to 0.00–0.10. The travel term is a potential, the
  same total at any speed; the success bonus scaled by the rounds left was
  the members' only speed signal, and the two-stream design put it on a
  planning stream with no planner. A member critic at explained variance
  0.95 on a return of potentials is a critic that predicts nothing about
  the outcome. Standing still is still never chosen (the hold declaration
  0.00 on five seeds of six; the policy shuffles inside, paid the same).
  The next lever is a completion INSIDE the plan — an arrival pot on the
  member stream when the mode flips, discounted per round so sooner pays
  more — never the lump back on the members. And a probe that tests
  "inside" by centre distance disagrees with the reward's base-edge rule at
  the rim: one definition, for probes too.
- ⚠ **THE COMMITMENT HEAD CAN MAKE A GOOD PLAN — WHAT IT COULD NOT DO WAS
  MAKE ONE WHILE ITS MEMBERS WERE LEARNING TO WALK; AND JUDGE A PLAN BY
  THE RUNG'S CRITERION WITH EXECUTION HELD FIXED, NEVER BY ITS RESEMBLANCE
  TO THE SCRIPT'S MECHANISM.** Stage 1's head jointly trained read 0.03 /
  0.00 / 0.00 and stacked; the same head architecture trained plan-only
  (`--members squad_march_committed`, the bar's soldiers walking whatever
  the head commits) reads **1.000 on six of six seeds** at the cap on the
  plan-only row of `just measure-plan`, faster than the bar walks its own
  plan, under the broadcast credit as shipped — the per-unit counterfactual
  (B6, built) is a paired null beside it and its planning critic does not
  fit (EV 0.05–0.07). The heads plan ADAPTIVELY: commit, watch, re-commit
  (persist 0.77–0.88), covering ≥ 0.93 of the objectives by the last turn
  and ~0.85 per turn on average. The pre-registered PLANS clause (distinct
  ≥ 0.95 per turn, claim ≤ 1.10) asked for the script's fixed assignment
  from turn one and read this as a fail; it is retired as a criterion and
  kept as a readout. Read the plan and the execution SEPARATELY on any
  head arm: the plan-only row beside the as-trained row, at every read.
  Two mechanics to carry: **a member policy that follows a plan must never
  be able to rescue an unplanned unit** (the first launch scored the
  script's fallback at 1.000 with half the objectives claimed — KEEP is
  now illegal on an empty slot), and **read the ablation through the
  plan-only chooser** on a plan-only head (its own members are untrained,
  so the as-trained columns are 0.00 and mean nothing).
- ⚠ **THE JOIN FAILED BECAUSE THE HEAD NEVER PLANNED — IT SEARCHED — AND
  NO LEARNED EXECUTOR ON A3 HAS EVER FOLLOWED A MARK: THEY WALK TO THE
  NEAREST OBJECTIVE AND HOVER.** Investigated 2026-09-24 with no training
  (`drafts/join_probe.py`, amendment 8). PL1's head, the "1.000 plan",
  commits four squads to four distinct objectives at the first opportunity
  in **9–42%** of episodes (the bar: 100%), re-commits half its squads
  before they arrive, and reaches 1.000 by watching the counts and moving
  the surplus — forbid re-commits at play and it reads **0.52–0.78**. The
  planning reward priced the search's outcome and a target switch
  re-anchors the members' potential, so re-committing is free; the bar's
  members arrive in 3.6 turns, so the search converged inside eight. The
  learned members (FH1, CM4, `a3_cm`) close more on the committed
  objective than on the nearest on 0.5–0.7 of moves where the two differ
  (the bar 0.96; a random mark 0.33), realise +0.03 to +0.06 per step of
  the +0.20 the mark pays, arrive in ~6 turns, and once on ANY objective
  hover there (leaving a wrong objective on 0.35–0.54 of moves, earning
  ~0 toward a committed objective that pays +0.48 per step): 81–88% of
  FH1's failures are the committed squad standing on another objective.
  ⚠ **`a3_cm`'s 0.82–0.93 was never plan-following** (chance, 0.48–0.57):
  its greedy writer re-derives the plan from where the bodies are, so the
  plan follows the members. The two halves trained each other into a
  search and a geometry walk: a churning mark predicts pay worse than
  the nearest objective, slow hovering members break the search. **Read
  a head's FIRST-plan coverage and its re-commits-before-arrival beside
  any plan-only row, and read the members' mark-following where the
  commitment is not the nearest** — a plan-only row without them reads
  1.000 on a head that does not plan. ⚠ The rule that stood here for a
  few hours ("the members want a plan that holds still"; hold the head's
  commitment still as the next arm) is RETRACTED: with re-commits
  forbidden at play FH1 moves 0.37 / 0.61 / 0.63 → 0.44 / 0.61 / 0.67,
  because a plan that holds still is a plan that stays wrong. What still
  stands from amendment 6: the members' half is the larger, a frozen
  planner is not a fixed plan (its plan is a function of its executor),
  no half-step under the joint head, no FH1 extension. Next: close the
  search route on the plan-only rung (commitments sticky IN TRAINING, the
  head must cover the board in one shot or fail), then the members under
  that planner, where mark-following and the hover are measured under a
  mark that does not move.
- ⚠ **SIZE IS NOT THE JOIN'S LEVER, AT EITHER END — DO NOT RUN ANOTHER
  SIZE ARM ON IT.** The whole-army trunk's size (8 × 256, ~5M parameters,
  CW1) and two-layer policy heads (CH1), each on CM4's recipe, read NULL
  ×3 and BEHIND the shipped 1.23M set network on every seed on both rows
  (as trained 0.00 / 0.00 / 0.21 and 0.02 / 0.00 / 0.00 against CM4's
  0.00 / 0.37 / 0.32; plan-only 0.00 ×3 and 0.09 / 0.05 / 0.00 against
  0.72 / 0.68 / 0.72). At both sizes the head's plan under learning
  members went BACKWARDS through training (CW1 0.40 → 0.00 on one seed,
  plan-only) where CM4's rose monotonically: more capacity fits the
  noisy planning return sooner, and what it fits is a stack. The members
  read the plan at no size (the ablation is flat) and leave at the same
  0.5–0.76. The set network plans at 1.000 with scripted members and
  holds the six-objective clone at this size; if capacity is ever the
  question again, ask it on a rung the network fails WITH scripted
  members, never on a join.
- ⚠ **A NORMALISED POTENTIAL UNDER A WRITER THAT RE-COMMITS IS A
  HEAVY-TAILED REWARD, AND THE BAR'S STABLE PLAN CANNOT EXPOSE IT.** The
  two-stream reward (progress as the fraction of the distance at commit,
  so every completed commitment pays the same; a capped staying term keyed
  to the commitment; no fallback) passed its desk check on the bar to four
  decimals and FAILED every criterion on both arms (FH2 split 0.39 / 0.29 /
  0.40 against FH1's 0.37 / 0.61 / 0.63; CM5 0.29 / 0.00 / 0.22 against
  CM4's 0.00 / 0.37 / 0.32): the head re-commits 85–98% of its squads before
  they arrive, a re-commit made near the new target sets a small anchor,
  and one step then pays a whole commitment (+0.17 per step, or −0.33 to
  −0.71 receding) against the bar's p99 of +0.07 — the members' explained
  variance fell to 0.06–0.55 from 0.84–0.96 and their advantages were
  noise. "Following pays the same for every plan" has to hold per STEP
  under every plan the writer can produce, not per completed commitment
  under the bar's. **Read a member reward's per-step pay distribution
  under a CHURNING plan in every desk check** (CM4's own checkpoints are
  the writer to use), never only under the bar. The stay term did its
  job at the scale it was given (keeping an objective pays +0.01 to +0.05
  against −0.06 to +0.02 for leaving; the members set out from a wrong
  objective on 0.66–0.85 of moves on two CM5 seeds) and is not what
  failed. Do not sweep the weights on this build: the tail scales with
  them. Anchor the fraction to the unit's first commitment (or floor the
  anchor) before any re-run, and expect the shared eastward geometry (cos
  0.5–0.7 between the committed and the nearest direction) to cap
  mark-following well below the legibility rung's 0.90 even then.
- ⚠ **WITH THE ANCHOR FLOORED THE MEMBERS FOLLOW THE PLAN, AND THE JOIN'S
  FAILING HALF IS THE PLANNER: A BROADCAST OUTCOME STREAM PAYS A STACKING
  HEAD THE SAME AT EVERY UNIT.** The floored two-stream reward
  (`a3_head_rf.yaml`) restored the members' critic (EV 0.67–0.94 from
  0.06–0.55) and, for the first time on this ladder, produced learned
  members who go where their unit's commitment points: on CM5b they arrive
  at a committed objective that is NOT their nearest on 0.86 / 0.94 / 0.64
  of squads and realise +0.13 to +0.19 per step of the +0.20 the mark
  pays (the un-floored stream +0.01 to +0.03); on FH2b, under PL1's head,
  success reads 0.84 / 0.79 / 0.46 against FH1's 0.37 / 0.61 / 0.63 — a
  FAIL on the letter by a hundredth on the success mark, with
  mark-following 0.60–0.75 where the plan agrees with the nearest
  objective for half the squads and shares most of the walk with it (the
  legibility rung's 0.90 came from a plan that disagreed 83% of the
  time). Do not sweep the weights on it. Faithful members on a stacked
  plan hold one or two objectives, and two of CM5b's three heads never
  left the stack (plan-only 0.00 / 0.01 at the cap; the third found a
  covering plan in the last quarter, 0.82): the planner's stream is the
  outcome broadcast to every unit's commitment decision, so a stacking
  head has no unit-level reason to send one squad elsewhere, and members
  who take five or six turns cannot repair a turn-one stack inside eight
  as the bar's could. **Keep the floored stream as the members' reward;
  the next lever is on the planner's side** — the per-unit counterfactual
  planning credit (B6, built, a paired null under the bar's members where
  the search converged regardless) has never been read under members who
  follow, and a plan-shape term on the planning stream is the lever after
  it. Read a head arm's first-plan coverage and the members'
  arrival-at-committed beside its success, always.
- ⚠ **A PLAN-SHAPE TERM ON THE PLANNER'S STREAM IS WHAT MAKES THE HEAD
  PLAN, AND WITH IT THE JOIN WORKS WHEREVER THE MEMBERS READ THE MARK.**
  `commitment_coverage` (the share of objectives some unit is committed
  to, paid at every close to the planning stream and never to a member)
  took first-plan coverage from 0.00 / 0.00 / 0.09 (CM5b) to 0.96 / 0.98 /
  0.66 (CM6) and the joint arm from 0.00 / 0.00 / 0.17 to **0.00 / 0.88 /
  0.94** — the first joint arm ahead of A3 from scratch, one hundredth
  short of PASS on one seed. It respects the second constraint (it pays
  the planner for a property of its own decision) and its critic fits it
  (planning EV 0.72–0.75 from 0.09–0.45); the outcome-only stream, which
  the fork probe showed cannot see a redirected squad under slow members,
  never moved a head off a stack. **What still fails is the members
  reading the mark, and it is a per-seed coin flip**: CM6 s1's head plans
  perfectly and its members walk to the nearest objective (0.00); under a
  writer that NEVER agrees with the board (FH3) one seed of three becomes
  a full mark-reader by a third of the budget (arrival 0.94 on the
  rotated config, 0.98 under PL1's head at the bar's speed, pay
  differential +0.16 of the bar's +0.20 — the best executor on the ladder)
  and two never do (0.38 / 0.50; LR1 read 2 of 3 on the old reward). Read
  an executor at 40,960 on the rotated config and restart the seed that
  walks to the nearest; the join's next arm is CM6's recipe with the
  mark-reading executor as the members' warm start, and then the
  half-step. Keep `a3_head_rf_pc.yaml`'s reward as the reward from here.
- **A3'S JOIN IS CLOSED: PLANNER AND MEMBERS TRAINED TOGETHER PASS THE
  RUNG WHEN THE MEMBERS START FROM A PLAN-READER.** CM7 — `a3_head_rf_pc`'s
  reward (the floored two-stream member terms, `commitment_coverage` on
  the planner's stream), the network warm-started from FH3 s1's executor,
  the planner from its initialisation — reads **0.92 / 0.99 / 0.80** at
  the cap, the first PASS of a joint arm on this ladder, with the members
  arriving at a non-nearest committed objective on 0.96–0.98 of squads
  and the planner covering the board first time on 0.90–0.97. The same
  recipe from scratch (CM6) read 0.00 / 0.88 / 0.94 because one seed's
  members never read the mark. The two-stage recipe — the rotated writer
  first (a plan-reader on one seed of three), the head second — is the
  procedure until the seed lottery on the members' side is understood.
  The plan-reading walk survives a planner trained on top of it but can
  wobble while the planner churns (s1: rotated arrival 0.84 → 0.62 →
  0.86 as re-commits before arrival went 0.13 → 0.33 → 0.11). The residual
  on this shape is speed (6.9–7.6 turns against the bar's 5.3) and the
  walk-off; the question moves to the half-step (CM8 / CM8w), where the
  planner's churn (86–100% of squads re-committed before arrival at a
  third of the budget, the members arriving and leaving) is the first
  thing to read.
- ⚠ **The spread rung is an ASSIGNMENT problem from a random start, and
  the travel term's per-objective assignment half-contradicts it — but
  fixing the assignment made it WORSE.** Only 12% of A3 deployments give
  the four squads four distinct nearest points; under `closest_objective_v2`
  one squad owns two or more points on 49% of steps and the leftover squad
  is paid by `fallback_to_nearest` toward a point already claimed. A
  matching (`one_objective_per_group`, ships default off) took that to 0%
  and read **HARMFUL** (0.83 / 0.69 / 0.61): a matching re-solves every
  step, a target switch is unpaid, and the critic got worse (explained
  variance 0.26–0.38). Fifth empty-or-worse result on this term.
- **Warm-starting a rung from the rung below is the one lever that moved
  the spread rung: AHEAD 3/3** (0.91 / 0.94 / 0.96 against scratch's 0.70 /
  0.80 / 0.77 at the same rounds; in-run 80% at 11k–22k rounds against
  91k–never). The A2b one-point policy carried over and did not have to
  unlearn stacking, against the written prediction. Unpaired on init; one
  seed passes at n=100 and two sit within one SE. T1 (A5 from A4) keeps
  its own pre-registration; this is its prior, and the C rungs' rule
  ("warm-start from the rung below") now has a measurement behind it.
- ⚠ **THE ALLOCATION RUNG BEATS BOTH TRAINERS, AND THEY FAIL IN
  OPPOSITE WAYS.** A5 (eight squads of three over six points, the
  spare-squads shape): the whole-army control puts 22 of 24 bodies on
  points and leaves one empty (0.80 / 0.94 / 0.98 at 245k); the
  per-model arm from scratch puts **8–10 of 24 on points**, holds
  4.1–4.6 of 6, and reads **0.26 / 0.18 / 0.16** at the same rounds
  having never reached a rolling 50% in-run — under-arrival with the
  squads dissolved (coherency 0.10), not stacking. It is the first rung
  where the per-model arm is worse than the control at equal rounds,
  and by a wide margin; more rounds is not the reading (peaks at
  133k–211k, no climb after). Read a per-model FAIL's census before
  naming it "allocation": `on_obj` separates "arrived and mis-spread"
  from "never arrived". Hypothesis on file, untested: the state terms
  paid as the mean over alive models dilute a body's own credit with the
  army (1/24 here against 1/12 on A3–A4) — the retimer build the speed
  screen named is the test. T1 (the same rung warm-started from A4x)
  reads whether a warm start buys arrival.
- **On the first rung with an enemy, the warm start carries the skill
  and the enemy's disc is not the failure.** C1's per-model arm, started
  from A3x, was at 80% in-run by 2,560 rounds on every seed where the
  whole-army control needed 131k–154k, and reads 0.990 / 0.930 / 0.990
  at 245,760 — a FAIL by one seed on a plateau, the control passing 3/3
  two rounds slower. The census puts every miss at the points *beside*
  the enemy's, never on its disc: the engagement-range endpoint rule is
  learned in a few thousand rounds, and what a C rung inherits from A3
  is A3's residual allocation error (0.96–0.97 at its best). Read a C
  rung's FAIL against the A rung it warm-started from before blaming the
  enemy.
- ⚠ **NO WHOLE-ARMY CONTROL CAN BE READ ON A RUNG WHERE THE OPPONENT
  CAN BE WIPED (#317).** The phase facade ends the battle when the
  opponent army is wiped, against the rules; the per-model facade plays
  on. C2 as #340 wrote it (both sides armed) had our twelve rifles wipe
  the three blockers in two shooting phases, and the same scripted bar
  read **0.380 on the phase facade against 1.000 on the per-model one**
  on the same seeds (`just measure-bridge`: BRIDGE DIVERGES) with
  `alive` 0.95 on both — the wipe rule ended the game before the last
  squad arrived. So C2 arms their side only, C3 carries our guns where
  the blocker is meant to die, the C and D rungs avoid wipes by design,
  and the E rungs' control is compared against itself on the phase
  facade's own evaluation family. Run `just measure-bridge` on every
  rung before its control launches; a divergence there is a design
  fault, not a finding — **bounded 2026-09-17 with E1: a divergence
  under a rule the per-model facade RECORDS (`FacadeDivergence`, e.g.
  `shooting.targets_judged_after_casualties`, which fires in every
  episode once a side fields several shooting units) is the per-model
  facade playing the more rules-faithful game; the rung stands and
  each trainer is read against the bar on its OWN facade.
  `measure-bridge` names the recorded rules and exits 2 only when
  nothing was recorded.** **Fixed 2026-09-17 with C3:** the phase facade ends
  on an opponent wipe only under `terminate_on_opponent_elimination`
  (default `False`, the mirror of the player-side switch), so the two
  facades agree by default and a wipe rung can carry a whole-army
  control. Any phase-facade episode before that date that ended by an
  opponent wipe was scored short; at the goldens' lethality that is
  rare and unmeasured.
- **With the blocker firing, the whole-army control fails and the
  per-model arm does not — the first rung where the arm is ahead.**
  C2: the control reads 0.79 / 0.69 / 0.78 at 120 epochs, arriving in
  round seven of eight (turns 14.2–14.8 of 16 against the script's
  9.60), where C1's control passed two rounds behind the script; three
  rifles turned it from slow into too slow. The per-model arm from C1
  reads 0.89 / 0.92 / 0.95 at the script's speed and the script's
  `alive` (0.83–0.86 against 0.842), 80% in-run by 2,560 rounds — a
  FAIL on the letter by one hundredth on one seed, whose misses are
  C1's spread allocation residual, not the enemy's disc. Read a
  whole-army control's turns beside its success on every guns rung: a
  control that arrives with the game nearly over is measuring its
  speed, and the per-model arm's lead here is speed.
- ⚠ **A SUCCESS CRITERION THAT ENDS THE EPISODE THE INSTANT IT HOLDS
  REWARDS WHATEVER REACHES IT FIRST.** C3 (one armed squad must shoot
  three tough, lethal blockers off a point before the unarmed squads
  walk in): the per-model arm warm-started from C2 read 0.66 / 0.92 /
  0.83 by dashing a last body onto the blockers' point the moment the
  other three were held — `terminate_on_success` ended the game before
  the blockers fired again, the blockers alive in 70–84% of episodes, a
  third of the bodies dead, the pre-registered ordering clause (kill
  before arrival ≥ 0.90) at 0.62–0.71. With success read on the FINAL
  board the escort still reads 0.990 and plain `take` falls 0.590 →
  0.280. On any rung where reaching a state and keeping it are
  different skills, set `terminate_on_success: false` and judge the
  end (C3b). And **the C rungs' warm-start rule has its counter-example**:
  the C2 policy's rush-in was the thing to unlearn, and the from-scratch
  companion — not the warm start — learned the order (0.95–1.00) while
  being slow at the walk (0.74 / 0.74 / 0.09). Run a from-scratch
  companion beside every warm-started arm from here; a warm start
  carries the habit as well as the skill.
- ⚠ **NEITHER TRAINER LEARNS AN ORDERED PLAN FROM REWARD ALONE, AND
  THE PER-MODEL ARM STOPS HALFWAY.** C3b (C3 judged on the final
  board): the arm from C2 reads 0.20 / 0.13 / 0.23, the from-scratch
  companion 0.02–0.06, the whole-army control 0.28 / 0.00 / 0.45,
  against the escort's 0.990. Removing the instant win taught the arm
  the first half of the plan — it wipes the blockers in 75–79% of
  episodes where on C3 it did in 16–30%, and fires first in 77–86% —
  and it then holds the two near points it reached by round four and
  parks (47% of openings stationary, explained variance 0.20–0.25, the
  ladder's lowest), the cleared point and the far point empty in half
  the episodes. Nothing in the reward pays for going on after the half
  it was paid for. **Read a multi-phase rung's census by phase** —
  killed, arrived, held — before naming what a trainer learned; success
  reads 0.20 for half the plan and 0.02 for none. The continuation is
  the D rungs' question: clone the escort into the set network (D1,
  #331) and ask whether PPO can improve a policy that already carries
  the whole plan (D2).
- **THE SET NETWORK HOLDS AN ORDERED PLAN FROM IMITATION THAT REWARD
  COULD NOT TEACH IT.** D1: the escort cloned from 240 games reads 0.83 /
  0.85 on C3b's final-board scenario with the order intact (kill before
  arrival 0.99–1.00, blockers wiped 98%, alive 0.90) where 245k rounds
  of PPO reached 0.20 from C2 and 0.02–0.06 from scratch. Two clones
  agree to a thousandth. ⚠ **Do not score a per-model clone by a joint
  match that includes whose turn it is**: which unit a script opens
  next is its own plan order, not a function of the observation — the
  clones are at chance on it held-out (one in four) and at 0.60 on the
  episodes they were fitted on — so a joint-match bound of 0.95 is
  unattainable for any clone of such a teacher. Score per head given
  the teacher's model (here declaration 0.93, unit-pointer 1.00,
  displacement 0.61) and by the rung's own criterion. The displacement
  head over-fits 240 games (0.95 training, 0.61 held-out): a
  continuous move quantised into a column needs games, not epochs.
  **D1b (1,200 games) reached the rung's bound — 0.96 / 0.96, ordering
  1.00, alive 0.95 — while the displacement match rose only 0.61 → 0.72:
  the fidelity bound is RETIRED for the D rungs.** Score a clone by the
  rung's own criterion and the ordering; report the per-head match as a
  readout and never bound it where the teacher's quantisation forbids.
  The 1,200-game clone is the best policy on the per-model facade on
  any rung with guns, and D2 starts from it.
- ⚠ **WARM-START THE PER-MODEL PPO FROM A CLONE ONLY UNDER A KL ANCHOR
  TO IT; THE CRITIC IS NOT THE LEVER.** D2, the 2×2 on C3b from D1b's
  clone (0.95): plain PPO reads **0.03 / 0.18 / 0.03**, gone by 2,560
  rounds; with the clone's value head fitted first (explained variance
  0.74, policy bit-identical) **0.06 / 0.01 / 0.03**, the same collapse
  at the same speed — the whole-army record's cold-critic diagnosis is
  not the mechanism on this trainer, and was the obvious wrong guess;
  with the KL anchor (`--kl-ref-coef 10 --kl-ref-target 0.03`, #332)
  **0.94 / 0.96 / 0.96**, paired vp within 1.5 SE of the clone, half the
  episodes identical, the critic learning to 0.66 from cold; both
  together 0.94 / 0.91 / 0.91. The destroyed policies keep the order
  (fire first in 87–97% of episodes) and lose the walk after it. Under
  the anchor reward changes nothing measurable: the coefficient climbs
  to its cap (Adam normalises the penalty's gradient, so the measured
  drift sits at ~0.09 nats per decision whatever the coefficient) and
  the policy sits where it started. **The IMPROVES question is the
  anchor's coefficient**, not more rounds and not the critic. Read a
  2×2 before naming a mechanism the record already offers.
- **Six per-model seeds on two rungs all arrive within 0.2 turns of the
  script; six whole-army control seeds are all a round or more behind**
  (per-model 4.88–5.11 v script 4.93–4.96 v control 6.07–7.43). Movement
  only, three bodies, one point — a pattern about the two trainers on this
  reward, not yet a result about the game.
- **The sampled policy walks the squad apart; the greedy one does not.** On
  A1 the policy training rolls out has coherency 0.23–0.62 where the policy
  a score reports has 0.84–0.94 — nothing on the A rungs pays for
  formation. The E rungs' referee will price that gap; read the sampled row
  before trusting a per-model coherency figure.
- **Measure the bar before fixing the disc.** At objective radius 3 the
  script itself failed 5% of A0 episodes with one model frozen behind a
  friend; at radius 4 it is 1.000. A rung whose bar fails its own criterion
  goes back to design, and the change is recorded in the pre-registration
  before any training number exists.

### How to measure here

- **Score on the refereed eval configs, at K=3 with `verify_moves`, and say how a
  score was decoded.** Unrefereed scoring flatters the scripts by ~16 vp.
- **n=100 minimum; n ≥ 180 on both sides for a 5–15 vp claim.** Per-episode
  `vp_margin` sd is ~45–50 on the generated-terrain configs and **81–89 on the
  map-pool configs**; n=45 is ±12.7.
- **Measure the comparator at the same n as the arm and propagate its SE.** A
  deterministic script is a fixed policy sampled over scenarios, not a constant
  (a bar remeasured at n=180 moved +11.8 → +36.8).
  [Report](reports/2026-09-06-the-ladder-was-measurement-noise.md).
- **Never quote an across-seed SE when every seed shares the scenarios** —
  scenario noise is common-mode. Report the paired per-scenario estimator.
- **Raising n moves the mean, not only the interval.** Predicting an interval
  from a small-n point assumes it was unbiased.
- **Quote a t AND a sign count** — and check what the count would be under the
  effect claimed; at sd ~85 a true +10 predicts 24 of 45.
- **Tune on one seed band, confirm on a disjoint one.** Bands: evaluation
  700000+, in-run eval 500000+, baselines 10000+, clone 800000+, tuning
  900000+. A sweep at n=90 put an effect at +5.2 (t≈2.9) that confirmed at +0.5.
  [Report](reports/2026-09-06-the-tuning-band-picked-the-wrong-cells.md).
- **Difference against the same seed set, never against a published row.**
- **Fix the comparator by name before measuring**, on the statistic you will
  report. Argmax-selected "best script" changes identity between cells and
  inflates the script by +1.4 to +2.9.
- **Measure the configuration that SHIPS**, not an intermediate one. Twice a
  partial change pointed the opposite way from the whole.
- **Pair your arms.** Same seed = same init (`train.py` seeds before building
  the model), so the per-seed difference is a paired estimator worth an order of
  magnitude. Report the per-seed difference, its sd and the correlation. A shape
  change (added action, wider tensor) cannot be paired; a zero-initialised
  conditioning path restores pairing for an added *input*, never an added action.
- **Three seeds minimum, and do not carry one config's seed spread to another**
  (11.2 vp on `two_mode`, 26 vp on `coherency`). Six seeds at 1000 epochs to
  resolve anything a three-seed screen reversed.
- **Two seeds off one warm start are not two samples** — training amplifies the
  initialisation. Vary the warm start and record which checkpoint each run
  descended from. [Report](reports/2026-08-16-enforcement-is-a-referee.md).
- **Screen at ~300 epochs, quote effect sizes at 1000+.** A marginal 300-epoch
  result means "run it longer". But ⚠ a three-seed screen was read as a result
  twice and reversed both times at 1000 — nothing from one should move a design
  decision.
- **Power-check a per-seed bound against the expected spread before writing it
  down.** A "−8 on 3/3" bound failed 56% of the time for a lever costing zero.
- **Compare arms at `last.ckpt`, not the newest `ppo-NNN` checkpoint** (which
  records the last epoch whose *training* reward improved). **Score a killed run
  from its highest `ppo-NNN-*.ckpt`** — SIGKILL writes no `last.ckpt`, so it is
  up to 25 epochs stale.
- **Never read a launcher's exit code as evidence a run happened**; check `ps`,
  the GPU, and that checkpoints advanced. `train-arm` and a broken
  `--resume-ckpt-path` both exited 0 with every run dead.
- **Every score carries `coherent` and `adrift`.** A `vp_margin` alone is a
  result plus an unstated claim that the moves were legal
  ([docs/metrics.md](docs/metrics.md)).
- **`held` ranks, `vp_margin` decides, `on_obj` does neither.** `held` is an
  end-state snapshot while VP accrues every round. `just measure-objective-split`
  gives a redistribution ceiling that can rule re-allocation *out*, never in.
- **Prefer `vp_margin` to win rate** (win rate cannot resolve under ~7pp).
- **Raw vp is not comparable across horizons** (five-round sd 12, twenty-round
  91); quote within a horizon or normalised.
- **A checkpoint's own metrics are seed-set dependent**; never quote one without
  its seeds. Training's `eval/baseline_*` (20 episodes, seeds 10000+) and
  `eval/win_rate` (10, seeds 500000+) are not comparable to each other.
- **Every `measure-*` recipe takes trailing `key=value` scenario overrides**
  (`rounds=5`, `weapon_range=24`, …); the printed header names them.
- **Run the clone control on any behavioural statistic** before building a
  diagnosis on it. A clone of the *winning* script scored near the losing agent
  on the squad-heading statistic: it measured the architecture.
- **A scripted policy is not a control for what a reward term does** — it does
  not learn from reward.
- **Split a statistic by where a model ends, not where it starts, and give every
  behavioural statistic its within-policy control.** `random` is not a control
  for action-slice usage (it never chooses an advance) or for movement delivery
  (a blocked random policy tries another direction; a purposeful one re-issues).
- **Read the census before naming a failure**: killed / arrived / held, and
  `alive` beside `held` (a coherency rate rises whenever an army dies).
- **Read the traces, not just the aggregates** — `just analyze-compare`. Only
  `vp_per_step` ranks policy quality there.
- ⚠ **Don't query the Wandb API while runs are training** — four runs segfaulted
  within 20 s of two `wandb.Api()` calls. Read `wandb/run-*/files/output.log`.
- ⚠ **Training is NOT bit-reproducible** (GPU float nondeterminism amplified by
  sampled actions: 0 of 222 tensors identical on a map-pool config). Pairing
  holds at initialisation only; re-running a control is legitimate, and a no-op
  digest for a training-loop change has to be argued from static reading.
- ⚠ **Every eval metric and training video is the UNDECODED policy**, and the
  rollout regime (K=1, undecoded) is not the scored one. Measure a policy in the
  regime it trains in whenever a play-time decode stands between the two.

- ⚠ **STATE THE ROUNDS PER UPDATE ON EVERY PER-MODEL ROW, AND NEVER LET THE
  ENV COUNT COME FROM CPU AFFINITY.** `train_per_model.py` updates every
  `rollout_rounds × num_rollout_envs` rounds, and auto-detection clamps the
  env count to `os.sched_getaffinity` — so a pinned launch trains on ONE
  env. Every per-model run before 2026-09-14 did, updating every 8–32 rounds
  against the phase facade's 1024: under one episode per update, re-read four
  times. Their flat curves are **void for regime, not a verdict on the
  architecture** ([report](reports/2026-09-14-one-episode-per-update.md)).
  A per-update statistic at 32 rounds per update is not the same quantity as
  at 1024; `train/rounds_per_update` is on every row, the driver warns when
  the clamp bites, and `just train-per-model-arm` refuses a flag string
  without `--num-rollout-envs`.
- ⚠ **SCORE A PER-MODEL CHECKPOINT GREEDY WITH THE PASSIVE FINGERPRINT, AND
  READ THE SAMPLED SCORE BESIDE IT BEFORE CALLING A FLAT CURVE "NOTHING
  LEARNED".** The per-model facade decides "do nothing" as explicit steps
  (`stationary`, `hold_fire`), so every `EvalResult` from it carries
  `stationary_share` / `hold_fire_share` (`stat` / `hold` on every scoring
  table, `eval/stationary_share` in a run). On the void runs the greedy
  share swung **0.00 ↔ 0.94 between consecutive evaluations**: a diffuse
  declaration head whose argmax flips, not a policy that learned to stand
  still. `just measure-per-model-eval-mode` scores greedy and sampled on
  identical seeds, paired.

### Designing a reward lever

- **Check the agent can OBSERVE what the lever keys on.** Ask: if two states
  differ only in what this term keys on, do they differ in the observation?
  Two opposite levers halved occupancy because both keyed on counts the agent
  could not see.
- **Per-model is necessary, not sufficient — the number must VARY across the
  choice the model is making.** Flat `objective_hold` paid the thirteenth model
  on a point the same as the first.
- **An anti-concentration lever must REDISTRIBUTE reward, not destroy it.**
  `overstack_penalty_per_extra` and `surplus_value` both lowered total income and
  both halved occupancy; `crowding_exponent` conserves the pot and is worth 68 vp
  at fixed weight. Ask whether the wanted behaviour pays *more in total*.
  [Report](reports/2026-08-08-paying-the-pot-beats-the-bar.md).
- **Sign does not separate winners from losers; total income does** — the
  positive twin of the overstack penalty failed identically. Magnitude matters
  more (`group_cohesion` −0.2 inverted the ranking; −0.05 is in the winner).
- **Raising an objective weight alone is catastrophic** (weight 1.25 with the
  exponent off: −40.4 against +3.25, 20 of 21 survivors on one point).
- **A term with negative net income is not thereby broken**: `measure-income-share`
  shows what a term costs, never what it prevents. Removing the small overstack
  penalty lost −12.2 ± 5.5 paired.
- **Write reject rules for the failure the lever actually risks**, not the one
  you are already worried about (`alive` *fell* where the rule watched for it rising).
- **Read the doctrine entry's verdict before writing the term**, and price the
  claim as a scripted policy (`just measure-paired`, no GPU) first.
- **Treat any precision or numerics setting as a reward-affecting change.**
- Register new calculators and criteria in [docs/reward-phases.md](docs/reward-phases.md).

### Running a run

- `just train <config.yaml> [epochs]`; parallel arms with `just train-multi`
  (unique `--run-suffix`, shared `--wandb-group`).
- Copy a `golden/` config into `experiments/` to make an arm — **never edit a
  golden config** ([configs/README.md](configs/README.md)).
- **`checkpoints/` is the only copy of the weights; `just clean` is destructive.**
  Checkpoints are deliberately not uploaded to Wandb.
- Key options: `--record-during-training`, `--max-epochs`, `--n-eval-episodes`,
  `--seed`, `--tf32`, `--precision`, `--eval-every-n-epochs`, `--lr`,
  `--max-grad-norm`, `--no-wandb`, `--run-suffix`, `--wandb-group`,
  `--warm-start-ckpt-path`, `--resume-ckpt-path`, `--kl-ref-coef` /
  `--kl-ref-target`, `--n-layers` / `--embedding-size` (⚠ change the network;
  such a checkpoint is comparable to nothing).
- **Self-play is opt-in** (`--self-play`, `--pool-*`, `--pfsp-mode`; off builds
  no scheduler, so a control is bit-identical). ⚠ Do not start one on a scenario
  whose seat-parity gate fails. See [docs/self-play.md](docs/self-play.md).
- `--eval-every-n-epochs 4` cuts wall-clock ~16% — **single-phase configs only**
  (on a curriculum it changes which epoch a phase advances on).
- **Training logs the bar** (`eval/baseline_*`: `random`, `squad_march`,
  `squad_march_shoot`). Read the shooting one.
- Inspect a run with `just run-summary <run_id>` (rolling means; a single-epoch
  `success_rate` is a binomial sample) and `just measure-phase-gates`.
  [docs/metrics.md](docs/metrics.md) says what each key means.
- `just profile <config.yaml>` writes `profile.html`; `just simulate-latest` runs
  the newest checkpoint.

### Performance and numerics

- ⚠ **TF32 is off by default: it costs ~8.5 vp** (paired, bit-identical control),
  and the speed is 17.8% of an epoch. `--tf32` for smoke and profiling only.
  [Report](reports/2026-08-09-tf32-costs-eight-vp.md). `bf16-mixed` is opt-in
  with only its speed measured.
- **`torch.compile` is deliberately not wired**: it prefixes `state_dict` keys
  and the warm start loads with `strict=False`, so such a checkpoint would load
  as nothing and score a random network.
- `just measure-throughput <config>` splits `env.step()` by section and
  calculator. Any reward-pipeline change must keep `tests/test_reward_golden.py`
  bit-identical. See [docs/training-throughput.md](docs/training-throughput.md).
- `tests/test_network_size.py` pins the shipped 8/8/256 trunk.

### Where a finding lands — all four, or it is not landed

1. **`reports/<YYYY-MM-DD>-<slug>.md`**, carrying its provenance: date · GPU or
   not · seeds and seed base · n · config and whether refereed · decode K and
   `verify_moves` · paired or not · the comparator by name · opponent · epoch ·
   code revision · checkpoint · coherency · t and a sign count · Wandb run id.
2. **A one-row index entry in `reports/README.md`**, verdict bold and first. On
   a retraction the report body gains an appended `## ⚠ RETRACTION`; the index
   row is rewritten.
3. **One row in the findings table below, or one bullet in a rules section** —
   the verdict and the rule it left, never the narrative. A retracted rule is
   deleted here, not annotated. If no rule changed, write that explicitly.
4. **The live doc** — `docs/play-doctrine-findings.md` if a `D-NN` was priced,
   `configs/README.md` if a config was added or retired,
   `docs/rules/implementation-status.md` if a gap-map row moved.

Then set the `outcome:` label and remove `needs:writeup`. `kind:bug` and
`kind:build` use a shorter list: a pinning test, a "What voids a number" bullet
if it voids, a gap-map row, and goldens byte-identical or deliberately regenerated.

### What voids a number

Results either side of each are not comparable. Re-measure rather than carry a
figure across one.

- **2026-08-10** — the board stopped being a chessboard (real points, exact
  move distances, sampled-ray sight, base radii). Every specific figure before it
  is stale.
- **2026-08-13** — models no longer block line of sight; `eval/exposure_rate`
  changed definition the same day. `squad_march_shoot` moved +38.0 → +17.0.
- **2026-08-19** — a corpse used to pin your shooting (92% of engagement
  suppressions were spurious). Worth +7.0 vp to the agent, every config affected.
- **2026-08-20/21** — the eval tables were regenerated and re-measured.
- **2026-08-22** — the observed control count now matches the scored one (7.6%
  of slots disagreed). `observation_golden_25v25_shooting_opponent.npz`
  regenerated; +0.84 ms/step.
- **2026-08-23** — the scripts learned to Advance, a move must end unengaged,
  and the wholly-within deployment check (`d607561`) landed. The scripted bar
  moved +1.3 to +32.6 on advance configs and **+7.6 on non-advance
  `take_opponent_refereed`**; every 2026-08-21 row is stale.
- **2026-08-23** — the command phase is a real agent step on advance configs;
  episodes ending early by elimination lose one scoring event (−1.5 on 10 of 45).
- **2026-08-31** — `group_span` rounds up, so `max_groups` is a real cap. The
  goldens are bit-identical; the three `30v15` / `15v30` arms and
  `25v25_maps_take_small_units` were aliasing units and are void.
- **2026-09-06** — a fully hidden model no longer denies its unit cover (#289).
  Scripted bar +1.9 / +2.1 / +3.0 on the three terrain configs.

- **2026-09-14 — every per-model training run before this date is void.**
  The stage-1 calibration sweep (`checkpoints/per_model/calibration_stage1/`,
  `VOID.md` there) and the three observe runs trained on one rollout env at
  8–32 rounds per update, and their checkpoints predate the audited set
  network so cannot load on `main`. Void for **regime and code**, not as
  evidence about the architecture. See
  [the report](reports/2026-09-14-one-episode-per-update.md).

### Settled — do not re-run

- **The agent does not use terrain for cover; it manages range.** Deleting all
  terrain left exposure unchanged; doubling range collapsed the win rate.
  `observe_threat_count` was null and removed.
  [Terrain](reports/2026-08-05-stochastic-terrain-and-cover.md) ·
  [cover](reports/2026-08-06-cover-signal-reason-geometry.md) (read its corrections).
- **Terrain: count dominates size** — tune a profile with `just measure-terrain`.
- **The dice contribute more outcome spread than the scenario** (sd 50.6 within
  a layout, 45.0 between). `just measure-noise-floor`.
- **`eval/firepower_ratio` measures the firefight, not policy quality**; read it
  beside `vp_margin`. It replaced `firepower_advantage` on 2026-08-06.
- **~37% of objectives get zero models across five weightings and two
  scenarios.** Stop tuning weights at abandonment.
- **PPO cannot improve a behaviour-cloned policy from a cold critic**, at any
  `ent_coef`; the gamma explanation was refuted.
- **Four coherency levers are nulls**: `observe_unit_centroid` (−62.1 refereed),
  unit-level action spaces (0.444), smaller units (a casualty confound), and
  rescaling the nearest-squadmate observation (+3.5 ± 5.3, sign flips). The
  remaining coherency gap is not perceptual.
- **Four movement-side fixes for freezing are measured away**: the tangential
  slide, bisection on travel, a descending scan — "fix freezing" reduces to "fix
  allocation". Do not attempt a fourth.
- **Six attempts to shape offence or the travel term are empty or worse**:
  `contest_deficit`, removing the overstack penalty, the potential-invariance
  defect, the fallback mechanism, `one_objective_per_group`, `surplus_value`.
  Stop nominating `closest_objective_v2`.
- **"Make the squad agree" is a null**: consensus decoding on frozen weights,
  −4.8 / −4.1 / −9.1, 3/3.
- **Iterating the reallocation decode is negative** (−1.28 ± 0.76); `decode_stay`
  is a null; `min_stack` 4 → 2 is +0.5 on `refereed`.
  [Report](reports/2026-09-06-three-decode-knobs-and-none-of-them-pays.md).
- **Do not add scripted policies whose purpose is to advance** — the two on file
  cost their users −78 and −11.9 vp.
- **Five rounds is not closed as a training scenario** but nothing has trained
  there; `held` is horizon-invariant, so the allocation failure is not the clock.

### Findings

One row per finding: the verdict, the rule it left, the report. Numbers are the
ones a reader needs to act; everything else is in the report. Provenance for
every agent row unless stated: `25v25_maps_two_mode` lineage, three seeds, 300
epochs, `ent_coef` 0.003, refereed at K=3, held-out nine.

**Coherency** — the game's formation constraint, and where most effort went.

| date | verdict | rule | report |
|---|---|---|---|
| 08-16 | Enforcing coherency in training is a referee: every mode costs ~26 vp, and `repair` lost too (−57.6 v −34.8) | Enforce at PLAY, never in training; `objective_hold.require_coherent` lifts coherency 0.55 → 0.78 for free; `coherency.attrition: true` in every play config and no training config (+15 vp; alone in training it deletes the army) | [report](reports/2026-08-16-enforcement-is-a-referee.md) |
| 08-18 | The 2" chain binds, not the 9" spread: a 7.8% per-model tail becomes a 33% unit veto; the 0.89 plateau is per-model p = 0.977 | Train the per-model tail, read it beside the unit rate; freezing under `revert_unit` alone is absorbing (0.62) | [report](reports/2026-08-18-the-chain-tail-and-the-frozen-army.md) |
| 08-18 | Coherency rate does not predict the referee tax; the STAY rate does (agent 0.4% v scripts 38–57%) | Treat the deliberate-stay share as a first-class diagnostic | [report](reports/2026-08-18-the-lead-was-rule-breaking.md) |
| 08-20 | Joint constrained decoding is worth +40.5 vp for no weights, 45/45 tables; `verify_moves` another +11.4 (49.8% of models did not land where the relaxed model predicted) | `decode_topk=3` at play, default 1 so history stands; decoding in training is −51.8 and K^k caps K at 5; never quote a score without its decode | [report](reports/2026-08-20-decoding-does-not-belong-in-training.md) |
| 08-18 | `ent_coef` 0.003 beats 0.03: +5.9 ± 2.5 paired, coherency 0.771 v 0.674 | 0.003 is the setting; the entropy explanation for the stay rate is refuted (STAY got 30× rarer) | [report](reports/2026-08-18-the-chain-tail-and-the-frozen-army.md) |
| 09-04 | PPO trains undecoded and is scored decoded, and spends the decode's headroom: +19.7 in its own regime, −14.9 scored; headroom +74.9 → +40.2 | A play-time decode makes the matching training-time skill worthless; the lever is the joint coherent decode and it cannot be moved into training | [report](reports/2026-09-04-ppo-spends-the-decodes-headroom.md) |
| 09-04 | The KL anchor (`--kl-ref-target 0.03`) is the best melee policy on file: wins 2 of 4 ladder cells, ties `vs_shoot`, +29.6 ± 1.7 over unanchored self-play; the melee goal is NOT met | Self-play alone is the control, not the treatment (−27.2, worse than a fixed opponent); `require_coherent: false` in training is rejected — a decode substitutes for a skill's execution, not for the pressure that makes it representable | [report](reports/2026-09-04-the-anchor-holds.md) |
| 09-06 | Distillation at house fidelity (1200 games × 60 epochs) carries a decoded policy: action-match 0.973, not worse than its teacher on any cell | The policy-improvement loop (decode, distil, repeat) is viable; whether it compounds is untested | [report](reports/2026-09-06-the-clone-carries-the-decoded-policy.md) |

**Allocation and offence** — the agent stacks, hoards and cannot attack.

| date | verdict | rule | report |
|---|---|---|---|
| 08-16 | The VP cap makes it a denial game: all margin is denial and reward levers are null there | Cloning a better script beat the bar by +1.8 ± 0.9 | [report](reports/2026-08-16-the-cap-makes-it-a-denial-game.md) |
| 08-22 | Five squads against five or six points pose no allocation question (`take` v `deny` changes sign across seed sets); eight squads of three do (+16.0, 3/3); mixed weapon profiles are a null once shots are matched; trained there, offence did not move | The agent HOARDS: 52.9% alive v the scripts' 27–31% while holding less; the VP cap taxes the scripts (10.1% of VP) not the agent (1.1%) | [report](reports/2026-08-22-spare-squads-pose-the-question-the-agent-still-cannot-answer.md) |
| 08-22 | `overstack_penalty_per_extra: 0.0` REJECTED, −12.2 ± 5.5 paired, 3/3 | Discouraging stacking was making models spread to deny | [report](reports/2026-08-22-the-overstack-penalty-was-paying-for-itself.md) |
| 08-22 | Objectives are ruins, so standing on one is cover: excess death hazard negative in 5 of 5 policies. The agent spends 54.4% of model-steps on objectives v 75.5% and stacks 4.90 on its top point v 2.73; 53.7% of its income is global | "Hiding is correct play" is refuted; the error is allocation, not risk; do not reach for anti-stacking shaping | [report](reports/2026-08-22-holding-pays-and-the-agent-stacks.md) |
| 08-22 | `contest_deficit` (widening the travel term's candidate gate) REJECTED, −2.7 ± 4.8; offence went −61 → −72 — the third consecutive term to leave offence flat or worse | Stop shaping offence; the diagnosis is a difference-reward problem: `vp_gain` is net but global, so no model can prefer "take theirs" to "stand on ours" | [report](reports/2026-08-22-the-agent-is-never-paid-to-attack.md) |
| 08-23 | The critic already knows: it prices spreading a surplus squad +2.63 (realised +3.85) and stacking −7.18 (realised −11.52), 6 of 6 cells | The survival-premium diagnosis is refuted; the failure is SEARCH — spend on directed exploration and representation, not reward attribution; `corr(dV, dVP)` ≈ 0; the exact-assignment script LOST to greedy by 33.7 | [report](reports/2026-08-23-the-critic-already-knows.md) |
| 08-23 | The agent allocates worse than chance: as many squads on objectives as the script (4.0–4.5 v 3.97) crammed onto 2.1–2.3 points v 3.28, while travelling 40% MORE | The squad-heading statistic measured the factored architecture (83% of the gap), not skill | [report](reports/2026-08-23-a-squad-cannot-agree-where-to-go.md) |
| 08-23 | The travel term is inert: 43.5% of paid steps are already inside their target; it does not pull to the centre; squadmates paid apart on 8.0% of squad-steps | Fourth empty result on `closest_objective_v2`; no gate explains the allocation gap | [report](reports/2026-08-23-the-travel-reward-audit.md) |
| 08-24 | Five rounds: `held` is horizon-invariant (−0.81 and −1.15 short, 0 of 9), and the agent is WORSE at five rounds where it wins at twenty; the board is static after round 8 and the agent's allocation is fixed by round 2 | Its edge is denial, which accrues per scoring event; the comparator switched identity between cells and every magnitude first published was an artefact | [report](reports/2026-08-24-five-rounds-does-not-rescue-the-agent.md) |
| 08-22 | Freezing is friendly gridlock: 91.8% of frozen model-steps touch a friendly base; purposeful policies are absorbing (+0.86), `random` is not (+0.09) | Read `absorbing` beside any movement feature; trained agents freeze 26% because they stack; the solver is not the bug | [report](reports/2026-08-22-freezing-is-friendly-gridlock.md) |
| 08-22 | Asymmetric 30v15 horde: agent +48.5 over the best script (t=3.65, 8–9 of 9), all of it defence; `held` inverts because control is a headcount; the referee reorders the bar | ⚠ VOID as a matchup claim: the elite side's unit encoding was aliased for every number (fixed 08-31); re-measure before quoting | [report](reports/2026-08-22-the-horde-is-the-agents-best-matchup.md) |

**The advance move** — a core rule, re-encoded, and a lever the agent uses badly.

| date | verdict | rule | report |
|---|---|---|---|
| 08-22 | Advance as extra speed bins REJECTED at 300 epochs: −26.7 ± 8.3 unpaired; forbidding it at play on the same weights recovers +8.5 | Adding actions cannot be paired; bridge the two configs with a non-advancing script first; not caused by freezing (the control freezes as much) | [brief](docs/advance-move-problem.md) |
| 08-23 | The scripted "run while far" heuristic cost its user ~78 vp; the published +15.5 was two self-inflicted wounds cancelling; the endpoint rule removes 100% of ending-engaged | NEVER measure a symmetric change with both sides changed; run the 2×2. `advance_when_out_of_reach` defaults False | [report](reports/2026-08-23-the-bar-was-playing-a-different-game.md) |
| 08-23 | Advance is a short-game move: worth ≤ 0 at twenty rounds (three scripted rules, three rejections, converging on plain walking from below) and +0.14 sd at five; all four nominated encoding defects fail to bind | A move type is a lever, not an advantage — do not add scripts whose purpose is to advance; ask whether carrying it costs the agent, via a `dark_action_slices` control | [report](reports/2026-08-23-three-prices-for-the-advance-move.md) |
| 08-23 | Shipped: absolute rungs (8/10/12" at three bins) gated by the unit's roll, the move type declared by the unit's LEADER in the command phase (a 2-action slice, STAY = `normal`), 0.0% dominated advances | `n_advance_speed_bins` defaults 0 so no golden moves; the roll is at the start of the side's turn; a leader-bind inside the movement slice would shatter formation | — |
| 08-24 | Carrying the re-encoded lever costs −2.9 ± 0.7; USING it costs −13.4; the 300-epoch read (+2.2, signs flipping) reversed at 1000 (−16.3 ± 8.9, 3/3 negative); two seeds drifted into using it more with more training. REJECT, unresolved | "The encoding costs ~12 vp" is refuted — do not re-open the encoding; lever usage is a convergence signal; resolving it needs six seeds at 1000, not three | [report](reports/2026-08-24-carrying-the-lever-is-free-using-it-is-not.md) |

**Scenarios and configs**

| date | verdict | rule | report |
|---|---|---|---|
| 08-08 | `25v25_shooting_opponent` beat the shooting bar (+30.8 / +27.4 v +17.0) via `objective_hold.crowding_exponent`; effectively a two-objective mission | The exponent has only been measured there; `25v25_cover_control` is its control | [report](reports/2026-08-08-paying-the-pot-beats-the-bar.md) |
| 08-19 | Seat parity: the shooting config fails (−24.6 ± 9.4), `two_mode` passes (+6.5 ± 6.1 at n=120) | Run the gate at n ≥ 100 before any self-play run | [report](reports/2026-08-19-the-two-seats-are-not-the-same-game.md) |
| — | `25v25_single_phase` and `25v25_curriculum` share a scenario and final phase; comparing them isolates the curriculum | Every phase keeps `vp_gain` and one per-model calculator (`tests/test_curriculum_configs.py`) | — |
| 08-25 | Melee ships behind `melee.enabled`; the charging script is worth +62.5 ± 14.7 and its value is entirely the shooting shield | A vp gate on a lethality-neutral mechanic is unpowered by construction (MDE 26 vp at n=3); read mechanism counts | [prereg](reports/2026-08-25-melee-preregistration.md) · [docs/melee.md](docs/melee.md) |

## Tracking work

**GitHub Issues is the tracker.** ⚠ From 2026-09-06 there is no planning
directory: `.planning/` was a GSD artifact for `/gsd-*` slash commands that no
longer exist, and it was removed. `git show <sha>^:.planning/` recovers any of it.

Four kinds, one per issue, and the label name is the enforcement:

| `kind:` | is | closes when |
|---|---|---|
| `question` | a question the record will answer; owns arms | its falsifier is answered — **including with a null** |
| `arm` | ONE change, ONE comparator, ONE provenance tuple | its finding lands in all four places |
| `bug` | behaviour diverges from `docs/rules/` or its own spec | the fix ships with a test pinning it |
| `build` | feature or refactor; ships behaviour, no verdict | it ships |

⚠ **There is deliberately no `experiment` label.** An experiment is a question
with N arms. You cannot file a bundle, because a bundled change cannot be
attributed.

- **`kind:arm` opens carrying `needs:prereg`.** Remove it only once the
  pre-registration path resolves on `main` — the commit timestamp is what proves
  the criterion predates the numbers. `gh issue list --label needs:prereg` is the
  list of arms that are unsafe to launch.
- **Every issue opens with a TL;DR: two or three sentences, no jargon**, for
  someone with no context on this repo — what it is about and why it matters.
  The body below it can be as technical as it needs to be; the TL;DR cannot. An
  issue nobody can grasp in fifteen seconds is a note to yourself, not a tracker
  entry. The test: if it contains `vp`, `K=3`, `paired`, `refereed`, `decode` or
  a config filename, it does not pass.
- ⚠ **`gh issue create` BYPASSES the issue forms** — `.github/ISSUE_TEMPLATE/`
  only runs in the web UI, so a CLI-created issue gets no TL;DR, none of the
  required fields and no board placement. That is how #277, #278 and #281 were
  opened without any of them. **Use the `issue` skill**
  (`.claude/skills/issue/SKILL.md`), which is the CLI's version of the contract.
  `.github/workflows/issue-tldr.yaml` is the backstop: it labels `needs:tldr` and
  comments once, and the label clears itself when the body is fixed.
- **An issue is for work intended and not done.** Shipped behaviour goes in
  `docs/`; what was measured goes in `reports/`; the roadmap is
  `docs/goals-and-roadmap.md`. When work lands, close the issue and write the doc.
- **Unimplemented design proposals go in issues, not `docs/`** — `docs/**/*.md` is
  the drift check's live set and describes behaviour the env actually has, so a
  proposal parked there gets "corrected" to match code that does not implement it.
  If a live doc must point at one, link the full issue URL.
- **Link the PR:** `just ship <branch> "<msg>" <issue>` appends a `Closes #N`
  trailer to the commit body. It goes in the commit, not only the PR — most merges
  here are merge commits, so the trailer is what reaches `main`.
- **Retraction mirrors the report rule.** Never edit an issue body; comment. Edit
  the **labels**, which are the issue's index row. ⚠ **Never reopen** — open a new
  issue saying `Retracts #N`. Reopening destroys the close date, the timestamp
  that orders the finding before its retraction. The tripwire is the index row: if
  a claim reached `reports/README.md` it gets the full ceremony; if not, it is a
  draft, so correct it in place.
- **The board is [Wargame RL](https://github.com/users/sashman/projects/2)**
  (`gh project item-list 2 --owner sashman`), linked to the repo. Columns:
  Backlog · Blocked · In progress · **Measured, not landed** · Done.
  ⚠ **That fourth column is the one this repo actually needs** — a generic board
  files a finished-but-unwritten experiment under Done and the debt disappears.
  It is the board view of `needs:writeup`, as `Blocked` is of `hold`.
  ⚠ Projects needs the `project` token scope: `gh auth refresh -h github.com -s project`.
- Labels and required fields live in `.github/ISSUE_TEMPLATE/` and `gh label list`,
  and are **not restated here**, so they cannot drift.

## Git Workflow

- Always verify the current branch before committing (especially after a PR merge)
- Create feature branches for all changes; avoid committing directly to `main`
- Branch naming: `feature/<topic>`, `fix/<topic>`, `refactor/<topic>`
- Commit messages: imperative mood, concise summary (e.g. "Add reward shaping for distance")
- If pre-commit hooks reject a commit, fix the issues and make a new commit — no `--amend`, no `--no-verify`
- After pushing a new feature branch, always create a PR using `gh pr create`
- Run `just validate` (format + lint + test) before pushing; `just format && just lint` for quick iteration
- **Shipping:** always create a new branch from up-to-date `main` — never reuse an existing feature branch for a new PR. Checkout `main`, pull latest, then branch. Never push directly on an in-progress branch from another workflow. The `/ship` skill (`.claude/skills/ship/`) automates this via `just ship`
- **Docs-drift check:** a `PostToolUse` hook (`.claude/settings.json` → `.claude/hooks/docs_check.py`) fires after `gh pr create` and `just ship`. It diffs the branch against `main` and names the live docs that cite the changed paths, symbols, recipes or config fields. Fix mechanical drift (renamed symbol, changed default, missing table row) directly; only *suggest* anything asserting behaviour. It is silent when nothing is implicated, and never fails a ship. `reports/`, `ratings/` and `configs/` are exempt — they record what was measured or believed at the time, under a named code revision. Run it by hand with `python3 .claude/hooks/docs_check.py --dry-run [<base>..<head>]`

## CUDA Environment

- Do NOT preemptively disable CUDA — only set `CUDA_VISIBLE_DEVICES=""` when training actually fails with CUDA errors
- By default, let PyTorch use the GPU
