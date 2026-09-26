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
