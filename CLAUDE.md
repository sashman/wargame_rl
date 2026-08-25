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
│       │   └── opponent/          # A checkpoint seated on the opponent side
│       ├── rating/                # Elo: margin score, Bradley-Terry fit, schedule,
│       │                          #   arena, ledger, table
│       ├── selectors.py           # One policy-name-or-checkpoint-path resolver
│       └── types.py               # Experience
├── configs/                       # Env configs, tiered by what breaks if edited
│   ├── golden/                    #   backs a published number
│   ├── experiments/               #   arms; deleted once answered
│   ├── evaluation/maps/           #   the real table layouts
│   └── dev/                       #   fixtures and demos
├── tests/                         # Pytest suite with conftest.py fixtures
├── docs/                          # Design docs (movement, reward phases, missions-and-vp,
│                                  #   roadmap, metrics, shooting, expected-damage,
│                                  #   terrain, training-throughput, play-doctrine, elo)
│   └── rules/                     # Rules specification + constants.yaml + gap map
├── reports/                       # Experiment findings, kept for retrospection
├── ratings/                       # Rating ledgers, one per scenario fingerprint
├── scripts/                       # Run-inspection tooling (fetch_map_layouts,
│                                  #   run_summary, measure_phase_gates,
│                                  #   measure_baselines, measure_checkpoint, measure_terrain,
│                                  #   measure_noise_floor, measure_objective_split,
│                                  #   measure_income_share, measure_maps,
│                                  #   behaviour_clone, measure_seat_parity,
│                                  #   measure_elo, elo_table,
│                                  #   measure_throughput)
├── train.py                       # Training entry point (Typer CLI)
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
| Ship (branch → commit → push → PR) | `just ship <branch> "<message>"` |
| Simulate latest | `just simulate-latest` |
| Simulate / record a checkpoint | `just simulate <ckpt> <config.yaml> [overlays]` · `just record-sim <ckpt> <config.yaml>` |
| Regenerate the eval tables from the layout API | `just fetch-maps [owner] [maps_dir]` |
| Record the README's GIFs (exact colours, median of N) | `just record-gifs <policy\|ckpt> <config> [tables]` |
| Test env (random) | `just test-env` |
| Watch a scripted policy play (no checkpoint) | `just play [config.yaml] [policy] [theme] [overlays]` |
| Step a match by hand and rewind it | `just debug [config.yaml] [policy\|ckpt] [theme] [overlays]` |
| Recreate a recorded match exactly and step it | `just debug-recording <file> [policy\|ckpt] [theme] [overlays]` |
| Record a match event log | `just record <config.yaml>` |
| Replay / narrate a log | `just replay <file>` · `just replay-summary <file>` |
| Replay a log visually (window or MP4) | `just replay-render <file> [out.mp4] [theme] [overlays]` — tabletop by default |
| Analyse a log | `just analyze <file>` · `just analyze-compare <files...>` |
| Inspect a Wandb run | `just run-summary <run_id> [bucket]` |
| Measure reward-phase gates | `just measure-phase-gates <ckpt> <config.yaml> [n_episodes]` |
| Scripted baselines (floor + bar) | `just measure-baselines <config.yaml> [n_episodes] [record] [seed_base] [key=value...]` |
| Score a checkpoint (baseline-comparable) | `just measure-checkpoint <ckpt> <config.yaml> [n_episodes] [record] [decode_topk] [key=value...]` |
| Score on the real table layouts | `just measure-maps <policy\|ckpt> <config.yaml> [n_episodes] [maps_dir] [decode_topk] [key=value...]` |
| Why an objective was not held | `just measure-objective-split <policy\|ckpt> <config.yaml> [n_episodes]` |
| What a policy buys with the advance move, and what it pays | `just measure-advance-use <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| How often the VP cap binds, and what it discards | `just measure-vp-cap <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| What holding a point earns against what it costs | `just measure-hold-hazard <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| How often a policy is in unit coherency | `just measure-coherency <policy\|ckpt> <config.yaml> [n_episodes]` |
| Which calculator pays, and how much is global | `just measure-income-share <policy\|ckpt> <config.yaml> [n_episodes]` |
| Clone a scripted policy into the network (warm-start checkpoint) | `just behaviour-clone <policy> <config.yaml> [n_episodes] [epochs] [out]` |
| Two policies on identical layouts, paired per episode | `just measure-paired <policy\|ckpt> <policy\|ckpt> <config.yaml> [n_episodes] [seed_base] [key=value...]` |
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

- `WargameEnv` — Gymnasium env with configurable board, models, objectives
- **Polar movement** — actions encoded as (angle × speed) per model
- **Reward phases** — curriculum learning with phased reward configs
- **VP reward and success** — `vp_gain` calculator, `player_vp_min` success criteria, optional terminal VP bonus; observation includes `player_vp_delta` for step-wise VP signal
- **Deployment zones** — configurable spawn areas for player and opponent
- **Group cohesion** — optional penalty for unit separation
- **Melee, off by default** — `melee.enabled` steps the charge phase, resolves fights,
  lets a move end in contact, costs an engaged unit that withdraws its shooting, and stops
  engaged models being shot at. Off it registers no slice, draws no dice and leaves
  `skip_phases` alone, so every golden config and every golden fixture is bit-identical.
  ⚠ Engagement was 0.0000% of model-pairs not because contact is unreachable but because
  `back_off_to_unengaged` parks the CLOSEST pair **8.7 micro-inches** outside it — a charge
  needs the *exemption* first. ⚠ **RETRACTED: it needs the distance too.** That minimum was
  read as a typical value; the median charge-eligible unit is **5.99"** from its nearest
  enemy and **0.0%** of declarations are within one speed bin. ⚠ **Nothing has been measured
  with melee on, and a training arm launched today would measure the BAR, not the agent** —
  no scripted baseline or opponent can charge. An expert panel measured a charging script at
  **+62.5 ± 14.7** whose value is **entirely the shooting shield** (−4.0 ± 17.4 with the
  target gate ablated). ⚠ **The three defects an audit found in it are now all
  closed** — the charge roll is an observation column and not just a logit mask; the joint
  decoder runs in the charge phase against the charge's OWN referee, so a melee score at
  K=3 is one; and the shooter-side engagement gate reduces over the shooter's UNIT, closing
  the "send one model to lock them and keep four firing" exploit. All three are
  **unpriced** — engagement stays 0.0000% without a charging policy, so the seeded digest
  is 9 of 9 identical to `main`. [docs/melee.md](docs/melee.md)
- **DDD layering** — `domain/` owns the rules (Battle aggregate, clock, placement, termination, LOS, shooting); `wargame.py` is a facade; reward/renders depend only on the `BattleView` protocol. See [docs/ddd-envs.md](docs/ddd-envs.md)
- **Rules specification** — [docs/rules/](docs/rules/README.md) is the game's rules authority: a self-contained spec written for this project, with `constants.yaml` (every number, in inches) and [implementation-status.md](docs/rules/implementation-status.md) (per-rule: implemented / partial / divergent / absent). Before implementing a mechanic, read its chapter and its gap-map row. `tests/test_no_ip_references.py` keeps the repo free of references to the commercial product the rules derive from — the spec names no product, publisher, edition or faction, and neither should anything else
- **Play doctrine** — [docs/play-doctrine.md](docs/play-doctrine.md) is how this game is *won*, as `docs/rules/` is how it is *played*: 43 numbered entries, each stating a claim, whether the environment can express it, which extension point it lands in, and what has already been measured about it. It is a store of **hypotheses, never of evidence** — price an entry as a scripted policy (`just measure-paired`, no GPU) before it becomes a reward term or a training run, and where an entry disagrees with the record below, **the record wins**

### Game State I/O (`envs/state/`)

Snapshot/event pipeline for recording and inspecting matches — `GameStateSnapshot`, event-log deltas, `StateExporter` (wired into `step()`), replay, narration, and `analyze_match` metrics. Driven by `replay_events.py` / `analyze_events.py` and the `record` · `replay` · `analyze` · `analyze-compare` recipes. See [docs/game-state-io.md](docs/game-state-io.md)

### Ratings (`rating/`)

Puts scripted baselines and learned checkpoints on **one scale**, so "did this get better" has an answer that does not depend on which opponent it happened to face. Bradley-Terry maximum likelihood with the deployment-zone and first-turn advantages as explicit fitted terms, a bootstrap over layouts, and an append-only ledger in `ratings/` keyed by a scenario fingerprint that **refuses** to mix scenarios. `score.py` and `elo.py` import numpy and nothing from this repo; `arena.py` is the only module that touches a live env, and it wraps `evaluate_selector` rather than reimplementing it. Recipes: `measure-seat-parity` · `measure-elo` · `elo-table`. See [docs/elo.md](docs/elo.md)

⚠ **A rating assumes the two seats are the same game, and nothing enforces that.** On `configs/golden/25v25_shooting_opponent.yaml` they are not — one policy played from both seats loses from the *player* seat by **−24.6 ± 9.4 vp**, and every number in this file is quoted from that seat. `just measure-seat-parity` is the gate and it is **advisory**: entrant A always takes the player seat and `pairings` lists each pair once in input order, so on a config that fails the gate, ratings are confounded by command-line position. No rating is published for this reason. See [the report](reports/2026-08-19-the-two-seats-are-not-the-same-game.md)

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

PPO on a `TransformerNetwork` is the only thing that trains — there is no
algorithm or network to choose, and `just train <config> 800` means 800 *epochs*.

### The board

`configs/evaluation/maps/` is **generated by `just fetch-maps`** from the public
layout API — the same 45 layouts the hand-traced tables were, matched 45/45 by
piece bounds, so the numbering and the held-out nine are unchanged. Each table
carries 16 pieces as 8-vertex silhouettes, 5 or 6 objectives, and its own
deployment zones. Full derivation, and every superseded measurement, in
[the report](reports/2026-08-20-the-tables-are-generated-now.md).

Five things about these tables are live decisions, not history:

- ⚠ **The API is NOT the source for objectives — only for terrain and zones.**
  Its objective markers disagree with the published layout cards on **6 of 45
  tables by 12–18 inches**. Objectives come from `scripts/objective_markers.json`,
  carried over from the hand tracing and right on **45 of 45, worst error 1.5"**.
  An objective is a **RUIN** — pieces sharing ≥1.0" of boundary — and a tie
  designates both, which is why a table carries 5 or 6. Pinned in
  `tests/test_map_objective_counts.py`.
- ⚠ **Do not "tidy" `objective_budget` down to 5.** It stays 6 and
  `terrain_budget` 16; changing either changes the tensor width and orphans every
  checkpoint in `checkpoints/`.
- ⚠ **`long_edges` puts the armies 20" apart across the SHORT axis** against
  24–40 elsewhere. At a 12" weapon range that is a different game from turn one,
  and it is 6 of 45 tables.
- **Zones are polygons, not the `deployment_zone` rectangle** — only two of the
  six real deployments are axis-aligned bands (`long_edges` and `short_edges`,
  11 of the 45 tables); the other 34 are triangles, staircases and arcs. Both
  the placement wiring and the *rendering* wiring are needed; see
  `envs/CLAUDE.md`.
- **The pool has a ~6 vp resolution floor.** Per-table `vp_margin` sd is
  18.5–20.6, so even n=45 gives SE 2.75–3.07. More episodes per table cannot
  help: the variance is *across tables* and only 45 tables exist.

**The bar on the generated tables, all 45, n=30, seeds 700000+:** `random`
**−222.5** · `squad_march_take` **+5.9** · `squad_march_shoot` **−5.9** ·
`squad_march_deny` **+5.4**. Note `shoot` is the *weakest* of the three scripts
here while convention calls it "the bar" — **name the policy, never say "the
bar"**.

**The opponent is worth ~120 vp, measured with one policy on both sides.**
`squad_march_take` scores **+126.2** against `scripted_advance_and_shoot`
(`25v25_maps_coherency`) and **+5.9** against `squad_march_take` itself
(`25v25_maps_two_mode`) — same 45 tables, same n, both unrefereed. That gap is
why every pre-2026-08-16 figure is incomparable, and it is the honest form of the
comparison: one policy, two opponents, rather than swapping policies between
columns.

**On the other four golden configs** (they generate their own terrain and are
unaffected by the table change), re-measured 2026-08-19 post corpse-fix, n=100 at
seeds 700000+, `squad_march_shoot`: `25v25_shooting_opponent` **+13.3**,
`25v25_cover_control` **+15.9**, `25v25_single_phase` / `25v25_curriculum`
**+70.3** (identical, as they share a scenario — a useful consistency check).
Floors (`random`): −124.9, −133.5, −256.6 / −256.0.

**`25v25_maps_coherency` re-measured 2026-08-21 on the generated tables**, all
45, n=30, seeds 700000+ (it is *unrefereed* — `enforce_at_deployment` only — so
these are not comparable to a refereed number): `squad_march_take` **+126.2**,
`squad_march_deny` +121.6, `squad_march_shoot` **+116.7** (was +105.7),
`contest_and_spread` +112.8, `random` **−59.6** (was −14.7). `take` is the
strongest here, not `shoot`. ⚠ **`random` lost 45 points**, by far the biggest
mover — the same mechanism as on `two_mode`: it used to score by deploying onto
home objectives and standing there, and the generated tables put the objectives
somewhere else.

⚠ **Nothing trains on `25v25_maps_coherency` any more, and that is deliberate.**
Comments stripped it is byte-identical to `25v25_maps_two_mode` but for
`config_name` and the opponent (`tests/test_map_config_pairs.py` pins this), so
it is not a second scenario — it is the same one against an opponent worth ~120
vp less. Its agent column comes from scoring the `two_mode` lineage on
`configs/evaluation/25v25_maps_vs_advance_and_shoot.yaml`, the refereed member
of the eval family, where the agent is **−75.9 behind the best script on 0 of 9
tables**. Training against the weaker opponent would produce a weaker agent at
the matchup it is already worst at; spend the GPU elsewhere.

### Where the agent stands

⚠ **REISSUED 2026-08-24 at `f741e14`. FOUR OF FIVE ROWS MOVED and the headline claim
is now carried by ONE row.** Re-measured independently twice (an audit panel and by
hand), agreeing row for row. Bisected: `squad_march_deny` on the take config reads
**−1.1 at the publishing commit** — the published value to the decimal — and +6.5 at HEAD,
in two steps: the endpoint rule **+5.0** and **`d607561`** (the wholly-within deployment
check, a fix **nobody named**) **+2.6**. The command-phase change contributes **0.0**.
⚠ **ALWAYS STAMP A REVISION ON A QUOTED TABLE**, and bisect a staleness claim — scripted
policies are deterministic and git is free, so it costs about a minute per point.

| opponent | agent | best script | gap | t | sign | was | moved |
|---|---|---|---|---|---|---|---|
| `squad_march_deny` | **+20.0** | −6.1 (`take`) | **+26.1** | 3.51 | 7/9 | +35.4 | −9.3 |
| `squad_march_take` | +19.4 | +6.5 (`deny`) | +13.0 | 1.44 | 7/9 | +26.1 | −13.1 |
| `squad_march_shoot` | +33.2 | +27.7 (`deny`) | **+5.5** | **0.58** | **3/9** | +16.2 | −10.7 |
| `contest_and_spread` | +16.7 | **+30.5** (`take`) | −13.8 | −1.61 | 4/9 | −9.5 | −4.3 |
| `advance_and_shoot` | +61.4 | **+135.6** (`take`) | **−74.3** | **−6.98** | **0/9** | −75.9 | +1.6 |

⚠ **THE AGENT NOW CLEARS THE BEST SCRIPT SIGNIFICANTLY ON ONE OF FIVE OPPONENTS, NOT
THREE.** The `shoot` row is a **null** (t=+0.58, 3 of 9). "A better defensive player than
any script" now rests on `squad_march_deny` alone. Coherency is unchanged and still wins
everywhere (agent 0.937–0.954 against 0.863–0.911).

The agent moved **−6.4 / −5.7 / −6.0 / −4.1** on the four `squad_march` opponents and
**exactly 0.0** on `advance_and_shoot` — a one-directional signature: the changed policy
is on the *opponent* side in the first four and is a different family in the fifth.

The offence/defence split below and the r=+0.991 correlation were fitted on the OLD rows
and have not been refitted. Treat both as provisional.

**Six seeds** of the documented recipe (`configs/golden/25v25_maps_two_mode.yaml`,
`ent_coef` 0.003, 300 epochs, `just train-coherency-baseline`), held-out nine,
n=30, verified top-3 decode, **refereed** eval configs, scripts re-measured per
opponent. Measured 2026-08-21:

| opponent | agent | best script | gap | t | sign |
|---|---|---|---|---|---|
| `squad_march_deny` | **+26.4** | −8.9 (`take`) | **+35.4** | 4.49 | **9/9** |
| `squad_march_take` | **+25.1** | −1.1 (`deny`) | **+26.1** | 3.32 | 8/9 |
| `squad_march_shoot` | **+39.2** | +23.0 (`take`) | +16.2 | 1.64 | 7/9 |
| `contest_and_spread` | +20.8 | **+30.2** (`take`) | **−9.5** | −1.18 | 4/9 |
| `advance_and_shoot` | +61.4 | **+137.2** (`deny`) | **−75.9** | **−7.12** | **0/9** |

Coherency **0.938–0.955** on every opponent against a scripted 0.867–0.908 —
formation holds even in the matchups it loses. Seeds 4–6 moved every pre-existing
row by under 2 vp, so the table is now a replication rather than a first read.

**ONE trait explains all five rows: the agent's defence is excellent and its
offence is capped, so its lead is whatever denial happens to be worth.** Split
the gap into what it scores minus what the script scores (offence) and what the
script concedes minus what it concedes (defence):

| opponent | script concedes | offence | defence | gap |
|---|---|---|---|---|
| `squad_march_deny` | 223.5 | −60.8 | **+96.1** | +35.3 |
| `squad_march_take` | 219.6 | −56.3 | **+82.3** | +26.1 |
| `squad_march_shoot` | 197.1 | −42.0 | +58.2 | +16.2 |
| `contest_and_spread` | 184.2 | −48.0 | +38.5 | −9.5 |
| `advance_and_shoot` | 128.0 | −71.3 | **−4.5** | −75.8 |

Offence is flat at −42 to −71 everywhere; defence runs +96 down to **zero**, and
the gap tracks *what the best script concedes* at **r = +0.991**. `held` is
1.9–2.1 against every opponent while the scripts reach 2.9–3.9 against the weak
ones. The agent plays the same game regardless and cannot tell which game it is
in.

- ⚠ **Absolute score measures the OPPONENT, not the agent.** The agent scores
  *higher* against weaker opponents while being further behind the scripts. Only
  the same-row comparison means anything, and swapping the opponent voids every
  baseline on that config.
- ⚠ **The weaker the opponent, the worse the agent does — relative to a script.**
  Against `advance_and_shoot` both sides concede ~130, so the defensive edge is
  worth nothing and only the offensive deficit is left: **−75.9 on 0 of 9
  tables**, the largest and most significant deficit ever measured here. This is
  not a different failure from `contest_and_spread`'s −9.5; it is the same one
  with denial priced at zero.
- ⚠ **`contest_and_spread` is unchanged at six seeds** (−8.4 → −9.5) and still
  not statistically settled (t=−1.18, 4/9). What six seeds *did* settle is that
  it is not a one-seed artefact — the per-seed band is +8.1 to +30.9, all six
  behind the script's +30.2. The claim is **"a better defensive player than any
  script"**, not "a better player".
- ⚠ **ALWAYS state which config a quoted agent number was trained on.** Missing
  provenance is why two lineages sat side by side in the docs undetected.

### Allocation: the scenario was not asking, and fixing it did not fix the agent

Measured 2026-08-22, [report](reports/2026-08-22-spare-squads-pose-the-question-the-agent-still-cannot-answer.md).

- **Five squads against five or six objectives pose NO allocation question.**
  `squad_march_take` and `squad_march_deny` differ only in what a spare squad
  does; paired at n=100 their difference is **+5.7 / −9.2 / +9.8** across three
  layout sets — **it changes sign**, mean +2.1. ⚠ A single seed set reading +5.7
  says the opposite. 25–31 episodes in 100 are *identical*.
- **Eight squads of three do pose it: +16.0, positive on 3/3 sets**, and only
  **2–5** episodes in 100 identical. `configs/experiments/24v24_maps_spare_squads.yaml`
  is the golden config with only the squad structure changed.
- ⚠ **Mixed weapon profiles are a measured null — and the first two arms measured
  their own lethality instead.** `25v25_maps_mixed_roles` fires **45 shots a round
  against the control's 25**; `alive` collapses 0.432 → 0.203 → 0.135 at 40
  models, and an army of five survivors cannot spread over six objectives. Held
  at **exactly 25 shots** (`..._matched.yaml`), roles reproduce the control's
  paired difference **to one decimal**. It was the squad count, never the guns.
- **Trained on the config that does ask, offence did not move.** Three seeds, 300
  epochs, `ent_coef` 0.003, scored refereed at K=3, all at `last.ckpt` (epoch
  299 — the highest `ppo-NNN` is **145** for s1 against **292** for s3):
  agent **+15.1 ± 5.6** against `squad_march_take` **+6.0 ± 3.0**, gap **+9.1,
  t≈1.44, UNPAIRED** (`max_groups` 5→8 is a shape change). Offence **−50.5**,
  defence **+59.6** — still entirely denial. `held` 2.17 v 2.80, a 0.63 shortfall
  against 0.58 before. Coherency 0.964–0.967 against the scripts' 0.941–0.945.
- ⚠ **THE AGENT HOARDS.** It finishes with **52.9% of its army alive against the
  scripts' 27.4–30.9% while holding fewer objectives.** Nearly twice the
  survivors, less ground. That, not the scenario and not the profiles, is where
  the offence deficit lives.
- **The VP cap taxes the SCRIPTS, not the agent.** `min(15, controlled × 5)`
  means the *fourth* objective pays zero while the tables carry five or six.
  `just measure-vp-cap` on `25v25_maps_two_mode`: `squad_march_take` is above the
  cap on **23.9%** of steps and loses **10.1%** of its VP; the agent loses
  **1.1%** and reaches three objectives on only **22.3%** of steps against the
  script's 55.6%. So the agent's shortfall is **fully payable**, and the cap
  compresses exactly the `take`-vs-`deny` difference used to detect allocation.
  `held` cannot see any of this — it is an end-state snapshot with no notion of
  which points were paid.

### The overstack penalty was paying for itself

Measured 2026-08-22, paired, [report](reports/2026-08-22-the-overstack-penalty-was-paying-for-itself.md).

- **`overstack_penalty_per_extra: 0.0` is REJECTED: −12.2 ± 5.5 paired, t=−2.23,
  3/3 seeds negative**, `held` 2.19 → 2.05. Three seeds, 300 epochs, scored
  refereed at K=3 on `24v24_maps_spare_squads_refereed.yaml`.
- The whole of `closest_objective_v2`'s **negative** net income is this penalty
  (progress +0.08, penalty −0.90). Removing it flips the term to +0.29 and every
  other calculator is bit-identical — the mechanism was exactly as diagnosed.
- **And it still lost.** Offence **+2.9**, defence **−15.1**: the travel term did
  pay more for movement, and the agent conceded fifteen VP for it. Discouraging
  stacking was making models spread out to *deny*.
- ⚠ **A term with negative net income is not thereby a broken term.** What a term
  costs shows up in `measure-income-share`; **what it prevents does not**. The
  rule "an anti-concentration lever must redistribute, not destroy" came from
  levers that halved occupancy — it does not license removing a small one that
  is not doing that. This one is 1/5 the magnitude of the lever that failed and
  sits *alongside* `crowding_exponent`, not instead of it.

### ⚠ The observed control count was not the scored one (fixed 2026-08-22)

- There were **three** implementations of "on an objective". Scoring, `objective_hold`
  and every control read use `norms_offset <= obj_radii`, measured from the model's
  **base edge**; `observation_builder` had its own `area.contains_points` test on the
  model **centre**. Measured on the held-out nine: **206 of 2,700 (objective, step)
  slots disagreed — 7.6%**, 215 models miscounted.
- **`player_count` on the objective token is the feature every objective-keyed reward
  term and every proposed mission primitive reads**, so the standing rule *"check the
  agent can observe what the lever keys on"* was quietly false for all of them.
- Now one definition, `objective_counts_from_norms_offset`, shared by all three.
  Pinned by `tests/test_observed_control_matches_scoring.py`, verified to fail on the
  old builder.
- ⚠ **This changed the observation**, so `observation_golden_25v25_shooting_opponent.npz`
  was regenerated deliberately (the other two are byte-identical — the change bites only
  where `observe_objective_control: true`). Checkpoints trained before this saw the old
  feature; scores across this date are not strictly comparable.
- ⚠ **It cost +0.84 ms/step (+16.1%)** on `25v25_maps_two_mode` — observation build
  0.659 → 1.257 ms — because the counts are now computed with the alive mask the
  observation path never had. The caches already passed to `_get_obs` are built
  **without** an alive mask, so reusing them would count the dead. The planned
  throughput step (one shared opponent cache, batched `_distances_to_objectives`)
  recovers this and more; see [docs/missions-design.md](docs/missions-design.md).

### Holding pays — the agent stacks, it does not hide

Measured 2026-08-22 with no GPU, [report](reports/2026-08-22-holding-pays-and-the-agent-stacks.md).

- ⚠ **OBJECTIVES ARE RUINS, so standing on one is COVER.** All 270 markers in
  `configs/evaluation/maps/` sit inside a terrain piece. `just
  measure-hold-hazard` prices the trade per model-step: standing on an objective
  pays **+0.37 to +0.44** more and its excess death hazard is **negative in 5 of
  5 policies** (−0.13% to −1.43%) against a break-even of +3.4% to +6.0%. The
  exposed models are the ones walking between points. **"Hiding is correct play"
  is refuted** — the agent is leaving return on the table.
- **The error is ALLOCATION, not risk.** The agent spends **54.4% of model-steps
  on objectives against the scripts' 75.5%**, and stacks **4.90 models on its top
  point where `squad_march_take` puts 2.73** — 8.6 of 12.5 survivors on
  objectives, 55.3% of points empty, **redistribution ceiling +2.20**, the
  largest recorded here.
- **It earns exactly half the script's `objective_hold`** (6.76 v 13.48) from a
  pot it splits over half as many points, and **53.7% of its income is global**
  against the script's 25.8% — `vp_gain` and `objective_coverage` are broadcast
  whole to every alive model, so more than half of what it earns asks nothing of
  any individual model.
- ⚠ **Do not reach for anti-stacking shaping.** `crowding_exponent: 1.0` is the
  measured-good lever, it is already on, and the agent ignores it. Measure
  **squad dispersion** first — squads of three under a 2" chain make the squad the
  allocation quantum — and check `closest_objective_v2`'s `fallback_to_nearest`,
  which *pays* an unassigned group to close on the nearest point, usually one
  already held.
- The observability desk check **passes** here: `observe_objective_control: true`
  and `_objectives_to_obs` supply per-objective alive counts for both sides.

### Asymmetric armies — the agent's best matchup, for the reason already on file

Measured 2026-08-22, [report](reports/2026-08-22-the-horde-is-the-agents-best-matchup.md).
`configs/experiments/30v15_fast_horde_vs_elite.yaml` — 30 bodies at 12" reach and
Move 12 against 15 elites at 24" reach — trained three seeds, 300 epochs, scored
**refereed** at K=3 on the held-out nine, n=30.

- **The agent beats the best script by +48.5, the largest margin recorded here.**
  Agent **+16.2** (+29.5 / +8.2 / +11.0) against `squad_march_deny` **−32.3**.
  ⚠ **t = 3.65, not the 7.26 first published** — that divided by the seed spread
  alone and treated the script's own **±11.5** as zero; propagating both gives
  SE 13.30. The "two independent estimators" were one dataset sliced two ways
  (their agreement is arithmetic), and **8–9 of 9 tables**, not 9 of 9 — the
  per-seed counts are 9/9, 8/9, 8/9 and averaging before counting signs flatters
  it. ⚠ **UNPAIRED** on init — though the layout pairing that matters here IS
  present (identical tables and seeds).
- ⚠ **This is NOT the agent getting better.** Offence is **negative on 3/3**
  (−11.4 / −43.9 / −31.1); defence carries all of it (+73.3 / +84.4 / +74.4).
  ⚠ **RETRACTED: this does NOT confirm the r=+0.991 rule.** Refitting the 25v25
  rows predicts **−5.1** at this concede level against +48.5 observed — a miss
  bigger than the effect. The correlation was fitted on one scenario and does not
  transfer. The offence/defence split is also an **identity**, not a
  decomposition, so read it as bookkeeping rather than as a cause. The elite concedes 187.4 to a
  script and 103–114 to the agent, so ~80 vp of denial exists here; against
  `advance_and_shoot`, where both concede ~130, the same trait was worth −75.9.
- **`held` INVERTED, and that is the one new observation.** The agent holds *more*
  than the scripts (1.38–1.72 v 0.60–1.00) while keeping 2.9x the army alive
  (0.287–0.374 v 0.077–0.125). The hoarding did not stop — it stopped costing
  ground, because control is a **headcount** and 30 survivors outnumber 15 elites
  wherever they arrive. ⚠ So **the horde side MASKS the offence deficit**; do not
  read a healthy `held` here as allocation being solved.
- ⚠ **The referee tax here is enormous and REORDERS the bar.** `take` −6.9 →
  **−46.4**, `shoot` −33.1 → −66.3, `contest_and_spread` −47.8 → −90.7, `deny`
  −15.4 → −32.3. Unrefereed `take` leads; refereed **`deny`** leads. Thirty models
  in six squads of five at Move 12 shatter formation constantly. Every scripted
  screen number is void as a bar — "measure what ships" applies to a **scenario**,
  not just a config field.
- **Untested: the elite side.** Only the horde was trained. The denial-price account
  predicts an elite agent wins by *less*; that is one training run.

### Freezing is friendly gridlock, and only deterministic policies suffer it

Measured 2026-08-22, no GPU, [report](reports/2026-08-22-freezing-is-friendly-gridlock.md).
`just measure-freezing` counts the movement orders that produce no movement — a
class of failure `vp_margin` and `coherent` are both blind to.

- **A frozen model stays frozen 89% of the time; a moving model freezes 3%.**
  `squad_march_take` / `shoot` / `contest_and_spread` all land at P(f|f)
  **0.888–0.893** against P(f|moved) 0.028–0.035, i.e. **absorbing +0.86**.
  ~11% of orders freeze and ~12% truncate, but **92% of ordered inches are
  delivered** — which is why it went unseen. The loss is a small population that
  never recovers, not a general slowdown.
- ⚠ **`random` is the control, and it inverts.** It truncates **more than twice
  as often** (27.5%) and delivers **less** (86.3%), yet is barely absorbing at
  **+0.086**. So the collision system is not at fault: a blocked random policy
  tries another direction next phase, a **purposeful policy re-issues the same
  blocked order forever**. Freezing is determinism meeting an obstacle, not the
  obstacle. **Any movement-delivery comparison against `random` reads backwards.**
- **The obstacle is FRIENDLY.** 91.8% of frozen model-steps have a friendly base
  touching against 27.7% of moving ones (1.27 v 0.32 friendlies; enemies 0.22 v
  0.03). Friendly bases may be crossed but not *ended on*, so a model whose
  destination is taken backs off to zero.
- **This is the stacking finding's mechanical consequence** — 4.90 models on the
  agent's top point against `squad_march_take`'s 2.73 — and it undermines
  **"the agent never stands still"**: some of that 0.4% STAY rate is models that
  are stuck, and the statistic cannot separate the two.
- ⚠ **Do not re-run the tangential slide.** Measured 2026-08-10 and **worse**
  (0.70/+20.6 → 0.57/+1.0): a fully blocked model spends its whole move sliding
  into the open.
- **Not measured:** the vp cost, or any trained agent. A model frozen on an
  objective it already holds loses nothing.
- **Read `absorbing` beside any movement feature's result** — an advance is the
  longest move in the game and so the most likely to be stopped.
- ⚠ **THE SOLVER IS NOT THE BUG — two variants tried and REVERTED.** Bisection
  on travel made it worse (delivery 91.8% → 90.4%): the legal set is not an
  interval, since travelling further can leave one base without entering
  another. A correct descending scan froze less (11.1%) but truncated more
  (13.3%) and delivered less (91.1%) — it converts freezes into short moves
  without buying ground. **75.5% of frozen model-steps have no legal shorter
  move along that heading at all.** So **"fix freezing" reduces to "fix
  allocation"** — the same wall three reward terms failed against — and this is
  the third movement-side fix measured away after the tangential slide.
  **Do not attempt a fourth.**

### The advance move is REJECTED at 300 epochs, and the loss splits in two

Measured 2026-08-22, [brief](docs/advance-move-problem.md).
`configs/experiments/25v25_maps_advance.yaml` — the golden config with only
`n_advance_speed_bins: 3` — three seeds, 300 epochs, scored refereed at K=3.

- **arm −3.3 (+10.8 / −12.4 / −8.4) against the control's +23.4. UNPAIRED
  −26.7 ± 8.3, t = −3.20.** The control beat the best script by +24.5; the arm
  is 2.2 *behind* it.
- ⚠ **UNPAIRABLE BY CONSTRUCTION.** Adding actions (102 → 150) changes the
  output head, so no init is shared. A zero-initialised conditioning path fixes
  an added *input*, never an added *action*. The layouts and seeds are shared,
  and the two configs are verified the same game for a non-advancing policy
  (scripts score to the same decimal on both) — that cross-config bridge is what
  makes the comparison legitimate at all.
- **Forbidding advance at PLAY, on the same weights, is worth +8.5 vp** (+10.9 /
  +3.9 / +10.8, 3/3). So the weights are not broken — but it reaches only
  **+11.1** against the control's +23.4. **~8.5 vp is the agent choosing a bad
  option; ~12 vp is a worse learned policy.** Both explanations are true.
- ⚠ **NOT caused by freezing, and that explanation was published before being
  checked.** The arm freezes 18–28% and delivers 70–77% — but **the control
  agent freezes 26.3% and delivers 76.4%**. Trained agents freeze at that rate
  *because they stack*, advance or not. The comparison had been made against the
  **scripts** (11%), which was never the right control.
- ⚠ **RETRACTED: usage is NOT monotone in the damage.** s2 advances 23.1% and
  gains least from giving it up (+3.9); s1 advances 8.1% and gains most (+10.9).
- **Open:** whether the ~12 vp is fixable by training longer or is a permanent
  cost of a 47% larger action space. 300 epochs is a screen, and this project's
  own rule is that a marginal screen means "run it longer".
- ⚠ **Read this beside the next section.** The arm's use of the slice is *sane* at
  convergence — it avoids dominated actions and agrees at unit level — and the move
  itself is worth ≤ 0 at twenty rounds for a script too. So the ~8.5 vp "choosing a
  bad option" half is the agent using a move that does not pay at this horizon, not
  a decode failure.

### Advance is a SHORT-GAME move, and nothing in the encoding was the problem

Measured 2026-08-23, no GPU,
[report](reports/2026-08-23-three-prices-for-the-advance-move.md). `just
measure-advance-use` censuses what a policy buys with the advance and what it pays.

- ⚠ **All FOUR nominated encoding defects fail to bind at convergence.** Three
  seeds of the rejected arm, held-out nine, n=10, at **both** `decode_topk` 1 and 3
  (K=1 ≈ K=3 in every cell, so none of it is the decoder's): **dominated** advances
  **0.4–5.9%**, unanimous 5-of-5 unit triggers **64–81%** with one model dragging
  four on only 7–11%, waste **1.8–4.0%**. The policy learned the unit-level move
  type without being given the structure, and learned to avoid the half of the
  slice that is strictly dominated.
- ⚠ **THREE SCRIPTED RULES, THREE REJECTIONS, each narrower than the last.** Paired
  against `squad_march_take`, n=100, three seed bases: pricing nothing ("run while
  far") **≈ −78** in the 2×2; pricing the forfeited shooting
  (`squad_march_take_advance`, 11.2% of unit-turns) **−18.4, 0 of 3**; pricing the
  shooting *and* requiring the run to land the squad on the point
  (`squad_march_take_arrive`, 2.2% of unit-turns) **−11.9, 0 of 3**. The family
  converges on the non-advancing control **from below** — the signature of a move
  whose value is negative wherever it is spent.
- ⚠ **The mechanism I proposed was REFUTED by the statistic built to test it.**
  "It ends inside their reach a turn early" (D-14) predicts advancing moves are
  exposed; they end inside an alive enemy's weapon reach on **4.1%** of model-moves
  (script) and 8.6% (agent) against **22.4%** and **44.7%** for *walking* moves.
  Five times safer. The rule only advances when nothing is in range, and nothing is
  in range when you are far away.
- **The real cost is WHOLE-EPISODE and no end-of-move statistic can see it.** All 45
  tables, n=10: exposure 0.2156 → **0.2388** (+10.8%), firepower 1.091 → **1.004**,
  `alive` 0.396 → 0.349, `held` 2.573 → 2.276, opponent VP **+13.2** against own
  −7.7. Coherency *rises* (0.845 → 0.859), so it is not a formation failure.
- **PRE-REGISTERED AND CONFIRMED: the advance's value is monotone in the round
  count.** `squad_march_take_arrive` v `squad_march_take`, n=100, three seed bases,
  **positive means plain walking wins**: rounds **5 → −1.7 (3 of 3 to advancing,
  t up to −2.73)**, rounds 10 → +1.3, rounds 20 → **+11.9 (0 of 3)**. ⚠ **Absolute
  vp are NOT comparable across horizons** — the five-round outcome sd is 12 against
  twenty's 91 — so read it normalised: **+0.14 sd → −0.04 → −0.13**. And the
  five-round game is **not degenerate**: `hold_deployment` scores −33.1 with `held`
  0.79 against the marcher's −0.7 and `held` 2.50.
- **So the config that trains runs 20 rounds, and there the advance is worth ≤ 0.**
  That cuts both ways for the action space: it lowers the value of *re-encoding* a
  move the policy already uses sanely, and raises the value of *shrinking* it —
  32% of the action space is an option the policy must spend samples learning to
  decline, half of it strictly dominated.
- ⚠ **A MOVE TYPE IS A LEVER, NOT AN ADVANTAGE — and the gate that assumed otherwise is
  RETIRED.** The standing rule was "a scripted advance rule that prices the forfeited shooting
  has to beat `squad_march_take` before anything trains". It bakes in the assumption the
  evidence refutes, so no correct implementation can satisfy it. **Do not add scripted policies
  whose purpose is to advance** — the two on file cost their own users −78 and −11.9 vp. The
  right question is not *does the lever pay* but **does carrying it cost the agent anything**,
  which needs no advance-seeking script: train against a `dark_action_slices` control of
  identical shape and read the paired difference. See D-43.
- ⚠ **NO MELEE IN ANY MEASURED CONFIG, so every movement measurement here is
  PROVISIONAL.** A shooting army has no reason to close except to stand on an objective, so
  closing is priced only by what it captures and never by what it threatens. Any move type
  whose value is "arrive sooner" is being measured in a game that does not yet reward
  arriving. ⚠ The charge and fight phases now **exist** behind `melee.enabled` (default
  **False**, an exact no-op verified byte-identical to a pre-melee `main`) — but nothing has
  been measured with them on, and turning them on voids every baseline and every agent score
  on that config. See [docs/melee.md](docs/melee.md), which also records what is still
  outstanding and why a **vp gate is unpowered by construction** for a lethality-neutral
  mechanic.
- ⚠ **The only live explanation left for the −26.7 is the PATH, and nothing above
  prices it.** Every statistic here is taken at convergence; a 300-epoch screen
  prices sample efficiency.
- **Cautions earned.** Split a statistic by where a model *ends*, not where it
  starts ("advances from inside an objective" reads 21–31% and is 1.8–4.0% waste
  plus 17–20% reallocation). Every behavioural statistic needs its **within-policy**
  control — within-unit distance spread looked like an advance defect at p90 4–6"
  against a 2" chain until the same policy's *walking* turns came out the same.
  And ⚠ **`random` is not a control for action-slice usage**: `RandomBaselinePolicy`
  samples `0..n_move_actions` and can never choose an advance.

### The advance slice, re-encoded: absolute rungs, gated by a mask

Shipped 2026-08-23. `n_advance_speed_bins` defaults to **0**, so **no golden config
is touched** and every reward and observation golden stays bit-identical.

- **Rungs are absolute**: `M + (bin + 1) x (6 / bins)` — at `M = 6` with three bins,
  **8" / 10" / 12"**. The unit's D6 now decides which rungs are **legal**
  (`ActionHandler.advance_legality`, masked on **both** seats) instead of deciding
  what an action means.
- **Two defects go with it.** No action can spend the unit's shooting for a distance
  a normal move reaches — **dominated advances measured 0.0%**, against 3.5–13.8%
  for scripts and 0.4–5.9% for agents under the old ladder. And it was the only
  slice in the game whose indices changed meaning turn to turn, so a policy had to
  read `advance_roll` to know what its own action did.
- **Exploration burden, measured** (120 movement phases): **25.1 of 48** advance
  actions legal per model, **0.00** of them dominated — against roughly **24 of 150
  actions, 16% of the whole space, strictly dominated and always legal** before. That
  is the whole of what the re-encoding buys, and only training can cash it.
- ⚠ **The reason on file for admitting dominated bins does not hold**, and was
  checked against `env.step`: only ONE model need choose an advance for the unit to
  advance, so its squadmates keep the whole normal slice and stop where they like.
  Two tests that pinned the old behaviour were replaced, each naming what it
  replaced and why.
- **Cross-config bridge verified.** `squad_march_take` — which never advances —
  scores **−2.8 / `held` 2.57 / `alive` 0.396 / `coherent` 0.845** on all 45 tables
  either side of the change, identical to every printed digit.
- ⚠ **It VOIDS the advance arm's checkpoints behaviourally.** The tensor width is
  unchanged so they still load; their action indices now mean different distances.
- ⚠ **At three bins a roll of 1 leaves NO legal rung.** Deliberate: the rules would
  permit a 7" advance, the ladder cannot express it, and a 1" gain never repays a
  turn of fire.
- ⚠ **Leader-binds inside the movement slice would SHATTER formation** — move type
  and displacement were the same action, so a leader-only advance caps every
  squadmate at `M`. The scripts advance **5-of-5 at a within-unit spread of 0.00"**;
  leader-binds forces ~6" against a 2" chain. That is why the declaration had to be
  split out into its own phase rather than masked inside the movement slice.

### The move type is declared in the command phase, by the unit's leader

Shipped 2026-08-23. This is the "unit declaration" and "additive cost" half of the
movement goal, and it only works because the declaration is **separate from the
displacement**.

- **A `move_type` slice of 2 actions** (`normal`, `advance`), valid in
  `BattlePhase.command`, registered **last** so no existing index moves. Action
  space **150 → 152** with advance on; **unchanged at 102** with
  `n_advance_speed_bins: 0`, which is every golden config.
- **The unit's LEADER decides** — its lowest-indexed alive model — and the whole
  unit is bound. ⚠ This replaces an **OR over five per-model movement actions**, in
  which any one model choosing a long rung spent all five models' shooting (85.5%
  of five-model unit-turns at initialisation).
- **STAY declares `normal`**, so every policy written before the declaration
  existed behaves exactly as it did. Verified: a non-advancing script scores
  **bit-identically** on 10 of 10 seeds across the change.
- **A rung is legal only for a unit that declared**, and only within `M + roll`.
  Masked on **both seats**.
- **Declaring costs the shooting immediately**, whether or not a member then uses a
  long rung. That is the rules' cost: it attaches to the move type, not the
  distance.
- ⚠ **The roll moved to the START of the side's turn.** It used to happen on the
  command→movement boundary, which was right while the type was chosen during
  movement — but a declaration made in the command phase would then be **blind**,
  and since legality is gated on `M + roll`, no rung would ever be legal. It is
  idempotent and keyed on `(battle_round, active_player)` rather than hung on a
  phase transition, because command is the FIRST phase of a turn and the first turn
  of an episode never advances into it.
- **Config validation**: `n_advance_speed_bins > 0` with `command` in `skip_phases`
  is rejected at construction — otherwise the rungs exist and no declaration is
  ever legal, and a training run measures a feature it never had.
- **Adding fall back or charge now costs one value in `move_type`**, not another
  48-action slice and another unit-resolution hack.

⚠ **What it voids.** The command phase is now a real agent step on advance configs.
Verified neutral on the game itself — the golden config scores **bit-identically on
8 of 8 seeds** with command skipped or active — **except in episodes that end EARLY
by elimination**, which lose one player scoring event. Measured: 10 of 45 tables
moved by exactly **−1.5** at n=10, i.e. 15 VP in one episode each. The skipped
command phase used to be traversed *inside* the terminating step and scored there;
now it is a phase the agent never gets to leave. **Arguably more correct** — a
scoring event that needs your next turn should not fire in a game already over —
but it is a change, so re-measure rather than carry a figure across it.

**Throughput: ~15% more wall-clock per battle round, not 50%.** Per-step cost
*falls* 4.338 → 3.334 ms because command steps do almost nothing, so 1.5x the steps
nets 8.68 → 10.00 ms per round. A 2048-step epoch is 9.5 s → 7.5 s but covers a
third fewer rounds.

### The advance lever at 300 epochs said FREE; at 1000 it says −16.3 (unresolved)

Measured 2026-08-24, three seeds, 300 epochs, **paired**,
[report](reports/2026-08-24-carrying-the-lever-is-free-using-it-is-not.md).
`25v25_maps_advance` against `..._advance_dark` — identical but for
`dark_action_slices`, so both are 152 actions with a bit-identical init.

| seed | advance | dark | paired | usage | forbid-at-play |
|---|---|---|---|---|---|
| s1 | +32.3 | +19.8 | **+12.5** | **0.0%** | +31.2 (−1.1) |
| s2 | −18.2 | −8.4 | **−9.8** | **10.9%** | **+8.1 (+26.3)** |
| s3 | +19.9 | +15.9 | **+4.0** | **0.6%** | +22.6 (+2.7) |

- **Paired +2.2 ± 6.5, t=+0.34, signs flipping.** The old encoding cost **−26.7**
  and never flipped. **Pre-registered verdict: FAIL** — 2 of 3 seeds cleared the
  −8 bound.
- ⚠ **THE ACCEPT CRITERION COULD NOT HAVE PASSED RELIABLY.** Per-seed paired sd is
  **11.3**, so a lever costing *exactly zero* lands a seed below −8 on 23.9% of
  tries and fails "−8 on 3/3" **56% of the time**. The bound was tighter than the
  estimator's own noise. Recorded as a defect in the rule, **not** as grounds to
  overturn the verdict. **Power-check a per-seed bound against the expected spread
  before writing it down.**
- **The cost is in USING it, tested not inferred.** Usage and score order perfectly
  (0.0/0.6/10.9% against +12.5/+4.0/−9.8), and that ordering was *not*
  pre-registered — so it was checked by forbidding advance at PLAY on the same
  weights (the dark config shares the 152-action shape). Prediction written first,
  **confirmed 3/3**: the advancing seed recovers **+26.3**, the declining seeds move
  −1.1 and +2.7.
- **Two of three seeds learned to decline it entirely** — s1 chose **0 advances in
  7,227 unit-turns**. ⚠ Verified this is refusal and not a mask: all 25 alive models
  are offered the declaration in every command phase.
- ⚠ **The result MIXES converged and unconverged runs.** s2's usage across its last
  50 epochs is **7.9% → 4.8% → 7.8%** — oscillating, not decaying. Nearly all the
  noise in ±6.5 is that one seed; the other two agree at +12.5 and +4.0.
- ⚠ **RETRACTED BY THE 1000-EPOCH RUN, SAME DAY.** Resumed to epoch 1000 and
  rescored: paired **−16.3 ± 8.9, t=−1.84, all three seeds negative**, against
  +2.2 ± 6.5 with flipping signs at 300. **s1 and s3 both flipped sign** (+12.5 →
  −6.9, +4.0 → −34.0), so the 300-epoch reading was not a noisier version of this
  one — it pointed the other way. Verdict against the criterion committed to git
  before the scores existed: **UNDERPOWERED** (lower bound −42.2). Not a pass.
- ⚠ **"Two of three seeds learned to decline it" is RETRACTED.** At 1000 only s1 is
  near zero (0.3%); s2 and s3 sit near 5%. **More training made them WORSE at
  leaving the option alone.** The prediction that s2 would fall to the others' floor
  failed: 7.8 → 6.6 → 3.2 → 4.2 → 4.9%, a plateau and drift back up, i.e. a second
  mode rather than under-training. **More epochs is not the lever; more seeds is.**
- **What survives**: the usage/score relationship, now stronger (0.3% usage → −6.9;
  ~5% → −8.0 and −34.0), and the forbid-at-play falsifier (**+26.3** recovered on
  the seed that used it, −1.1/+2.7 on those that did not). The four structural
  criteria are untouched by any of this.
- ⚠ **HYPOTHESIS, NOT A FINDING: the extra option may SLOW learning.** The control
  gained more from the extra 700 epochs than the arm on every seed (+13.8 v −5.6,
  +26.3 v +28.1, **+19.7 v −18.3**). Three seeds at sd 15.3 cannot establish it.
- **VERDICT: REJECT.** Arm **+12.7** against the control's **+29.0** at 1000 epochs
  (beats the old advance arm's −3.3, does not clear the control on any seed).
- ⚠ **But the reject clause's own explanation — "~12 vp is the permanent cost of a
  larger action space" — is REFUTED by the same run.** A pre-predicted falsifier,
  confirmed 3/3: advance-trained weights with the lever masked at **play** land
  within **1.8–4.1 vp** of the control (s3 alone recovers **+32.2**). Decomposed:
  **carrying the option −2.9 ± 0.67 (lower bound −4.8)**; using it at play −13.4.
  **Do not carry forward "the encoding costs ~12 vp", and do not re-open the
  encoding** — the structure is not what loses. The open problem is that two of three
  seeds drift into *using* a move that does not pay, and 700 extra epochs made that
  worse, not better.
- ⚠ **A THREE-SEED SCREEN WAS READ AS A RESULT TWICE AND REVERSED BOTH TIMES.** The
  per-seed paired difference is unstable across seeds *and* across epoch budgets.
  **Nothing here should move a design decision** — resolving "free" versus "−16"
  needs **six seeds at 1000 epochs**, not another three.
- **NEW DIAGNOSTIC: lever usage is a convergence signal.** When the right answer is
  "rarely", a lever whose usage is still oscillating means the run has not settled,
  whatever the reward curve says. One inference run at two checkpoints. It would
  have flagged s2 as not-comparable before it entered the average, and it
  generalises to every move type the rules add.

### Offence is not reward-shapeable here — three arms, one conclusion

Measured 2026-08-22, [report](reports/2026-08-22-the-agent-is-never-paid-to-attack.md).

`closest_objective_v2`'s candidate gate asks whether an arrival improves the
control label, imagining exactly ONE model arriving — so an objective the
opponent holds by two or more could never be a travel target. `contest_deficit`
widens that. Three seeds, 300 epochs, `ent_coef` 0.003, scored refereed at K=3,
**paired against the `-newmaps` controls**.

- **REJECTED. −2.7 ± 4.8 paired, t=−0.55, 1 of 3 seeds positive**; across tables
  −2.7 ± 3.7, t=−0.72, **ahead on 2 of 9**.
- ⚠ **It failed on the ACCEPT criterion, which was OFFENCE.** Offence went
  **−61.2 → −71.5**, backwards on 2 of 3 seeds. The lever built to fix offence
  made it worse.
- ⚠ **`alive` fell 3/3 (−0.048) with `held` flat** — the *reverse* of hoarding,
  and the original reject rule (`alive` **rises**) would have missed it entirely.
  The symmetric clause was added at epoch 290, before any score existed, after
  adversarial review. **Write reject rules for the failure your lever actually
  risks, not the one you are already worried about.**
- **The mechanism is the 2026-08-11 teleport audit reproduced by gradient.** That
  audit force-moved a squad onto contested ground and measured **−1.69 of 5
  models** and **−29.41 of its own income** against 4.91 defenders. Paying a
  policy to walk at defended ruins gets it shot crossing open ground. **The
  one-model gate was load-bearing**, exactly as the overstack penalty was.
- The gate change *worked mechanically* — "they hold it by 2+" exclusions fell
  43.4% → 3.9%, units with their own objective rose 32.0% → 48.1% — and bought
  nothing. Observability passed. The scenario was not at fault either
  (`24v24_maps_spare_squads` was built to pose the question; offence did not move
  there).

⚠ **THIS IS THE THIRD CONSECUTIVE REWARD TERM TO LEAVE OFFENCE FLAT OR WORSE**
(−50.5, −42, −71.5). Stop shaping offence. **The diagnosis the evidence supports
is a DIFFERENCE-REWARD problem:** `vp_gain` is net, so denial *is* paid — but it
is **global**, broadcast identically to every alive model, so no model can prefer
"take theirs" to "stand on ours"; both move the same shared scalar identically.
The per-model term prices only *distance closed*; the term that prices *outcome*
is global. No candidacy gate can reach that.
### The critic already knows the stack is wrong — the failure is SEARCH

Measured 2026-08-23, no GPU, three seeds, 634 forked games,
[report](reports/2026-08-23-the-critic-already-knows.md). `just measure-critic-probe`
forks a live game, rigidly translates one SURPLUS squad off an over-stacked
objective onto an empty one, and prices the move twice — `dV` is the critic's
summed army value, `dVP` the realised `vp_margin` from playing both branches out.

| direction | n | dV (critic) | dVP (realised) |
|---|---|---|---|
| **spread** a surplus squad onto an empty point | 397 | **+2.63 ± 0.32** (t=+8.3) | **+3.85 ± 1.81** (t=+2.1) |
| **stack** another squad onto the pile (the control) | 237 | **−7.18 ± 0.58** (t=−12.4) | **−11.52 ± 2.51** (t=−4.6) |

- ⚠ **THE SURVIVAL-PREMIUM DIAGNOSIS IS REFUTED.** Two independent expert panels
  converged on it — the global stream is paid only to `alive_models`
  (`reward/phase_manager.py:272-274`), so the agent supposedly learned to
  over-price survival. That predicts the critic prefers the surplus model staying
  put. **It prefers the opposite, 6 of 6 seed-round cells, t=+8.3.** Do not fund
  `dead_share_fraction` or pivotality redistribution *on that rationale*.
- **The reverse direction is the control that makes it mean anything.** The
  counterfactual is off-distribution and critics are optimistic there — but that
  predicts BOTH directions positive. The critic is directionally correct both
  ways and gets the asymmetry approximately right (2.7× against a realised 3.0×).
- **What is left is a SEARCH failure.** Reward and critic both value spreading
  correctly; the policy does not do it. Spend on directed exploration and
  representation, not on reward attribution.
- ⚠ **The gradient out is SHALLOW and the gradient in is STEEP.** Marginal
  spreading gains +3.85; marginal stacking loses −11.52. The agent is *slightly
  past* a broad optimum, not parked in a basin the reward dug. Any lever will
  therefore move top-stack occupancy a lot for a small score change — **read
  `dVP`, not occupancy.**
- **`corr(dV, dVP)` is ~0** (+0.07). The critic has the direction and no grip on
  *which* redistribution pays. A search method that needs the critic to rank
  candidate reallocations will not work; one that needs only the direction will.
- ⚠ **Optimal allocation LOST.** `assignment_optimal` is `squad_march_take` with
  the greedy matching replaced by an exact minimum-cost assignment (subset DP,
  verified against brute force on 300 instances): **−26.1 ± 9.4 against greedy's
  +7.6 ± 3.8**, `held` 2.21 v 2.80. This is **not** proof allocation is at its
  ceiling — it is one untuned cost model losing to greedy — but an
  allocation-aware decode would be replacing a rule that just beat its own exact
  counterpart by 33.7 vp. Re-cost before funding. Tune on the 36, never the nine.

### Squad heading disagreement is a SYMPTOM, and the statistic measured architecture

Measured 2026-08-23, no GPU, three seeds,
[report + correction](reports/2026-08-23-a-squad-cannot-agree-where-to-go.md). Two expert
panels were given the first version and refuted its causal half the same day.

**What SURVIVES — the agent allocates worse than chance.** Both panels reconstructed the
numerator independently: the script puts **3.97** squads on objectives, the agent
**4.03–4.51** — essentially identical — and the agent crams them onto **2.08–2.30** distinct
points against the script's **3.28**. Per alive squad, 0.35 objectives against 0.685.
Correcting for squad count makes the gap **larger**.

- ⚠ **RETRACTED: "a squad cannot agree where to go, so it never gets anywhere."** Executed
  squad-centroid travel is **2.82" per squad-step against the script's 2.05"** — the agent
  covers ~40% MORE ground while holding a third fewer objectives. It is not failing to go
  anywhere; **it is going somewhere useless, constantly.**
- ⚠ **THE HEADLINE STATISTIC MEASURED THE ARCHITECTURE.** `clone_squad_march_take.ckpt` is a
  factored per-model network cloned from the winning script. All-on-one-heading: teacher
  **91.8%**, its own clone **42.2%**, agent 35.1%. Normalised to per-model modal agreement,
  **83% of the script-to-agent gap is the factored architecture** and only 17% is the agent.
  A product policy cannot reproduce a shared vector. **Report per-model modal agreement, and
  always beside a clone control.**
- ⚠ **"Make the squad agree" is a MEASURED NULL.** Consensus decoding on frozen weights
  drives within-squad variance to **0.0000** and buys 7.8% more travel for
  **−4.8 / −4.1 / −9.1 vp, 3/3 seeds negative** (two independent implementations).
- ⚠ **`measure_angle_collapse` had NO movement-phase filter** and decoded the shooting slice
  as headings (bin 16 of a 16-bin wheel); squadmates shoot the same target, so those rows
  read unanimous and diluted whichever policy shoots more. Fixed with a phase guard and a bin
  assert. Corrected: script 0.0006 / 97.9%, agent **0.142–0.190 / 41.6–47.7%**.
  ⚠ **The "stay share 33.5% v 65.5%" line is RETRACTED** — movement-phase only it is ~0% v
  56.9%, which *confirms* the standing 0.4%-v-38–57% figure.
- **THE LEAD CANDIDATE, and CLAUDE.md already said to check it.** `closest_objective_v2` +
  `fallback_to_nearest: true` pays **+0.081 per inch closed on the CENTRE POINT** of each
  model's *own* nearest objective — saturating within ~0.63" of the centre, **not** at the
  control radius. So **STAY is strictly dominated** (hence the ~0% stay rate), every model is
  pulled to a point one or two bases can occupy, and — with 8 squads over 5–6 markers leaving
  2–3 unassigned each step — **two members of one squad are paid to walk apart**. A target
  switch returns progress 0.0 and re-anchors, so **abandoning a target is free**. Check this
  before anything else.
- ⚠ **RUN THE CLONE CONTROL ON ANY BEHAVIOURAL STATISTIC before building a diagnosis on it.**
  It costs one inference run. If a clone of the *winning* policy scores near the *losing* one,
  the statistic is measuring your architecture, not skill. This is the second time a published
  explanation here was checked against the wrong control.

### The travel reward, audited: the mechanism was wrong and the term is mostly inert

Measured 2026-08-23, no GPU, held-out nine,
[report](reports/2026-08-23-the-travel-reward-audit.md). `just measure-shaping-gates` on the
config that trains, agent at K=3.

- ⚠ **REFUTED: `closest_objective_v2` does NOT pull models to an objective's CENTRE POINT.**
  `_distances_to_objectives` measures an **area's outline, zero inside**, and the training
  config's objectives are all areas (`radius_size 0.0`). The pull saturates at the boundary.
  The panel read `norms_offset` as a centre distance; for an area it is not.
- ⚠ **REFUTED: "STAY is strictly dominated".** **43.5% of paid model-steps are already
  INSIDE their target**, earning exactly zero however the model moves. This term cannot be
  what drives the ~0% stay rate.
- **REAL BUT SMALL: squadmates paid to walk apart.** 8.0% of squad-steps have members on 2+
  targets (script 4.8%). Too rare to explain a 33% shortfall in objectives held.
- ⚠ **A SCRIPTED POLICY IS NOT A CONTROL FOR WHAT A REWARD TERM DOES.** It does not learn
  from reward, so its column says where its models *stand*, not what it was *paid*. Here that
  matters: the script is **worse on every gate** (points at 16.0% of objectives v 35.8%,
  assigns 9.8% of units v 21.0%, takes 84.2% fallback v 73.6%) **and allocates better** (3.28
  objectives v 2.08–2.30). **No gate explains the allocation gap.**
- **The term is largely inert and nets negative**: 43.5% of paid steps pay zero, 64.2% of
  objectives are not candidates for anybody, and net income is progress +0.08 against the
  overstack penalty's −0.90.
- ⚠ **FOURTH consecutive empty result on this term** — the candidate gate (`contest_deficit`,
  rejected), removing the overstack penalty (rejected, −12.2 ± 5.5), the potential-invariance
  defect (real, term nets negative anyway), and now the fallback mechanism. **Stop nominating
  `closest_objective_v2`.** There is no working travel gradient and four attempts to build one
  have failed.

### How to measure here

The single most expensive class of error in this project. Every rule below was
paid for.

- **Measure the configuration that SHIPS, not an intermediate one.** Twice in two
  days a partial change pointed the opposite way from the whole: new terrain under
  the *old* rectangular deployment read "the tables are harder to hold" when the
  shipped tables are not, and scoring on the *training* config (no referee) read
  the agent +6.8 clear of the best script when refereed it is +23.7. The referee
  taxes each policy by how often it breaks coherency, so turning it off flatters
  the scripts by ~16 vp. **Score on the refereed eval configs.**
- ⚠ **PAIR YOUR ARMS.** `train.py:303` calls `seed_everything`, `:374` constructs
  the model, so two arms differing only in a scalar start from *identical*
  weights and the per-seed difference is a paired estimator. Measured: two seed
  pairs differing by **+7.5 and +7.2 vp — 0.3 apart** — where the unpaired spread
  on the same arm is 26. Worth roughly an order of magnitude; at least one claim
  was recorded "not significant" purely for lack of it. Report the per-seed
  difference, its sd and the correlation; if the correlation is negative say so
  and fall back to unpaired. Pairing is unavailable whenever a change moves a
  parameter shape — those are the least measurable class here. A **zero-initialised**
  conditioning path restores it, since the logits are bit-identical at step 0.
- **Three seeds minimum.** Seed spread is **11.2 vp on `25v25_maps_two_mode`**
  (19.9 / 21.4 / 31.1) but **26 vp and 0.202 coherency on `25v25_maps_coherency`**
  — do not carry one config's spread to another. On the older scenario a single
  seed *inverted* the ranking of a 2×2, and three claims were retracted in one day
  for exactly this.
- **Quote a t AND a sign count on the map pool.** Per-table differences are
  heavy-tailed and the two disagree often enough that either alone misleads.
- **Every `measure-*` recipe takes trailing `key=value` scenario overrides** —
  `rounds=5`, `weapon_range=24`, `turn_order=player` — so one config can be scored
  at several settings of one number without copying it (`scripts/scenario_overrides.py`).
  With no override token the load is exactly a plain parse, so every existing
  invocation is unchanged. The printed header names the overrides, because a
  table that does not say which scenario it measured gets compared to the wrong
  one.
- **n=100.** `measure-checkpoint` and `measure-baselines` default there, not 30:
  per-episode `vp_margin` sd is ~45–50, so n=30 gives SE ~8–9, larger than most
  arm differences ever measured here. ⚠ **That ~45–50 is LOW BY ~1.7x on the
  map-pool configs** — measured 2026-08-24 it is **80.9–83.1 for the scripts** and
  62.3–67.1 for the agent on `take_opponent_refereed`. Every n and every gate sized
  off the doctrine number is under-powered there.
- ⚠ **The ~6 vp resolution floor is TRUE FOR THE AGENT AND FALSE FOR THE SCRIPTS**,
  which is the inverse of the reason on file: between-table sd is **0–6** for the
  scripts against **8.5–22.0** for the agent (F 1.49–4.60). The learned lineage is
  table-dependent; the scripts are not.
- **Fix the comparator BY NAME before measuring, and select it on the statistic you
  will report.** A "best script" chosen by argmax on the same data changes identity
  between cells and turns a magnitude into an artefact — it did exactly that in the
  five-round report. Winner-selection bias measured **+1.4 to +2.9**, and it
  **inflates the script**, so it flatters nothing about the agent.
- **Score agent and baseline on identical layouts.** `just measure-checkpoint
  <ckpt> <config> 30` uses seeds 700000+, so pair it with `just measure-baselines
  <config> 30 "" 700000`. **The bar is a distribution over layout sets, never a
  single number** — `squad_march_shoot` scores 0.45, 0.53 and **0.77** on the same
  config from different `seed_base` values. Training's own `eval/baseline_*` uses
  20 episodes at seeds 10000+ while `eval/win_rate` uses 10 at seeds 500000+;
  those two are not comparable to each other either.
- **A checkpoint's own metrics are seed-set dependent** — the same checkpoint
  measured 0.505 to 0.651 unit coherency across five seed sets of 30. Never quote
  one without its seeds.
- **Prefer `vp_margin` to win rate.** Win rate cannot resolve differences under
  ~7pp here; TF32 cost 8.5 vp while moving win rate only 0.705 → 0.65.
- **Two seeds off one warm start are not two independent samples.** Training
  *amplifies* a small initialisation difference: +0.067 coherency became +0.19
  after 300 epochs and ~9 vp held out. Seeds sharing a warm start agree *tightly*
  with each other while both report their initialisation — which is exactly what a
  real effect looks like. Vary the warm start, and record which checkpoint each
  run descended from. See [the report](reports/2026-08-16-enforcement-is-a-referee.md).
- **Screen at ~300 epochs, quote effect sizes at 1000+.** Epochs 0–300 move
  `vp_margin` −76 → −2 but 300–1000 add another +8, the same size as arm
  differences. The *ordering* separates early. Treat a marginal 300-epoch result
  as "run it longer", not "rejected".
- **Screening arms by their newest top-k checkpoint compares different epochs** —
  `ppo-NNN-*.ckpt` records the last epoch whose *training reward* improved, which
  differs per arm. Matching a comparison to `last.ckpt` once **reversed a ranking**.
- ⚠ **`--resume-ckpt-path` was BROKEN for every checkpoint this repo writes, and
  failed SILENTLY** (fixed 2026-08-24). Torch 2.6 flipped `torch.load`'s
  `weights_only` default to True; a checkpoint pickles the whole `WargameEnv` as a
  Lightning hparam, so Lightning's restore raised `UnpicklingError: Unsupported
  global ... WargameEnv`. Every run died in ~6 seconds **and the launcher exited 0**,
  printing its per-seed "done" lines — the same shape as `train-arm`'s silent
  failure. Caught only by checking process count and GPU memory. **Never read a
  launcher's exit code as evidence a run happened**; check `ps`, the GPU, and that
  checkpoints advanced.
- ⚠ **Score a killed run from its highest `ppo-NNN-*.ckpt`, not `last.ckpt`.**
  `PeriodicLastCheckpoint` writes every 25 epochs, at `on_train_end` and on
  `KeyboardInterrupt` — but **`SIGKILL` triggers none of those**, and SIGKILL is
  the prescribed way to stop these trainers, so `last.ckpt` is routinely up to 25
  epochs stale. (Before 2026-08-08 `last.ckpt` was not the last epoch at all;
  every score labelled "at N epochs" before that date is "at whatever epoch that
  run last improved". `tests/test_checkpoint_callback.py` is now behavioural.)
- **Every score carries coherency.** `measure-baselines`, `measure-checkpoint` and
  `measure-maps` all print `coherent` and `adrift` unconditionally. A `vp_margin`
  alone is a result *plus* an unstated claim that the moves earning it were legal
  — the rule is measured everywhere and enforced almost nowhere, so a table
  without that column reads as compliance and is not. See
  [docs/metrics.md](docs/metrics.md) § Coherency.
- **`held` ranks policies, `on_obj` does not, and `vp_margin` decides.** `on_obj`
  is a fraction of alive models on *any* objective and cannot tell 15 models on
  one point from 5 each on three. But `held` is an *end-state* snapshot while VP
  accrues every round, so a flat `held` is not a null result — `share_soft` gained
  +19.3 vp with `held` unchanged. Read `held` to rank, `vp_margin` to decide, and
  `just measure-objective-split` before shaping a reward against either — it
  reports per-objective `(player, opponent)` counts plus a **redistribution
  ceiling**, what the same survivors would hold if surplus models moved to the
  cheapest lost point. It is deliberately optimistic (no travel time, no return
  fire), so a ceiling near current `held` *rules re-allocation out*; a large one
  does not rule it in.
- **Read the traces, not just the aggregates** — `just analyze-compare <agent>
  <baseline>`. Only `vp_per_step` ranks policy quality; `idle_rate`,
  `objective_approach_rate` and `tactical_score` are structurally misleading here.
- ⚠ **Don't query the Wandb API while runs are training.** Four concurrent runs
  segfaulted within 20 seconds of two `wandb.Api()` calls. Causation is unproven
  but the shared wandb service is the only thing that explains it; read
  `wandb/run-*/files/output.log` instead.
- **Training is deterministic given seed + config + code** (within one setting of
  `--tf32`). Never retrain a control that already exists at the same epoch budget.

### Coherency

The rule is the game's own formation constraint. It is where most of the
project's effort has gone, and the shape of the problem is now settled.

- **It is an AGGREGATION problem, and the fix is the DECODE — +40.5 vp for no
  weights at all, positive on 45 of 45 tables.** Legality is a property of the
  *combination* of 25 independent per-model moves, so a per-model policy is
  punished by `p^k` arithmetic rather than by judgement. `decode_topk=3` takes
  each model's top 3 moves, enumerates the 243 combinations per five-model unit
  and executes the most probable legal one. Under `revert_unit` + `attrition`,
  held-out nine, three seeds: argmax **−38.5 / 0.639** → rerank **−10.3 / 0.851**
  → verified **+1.1 / 0.936**. **K=3 is the setting** (K=5 was better on one seed
  of three at 3.4x the cost) and `decode_topk` defaults to **1**, so every
  historical number stands. **Never quote a score without saying how it was
  decoded.**
- ⚠ **`verify_moves` is the default and matters.** The decoder's forward model
  judged candidates on `position + displacement` while the env clamps and runs
  `resolve_move`: **49.8% of models did not land where it predicted** and 9.3% of
  certified-legal unit-moves landed incoherent. Worth **+11.4 vp**. Seven tests
  covered the module and **none called `env.step`**, so every one asserted the
  decoder against its own relaxation.
- ⚠ **Decode at PLAY only.** Folding it into training means renormalising over the
  legal combinations and *sampling* — a post-hoc filter breaks PPO because the
  executed action is not the sampled one. From scratch it is **−51.8 vp**, and
  annealing K is impossible because `K^k` caps K at 5. See
  [the report](reports/2026-08-20-decoding-does-not-belong-in-training.md).
- **The 2" chain binds, not the 9" spread.** Median gap to nearest squadmate
  **0.09"**, p90 1.75", **7.8% beyond the limit**; spread breaches are 3–5%. On a
  five-model unit `1 − 0.922⁵ = 0.32` against a measured 0.331 — so an
  all-or-nothing revert converts a 7.8% *per-model* tail into a **33% unit veto**,
  and the training target is the tail, not the unit rate. The 0.89 plateau every
  reward lever hits implies per-model `p = 0.977`: an entropy floor raised to the
  fifth power, not under-tuning.
- ⚠ **Do not answer a low coherency number by training under
  `coherency.enforce_move`.** That is a referee for *play*: it supplies no
  gradient and makes formation **worse** (0.569, against 0.756–0.886 for
  `objective_hold.require_coherent` alone, which lifts coherency 0.55 → 0.78 for
  free). Reach for `require_coherent` instead.
- **Enforce at PLAY, never in training — whether or not the mode aliases.**
  `revert_unit` costs ~26 vp in training; `repair` was the obvious exception
  because the aliasing argument does not apply to it, and it **still lost**
  (−57.6 vp / 0.489 coherent against a never-enforced −34.8 / 0.651; read paired,
  the coherency gap is −0.162 ± 0.091 but the vp gap is not significant). What
  fits every arm is that *any* referee substitutes for the skill, and repair is
  the most helpful referee, so it learns the least.
- **`coherency.attrition: true` belongs in every play/eval config and in no
  training config.** Under `revert_unit` alone, 33% of unit-moves are cancelled,
  48.9% of intended movement inches destroyed, and freezing is an **absorbing
  state** (`P(frozen | frozen) = 0.62` against 0.17), so a deterministic policy
  hard-deadlocks. Attrition is the rules' own fix, worth **+15 vp**. Alone in
  training it deletes the army (−105.5 vp, 15.4% alive).
- **Coherency rate does not predict the referee tax — the STAY rate does.** Agent
  s1 intends 0.809, indistinguishable from `squad_march_take`'s 0.800, and pays
  −34 vp where the script pays −0.7. The separator: the agent stands still on
  **0.4%** of unit-moves against the scripts' 38–57%, and standing still is
  trivially legal. Treat "share of unit-moves that are a deliberate stay" as a
  first-class diagnostic.
- ⚠ **A coherency rate rises whenever an army dies** — a unit reduced to one model
  is coherent by definition. Read the per-model tail, which is invariant to unit
  size, beside it.
- **`ent_coef` 0.003 is the better setting for this goal**: coherency 0.771 ±
  0.060 against 0.674 ± 0.104, and **paired** the vp differences are +3.1 / +7.5 /
  +7.2, i.e. **+5.9 ± 2.5, t≈4.1**. What is refuted is the *entropy explanation*
  for the stay rate: 0.003 concentrated the policy exactly as predicted and made
  STAY 30× rarer.
- **Four levers are measured nulls — do not re-run them.** `observe_unit_centroid`
  (−62.1 refereed, the worst arm ever measured here); unit-level action spaces
  (rigid translation preserves coherency but cannot restore it — 0.444); smaller
  units (the `p³` gain is cancelled by a worse per-model tail, and the apparent
  +0.081 was a casualty confound); and rescaling the nearest-squadmate observation
  to the chain band (+3.5 ± 5.3, sign flips across seeds — kept because the old
  scaling was indefensible on inspection, **not** because it bought anything). The
  useful negative from the last: **the remaining gap is not perceptual.**

### Designing a reward lever

- **Check the agent can OBSERVE what the lever keys on.** A desk check that costs
  seconds and has burned ~10 GPU-hours. Two mechanically opposite levers both
  halved objective occupancy because both keyed on per-objective model counts the
  agent could not see. Ask: *if two states differ only in what this term keys on,
  do they differ in the observation?*
- **Per-model is necessary and nowhere near sufficient — the number must VARY
  across the choice the model is making.** Flat `objective_hold` is per-model, yet
  the thirteenth model on a point earned the same as the first, so no model ever
  had a private reason to leave. Price what the model's presence actually changes.
  Counterweight: some quantities genuinely cannot be per-model — `models_lost`
  must be global, because `phase_manager` iterates *alive* models and at
  `max_wounds: 1` a per-model loss penalty is identically zero.
- **An anti-concentration lever must REDISTRIBUTE reward, not destroy it.**
  `overstack_penalty_per_extra` (occupancy 0.925 → 0.520) and
  `objective_hold.surplus_value` (0.784 → 0.284) both lower total objective
  income, so the policy experiences either as "objectives pay less".
  `crowding_exponent` at a=1 conserves the pot — k models on one point earn it
  once, k/2 on each of two earn it twice — so spreading strictly *raises* income.
  Before training a shaping term, ask whether the behaviour it wants pays **more
  in total** than the behaviour it replaces.
- **Don't reach for "positive rewards beat penalties" — tested and refuted.**
  `surplus_value` *is* the positive version of the overstack penalty and failed
  identically. Sign does not separate the winner from the losers; total income
  does. Magnitude matters more than sign (`group_cohesion` at −0.2 inverted the
  baseline ranking; at −0.05 it is in the winning config). Fair residue: a penalty
  is *more likely* to destroy total income by accident, so run the check.
- **Raising an objective weight alone is catastrophic**: weight 1.25 with
  `crowding_exponent` back to 0.0 scored **−40.4** against the control's +3.25 —
  20.2 of 20.8 survivors on one objective. At fixed weight the exponent alone is
  worth **68 vp**. See [the report](reports/2026-08-08-paying-the-pot-beats-the-bar.md).
- **Treat any precision or numerics setting as a reward-affecting change** and
  screen it like a shaping term.
- **Read the doctrine entry's verdict before writing the term.**
  [docs/play-doctrine.md](docs/play-doctrine.md) carries one per claim, and several
  of the terms tried here restate an entry now marked `refused`. The cheapest form
  of any entry is a scripted policy, not a calculator.
- Register new calculators and criteria and document them in
  [docs/reward-phases.md](docs/reward-phases.md).

### Running a run

- `just train <env_config.yaml> [epochs]` · **parallel arms**: `just train-multi
  config1.yaml config2.yaml` (unique `--run-suffix`, shared `--wandb-group`).
- Copy a `golden/` config into `experiments/` to make an arm — **never edit a
  golden config** to try something ([configs/README.md](configs/README.md)).
- **`checkpoints/` is the only copy of the weights, so `just clean` is
  destructive.** Checkpoints are deliberately not uploaded to Wandb
  (`log_model=False`) — nothing ever read a model artifact back, while each run
  uploaded ~591 MB and filled the quota.
- Key options: `--record-during-training`, `--max-epochs`, `--n-eval-episodes`,
  `--seed`, `--tf32`, `--precision`, `--eval-every-n-epochs`, `--lr`,
  `--max-grad-norm`, `--render-mode`, `--no-wandb`, `--run-suffix`,
  `--wandb-group`, `--warm-start-ckpt-path`, `--resume-ckpt-path`,
  `--record-threat-range`, `--record-engagement-range`.
- `just profile <config.yaml> [max_epochs]` writes `profile.html` (`--no-wandb`,
  capped at 5 epochs by default); `just simulate-latest` runs the newest
  checkpoint.
- Curriculum runs log `reward_phase` and `phase_advanced_at_epoch`, so phase
  transitions show up in the dashboard beside the reward curves.
- **Training logs the bar.** `eval/baseline_*` covers `random`, `squad_march` and
  `squad_march_shoot` (`BASELINE_POLICIES` in `model/common/lightning_base.py`).
  Read the **shooting** one — beating the movement-only 0.78 is not beating 1.00.
  `just measure-baselines` adds the middle rungs.
- **Inspecting a run:** `just run-summary <run_id> [bucket]` for rolling means —
  a single-epoch `success_rate` is an `n_episodes`-sample binomial, never read a
  point value. `just measure-phase-gates <ckpt> <config> 40` for per-phase
  criteria rates. See [docs/metrics.md](docs/metrics.md) for what each key means.

### Performance and numerics

- ⚠ **TF32 is off by default because it costs ~8.5 vp.** At epoch 1000, n=100
  identical layouts: s1 **+30.8 → +21.2**, s2 **+27.4 → +19.9**. The `--no-tf32`
  control reproduced the pre-TF32 run *bit-identically* (222/222 tensors), which
  both proves TF32 is the whole effect and confirms nothing else in that window
  changed training. The speed was oversold too: 1.34x on the *update* is 17.8% of
  an epoch. Pass `--tf32` for smoke, profiling and throughput runs only. See
  [the report](reports/2026-08-09-tf32-costs-eight-vp.md).
- **`--precision bf16-mixed` is another 1.8x on the update and is opt-in because
  only its SPEED has been measured** — A/B it over two seeds before trusting it.
- **`torch.compile` is deliberately not wired**: it prefixes every `state_dict`
  key with `_orig_mod.`, and `_apply_warm_start_weights` uses `strict=False`, so
  such a checkpoint would load as *nothing at all* and score a random network as a
  trained one.
- **Evaluation is ~22% of a real epoch** and is not counted in `perf/epoch_s`.
  `--eval-every-n-epochs 4` cuts wall-clock ~16%. **Single-phase configs only** —
  on a curriculum config it changes which epoch a phase advances on, and therefore
  what the run trains.
- `just measure-throughput <config>` gives the per-section and per-calculator
  split of `env.step()`. Two calculators were once ~80% of a 25v25 step by
  recomputing a model-independent quantity per model. Any change to the reward
  pipeline must keep `tests/test_reward_golden.py` **bit-identical** — it is
  verified to catch a one-ULP change. See
  [docs/training-throughput.md](docs/training-throughput.md).

### Five rounds is not a training scenario, and the offence deficit is not the clock

Measured 2026-08-24, no GPU, six seeds x four scripts x two opponents x two horizons,
[report](reports/2026-08-24-five-rounds-does-not-rescue-the-agent.md). Pre-registered
before the numbers existed; the verdict against its own criteria is **MIXED**.

- **`held` is nearly horizon-invariant.** Quarter the game and the agent goes 1.98 →
  1.80 and 2.03 → 2.14 while the scripts go 2.46 → 2.61 and 3.84 → 3.51. The shortfall
  is **−0.81 ± 0.04** and **−1.15 ± 0.11**, behind on **0 of 9** on both opponents. The
  agent is not failing to *arrive*; it fails to spread just as badly when spreading is a
  four-round problem. **The critic-probe conclusion needs no horizon caveat.**
- **Shortening the game makes the agent WORSE where it currently wins**: +13.0 ahead on
  7/9 against `squad_march_take` at twenty rounds, **−5.8 behind on 0/9** at five. Its
  edge is denial and denial accrues per scoring event.
- ⚠ **RETRACTED SAME DAY: "five rounds cannot tell six trained agents apart."** Wrong
  twice. The noise term omitted the **seed x map interaction** (sd 4.32 and 18.32 at
  twenty rounds, **0.00** at five), so the two biases run in opposite directions by
  horizon and both inflate the ratio; corrected the collapse is 12.13 → 0.81, not
  12.27 → 0.72. And **on `held` — the primary readout the pre-registration designated —
  the seeds separate slightly BETTER at five rounds** (F 7.96 → 9.70), as do four
  scripted policies, the fixed-policy control that should have been run (F 33.68 →
  **57.54**). The decisive table also compared **raw vp across horizons**, which the
  pre-registration forbids in bold; normalised the collapse is 3.2–5.9x.
- **What survives is a claim about SCORING, not resolution**: the agent's edge is denial,
  denial accrues per scoring event, and five rounds has four events against twenty's
  nineteen. Five rounds may still be wrong to train at — **it is not closed by this
  evidence**, and nothing has ever been trained there.
- ⚠ **The comparator was selected by `vp` while the readout was `held`, and switched
  identity between the cells being compared.** Fixed to `squad_march_take` the shortfall
  reads −0.73 → −0.82 (grew 12%, not 67%) and −1.81 → −1.37 (shrank 24%, not 37%); fixed
  to `deny` the sign of the change flips. The **verdict** is robust — no comparator rule
  gives a ≥50% shrink on both opponents — but every magnitude was an artefact.
- **NEW, and the sharpest statement of the search failure on file: the board is STATIC
  after round 8.** `held` by round (2/5/8/12/16/20) is 2.28 / 2.61 / 2.81 / 2.74 / 2.73 /
  2.70 for `squad_march_take` and 1.92 / 1.93 / 2.06 / 2.19 / 2.13 / 2.10 for the agent.
  Twelve of twenty rounds are a constant-rate replay of a frozen board, and **the agent's
  allocation is fixed by round 2** — it gains +0.18 objectives over the remaining eighteen
  rounds against the script's +0.53 by round 8.
- ⚠ **Raw vp is NOT comparable across horizons** (per-episode sd 61.7 → 12.6). Quote it
  within a horizon, or normalised.

### The other scenarios

`25v25_maps_two_mode` and `25v25_maps_coherency` draw from the eval tables;
the other four generate their own terrain and are a different game.

- **`25v25_shooting_opponent.yaml` is the config that beat the shooting bar**:
  **+30.8 (s1) and +27.4 (s2) vp_margin against `squad_march_shoot`'s +17.0**,
  n=100 identical layouts, epoch 1000, **`--no-tf32`**. The lever is
  `objective_hold`'s `crowding_exponent` — a point pays a fixed pot split between
  its occupants instead of paying every occupant the same wage. ⚠ **The exponent
  has only ever been measured on this scenario; do not port it elsewhere without
  measuring there.** This scenario is effectively a two-objective mission — both
  policies concede the third point in nearly every episode — so `held` is bounded
  near 2. `25v25_cover_control.yaml` is the control it was developed against. See
  [the report](reports/2026-08-08-paying-the-pot-beats-the-bar.md).
- **`25v25_single_phase.yaml` and `25v25_curriculum.yaml`** share a scenario and a
  final phase, so comparing them isolates the curriculum. Every phase must keep
  `vp_gain` and at least one per-model calculator —
  `tests/test_curriculum_configs.py` enforces both.
- ⚠ **A bar of 1.00 is an artefact of an opponent that never fires.** The original
  25v25 configs face `scripted_advance_to_objective`, which does not shoot;
  against `scripted_advance_and_shoot` on the same terrain `squad_march_shoot`
  falls to 0.60 and `squad_march` 0.80 → **0.24**. Switching a config's opponent
  invalidates every baseline *and* every agent score on it. See
  [docs/opponent-policies.md](docs/opponent-policies.md).

### What voids a number

Each of these changed the dynamics, so results either side are not comparable.
Re-measure rather than carry a figure across one.

- **2026-08-10 — the board stopped being a chessboard.** Positions are real
  points, a move covers exactly the distance its speed bin says (a "speed 1"
  diagonal used to travel 1.41), sight is a sampled ray rather than a Bresenham
  walk, and models can carry a base radius. The *qualitative* lessons survive;
  every specific figure needed re-measuring.
- **2026-08-13 — models no longer block line of sight**, only terrain does. A
  deliberate divergence from the rules, on the grounds that no model here has an
  opaque silhouette (see
  [docs/rules/implementation-status.md](docs/rules/implementation-status.md)).
  Large: `squad_march_shoot` moved **+38.0 → +17.0** on the shooting config. ⚠
  **`eval/exposure_rate` changed *definition* at the same time** — it now uses the
  same centre ray the shooting mask does — so exposure is not comparable across
  this date at all.
- **2026-08-19 — a dead model used to stop yours shooting.** The engagement gate
  took the nearest opponent over *all* opponents and only then applied
  `opponent_alive`, so a corpse pinned a model for the rest of the episode. It
  fired on **8.74%** of model-steps against the real rule's 0.80%: 92% of
  suppressions were spurious. `engagement_range` is `gt=0` and defaults to 1.0, so
  **every config was affected**. Worth **+7.0 vp** to the agent, paired, 3/3
  seeds — it cost the agent more than the scripts, plausibly because it
  concentrates its models and so stands near its own casualties more often. A
  second fix the same day (LOS symmetry, #211) is a **measured null** on score and
  voids nothing.
- **2026-08-23 — the scripts learned to Advance, and a move must end unengaged.**
  The scripted bar moved **+1.3 to +32.6 vp** (4 of 4) and the movement rule changed
  on every config, so every scripted-bar figure on an advance config and every agent
  score compared against one is void. ⚠ **This under-scopes itself, and the
  counter-example is measured**, and **bisecting found a SECOND cause nobody named**:
  `d607561`, the wholly-within deployment-zone check, worth **+2.6** beside the
  endpoint rule's +5.0. The endpoint rule is global, and on the
  *non-advance* `take_opponent_refereed` config the scripts moved **+7.6 vp** and the
  published agent gap **halved** (+26.1 → +13.0). Treat every 2026-08-21 row as stale
  until re-measured. See [the report](reports/2026-08-24-five-rounds-does-not-rescue-the-agent.md). Three goldens were regenerated deliberately;
  the other three are byte-identical, which is the check that the movement change is
  targeted rather than global.
- **2026-08-20 — the eval tables were regenerated**, and 2026-08-21 they were
  re-measured against their own deployment zones. See § The board.

### The bar was playing a different game — Advance, and the endpoint rule

Measured 2026-08-23, no GPU, **corrected the same day after two audit panels**,
[report + correction](reports/2026-08-23-the-bar-was-playing-a-different-game.md).

- ⚠ **ADVANCE IS A CORE RULE, NOT AN ARM.** It was framed as accept/reject against a
  control without it. Wrong question: it is staying.
- ⚠ **NO SCRIPTED BASELINE AND NO OPPONENT POLICY COULD ADVANCE.** An advancing agent was
  scored against a walking bar. That premise was real and is fixed; the *magnitudes* first
  published were not.

⚠ **THE BAR NEVER MOVED, AND THE HEURISTIC IS REJECTED.** The 2x2 the first two
measurements skipped (`25v25_maps_advance_refereed`, held-out nine, n=10,
`squad_march_take` both sides, vp_margin to the player):

| | opponent walks | opponent advances |
|---|---|---|
| **player walks** | **−4.1** | +72.7 |
| **player advances** | **−81.8** | −3.6 |

- **"Run while far, walk once close" costs its USER ~78 vp**, and both-advance (−3.6) is
  indistinguishable from both-walk (−4.1). The published "+15.5 to the bar" was **two
  self-inflicted wounds cancelling** — both sides adopted the same bad heuristic in the
  same change.
- ⚠ **NEVER MEASURE A SYMMETRIC CHANGE WITH BOTH SIDES CHANGED AT ONCE.** Run the 2×2.
- ⚠ The first OFF column was also measured on **different code** (before the endpoint
  rule), so those deltas were the sum of two changes.
- ⚠ **"`shoot` gains most despite forgoing its shooting" is FALSE — it forgoes nothing.**
  Declared shots 8,132 walking v **8,375 advancing**. At range 12 with objectives 20–40"
  away, squads only advance while already out of range.
- `advance_when_out_of_reach` now defaults **False** on both sides, pinned by a test. **The
  mechanism stays** — a bar that cannot use a core rule is not a bar. The heuristic is what
  is rejected: it never prices the forfeited shooting.
- ⚠ **Pricing it is NOT enough, measured 2026-08-23.** `squad_march_take_advance`
  advances only when a normal move would have left nothing in range and loses **−18.4
  paired, 0 of 3 seed bases**; adding the arrival clause (`squad_march_take_arrive`)
  reaches **−11.9, 0 of 3**. See § Advance is a SHORT-GAME move — at twenty rounds no
  advance rule pays, and at five rounds the same rule wins 3 of 3.

- **The endpoint rule works BETTER than first claimed: 7.52% → 0.00%, all of it removed.**
  The published 6.01% → 3.21% used a hardcoded 2.26" ring fractionally *larger* than the
  env's own predicate, while the back-off parks rescued models at `ring + epsilon` — so it
  counted every model the rule saved as still engaged. No model ever *starts* engaged
  either: `placement.py` enforces `hostile_separation = min_separation + engagement_range`.
- ⚠ **THE BATCH SHIPPED A MOVEMENT BUG, now fixed.** The back-off walked the endpoint
  backwards **without re-checking bases**, so a rescued model came to rest inside a
  friendly one: **0.18% of pairs, worst 0.68"**, against 0.0000% with the rule off. Six
  unit tests covered the function and **none called `env.step`**, so none could see it —
  verbatim the joint-decoder defect this project already paid +11.4 vp for. Occupied bases
  now contribute spans to the same backward walk.
- **Passing through an engagement range stays legal**; only ending inside is not. The
  reverted first attempt was a *path* constraint and cost 87% of opponent-held objectives
  their only legal spot.
- **The opponent advances too**, per unit from the unit's **centroid** (from the nearest
  member it almost never fires — the opponent deploys 3–12" from objectives at Move 6).
  The bar table was *already* symmetric: the advance configs set
  `opponent_policy: scripted_baseline` wrapping `squad_march_take`, which inherited Advance
  in the same change. An audit panel checked that correction and rated it SOUND.
- ⚠ **The opponent's advance columns are ZEROED, not dropped**, and #237's proposal does not
  work: player and opponent tokens share a feature width, so removing two columns from one
  side alone fails at the tensor.

### Settled — do not re-run

- **The agent does not use terrain for cover; it manages range.** Established by
  deleting all terrain (exposure 0.116 → 0.120) and by doubling weapon range (win
  collapsed to 6.8%). A second round with 19.8% of the board hidden, a per-model
  LOS input and priced losses left exposure at 0.092–0.110 across every arm.
  `observe_threat_count` was null and has been removed. Reports:
  [terrain](reports/2026-08-05-stochastic-terrain-and-cover.md),
  [cover](reports/2026-08-06-cover-signal-reason-geometry.md) — **read that one's
  corrections before reusing it**, the `models_lost` +7 reverses on held-out
  layouts.
- **Terrain: count dominates size.** `just measure-terrain` reports *cells hidden
  from a squad*; many small pieces beat few large ones at equal coverage. Tune a
  profile there, in seconds, rather than after a training run.
- **The dice contribute more outcome spread than the scenario does** —
  `vp_margin` sd 50.6 within a layout against 45.0 between layouts
  (`just measure-noise-floor`).
- **`eval/firepower_ratio` replaced `eval/firepower_advantage` on 2026-08-06 and
  the two are not comparable.** The old count difference scored `random` (0% win)
  top of the table. The ratio measures the *firefight*, not policy quality — read
  it beside `vp_margin`.
- **~37% of objectives get zero models across five weightings and two scenarios.**
  Abandonment is invariant to reward weight; stop tuning weights at it.
- **PPO cannot improve a behaviour-cloned policy here** — with a cold critic it
  destroys a 115.8 clone at every `ent_coef`. The gamma explanation was refuted.

**Past experiments:** [reports/](reports/README.md) records findings including
refuted hypotheses, and is where superseded figures live. **Start with
[the correction](reports/2026-08-04-correction-what-was-actually-broken.md)** — it
retracts most pre-2026-08-04 conclusions, including the claims that `gamma` 0.99
and `ent_coef` 0.01 were refuted (they were measured under a training loop that
never applied the reward being tuned).

## Git Workflow

- Always verify the current branch before committing (especially after a PR merge)
- Create feature branches for all changes; avoid committing directly to `main`
- Branch naming: `feature/<topic>`, `fix/<topic>`, `refactor/<topic>`
- Commit messages: imperative mood, concise summary (e.g. "Add reward shaping for distance")
- If pre-commit hooks reject a commit, fix the issues and make a new commit — no `--amend`, no `--no-verify`
- After pushing a new feature branch, always create a PR using `gh pr create`
- Run `just validate` (format + lint + test) before pushing; `just format && just lint` for quick iteration
- **Shipping:** always create a new branch from up-to-date `main` — never reuse an existing feature branch for a new PR. Checkout `main`, pull latest, then branch. Never push directly on an in-progress branch from another workflow. The `/ship` skill (`.claude/skills/ship/`) automates this via `just ship`
- **Docs-drift check:** a `PostToolUse` hook (`.claude/settings.json` → `.claude/hooks/docs_check.py`) fires after `gh pr create` and `just ship`. It diffs the branch against `main` and names the live docs that cite the changed paths, symbols, recipes or config fields. Fix mechanical drift (renamed symbol, changed default, missing table row) directly; only *suggest* anything asserting behaviour. It is silent when nothing is implicated, and never fails a ship. `reports/`, `.planning/` and `ratings/` are exempt — they record what was measured or believed at the time, under a named code revision; `configs/` is exempt too. Run it by hand with `python3 .claude/hooks/docs_check.py --dry-run [<base>..<head>]`

## CUDA Environment

- Do NOT preemptively disable CUDA — only set `CUDA_VISIBLE_DEVICES=""` when training actually fails with CUDA errors
- By default, let PyTorch use the GPU
