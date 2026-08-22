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
│       │   └── ppo/               # PPO: actor-critic, lightning module, agent, config
│       └── types.py               # Experience
├── configs/                       # Env configs, tiered by what breaks if edited
│   ├── golden/                    #   backs a published number
│   ├── experiments/               #   arms; deleted once answered
│   ├── evaluation/maps/           #   the real table layouts
│   └── dev/                       #   fixtures and demos
├── tests/                         # Pytest suite with conftest.py fixtures
├── docs/                          # Design docs (movement, reward phases, missions-and-vp,
│                                  #   roadmap, metrics, shooting, expected-damage,
│                                  #   terrain, training-throughput)
│   └── rules/                     # Rules specification + constants.yaml + gap map
├── reports/                       # Experiment findings, kept for retrospection
├── scripts/                       # Run-inspection tooling (fetch_map_layouts,
│                                  #   run_summary, measure_phase_gates,
│                                  #   measure_baselines, measure_checkpoint, measure_terrain,
│                                  #   measure_noise_floor, measure_objective_split,
│                                  #   measure_income_share, measure_maps,
│                                  #   behaviour_clone,
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
| How often the VP cap binds, and what it discards | `just measure-vp-cap <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| What holding a point earns against what it costs | `just measure-hold-hazard <policy\|ckpt> <config.yaml> [n_episodes] [decode_topk]` |
| How often a policy is in unit coherency | `just measure-coherency <policy\|ckpt> <config.yaml> [n_episodes]` |
| Which calculator pays, and how much is global | `just measure-income-share <policy\|ckpt> <config.yaml> [n_episodes]` |
| Clone a scripted policy into the network (warm-start checkpoint) | `just behaviour-clone <policy> <config.yaml> [n_episodes] [epochs] [out]` |
| Two policies on identical layouts, paired per episode | `just measure-paired <policy\|ckpt> <policy\|ckpt> <config.yaml> [n_episodes] [seed_base] [key=value...]` |
| Dice-vs-scenario noise floor | `just measure-noise-floor <config.yaml> [n_layouts] [n_combat_seeds] [policy] [key=value...]` |
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
  arm differences ever measured here.
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
- **2026-08-20 — the eval tables were regenerated**, and 2026-08-21 they were
  re-measured against their own deployment zones. See § The board.

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
- **Docs-drift check:** a `PostToolUse` hook (`.claude/settings.json` → `.claude/hooks/docs_check.py`) fires after `gh pr create` and `just ship`. It diffs the branch against `main` and names the live docs that cite the changed paths, symbols, recipes or config fields. Fix mechanical drift (renamed symbol, changed default, missing table row) directly; only *suggest* anything asserting behaviour. It is silent when nothing is implicated, and never fails a ship. `reports/` and `.planning/` are exempt — they record what was believed at the time. Run it by hand with `python3 .claude/hooks/docs_check.py --dry-run [<base>..<head>]`

## CUDA Environment

- Do NOT preemptively disable CUDA — only set `CUDA_VISIBLE_DEVICES=""` when training actually fails with CUDA errors
- By default, let PyTorch use the GPU
