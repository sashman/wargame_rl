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
| Simulate / record a checkpoint | `just simulate <ckpt> <config.yaml>` · `just record-sim <ckpt> <config.yaml>` |
| Regenerate the eval tables from the layout API | `just fetch-maps [owner] [maps_dir]` |
| Test env (random) | `just test-env` |
| Watch a scripted policy play (no checkpoint) | `just play [config.yaml] [policy] [theme]` |
| Step a match by hand and rewind it | `just debug [config.yaml] [policy\|ckpt] [theme]` |
| Recreate a recorded match exactly and step it | `just debug-recording <file> [policy\|ckpt] [theme]` |
| Record a match event log | `just record <config.yaml>` |
| Replay / narrate a log | `just replay <file>` · `just replay-summary <file>` |
| Replay a log visually (window or MP4) | `just replay-render <file> [out.mp4] [theme]` — tabletop by default |
| Analyse a log | `just analyze <file>` · `just analyze-compare <files...>` |
| Inspect a Wandb run | `just run-summary <run_id> [bucket]` |
| Measure reward-phase gates | `just measure-phase-gates <ckpt> <config.yaml> [n_episodes]` |
| Scripted baselines (floor + bar) | `just measure-baselines <config.yaml> [n_episodes] [record] [seed_base]` |
| Score a checkpoint (baseline-comparable) | `just measure-checkpoint <ckpt> <config.yaml> [n_episodes] [record]` |
| Score on the real table layouts | `just measure-maps <policy\|ckpt> <config.yaml> [n_episodes] [maps_dir] [decode_topk]` |
| Why an objective was not held | `just measure-objective-split <policy\|ckpt> <config.yaml> [n_episodes]` |
| How often a policy is in unit coherency | `just measure-coherency <policy\|ckpt> <config.yaml> [n_episodes]` |
| Which calculator pays, and how much is global | `just measure-income-share <policy\|ckpt> <config.yaml> [n_episodes]` |
| Clone a scripted policy into the network (warm-start checkpoint) | `just behaviour-clone <policy> <config.yaml> [n_episodes] [epochs] [out]` |
| Two policies on identical layouts, paired per episode | `just measure-paired <policy\|ckpt> <policy\|ckpt> <config.yaml> [n_episodes] [seed_base]` |
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

> ### ⚠ The eval tables were regenerated from the layout API on 2026-08-20, and every map baseline is void
>
> `configs/evaluation/maps/` is now **generated by `just fetch-maps`**, not traced
> by hand. The 45 tables are the same 45 layouts — matched 45/45 by piece bounds,
> so the numbering and the held-out nine are unchanged — but the geometry is the
> source's rather than an approximation of it. Each table now carries **16 pieces
> (was 15 or 16)** as 8-vertex silhouettes (was quads), and **5 objectives (was 5
> or 6)** resolved from the layout's own markers instead of chosen by eye for
> symmetry: on `table_01` only two of the five markers landed inside an objective
> the old file declared. **82% of objectives moved less than 3", but 18% moved
> further, up to 13.7"**. The zone split is still exactly balanced (82/82/82
> across player zone / middle / opponent zone).
>
> **Re-measured 2026-08-21 against the tables AS SHIPPED — terrain, objectives
> AND deployment zones — all 45, n=30, paired by table: THE TABLES ARE NOT
> HARDER.** `held` does not fall for any script (`take` +0.03 t=0.60, `shoot`
> +0.01 t=0.14, `deny` +0.09 t=2.36) and `on_obj` is flat or slightly up.
> `vp_margin`: `deny` **+12.8** (t=3.87, 31/45) and `take` **+7.5** (t=2.06,
> 27/45) both gain, the bar `squad_march_shoot` is a **flat null** (+1.8, t=0.55),
> and only `random` loses (−3.7, t=−3.28). Coherency untouched (≤0.02), as it
> should be for a geometry change. ⚠ **Quote a t AND a sign count on this pool** —
> the per-table differences are heavy-tailed and the two disagree often enough
> that either alone misleads. **The new bar, all 45 at n=30, seeds 700000+:**
> `random` **−222.5**, `squad_march_take` **+5.9**, `squad_march_shoot` **−5.9**,
> `squad_march_deny` **+5.4**. Note `shoot` is the *worst* of the three scripts
> here while convention calls it "the bar" — **name the policy, don't say "the
> bar"**; `take` and `deny` finish within 0.5 vp of each other, well inside the
> floor.
>
> ⚠ **RETRACTED: this banner previously said "the tables are harder to HOLD, and
> that is what is established"** — `held` down −0.11 to −0.29 for every script at
> t = −2.7 to −6.1, and `random` down **−18.9** (t=−6.6). Every one of those
> figures was measured with the new terrain under the **OLD rectangular
> deployment zone**, because the zones had not landed yet; that is not a board
> anyone plays. With the zones in, `random`'s −18.9 is −3.7 and its `held` *rises*
> on 41 of 45. **A second error in that measurement: its `on_obj` row was actually
> the `alive` column** (0.44/0.44/0.40 are survivor fractions; true `on_obj` for
> those scripts is 0.97), so "`on_obj` fell with `held`" really said models were
> dying more. **The lesson is to measure the configuration that ships, not an
> intermediate one** — a partial change can point the opposite way from the whole.
>
> ⚠ **Agent numbers still need re-measuring** — the four-opponent table in the
> README is flagged and unrevised. A 3-seed retrain on these tables is under way
> (`configs/golden/25v25_maps_two_mode.yaml`, wandb group `new-maps-baseline`).
>
> **This pool has a ~6 vp resolution floor.** Per-table `vp_margin` sd is
> **18.5–20.6**, so even n=45 gives SE **2.75–3.07**, and more episodes per table
> cannot help: the variance is *across tables* and only 45 tables exist. That is
> *better* than the traced tables' ~8 vp, and the deployment zones are why — every
> table now starts its armies in the shape its own layout specifies instead of one
> rectangle imposed on all 45. **Pairing does not rescue a map change**, because
> the table IS the treatment — `table_05` old and new are different boards, so
> pairing cancels only the part that did not change. That is the opposite of
> pairing two training arms, where identical initial weights make it worth an
> order of magnitude. A nine-table read of this same comparison made
> `squad_march_deny` look like the biggest mover; at n=45 it was not.
>
> **The tables also bring their own DEPLOYMENT ZONES.** `deployment_zone` is an
> axis-aligned band and **only one of the six real deployments is one** — two are
> triangles split by a board diagonal, two are stepped staircases, one is bounded
> by arcs. Maps carry outlines named for their shape (`diagonal_halves`,
> `long_edges`, `opposed_quadrants`, `short_edges`, `stepped_bands`,
> `stepped_columns`; 10/9/8/7/6/5 of 45). **Here the API IS trustworthy** — the
> published cards' tinted region is ≥98% inside its polygon on all 45 — unlike
> its objective markers. Sampling uses the rectangle as the outline's bounding
> box and rejects outside it; every model deploys inside its own zone on 45/45,
> where without the test an army spills out on five of the six shapes. ⚠
> **`long_edges` puts the armies 20" apart across the SHORT axis** against 24–40
> elsewhere — at a 12" weapon range that is a different game from turn one, and
> it is a fifth of the pool. A map with no `deployment` block still uses the
> rectangle, so generated-terrain configs are unchanged.
>
> **The observation width did NOT change** — `objective_budget` stays 6 and
> `terrain_budget` 16 — so every checkpoint still loads and paired arms still
> pair. Do not "tidy" the objective budget down to 5: that would change the
> tensor width and orphan every checkpoint in `checkpoints/`.
>
> **⚠ THE API IS NOT THE SOURCE FOR OBJECTIVES — only for terrain.** Its
> per-layout objective markers disagree with the published layout cards on **six
> of the 45 tables, by 12 to 18 inches**, which is a different ruin entirely;
> neither the layout's own copy of its deployment nor the deployment's canonical
> markers is right everywhere, and switching between them only moves which six
> fail. Terrain is unaffected — the piece geometry matched 45/45. Objectives are
> resolved from `scripts/objective_markers.json`, positions carried over from the
> hand-traced tables and **checked against the published cards: right on 45 of
> 45, worst error 1.5 inches.**
>
> ⚠ **RETRACTED: an earlier note here said the hand-traced objectives "were not
> the layout's".** That was wrong. The tracing was accurate — better than
> anything derived from the API — and the error was assuming the API's markers
> were authoritative and comparing the tracing against them.
>
> **An objective is a RUIN**, not a terrain piece: the layouts build one
> structure from several kit pieces, so pieces are grouped by **shared boundary ≥
> 1.0"** (nothing falls between 0.33" and 1.45"). A position takes the **largest
> ruin within 4.5"** — an *authoring* distance, **not** the rules' 3" control
> range — one ruin per position, and **a tie designates both**, which is what
> makes a table carry six and is drawn on the cards as two Centre icons. There
> are **no disc objectives**. Result: **45/45 objectives within 3" of a published
> one**, counts **24 fives and 21 sixes** as published, zone split exactly
> **82/82/82**. Pinned in `tests/test_map_objective_counts.py`, the only
> expectation in this work that is not our own reasoning.
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
>
> ### ⚠ A dead model used to stop yours shooting — fixed 2026-08-19, and it moves everything
>
> The engagement gate took the nearest opponent over **all** opponents and only
> then applied `opponent_alive`, so a corpse kept pinning a model for the rest of
> the episode. `engagement_range` defaults to 1.0 and is `gt=0`, so **every config
> was affected**. It fired on **8.74%** of model-steps against the real rule's
> **0.80%** — 92% of all engagement suppressions were spurious.
>
> Worth **+7.0 vp** to the agent, paired, 3/3 seeds. **Every baseline and every
> agent score measured before 2026-08-19 is void.** The tables below marked
> "re-measured 2026-08-19" are the current ones. Coherency is unaffected — it is
> a shooting change and did not move formation.
>
> A second fix the same day (**LOS symmetry**, PR #211) is a **measured null** on
> score (+0.1, signs mixed) and voids nothing, though it did eliminate 474
> direction-dependent sight answers out of 488,300.

> **Sight changed again on 2026-08-13: models no longer block line of sight.**
> Only terrain does — a deliberate divergence from the rules, on the grounds that
> no model here has an opaque silhouette (see
> [docs/rules/implementation-status.md](docs/rules/implementation-status.md) §
> Line of sight). It is a large change, not a tidy-up: on
> `golden/25v25_shooting_opponent.yaml` over 100 identical layouts the bar
> `squad_march_shoot` moves **+38.0 → +17.0 vp_margin** (win 0.75 → 0.65) and
> every other scripted policy loses ground too, because the opponent also shoots
> more freely. `eval/exposure_rate` changed *definition* as well — the exposure
> scan used the three-ray corridor and now uses the same centre ray the shooting
> mask does — so exposure is not comparable across this date at all. Both golden
> gates were regenerated deliberately. Re-measure any baseline before quoting it.

- PPO on a `TransformerNetwork` is the only thing that trains — there is no algorithm or network to choose. `just train` and `train.py` used to take `--algorithm` and `--network-type`; both are gone, so `just train <config> 800` now means 800 *epochs*
- Start training: `just train <env_config.yaml>` · with an epoch cap: `just train <env_config.yaml> 800`
- **Multiple configs in parallel**: `just train-multi config1.yaml config2.yaml` runs one training per config concurrently (PPO + transformer); each run gets a unique `--run-suffix` and shared `--wandb-group` so they appear grouped in the Wandb UI
- Env configs live in `configs/`, tiered by what breaks if you edit them ([configs/README.md](configs/README.md)). Copy a `golden/` config into `experiments/` to make an arm — never edit a golden config to try something
- Training logs to Wandb automatically; checkpoints saved to `checkpoints/`. Reward phase index and phase advancement are logged (`reward_phase`, `phase_advanced_at_epoch`) so curriculum runs show phase transitions in the dashboard
- **Checkpoints are not uploaded to Wandb** (`log_model=False` in `model/common/wandb.py`). Nothing in the repo ever read a model artifact back — `simulate`, `record-sim`, `measure-checkpoint`, `measure-phase-gates`, `--resume-ckpt-path` and `--warm-start-ckpt-path` all take a local path under `checkpoints/` — while each run uploaded ~591 MB (4 × 148 MB), which filled the storage quota. Metrics, history and videos still log normally. **`checkpoints/` is now the only copy of the weights, so `just clean` is destructive**
- **Reading run metrics:** see [docs/metrics.md](docs/metrics.md) for what each Wandb key means and the procedure for evaluating a run. Several metrics are means-of-means or change definition silently — check the reading rules there before drawing conclusions from `success_rate`, `terminal_success_bonus`, or any `reward/components/*` value
- **Coherency is an AGGREGATION problem, and the fix is the decode — worth +40.5 vp for no weights at all, on 45 of 45 tables.** Legality is a property of the *combination* of twenty-five independent per-model moves, so a per-model policy is punished by `p^k` arithmetic rather than by judgement. `just measure-maps <ckpt> <config> <n> <maps_dir> 3` (or `decode_topk=3` through `build_selector`) takes each model's **top-3** moves, enumerates the 243 combinations per five-model unit, and executes the most probable one that satisfies coherency — `a* = argmax over LEGAL combos of Σᵢ log πᵢ(aᵢ|s)`. On nine held-out tables, n=30, three seeds, under `revert_unit` + `attrition`: **−34.8 → −8.0 vp_margin and coherency 0.651 → 0.847** (per seed −20.3→+1.6, −37.9→−10.1, −46.3→−15.6 — every seed, spread 4 vp against a *seed* spread of 26). **⚠ That +26.8 is the RELAXED decode — it predates `verify_moves`, which is now the default, so it understates what ships.** Re-measured with all three arms under one code version, held-out nine, three seeds, n=30: argmax **−38.5 / 0.639** → rerank **−10.3 / 0.851** → verified **+1.1 / 0.936**, i.e. rerank **+28.2** and verification **+11.4** for **+39.6 together**. Across **all 45 tables** (the decode changes no weights, so the held-out split does not constrain the measurement) it is **+40.5 vp, positive on 45/45, coherency +0.309 on 45/45** — not carried by a minority, since the best 9 average +60.4 and the other 36 average +35.5. **The held-out nine are representative**: +39.6 (sd 15.7) against +40.7 (sd 14.7) for the other 36. That clears the scripted band 0.772–0.891 and lands level with the best script (`squad_march_deny`, −4.4), from thirty vp behind. **K=3 is the setting** — K=5 was better on one seed of three at 3.4x the cost. `decode_topk` defaults to **1**, so every historical number stands, and any past checkpoint can be re-decoded. **Never quote a score without saying how it was decoded.** ⚠ The policy is still *trained* under a decode it does not know about; folding it into training means renormalising over the ≤243 legal combinations and **sampling** — a post-hoc filter would break PPO, because the executed action would not be the sampled one. **⚠ AND THE DECODER'S FORWARD MODEL WAS WRONG UNTIL 2026-08-19.** It judged candidates on `position + displacement`; the env clamps to the board and runs `resolve_move`, backing models off bases sequentially. Measured: **49.8% of models did not land where it predicted** (p90 offset 2.005", the width of the whole 2" chain band) and **9.3% of certified-legal unit-moves landed incoherent**. `verify_moves` (now the default) re-checks the shortlist against the endpoints the env will actually produce — paired on three seeds plus a clone, **+6.4 vp (t=7.7) and +0.096 coherency (t=40)** — ⚠ re-measured as a one-flag difference under a single code version it is **+11.4 vp** (per seed +6.7, +16.2, +11.4), so treat +6.4 as the low end of a seed-dependent range, taking the three-seed agent to **+2.6 vp / 0.951 coherency against the best script's −4.4 / 0.891** (that is the `ent_coef 0.003` arm; the 0.03 arm reaches −1.6 / 0.942, and the +4.2 gap between them independently reproduces the paired ent_coef effect below) — legality decisively met (nothing here had previously held above 0.939), strength level inside the error bars. ⚠ A distilled clone reaches +13.8 / 0.976 but is **one lineage**: the mechanism it was credited to, cloning *decoded* rather than argmax demonstrations, **did not replicate** — worth +6.7 on its own teacher and −3.6 and −13.3 on the other two. Seven tests covered the module and **none called `env.step`**, so every one asserted the decoder against its own relaxation. See [the report](reports/2026-08-18-the-chain-tail-and-the-frozen-army.md) §§ 13, 16
- **`enforce_move: repair` beats `revert_unit` but is a scenario change, not a free win.** Instead of cancelling the move, it pulls stray models onto the unit body (≤8 passes) and falls back to reverting only when the unit cannot be gathered; a pure spread breach is declined. Held out, three seeds, the agent goes **−34.8 → −4.5** and finishes **ahead of every script** — but the referee change moves everyone (`squad_march_shoot` −36.7 → −9.2), so read the whole column before celebrating. It is a **documented divergence** from `03-moving.md` § Making a move and is opt-in for that reason. **Do not train under it either — and that result corrects the rule’s stated reason.** `repair` was the obvious exception, because the rule was justified by action *aliasing* and repair does not alias: different illegal actions produce different repaired configurations, so the gradient survives. It survives and it still loses. Three seeds, 300 epochs, nine held-out tables: **−57.6 vp / 0.489 coherent against a never-enforced control’s −34.8 / 0.651**, ⚠ **read it paired** — `seed_everything` precedes model construction, so arms at the same seed share initial weights: the coherency gap is **−0.162 ± 0.091** (same sign 3/3, non-overlapping ranges) but the **vp gap is −22.8 ± 23.0, t=−1.72, NOT significant**. Formation is what this establishes. The runs were healthy (`on_obj` 0.858, `held` 2.50 at epoch 100) so this is not the do-nothing collapse. **Aliasing was never the whole cause.** What fits every arm is that *any* referee substitutes for the skill — the policy is handed a legal board it never had to produce — and repair is the *most* helpful referee, so it learns the least. Enforce at play, never in training, **whether or not the mode aliases**
- **The referee cancels a third of all moves, and a play config without `attrition` deadlocks.** Under `enforce_move: revert_unit`, **33% of unit-moves are cancelled outright and 48.9% of all intended movement inches destroyed**; 91.5% of unit-episodes freeze at least once. Freezing is an **absorbing state** — `P(frozen next | frozen now) = 0.62` against 0.17 after a move — because a revert reproduces the same decision, and a *deterministic* policy hard-deadlocks (`squad_march_take` requests the identical move on all twenty rounds of one seed and ends where it deployed). A revert cannot **repair**, only refuse: a unit split by casualties is incoherent before it moves. **`coherency.attrition: true` is the rules' own fix and belongs in every play/eval config** — worth **+15 vp** and taking the strongest script 0.935 → 0.991 units coherent. **Never train with it**: alone it deletes the army (**−105.5 vp, 15.4% alive**). See [the report](reports/2026-08-18-the-chain-tail-and-the-frozen-army.md)
- **The 2" chain binds, not the 9" spread, and `revert_unit` amplifies it fivefold.** Median gap to nearest squadmate **0.09"**, p90 1.75", **7.8% beyond the limit**; spread breaches are only 3–5%. On a five-model unit `1 − 0.922⁵ = 0.32` against a **measured 0.331**. So an all-or-nothing revert converts a 7.8% *per-model* tail into a 33% *unit* veto, and the training target is the tail (7.8% → ~1%), not the unit rate. Squad compliance goes as **`p^k`**, which is also why the 0.89 plateau every reward lever hits implies per-model `p = 0.977` — it is an entropy floor raised to the fifth power, not under-tuning
- **Coherency rate does NOT predict the referee tax — the stay rate does.** Agent s1 intends **0.809** unit coherency, indistinguishable from `squad_march_take`'s 0.800, and pays **−34 vp** where the script pays −0.7. The separator: **the agent stands still on 0.4% of unit-moves against the scripts' 38–57%**. Standing still is trivially legal (positions do not change, so coherency cannot break), so the scripts collect half their moves legal for free while **98.8%** of the agent's cancelled moves were moves it wanted, against their 24–25%. A referee that cancels a move you were not going to make is free. Treat "share of unit-moves that are a deliberate stay" as a first-class diagnostic beside the rate
- **The nearest-squadmate observation was rescaled to the chain band, and it is a MEASURED NULL.** That column divided by the board diagonal, so the 2" band every coherency decision turns on had **2.7%** of its range; it now divides by 8" (`CHAIN_OBSERVATION_RANGE_IN`), giving the band a quarter of the range. Three seeds, 300 epochs, held out at n=30, read **paired** against `ctlE`: **vp +3.5 ± 5.3 (t=1.14, sign flips across seeds) and coherency +0.002 (flat)**. Kept because the old scaling was indefensible on inspection, **not because it bought anything** — do not cite it as a gain. ⚠ It regenerated `test_observation_golden`, so checkpoints from before it score differently (the width is unchanged, so they still load). **The useful negative: the policy was not held back by failing to see this distance, so the remaining gap is not perceptual**
- **The agent beats the best script against THREE of the four opponents, and LOSES to `contest_and_spread` — RE-MEASURED 2026-08-21 on the GENERATED tables.** Three seeds trained by the documented recipe (`25v25_maps_two_mode`, wandb group `new-maps-baseline`), held-out nine, n=30, verified top-3 decode, refereed eval configs, scripts re-measured per opponent: vs `squad_march_take` **+22.6** against −1.1 (**+23.7**, t=3.14, 8/9), vs `squad_march_shoot` **+40.2** against +23.0 (+17.2, t=1.78, 7/9 — **not settled**), vs `squad_march_deny` **+24.4** against −8.9 (**+33.4**, t=4.00, 8/9), vs `contest_and_spread` **+21.8 against +30.2 — BEHIND by 8.4** (t=−1.11, 3/9, so not established either, but it is the only matchup without a lead). Coherency **0.950–0.954** on every opponent against a scripted 0.903–0.908 — formation holds even in the matchup it loses. ⚠ **The seed spread is wide and one seed carries the loss**: +30.9 / +8.1 / +26.4 against a bar of +30.2. ⚠ **RETRACTED: the line below said the `contest_and_spread` loss "no longer exists".** It exists on the generated tables. That claim was measured on the hand-traced ones, and this is the second time in two days a conclusion from those tables reversed on these — treat every hand-traced agent number as history. ⚠ **Measure on the REFEREED eval configs**, not the training config: unrefereed, the same three seeds read **+20.6 against a best script of +13.7**, because the referee taxes policies in proportion to how often they break coherency and the scripts break it far more (0.90 v the agent's 0.95). Turning the referee off flatters the scripts by ~16 vp. **The superseded 2026-08-20 figures follow, kept only for the mechanism they describe.** ⚠ **The figures below understated the agent by ~16 vp because they came from checkpoints trained on `25v25_maps_coherency`, while the golden config of record and `just train-coherency-baseline` both use `25v25_maps_two_mode`.** Reproduced, three seeds, held-out nine, n=30, verified top-3 decode: vs `squad_march_take` **+24.1** against −6.2 (gap +30.3), vs `squad_march_shoot` **+36.2** against +13.4 (+22.8), vs `squad_march_deny` **+19.8** against −7.0 (+26.8), vs `contest_and_spread` **+35.3** against +25.9 (**+9.4**). The first three are 4–9x their standard error; the fourth is **2.2x — suggestive, not settled**. Coherency **0.958–0.960** on every opponent. **The `contest_and_spread` loss that motivated task #139 no longer exists** — it was a property of the weaker lineage, not of the agent. **ALWAYS state which config a quoted agent number was trained on**; that missing provenance is the whole reason two lineages sat side by side in the docs undetected. The superseded figures follow, kept only for the mechanism they describe. Post-corpse-fix, held-out nine, n=30, verified top-3 decode, scripts re-measured per opponent: vs `squad_march_take` **+8.0** against −6.2, vs `squad_march_shoot` **+25.0** against +13.4, vs `squad_march_deny` **+10.0** against −7.0, vs `contest_and_spread` **+21.8** against +25.9 — behind by **4.1**, down from 13.7, and inside the seed spread (agent seeds +30.0/+21.7/+13.6). Coherency **0.934–0.942** on every opponent, above the scripted band 0.777–0.895. **The corpse bug cost the agent more than the scripts**, plausibly because it concentrates its models and so stands near its own casualties more often. The pre-fix figures below are superseded and kept only for the mechanism they describe. Scored on nine held-out tables, n=30, verified top-3 decode, with the scripts **re-measured on each opponent** (swapping the opponent voids every baseline on that config): vs `squad_march_take` **+2.6** against −4.4, vs `squad_march_shoot` **+19.3** against +12.1, vs `squad_march_deny` **+4.0** against −3.1, but vs `contest_and_spread` **+17.4 against +31.1** — behind by 13.7. ⚠ **Absolute score measures the OPPONENT, not the agent** — the agent scores *higher* against the weaker opponents while being further behind the scripts; only the same-matchup comparison means anything. **Formation transfers completely**: intended coherency is **0.94–0.97 on every opponent**. The loss is one mechanism, visible in the VP split: against `contest_and_spread` the agent scores 192 / concedes 176 where `squad_march_take` scores 231 / concedes 200 — **it denies better and takes far less** (`on_obj` 0.68 v 0.91, `held` 2.11 v 2.47). Denial wins against opponents that hold ground; against one that spreads thin the ground is cheap and taking beats denying. So the claim is **"a better defensive player than any script"**, not "a better player"
- **⚠ PAIR YOUR ARMS — `seed_everything` runs BEFORE the model is built, so same seed means identical initial weights.** `train.py:303` seeds, `:374` constructs. Two arms differing only in a scalar (a reward weight, `ent_coef`, an enforcement mode) therefore start from the *same* network, and the per-seed difference is a paired estimator. Measured on `ctlE` − `ctl`, nine held-out tables: the two directly-measured seed pairs differ by **+7.5 and +7.2 vp — 0.3 apart** — while the unpaired spread on that same arm is **26 vp**. Pairing is worth roughly an order of magnitude in resolving power, and at least one claim in this file was recorded as "not significant" purely for lack of it. **Report the per-seed difference, its sd, and the correlation**; if the correlation is negative, say so and fall back to unpaired (coherency sometimes anti-correlates, because it is basin selection rather than gradient following). ⚠ Pairing is unavailable whenever the change moves a parameter shape — a widened token, a new head, a different action count — because the arms cannot share an initialisation. Those changes are the *least* measurable class here, which is exactly how `observe_unit_centroid` produced a retraction. A conditioning path that is **zero-initialised** restores pairing: the logits are bit-identical to the parent at step 0, so it warm-starts from existing seeds
- **⚠ SEED SPREAD IS 11.2 vp ON `25v25_maps_two_mode`, NOT 26 — the 26 was measured on `25v25_maps_coherency` and does not transfer.** Three seeds of the reproduced baseline span **19.9 / 21.4 / 31.1** (sd 6.1). That is less than half the spread the warning below describes, and it matters for what is worth running: a 6–10 vp effect is resolvable here and was not on the old config. Still take three seeds. ⚠ ON THE OLDER SCENARIO THE SEED SPREAD DWARFS EVERY LEVER — three seeds minimum before reading anything.** Three from-scratch seeds of the *same* config span **26.0 vp refereed** and **0.202 unit coherency**, against lever effects of 6 vp and 0.10. A single seed of the 2×2 said `ctl` reached 0.790 coherency and `ctlE` 0.703; replicated the means are **0.674** and **0.771** — the ranking *inverted*. Three separate claims were retracted in one day for exactly this, twice from n=4–6 interims and once from quoting seed 1 as a baseline
- **Three coherency levers are measured nulls — do not re-run any of them.** (1) **`observe_unit_centroid`**: on the 2×2 it is **−62.1 refereed**, the worst arm, and its early coherency lead evaporates by epoch 300. (3) **Unit-level action spaces** — `squad_march_take` already is one and reaches only 0.915; forcing a trained agent's moves rigid scored **0.444**, because rigid translation preserves coherency but cannot restore it. (4) **Smaller units** — the `p³` gain is cancelled almost exactly by a worse per-model tail (fewer squadmates ⇒ your nearest is further), and the apparent +0.081 was a casualty confound. **⚠ A coherency rate rises whenever an army dies**, since a unit reduced to one model is coherent by definition, so read the per-model tail — invariant to unit size — beside it. **`ent_coef` 0.003 is NOT on that list**: on three seeds it is the better setting for this goal — coherency **0.771 ± 0.060 against 0.674 ± 0.104** and refereed **−28.9 against −34.8**, with half the variance — and **paired it is significant** — see the pairing rule below; the per-seed differences are +3.1 / +7.5 / +7.2 vp, i.e. **+5.9 ± 2.5, t≈4.1**, where the *unpaired* spread on the same arm is 26 vp. This line previously said neither gap was significant; that was the estimator, not the data. What *is* refuted is the entropy explanation for the stay rate: 0.003 concentrated the policy exactly as predicted (entropy 3.545 → 1.893, top action 0.131 → 0.488) and made STAY **30× rarer**
- **Every score carries coherency, and no result is quoted without it.** `measure-baselines`, `measure-checkpoint` and `measure-maps` all print a **`coherent`** and an **`adrift`** column unconditionally. A `vp_margin` on its own is a result *plus* an unstated claim that the moves earning it were legal, and only that column carries the claim — the rule is *measured* on every config and *enforced* on almost none, so a table without it reads as compliance and is not. `random` scores `coherent` **0.008** with 22.5 models adrift on the real tables. The column is the **policy's own** figure: it reads `intended_coherency_rate` and falls back to the realised rate only where nothing is enforcing and the two are the same board. Do **not** respond to a low number by training under `coherency.enforce_move` — that is a referee for *play*, it supplies no gradient, and it makes formation worse (0.569 against 0.756–0.886 for `objective_hold.require_coherent` alone). See [docs/metrics.md](docs/metrics.md) § Coherency
- **The bar on every golden config, re-measured 2026-08-19 (post corpse fix), n=100 at seeds 700000+, as `vp_margin`:** `25v25_maps_two_mode` **−7.1**, `25v25_shooting_opponent` **+13.3**, `25v25_cover_control` **+15.9**, `25v25_maps_coherency` **+105.7**, `25v25_single_phase` / `25v25_curriculum` **+70.3** (identical, as they share a scenario — a useful consistency check). The bar is `squad_march_shoot` on all six. Floors (`random`): −216.3, −124.9, −133.5, −14.7, −256.6 / −256.0. ⚠ **The `25v25_maps_two_mode` and `25v25_maps_coherency` entries were measured on the HAND-TRACED tables and are both superseded** — those are the only two golden configs that draw from `configs/evaluation/maps`; the other four generate their own terrain and are unaffected. Re-measured on the generated tables, **`25v25_maps_two_mode` only** (all 45, n=30, seeds 700000+): `squad_march_shoot` **−5.9**, `random` **−222.5**, and `shoot` is no longer the strongest script there (`take` +5.9, `deny` +5.4). **`25v25_maps_coherency` has not been re-measured** — do not carry −5.9 across to it, it is a different scenario on the same tables.
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
- **`last.ckpt` was not the last epoch until 2026-08-08.** `get_checkpoint_callback` put `monitor` and `save_last=True` on one `ModelCheckpoint`, so `last.ckpt` was only rewritten on epochs entering the top-k — it held *the last epoch that improved*. One 1000-epoch batch's four files held epochs **970, 692, 948, 998**, and the 692 run's epoch-1000 weights were never written. **Every score in this repo labelled "at N epochs" before that date is really "at whatever epoch that run last improved by its own training reward"**, a spread worth ~13 vp_margin — and not a common rule across arms, since each arm's training reward is a different function. Now split into two callbacks. **A second bug hid inside that fix until 2026-08-09:** the unmonitored `ModelCheckpoint(save_top_k=0, save_last=True)` writes `last.ckpt` only at `on_train_end`, so completed runs were correct but a **killed run left no `last.ckpt` at all** — and runs here are routinely killed, leaving only the top-k files and reinstating the same selection bias. `PeriodicLastCheckpoint` now writes every 25 epochs, at `on_train_end`, and on `KeyboardInterrupt`, staging and renaming so a kill mid-write cannot truncate it. **A third variant survives: `SIGKILL` triggers none of those three, so a run killed that way leaves `last.ckpt` up to 25 epochs stale** — measured at 20 epochs behind the newest top-k file on all four arms of one batch. SIGKILL is the prescribed way to stop these trainers (SIGINT deadlocks them), so this is the normal case, not an edge one: **score a killed run from its highest `ppo-NNN-*.ckpt`, not from `last.ckpt`**, and never read the epoch off a top-k filename and attribute it to `last.ckpt`. The old tests asserted callback *configuration* and stayed green throughout; `tests/test_checkpoint_callback.py` is now behavioural — it runs a real `fit` and checks the file is on disk while training is still running
- **`held` ranks policies but is not the mechanism — a flat `held` is not a null result.** It is an *end-state* snapshot while VP accrues every round, so a policy can hold more during the episode and end level. `share_soft` gained **+19.3 vp_margin with `held` unchanged**, and `share` beats the bar by 11.4 vp while holding *fewer* objectives at the end (1.59 v 1.64). Read `held` to rank; read `vp_margin` to decide
- **Split `held` before shaping a reward against it.** `just measure-objective-split <policy|ckpt> <config>` reports per-objective `(player, opponent)` counts at episode end plus a **redistribution ceiling** — what the same survivors would hold if surplus models moved to the cheapest lost point. It is deliberately optimistic (no travel time, no return fire), so a ceiling near current `held` *rules re-allocation out*; a large one does not rule it in. On the batch-3 scenario the trained agent parks **12.9 of its 15.8 survivors on a point defended by 0.25 opponents** and loses the second 4.2 to 2.7 — ceiling 2.06 against the bar's 1.88, so allocation alone would clear the bar. Note both policies concede the third objective (the opponent stacks ~12 there, flipping it costs 13), so `held` is bounded near 2 and this is effectively a two-objective mission
- **`objectives_held` (`held`) is the metric that ranks policies, not `on_obj`.** Mean count of objectives controlled under VP's own strict count rule. Ordered by `held`, vp_margin is perfectly monotonic across every scripted and learned policy measured; ordered by `on_obj` it is not, because `on_obj` is a fraction of alive models on *any* objective and cannot tell 15 models on one point from 5 each on three. Three experimental rounds were aimed at an `on_obj` deficit that was mostly n=30 noise (0.925 vs 1.000 at n=30; 0.945 vs 0.960 at n=100) while the real 15 VP gap sat in `held` (1.42 vs the bar's 1.64) and was never measured. See [docs/metrics.md](docs/metrics.md)
- **Before training a reward lever, check the agent can observe what it keys on.** A desk check that costs seconds and has already burned ~10 GPU-hours. The overstack penalty and `objective_hold.surplus_value` are mechanically opposite levers that both halved objective occupancy, because both key on per-objective model counts the agent could not see — an objective reached the network as nothing but an `(x, y)` location. An unattributable reward is experienced only as "this pays less", so the policy does less of it. Ask: *if two states differ only in what this term keys on, do they differ in the observation?* If not, add the input first. See [docs/reward-phases.md](docs/reward-phases.md) § Design rules
- **Score with enough episodes to resolve the effect.** `measure-checkpoint` and `measure-baselines` now default to **n=100**, not 30. Per-episode `vp_margin` sd is ~45–50 on 25v25, so n=30 gives a standard error of ~8–9 — larger than most arm differences ever measured here (4–10 vp). Scoring costs minutes against a training run's hours; it was the cheap half being under-sampled. Training eval likewise runs 30 episodes in the seeded recipes rather than PPO's default 10
- **Screening arms by their latest top-k checkpoint compares different epochs.** `ppo-NNN-*.ckpt` records the last epoch whose *training reward* improved, and that epoch differs per arm — so "score each arm's newest checkpoint" silently scores one arm at 349 and another at 121 even when they started together and are at the same true epoch. Measured on the real-geometry screen: matching the comparison to `last.ckpt` (written every 25 epochs) moved the control from **−3.8 to +5.0** paired against the bar and hold25 from **+7.2 to +1.5**, **reversing the ranking**. Use `last.ckpt` for a mid-training comparison, check the runs actually started together, and treat the top-k files as what they are — a per-arm best-so-far, not a clock. Same family as the `last.ckpt` bug below, different mechanism
- **Screen at ~300 epochs, quote effect sizes at 1000+.** Measured on batch 4: epochs 0–300 move `vp_margin` −76 → −2, but 300–1000 add another **+8**, which is the same size as the arm differences — so an early cut is comparable to the signal, and 1000 epochs is if anything too few (the control is still climbing at 950). But the *ordering* separates early: the losing arm was already clearly behind by epoch 200–299. A 300-epoch screen is 3.3x cheaper and would have caught batch 4's failures in ~2h rather than 7. Treat a *marginal* 300-epoch result as "run it longer", not "rejected"
- **Training is deterministic given seed + config + code.** Two independently trained runs at the same seed reproduced bit-identical eval metrics on all ten fields (different checkpoint checksums; greedy evaluation collapses low-order weight differences). **Never retrain a control that already exists at the same epoch budget** — two of batch 4 round 1's eight runs bought nothing
- **Don't query the Wandb API while runs are training.** Four concurrent runs segfaulted simultaneously (signal 11) after `ConnectionResetError` in the wandb service client, minutes after two `wandb.Api()` queries. Causation is unproven — the same queries ran harmlessly during earlier rounds — but the shared wandb service is the only thing that explains four independent processes dying within 20 seconds, and mid-run reads are available from the local `wandb/run-*/files/output.log` instead
- **Win rate cannot resolve differences under ~7pp on these configs.** Measured within-arm seed spread across batch 3 was 6.0–7.3pp on win rate while `vp_margin` separated cleanly — prefer `vp_margin` for arm-to-arm comparisons, and never read a single-seed win-rate gap as an effect
- **Terrain: count dominates size.** `just measure-terrain` reports *cells hidden from a squad*, the only figure that matters, since exposure is "at least one enemy sees me". Many small pieces beat few large ones at equal coverage. Tune a profile there, in seconds, rather than after a training run
- **The dice contribute more outcome spread than the scenario does.** `just measure-noise-floor` holds layouts fixed and varies only `reset(options={"combat_seed": ...})`: on the batch-3 control, `squad_march_shoot` has a vp_margin sd of 50.6 within a layout against 45.0 between layouts. Run **two seeds per arm** before reading any difference smaller than ~10pp
- **Two seeds off one warm start are not two independent samples — vary the warm start too.** A small initialisation difference gets *amplified* by training: two checkpoints differing by **+0.067** in unit coherency (measured over five seed sets, n=30 each, the better one ahead on all five) produced descendants differing by **+0.19** after 300 epochs, and by **~9 vp_margin on held-out tables**. Sixteen runs, no overlap between the two bands, and a full crossover — every seed handed the *other* checkpoint — flipped every run into its new lineage's band, so the training seed explains none of it. **The failure mode is that it looks exactly like a real effect**: seeds sharing a warm start agree *tightly* with each other (sd 0.011–0.025) while both are reporting their initialisation. Vary the warm start across seeds, and record which checkpoint each run descended from. See [the report](reports/2026-08-16-enforcement-is-a-referee.md)
- **A checkpoint's own metrics are seed-set dependent, so never quote one without its seeds.** The same checkpoint measured 0.505 to 0.651 unit coherency across five seed sets of 30 episodes. Two claims here were wrong from exactly this — "it starts marginally behind" (n=20, one set) and "the gap exists from epoch 0" (the in-run metric, on the one set where the head start is doubled). This is the same trap as the scripted bar scoring +67.5 at seeds 10000+ and +93.5 at 700000+
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
