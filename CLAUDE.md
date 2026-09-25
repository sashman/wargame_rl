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
│                                  #   measure_throughput,
│                                  #   measure_per_model_eval_mode, measure_bridge,
│                                  #   measure_rung)
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
| Train one ARM of a per-model screen: N seeds, detached, `--num-rollout-envs` required | `just train-per-model-arm <rounds> <n_seeds> <group> <tag> <flags> <config.yaml>` |
| Score a per-model checkpoint GREEDY and SAMPLED on identical seeds, paired | `just measure-per-model-eval-mode <config.yaml> [n] [seed_base] <run/last.pt...>` |
| The WHOLE-ARMY control of a curriculum rung: N seeds of `train.py`, detached, on Wandb | `just train-curriculum-control <epochs> <n_seeds> <group> <tag> <flags> <config.yaml>` |
| The BRIDGE CHECK of a curriculum rung: one script through both facades, identical seeds | `just measure-bridge <config.yaml> [n] [policy] [seed_base] [key=value...]` |
| Read a curriculum rung: the bar, then every checkpoint, turns paired per episode | `just measure-rung <config.yaml> <n> <seed_base> <policy> <ckpt\|pt...>` |
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
| Clone a scripted policy into the SET NETWORK (per-model `.pt`, per-head match on held-out episodes) | `just behaviour-clone-per-model <policy> <config.yaml> [n_episodes] [epochs] [out] [seed]` |
| Fit only a per-model clone's value head to the teacher's returns (policy bit-identical) | `just fit-per-model-critic <clone.pt> <teacher> <config.yaml> [n_episodes] [epochs] [out] [seed]` |
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
  with melee on.** A training arm would have measured the BAR, not the agent, because no
  scripted baseline or opponent could charge — **fixed 2026-08-25**: `select_charge` is a
  hook defaulting to STAY (every baseline figure unchanged, digest 9 of 9 identical to
  `main`) and `squad_march_take_charge` overrides it on **both seats**, cleared on four
  mechanism gates written before it ran. The gate for the arm itself is **pre-registered**
  ([report](reports/2026-08-25-melee-preregistration.md)): a vp gate is unpowered by
  construction here (MDE **25.97** vp at n=3), so the primary readouts are mechanism counts. An expert panel measured a charging script at
  **+62.5 ± 14.7** whose value is **entirely the shooting shield** (−4.0 ± 17.4 with the
  target gate ablated). ⚠ **The three defects an audit found in it are now all
  closed** — the charge roll is an observation column and not just a logit mask; the joint
  decoder runs in the charge phase against the charge's OWN referee, so a melee score at
  K=3 is one; and the shooter-side engagement gate reduces over the shooter's UNIT, closing
  the "send one model to lock them and keep four firing" exploit. All three are
  **unpriced** — engagement stays 0.0000% without a charging policy, so the seeded digest
  is 9 of 9 identical to `main`. ⚠ **The charge is now DECLARED by the unit's leader in the
  command phase and the declaration BINDS the unit** — a charge used to be declared
  implicitly by picking a rung, so every model decided separately and a whole unit committed
  on only **23–35%** of its charges against a rigid script's 100%. The melee configs
  therefore step `command`: `max_turns` 60 → **80**, and **every melee figure measured before
  it is void on that config**. ⚠ **The fall back is refereed as of 2026-08-29**
  (`_enforce_fall_back`: unit must end unengaged AND coherent, whole-unit revert, both
  seats) — before it, a unit with a pinned member tore itself to 16.7–20" chain gaps
  "falling back" while a member stayed engaged, and **every melee figure measured before
  the referee is void on melee configs** (non-melee configs are untouched — the path is
  gated on engagement). [docs/melee.md](docs/melee.md)
- **DDD layering** — `domain/` owns the rules, as one bounded context shaped into sub-domains (`kernel/`, `battlefield/`, `sequencing/`, `movement/`, `attacks/`, `shooting/`, `melee/`, with the `Battle` aggregate and `BattleView` at the root); `wargame.py` is a facade; reward/renders depend only on the `BattleView` protocol. See [docs/ddd-envs.md](docs/ddd-envs.md)
- **Rules specification** — [docs/rules/](docs/rules/README.md) is the game's rules authority: a self-contained spec written for this project, with `constants.yaml` (every number, in inches) and [implementation-status.md](docs/rules/implementation-status.md) (per-rule: implemented / partial / divergent / absent). Before implementing a mechanic, read its chapter and its gap-map row. `tests/test_no_ip_references.py` keeps the repo free of references to the commercial product the rules derive from — the spec names no product, publisher, edition or faction, and neither should anything else
- **Play doctrine** — [docs/play-doctrine.md](docs/play-doctrine.md) is how this game is *won*, as `docs/rules/` is how it is *played*: 43 numbered entries, each stating a claim, whether the environment can express it, which extension point it lands in, and what has already been measured about it. It is a store of **hypotheses, never of evidence** — price an entry as a scripted policy (`just measure-paired`, no GPU) before it becomes a reward term or a training run, and where an entry disagrees with the record below, **the record wins**. ⚠ **DO NOT EDIT IT.** It was collected from external sources and its worth is that it still says what it said before we tested anything — an immutable reference. Results go in [docs/play-doctrine-findings.md](docs/play-doctrine-findings.md), one additive entry per claim priced, never rewriting the claim it answers
- **Threat field** — `envs/board/` is a **leaf** package of board-wide reads, and its first tool is the next-turn threat field (`just measure-threat-field`, the `[T]` overlay). ⚠ **Threat is a NEXT-TURN quantity: the opponent moves before it shoots**, so the `[R]` overlay — range ∩ sight from where models *stand* — reads **false-safe** for anyone choosing where to end a turn. Measured on the golden config's held-out nine, `[R]` calls **18.7%** of the board clear where the next-turn field does not, and mean expected casualties roughly doubles. Cover is not applied and *cannot* be, which biases the field **against objectives** — read it beside `just measure-hold-hazard`
- **Unit matchups** — the other `envs/board/` tool, and the one that has **no positions**: `just measure-matchups` reads the two armies' stat lines before a model has moved. It is a *reduction* of the per-model expected-damage matrix that already ships as an observation input, so the table a human reads and the number the network sees cannot disagree — attacker axis **sums**, defender axis does **not**. ⚠ **Range never enters the damage scalar**; it appears as `reach`, `free` (rounds of unanswered fire while the shorter gun closes) and an exchange ratio quoted at two distances. ⚠ **On the config that trains it is 1×1**, since both armies are one profile — it says something only where the profiles differ

### Game State I/O (`envs/state/`)

Snapshot/event pipeline for recording and inspecting matches — `GameStateSnapshot`, event-log deltas, `StateExporter` (wired into `step()`), replay, narration, and `analyze_match` metrics. Driven by `replay_events.py` / `analyze_events.py` and the `record` · `replay` · `analyze` · `analyze-compare` recipes. See [docs/game-state-io.md](docs/game-state-io.md)

### Ratings (`rating/`)

Puts scripted baselines and learned checkpoints on **one scale**, so "did this get better" has an answer that does not depend on which opponent it happened to face. Bradley-Terry maximum likelihood with the deployment-zone, first-turn and **player-seat** advantages as explicit fitted terms, a bootstrap over layouts, and an append-only ledger in `ratings/` keyed by a scenario fingerprint that **refuses** to mix scenarios. `score.py` and `elo.py` import numpy and nothing from this repo; `arena.py` is the only module that touches a live env, and it wraps `evaluate_selector` rather than reimplementing it. Recipes: `measure-seat-parity` · `measure-elo` · `elo-table`. See [docs/elo.md](docs/elo.md) · [docs/self-play.md](docs/self-play.md)

⚠ **A training run now logs two ratings about itself, and NEITHER is on this scale.** `eval/elo` inverts the Elo curve on the eval games against the config's own opponent (pinned at zero) — a **monotone transform of `eval/vp_margin`** that adds a bounded scale and no information; `self_play/learner_elo` is a **ladder** against the run's own pool, banked from finished rollout episodes so it costs no extra games, and it is the one that can move while a margin against a fixed script is flat. Both lack `h_seat`, `h_turn` and any bootstrap. ⚠ Never put one in a ledger or compare one to `just elo-table`. ⚠ **Filling that rating table also makes `--pfsp-mode` live for the first time**: `SnapshotPool.sample` reads an unrated entry as `p = 0.5`, nothing called `rate()`, and `pfsp_weights` on a constant vector is uniform — so `hard` (the **default**) and `even` both drew uniformly, verified directly. Nothing has been run under them, so nothing is voided, and `uniform` — the pre-registered screen's arm — is unaffected. See [docs/self-play.md](docs/self-play.md)

⚠ **The two seats are not the same game, and `h_seat` only partly answers it.** On `configs/golden/25v25_shooting_opponent.yaml` one policy played from both seats loses from the *player* seat by **−24.6 ± 9.4 vp**, and every number in this file is quoted from that seat. `h_seat` absorbs the confound so ratings no longer depend on command-line position — it is identified through a **cycle** in the pairing graph (so **three entrants are required and two are refused**) or directly by a **self-pairing**, which is what `just measure-seat-parity` plays and which roughly halves its standard error; the gate now appends its legs to the ledger for exactly that reason. ⚠ **But the gate is still advisory** — nothing refuses to rate a scenario that fails it — and `h_seat` assumes the advantage is *constant in Elo across pairs*, which `h_turn` (measured to change sign with shooting) suggests may be false. Cross-check the fitted term against the gate's own aggregate. ⚠ **The −24.6 is a property of THAT scenario, not of the engine: `25v25_maps_two_mode` — the config that trains — PASSES the gate**, measured 2026-08-31 at **+6.5 ± 6.1 vp** (`squad_march_take` both seats, 120 layouts; 95% bound roughly [−5.5, +18.5]). ⚠ **Run the gate at n≥100.** The same gate at n=30 read **+19.1 ± 11.2**, within 15% of failing, and the estimate did not survive quadrupling the layouts — at n=30 the threshold is 22.4 vp, so the −24.6 that condemned the shooting config would have been about a coin-flip to catch. ⚠ On a **map-pool** config the gate measures the seat **and the side of the table** lumped together — the drawn outlines stay bound to the seats, so the zone axis is a no-op and cannot separate them; there it plays the **turn-order pair only** (four legs would double-count every game) and appends **nothing** to a ledger. No rating is published. See [the report](reports/2026-08-19-the-two-seats-are-not-the-same-game.md)

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
- ⚠ **THE ELITE ARMY'S UNIT ENCODING WAS ALIASED FOR EVERY NUMBER ABOVE** (bug
  fixed 2026-08-31). `group_span` floored, so 15 elites at `max_groups: 6` split
  into **8 units, ids 0..7**, while `_group_ids_to_one_hot` **clips** to
  `max_groups - 1` — units 6 and 7 both encoded as column 5, silently, while
  `unit_count` sized the shooting slice at the true 8. The network could name
  three units its observation could not tell apart. The horde side (30 at cap 6)
  is clean, so the agent played a correct encoding against a scrambled one.
  **The margin as measured stands; "the horde is the agent's best matchup" now
  has a competing partial explanation.** Re-measure before quoting it, and note
  the fix *changes the scenario* — the elites become 5 units of 3, not 8 of 2 —
  so it voids rather than repairs those figures. The untested elite-side run
  would have trained straight through it.

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

The single most expensive class of error in this project. Every rule below was
paid for.

- ⚠ **MEASURE THE COMPARATOR AT THE SAME n AS THE ARM, AND PROPAGATE ITS SE.** A
  deterministic script is **not a constant**: it is a fixed policy sampled over
  scenarios, and it carries the same per-scenario noise the agent does. The
  melee ladder's bar was a single n=45 estimate quoted to one decimal as if
  exact; remeasured at n=180 its `vs_deny` value moved **+11.8 → +36.8**. Every
  ladder row ever published in that goal carried an unpropagated **±12.7**. See
  [the retraction](reports/2026-09-06-the-ladder-was-measurement-noise.md).
- ⚠ **NEVER quote an across-seed SE for a claim about the game when every seed
  shares the evaluation scenarios.** Scenario noise is then **common-mode** and
  does not shrink with seeds, so an n=6 SE omits it entirely — it read
  2.8–8.1 SE on four cells that a paired across-scenario estimator put at
  t = −0.03 to 1.12. Report the **paired per-scenario** estimator, or both.
  Seed variance was never the binding constraint on that ladder; **episodes
  were** (per-seed means spanned just +50 to +66 where scenarios spanned ±85).
- ⚠ **Raising n moves the MEAN as well as the interval.** A pre-registered
  prediction that `vs_shoot` would clear at t≈2.6 failed because n=45 → n=180
  moved its point estimate **−15.9**. Predicting only the interval assumes the
  small-n estimate was unbiased, which is the same error as trusting the
  comparator, one level up.
- ⚠ **A sign count does not always discriminate — check what it would be under
  the effect you are claiming.** At a per-scenario sd of ~85 a *true* +9.5
  effect predicts 24.4/45 and a true +15.6 predicts 25.9/45; observed 22 and 25.
  The standing "quote a t AND a sign count" rule assumes the count is
  informative, and at this noise level it is not.
- ⚠ **A TUNING SWEEP NEEDS THE SAME n DISCIPLINE AS THE ARM IT FEEDS.** A
  held-out sweep at n=90 x 3 seeds put `min_stack` 4 → 2 at **+5.2, 3/3,
  t≈2.9** on `vs_shoot` and slightly *negative* on `refereed`; confirmed at
  n=180 x 6 it is **+0.51 (t=0.45)** and **+1.51 (t=2.40)** — the tuning
  transferred in **neither magnitude nor location**. Tune-then-confirm on a
  disjoint seed band is what caught it (bands: evaluation 700000+, in-run
  eval 500000+, baselines 10000+, clone 800000+, tuning 900000+), and
  adopting a threshold from a sweep alone would have put a false positive on
  the record. See [the report](reports/2026-09-06-the-tuning-band-picked-the-wrong-cells.md).
- **n=45 cannot resolve this game.** Per-scenario sd is 81–89, so SE ≈ 89/√n:
  ±12.7 at n=45, ±6.3 at n=180, ±3.2 at n=720. Margins of 5–15 vp — which is
  every arm difference ever measured here — need **n ≥ 180 on both sides**.
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
- ⚠ **Training is NOT bit-reproducible, and this line used to say it was.**
  Measured 2026-09-04, two independent pairs of 2-epoch runs at identical code,
  seed, config and flags: on `configs/golden/25v25_shooting_opponent.yaml`
  **110 of 222** tensors came back bit-identical with a max difference of one
  ULP (1.19e-07), but on `configs/experiments/25v25_maps_melee_approach.yaml`
  **0 of 222** did, at a mean relative difference of **0.0064**. The rollout
  envs *are* seeded (`ROLLOUT_SEED_BASE + env_idx` — note that base does **not**
  depend on `--seed`), so this is GPU float nondeterminism amplified
  chaotically: a last-bit change flips a sampled action and the episode
  diverges. **Pairing two arms by seed still holds at INITIALISATION**, which is
  where its measured value came from — but the trajectories are not shared, so
  every paired difference here carries a rerun-noise term **nobody has
  measured**. Re-running a control is therefore legitimate and sometimes
  necessary; the old rule ("never retrain a control that already exists at the
  same epoch budget") is unsafe on map-pool configs. It also makes a
  bit-identical no-op digest impossible for any training-loop change — such a
  flag has to be argued from static reading instead (nothing constructed, no
  stream drawn, identical expression), which is the standard `--self-play`
  already meets.

### PPO spends the decode's headroom — §48's open question, answered

Measured 2026-09-04, **no GPU**, six paired seeds, n=45,
[report](reports/2026-09-04-ppo-spends-the-decodes-headroom.md).

- **PPO rolls out at `K=1` undecoded and is scored at `K=3` + charge decode, and
  the two objectives oppose each other.** On the refereed melee cell, PPO from
  the §46 clone gained **+19.72 vp (6/6 seeds, t=8.2)** in the regime it trains
  in and lost **14.92** in the regime it is scored in.
- ⚠ **Decode headroom is the quantity being spent: +74.87 vp for the clone,
  +40.23 after PPO.** It bought 19.7 vp of unaided skill with 34.6 vp of
  headroom. Unaided coherency rose 0.754 → 0.818 — real skill — but **both
  policies reach 0.96 once decoded**, so what it learned the decode already
  supplied.
- ⚠ **The lever is the JOINT COHERENT decode, not the charge decode** —
  pre-registered and falsified: Δ is **+18.22** at `K=1 cd=1` against a ≤+5.0
  bound. And the joint decode **cannot** be moved into training (−51.8 from
  scratch, −43.7 warm-started, scored decoded both times).
- **THE RULE: measure a policy in the regime it is TRAINED in, not only the one
  it is scored in, whenever a play-time decode stands between the two.** Its
  corollary — **a play-time decode makes the corresponding training-time skill
  worthless** — applies to all three decodes on file (formation, surplus
  reallocation, charge).
- A KL anchor to the warm-start weights ships as `--kl-ref-coef` /
  `--kl-ref-target` (adaptive, targets nats of drift; unanchored PPO drifts to
  **2.0–2.6 nats per model**). ⚠ **Its arm is in flight — no result yet.**

### The KL anchor — the best melee policy on file, and the goal is still not met

Measured 2026-09-04, six seeds, verified epoch 300,
[report](reports/2026-09-04-the-anchor-holds.md) · melee-teaching-goal §51.

- ⚠ **SELF-PLAY ALONE IS THE CONTROL, NOT THE TREATMENT.** Unanchored self-play
  from the §46 clones ends at **−27.17** against those clones' −9.07 — it
  destroys them slightly faster than a fixed opponent does.
- **With `--kl-ref-target 0.03` it wins 2 of 4 ladder cells and loses none**
  (`vs_take` +14.40 at 2.14 SE, `vs_deny` +20.90 at 6.95 SE), and the **refereed
  head-to-head moves from the clone's LOST to ahead** (+6.75, 1.11 SE). Against
  its own control, paired: **+29.63 ± 1.74, t=17.0, 3/3**.
- ⚠ **GOAL NOT MET** — it is conjunctive, and **`vs_shoot` ties at −0.15 SE**.
  Every route lands on that bar (clone +58.45, interpolation +56.50, arm +56.03,
  bar +56.6); winning it needs a real +8, not noise reduction. Charging is worth
  **+32.8 vp** there, so the deficit is **not** wasted declarations — that was
  measured and refuted.
- **Mechanism confirmed**: decode headroom **+78.43**, above the clone's +74.87
  (control +49.9, plain PPO +40.23); drift 0.039 against the control's 1.770.
- ⚠ **`require_coherent: false` in training is REJECTED, and it corrects the
  rule above it.** Its *decoded* coherency is only 0.906–0.921 — the decode
  cannot repair a policy that never learned formation, because it picks the most
  probable **legal** combination from each model's top-K and there is none
  there. **A decode substitutes for a skill's EXECUTION, not for the training
  pressure that makes the skill REPRESENTABLE.**

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
- **Self-play (opt-in, off by default):** `--self-play`,
  `--snapshot-every-n-epochs`, `--pool-capacity`, `--pool-anchor`, `--pfsp-mode`
  (`hard` | `even` | `uniform` — **`uniform` is the control**). Off builds no
  scheduler at all, so no stream is drawn and a control run is bit-identical.
  ⚠ **Do not start one on a scenario whose `just measure-seat-parity` gate
  fails** — the learner only ever trains the player seat, so a snapshot on the
  other seat plays a game it never practised. See
  [docs/self-play.md](docs/self-play.md).
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

### Where a finding lands — all four, or it is not landed

This is the step that slips. PR #259 rescued a report stranded on a branch for
nine days, absent from `reports/` and unlisted in the index the whole time.

1. **`reports/<YYYY-MM-DD>-<slug>.md`**, carrying its provenance: date · GPU or
   no-GPU · seeds and seed base · n · config path and whether refereed · decode
   `K` and `verify_moves` · paired or not · the comparator **named by name** ·
   opponent · epoch · code revision · which checkpoint · coherency · t AND a sign
   count · Wandb run id.
2. **A one-row index entry in `reports/README.md`**, verdict **bold and first**.
   ⚠ **The index row is the edited surface**: on a retraction the report body
   gains an appended `## ⚠ RETRACTION` and keeps its original text, while the row
   is rewritten so the top-level view is never stale.
3. **A distilled standing rule in `CLAUDE.md`** — § How to measure here, § What
   voids a number, § Settled — do not re-run, or a new section. If no rule
   changed, write that explicitly rather than skipping the step.
4. **The live doc** — `docs/play-doctrine-findings.md` if a `D-NN` was priced (one
   additive entry, never rewriting the claim), `configs/README.md` if a config was
   added or retired, `docs/rules/implementation-status.md` if a gap-map row moved.

Then set the `outcome:` label and remove `needs:writeup` (see § Tracking work).

⚠ `kind:bug` and `kind:build` do **not** use this list. Theirs is shorter: a test
pinning the behaviour, a § What voids a number bullet if it voids, a gap-map row,
and goldens byte-identical or deliberately regenerated. Ticking boxes that do not
apply is how the real list stops being read.

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
- **The trunk size is a parameter now, and the default is unchanged.** `train()` takes `--n-layers` / `--embedding-size`, threaded through `PPO_Transformer.from_env` to `TransformerNetwork.from_spec`. Omitting them is bit-identical to before they existed. ⚠ **They change the network**: a checkpoint trained at another size will not load into a default run and its scores are comparable to nothing here, which is why the flags warn. They exist so the test suite can stop building ~12.7M parameters on a 2-model 20x20 board — the five slowest tests were all trunk-bound. `tests/test_network_size.py` pins the shipped 8/8/256 so it cannot drift.
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
- **2026-09-06 — a fully hidden model no longer denies its whole unit cover** (#289).
  `_cover_mask` tested `visibility == COVER`, so a member terrain blocked
  *completely* — the best-protected model in the unit — stripped the unit's
  cover, against `docs/rules/13-terrain.md`'s "not fully visible". Now any
  blockage counts (`!= CLEAR`). Spurious denial measured at **2.4 / 5.2 /
  9.6pp** of declared (attacker, unit) pairs on `25v25_maps_two_mode` /
  `25v25_shooting_opponent` / `25v25_cover_control`; the scripted bar moved
  **+1.9 / +2.1 / +3.0 vp** at n=100 identical seeds (`squad_march_take`, seeds
  700000+, same direction 3/3 — small against per-episode sd, so treat pre-fix
  figures on terrain configs as suspect within ~3 vp rather than wrong).
  Goldens: the `25v25_single_phase` reward+observation pair regenerated
  deliberately; the other four byte-identical, the targeted-change check.
- **2026-09-14 — every per-model training run before this date is void.**
  The stage-1 calibration sweep (`checkpoints/per_model/calibration_stage1/`,
  `VOID.md` there) and the three observe runs trained on one rollout env at
  8–32 rounds per update, and their checkpoints predate the audited set
  network so cannot load on `main`. Void for **regime and code**, not as
  evidence about the architecture. See
  [the report](reports/2026-09-14-one-episode-per-update.md).
- **2026-08-31 — `group_span` rounds UP, so `max_groups` is a real cap.** It
  floored, and an army splitting into more units than the cap has one-hot columns
  had two units share a code (`_group_ids_to_one_hot` clips rather than raising)
  while `unit_count` sized the shooting slice at the true count. **Bit-identical
  wherever the cap divides the army — every golden, evaluation and dev config,
  both goldens verified unchanged.** It changes **seven configs under
  `configs/experiments/`**, all of which were aliasing: the three `30v15` /
  `15v30` asymmetric arms (15 at cap 6: 8 units → 5) and
  `25v25_maps_take_small_units.yaml` (25 at cap 8: 9 units → 7, **both armies**).
  Every figure measured on those seven is void — the split changed, so the units
  changed.

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

- **Distillation at house fidelity CARRIES a decoded policy — §46's contrary
  result is a power artefact.** A clone of the agent played with K=3 +
  reallocation + charge decode reaches **action-match 0.973 / unit-match 0.933**
  and, at n=180 paired, is **not worse than its teacher on any of four cells**.
  §46's "falls below the plain teacher by 3.9" used **120 demonstrations x 8
  epochs** against house fidelity's **1200 x 60**. ⚠ **The standing rule "a
  per-model fit does not inherit a joint property" is now bounded**: it holds
  for an under-powered fit, not for a properly-sized one. That makes the
  **policy-improvement loop** (decode to improve, distil to project back into
  the weights, repeat) viable here — the step believed blocked is not. Whether
  iterating it compounds is **untested**. See
  [the report](reports/2026-09-06-the-clone-carries-the-decoded-policy.md).
- ⚠ **Difference against the SAME SEED SET, never against a published row.**
  The clone (3 seeds) against the agent's published 6-seed row read +12.8 where
  the matched-subset gain is +6.9 — a doubling manufactured purely by comparing
  different seed counts.

- **The decode stack is at its ceiling for the melee ladder.** Three knobs
  measured at n=180, six seeds, paired, pre-registered: `decode_stay` is a null
  (**+0.52 ± 0.31**), **iterating the reallocation is NEGATIVE** (−1.28 ± 0.76,
  and −4.33 / t=−3.17 on `vs_deny`), and `min_stack` 4 → 2 buys **+5.2 on
  `vs_shoot` and ~0 on `refereed`**. ⚠ Do not re-run iteration — the author
  predicted +3 to +8, wrote the code, and it lost, with a held-out tuning band
  agreeing independently. See
  [the report](reports/2026-09-06-three-decode-knobs-and-none-of-them-pays.md).

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
