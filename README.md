# Wargame RL

Reinforcement learning model for playing table top wargames.

## Where we are

*Written for someone who knows the game but not the machine learning.*

The scenario: **25 models a side in five units of five**, both players carrying
the identical profile, over five or six objectives on real table layouts. **All
combat is shooting, to a maximum of 12 inches.** Every player — hand-written or
trained — is scored on **nine tables it has never seen**, reported as the average
victory-point margin, ours minus theirs.

### What a game looks like

Four of the nine held-out tables, one game each, played by the trained model
against `squad_march_take`. The tinted areas are each layout's own deployment
zones, and the four below are four of the six shapes the tables use.

| | |
|---|---|
| ![Table 45, long edges](docs/images/agent-table-45-long-edges.gif) | ![Table 30, opposed quadrants](docs/images/agent-table-30-opposed-quadrants.gif) |
| **table 45 — long edges.** Armies 20 inches apart across the *short* axis, at a 12-inch weapon range: a different game from turn one. Ends **120–60**. | **table 30 — opposed quadrants.** The zone boundary is an arc, 61 vertices. Ends **235–175**, 12 models left. |
| ![Table 35, stepped bands](docs/images/agent-table-35-stepped-bands.gif) | ![Table 15, diagonal halves](docs/images/agent-table-15-diagonal-halves.gif) |
| **table 35 — stepped bands.** A **loss**, 150–205, down to 6 models. | **table 15 — diagonal halves.** The zone is a triangle cut by the board diagonal. Ends **105–70**. |

*Each is the **median of eleven** games on that table — a rule chosen so the
picture is not the model's best game. Read them as one game each, not as a
score: over thirty games this model averages +21, +46, **+12** and +56 on these
four tables respectively. Table 35 is the honest one — its median game is a loss
even though its average is positive, so a single game there is a poor guide
either way.*

### The opponents

All five are hand-written. The middle three form a ladder — each adds exactly
one behaviour to the one above it — while the first and last play differently:

| opponent | what it does |
|---|---|
| `advance_and_shoot` | Every model steers for *its own* nearest objective, pulled slightly toward the group's centre, and fires at a target drawn at random from those in range. No squads and no allocation at all. **Weakest of the five**, and the only one that is not a squad player. |
| `squad_march_shoot` | Each squad marches to one objective as a body and holds it, firing at the nearest target in range. |
| `squad_march_deny` | The same, but squads with nothing left to hold go and **contest what the other side holds**. |
| `squad_march_take` | The same, but those spare squads instead go for the **most weakly held** objective — the cheapest to flip. **Strongest of the five**. |
| `contest_and_spread` | Allocates squads against **where the enemy has actually deployed** rather than by a fixed rule, and spreads its fire instead of always shooting the nearest. |

### What it scores

Nine held-out tables, **six** independently trained models, 20 rounds a game.
Trained by `just train-coherency-baseline` on
`configs/golden/25v25_maps_two_mode.yaml` and played with the top-3 joint decode
— **both matter**, and a figure quoted without them is not comparable. (The
figures published here before 2026-08-20 came from a different training config
and understated the model by about 16 points.)
Figures are the **average victory-point margin — ours minus theirs**, so 0 is a
dead heat and +20 means winning by twenty points a game.

Re-measured **2026-08-21** on the generated evaluation tables, with the
hand-written players re-measured against **each** opponent — swapping the
opponent changes the game, and voids every baseline on that config:

| opponent it faces | trained model<br>*avg VP margin* | best hand-written player<br>*avg VP margin* | |
|---|---|---|---|
| `squad_march_deny` | **+26.4** | −8.9 | **ahead by 35.4** |
| `squad_march_take` — the strongest | **+25.1** | −1.1 | **ahead by 26.1** |
| `squad_march_shoot` | **+39.2** | +23.0 | ahead by 16.2 |
| `contest_and_spread` | +20.8 | **+30.2** | behind by 9.5 |
| `advance_and_shoot` — the weakest | +61.4 | **+137.2** | **behind by 75.9** |

**Ahead of the best hand-written play in three matchups of five, and behind in
two.** Two of the leads are decisive — paired per table, t = 3.3 and 4.5, the
model ahead on 8 and on **all 9** tables. The lead against `squad_march_shoot`
is **not settled** (t = 1.6, 7 of 9), and neither is the `contest_and_spread`
loss (t = −1.2, 4 of 9).

**The last row is settled, and it is the important one.** Behind by 76 points on
**every one of the nine tables**. It is not a different failure from the
`contest_and_spread` loss — it is the same one, larger:

> **The model wins by conceding, not by taking.** It parks on about two
> objectives and holds them, against every opponent alike. When the opponent
> would otherwise score heavily, not conceding is worth 60–96 points a game and
> the model wins comfortably. When the opponent is weak, *everyone* holds it to
> the same score, that advantage is worth nothing, and all that remains is
> ground taken — where the hand-written players reach three or four objectives
> and the model still sits on two.

The two halves separate cleanly: across all five matchups the model scores 42 to
71 fewer points than the best hand-written player — flat, whoever it plays —
while its defensive advantage falls from +96 to **zero**. The margin tracks how
much the opponent would have scored, at a correlation of **0.99**.

⚠ **An earlier version of this table showed `contest_and_spread` as a 9.4-point
lead, measured on the hand-traced tables.** On the generated ones the matchup
reverses. Do not treat the loss as retired.

It also keeps its squads together better than any hand-written player in every
matchup — unit coherency **0.938–0.955**, against a scripted band of
**0.867–0.908**. That is the game's own formation rule, and it is measured on
what the model *intended*, not on what a referee corrected. Formation is the one
thing that holds in **all five** matchups, including the two it loses.

⚠ Read those numbers *down a column*, never across. A player's absolute score
mostly measures how weak its opponent is — the trained model scores *higher*
against the weaker opponents while doing relatively *worse*. The only meaningful
comparison is the two figures on the same row, which faced the same opponent.

Full detail: [reports/](reports/README.md), most recently
[the chain tail and the frozen army](reports/2026-08-18-the-chain-tail-and-the-frozen-army.md).

## Documentation

- [Goals & Roadmap](docs/goals-and-roadmap.md) — Project vision, current status, and phased development plan
- [Rules Reference](docs/rules/README.md) — The full rules specification for the game we're modelling, plus a per-rule map of what the environment implements
- [Movement System](docs/movement.md) — How polar coordinate movement works (action encoding, direction, speed, configuration)
- [DDD in wargame/envs](docs/ddd-envs.md) — Domain-driven design motivation and how to extend the environment
- [Metrics Reference](docs/metrics.md) — Semantics of every Wandb metric, plus an evaluation procedure for assessing runs

## How to add a feature to the environement?
1. Update types, states and space
2. Update the state_to_tensor
3. Update the reward
4. Be sure that pytest is working

# Development

## 🎯 Core Features

### Development Tools

- 📦 UV - Ultra-fast Python package manager
- 🚀 Just - Modern command runner with powerful features
- 💅 Ruff - Lightning-fast linter and formatter
- 🔍 Mypy - Static type checker
- 🧪 Pytest - Testing framework with fixtures and plugins
- 🧾 Loguru - Python logging made simple

### Infrastructure

- 🛫 Pre-commit hooks
- 🐳 Docker support with multi-stage builds and slim images
- 🔄 GitHub Actions CI/CD pipeline


## Usage

The template is based on [UV](https://docs.astral.sh/) as package manager and [Just](https://github.com/casey/just) as command runner. You need to have both installed in your system to use this template.

To get started, install `just`, you can run `brew install just`, then just run
```bash
just setup
```

Here are other useful `just` command setup for this repository...
```bash
just dev-sync
```

to create a virtual environment and install all the dependencies, including the development ones. If instead you want to build a "production-like" environment, you can run

```bash
just prod-sync
```

In both cases, all extra dependencies will be installed (notice that the current pyproject.toml file has no extra dependencies).

You also need to install the pre-commit hooks with:

```bash
just install-hooks
```

### Formatting, Linting and Testing

You can configure Ruff by editing the `ruff.toml` file (line length 88, double quotes).

Format your code:

```bash
just format
```

Run linters (ruff and mypy):

```bash
just lint
```

Run tests:

```bash
just test
```

Do all of the above:

```bash
just validate
```

### Executing

#### Training

PPO on a transformer policy is the only configuration:
```bash
just train configs/golden/25v25_shooting_opponent.yaml
```

Cap the run at a number of epochs:
```bash
just train configs/golden/25v25_shooting_opponent.yaml 800
```

Resume full training state (model + optimizer + epoch/step) from an existing checkpoint:
```bash
uv run train.py --env-config-path configs/dev/ci_smoke.yaml --resume-ckpt-path checkpoints/<run>/last.ckpt
```

Warm start from checkpoint weights only (fresh optimizer and training counters):
```bash
uv run train.py --env-config-path configs/dev/ci_smoke.yaml --warm-start-ckpt-path checkpoints/<run>/last.ckpt
```

#### Running a simulation

Latest checkpoint, will find the last checkpoint file and its related env config:
```bash
just simulate-latest
```


Specific checkpoint:
```bash
just simulate checkpoints/ppo-transformer-25v25-2026-08-09-10-30-00/last.ckpt checkpoints/ppo-transformer-25v25-2026-08-09-10-30-00/env_config.yaml
```

#### Testing Env

You can run the environment in isolation while random actions are fed to the agent.

```bash
just test-env
```

### Docker

The template includes a multi stage Dockerfile, which produces an image with the code and the dependencies installed. You can build the image with:

```bash
just dockerize
```

### Github Actions

There is one Github Actions workflow: it runs formatting, linting and tests on every push to `main` and on every pull request. You can find the workflow file in `.github/workflows/main-validate.yaml`.
