# Wargame RL

Reinforcement learning model for playing table top wargames.

## Where we are

*Written for someone who knows the game but not the machine learning.*

**The aim.** Teach a computer to play a tabletop wargame well by *playing* it —
not by being handed tactics. It deploys, moves, shoots and holds objectives, and
the only thing it is told is the score at the end. Anything that looks like
tactics has to be discovered.

**How we judge it.** Every player, human-written or learned, is scored the same
way: play 100 games on each of **nine tables the learner has never seen**, and
report the average victory-point margin — our score minus theirs. Nine unseen
tables matter more than any number of games on familiar ones; a player that only
performs on ground it has practised on has memorised the ground, not the game.

The scale, on those nine unseen tables:

| player | margin |
|---|---|
| deploys and never moves | −88.9 |
| moves at random | −26.4 |
| **walks squads onto objectives** | **+79.4** |
| walks onto objectives *and shoots* | +111.8 |
| **`squad_march_take` — best we have** | **+116.7** |
| the trained AI, playing on its own | **+75.9 to +84.7** |
| the trained AI, taught by copying | +113.6 |

**Where that leaves us.** A few dozen lines of hand-written orders still play
better than anything the computer has worked out for itself. The AI playing on
its own lands around +80 — respectable, comfortably better than random, and well
short of a competent player.

**The one real tactical discovery so far**, and it came from reading the mission
rather than the machine: scoring is capped at 15 points a round, which is three
objectives' worth. The tables carry five or six. **So a fourth objective you hold
scores you nothing** — its entire value is that the opponent does not have it.
Above three objectives this stops being a scoring race and becomes a game of
denial, and the whole gap between a decent player and a good one lives there.

That produced the current best player, `squad_march_take`, and one rule of thumb
worth having at the table: **holding an objective denies it far more reliably
than contesting one.** A raid that arrives but does not outnumber the defender
changes nothing at all — we measured a raiding player whose raids never once
flipped an objective. Walking onto weakly-held ground flips it on arrival, and
then denies for the rest of the game.

**The honest part.** The best *learned* player (+113.6) got there by **watching
the hand-written one play 1200 games and copying its decisions**, matching them
98% of the time. It is a good player, and it is a copy. Every attempt to let it
improve on its teacher from there made it worse — that is tested thoroughly and
written up, not glossed over.

**Why we think that is**, and it is the interesting problem rather than an
excuse: look at the scale above. "Walk your squads onto objectives and shoot"
already gets **96%** of everything available. All the cleverness — the cap, the
denial, the best player we have — competes over the last 4%. Against an opponent
that never changes its plan, with identical armies on both sides and one weapon
range, there may simply be very little tactical depth for a learner to find.

**So the next move is to make the game harder rather than the learner smarter**:
a stronger opponent first, then genuinely different armies. If a two-rule
heuristic stops being nearly optimal, there is something to learn.

Full detail: [reports/](reports/README.md), most recently
[the cap makes it a denial game](reports/2026-08-16-the-cap-makes-it-a-denial-game.md).

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
