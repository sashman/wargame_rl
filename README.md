# Wargame RL

Reinforcement learning model for playing table top wargames.

## Documentation

- [Goals & Roadmap](docs/goals-and-roadmap.md) — Project vision, current status, and phased development plan
- [Tabletop Rules Reference](docs/tabletop-rules-reference.md) — Condensed reference of the tabletop wargame mechanics we're modelling
- [Movement System](docs/movement.md) — How polar coordinate movement works (action encoding, direction, speed, configuration)
- [DDD in wargame/envs](docs/ddd-envs.md) — Domain-driven design motivation and how to extend the environment
- [Self-Play + Elo](docs/self-play-elo.md) — PPO self-play schedule, snapshot pool, and Elo workflow

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
- 🐳 Docker support with multi-stage builds and distroless images
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

You can configure Ruff by editing the `.ruff.toml` file. It is currently set to the default configuration.

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

Default algorithm is PPO:
```bash
just train examples/env_config/example.yaml
```

Train with DQN instead:
```bash
just train examples/env_config/example.yaml dqn
```

Train with DQN and a specific network type:
```bash
just train examples/env_config/example.yaml dqn transformer
```

Resume full training state (model + optimizer + epoch/step) from an existing checkpoint:
```bash
uv run train.py --env-config-path examples/env_config/ci_smoke.yaml --algorithm ppo --resume-ckpt-path checkpoints/<run>/last.ckpt
```

Warm start from checkpoint weights only (fresh optimizer and training counters):
```bash
uv run train.py --env-config-path examples/env_config/ci_smoke.yaml --algorithm dqn --warm-start-ckpt-path checkpoints/<run>/last.ckpt
```

#### Running a simulation

Latest checkpoint, will find the last checkpoint file and its related env config:
```bash
just simulate-latest
```


Specific checkpoint:
```bash
just simulate checkpoints/policy-dqn-env-v2-2025-10-24-22-50-54/last.ckpt checkpoints/policy-dqn-env-v2-2025-10-24-22-50-54/env_config.yaml
```

#### Elo evaluation

```bash
just evaluate-elo checkpoints/<run>/last.ckpt checkpoints/<run>/env_config.yaml 10
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

The template includes two Github Actions workflows.

The first one runs tests and linters on every push on the main and dev branches. You can find the workflow file in `.github/workflows/main-list-test.yml`.

The second one is triggered on every tag push and can also be triggered manually. It builds the distribution and uploads it to PyPI. You can find the workflow file in `.github/workflows/publish.yaml`.
