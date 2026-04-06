# Technology Stack

**Analysis Date:** 2026-04-06

## Languages

**Primary:**
- Python 3.13 — all application code, training scripts, environment, tests
  - `pyproject.toml` specifies `requires-python = ">=3.12"`
  - `.python-version` pins `3.12` (used by UV for env creation)
  - `Justfile` `setup` recipe creates venv with `uv venv --python 3.13`

**Secondary:**
- YAML — environment configuration (`examples/env_config/*.yaml`), pre-commit config, CI
- Markdown — documentation (`docs/`)

## Runtime

**Environment:**
- CPython 3.13 (development)
- Docker: Python 3.12-slim-bookworm (production image — `Dockerfile`)

**Package Manager:**
- UV (astral-sh) — fast Python package manager and virtualenv tool
- Lockfile: `uv.lock` (present, committed)
- Cache: `.uv_cache/` (local, gitignored)

**Task Runner:**
- Just (`Justfile`) — all dev commands run through `just` recipes
- Key recipes: `setup`, `dev-sync`, `format`, `lint`, `test`, `validate`, `train`, `simulate-latest`, `profile`, `ship`

## Frameworks

**Core:**
- Gymnasium 1.x (`>=1.0.0,<2.0.0`) — RL environment interface; custom env `WargameEnv` registered as `gymnasium_env/Wargame-v0` in `wargame_rl/__init__.py`
- PyTorch — neural network definition, training, and inference
- PyTorch Lightning (`>=2.5.2`) — training loop orchestration; `DQNLightning` and `PPOLightning` modules in `wargame_rl/wargame/model/dqn/lightning.py` and `wargame_rl/wargame/model/ppo/lightning.py`; `WargameLightningBase` shared base in `wargame_rl/wargame/model/common/lightning_base.py`

**Data / Config:**
- Pydantic — all config and type models (`WargameEnvConfig`, `DQNConfig`, `PPOConfig`, `TransformerConfig`, `RewardPhaseConfig`, `ModelConfig`, `ObjectiveConfig`, etc.)
- pydantic-yaml (`>=1.6.0`) — YAML ↔ Pydantic deserialization for env configs (used via `parse_yaml_raw_as` in `train.py` and `simulate.py`)

**CLI:**
- Typer (`>=0.16.0`) — CLI for `train.py` and `simulate.py`; both use `typer.Typer(pretty_exceptions_enable=False)`

**Logging:**
- Loguru (`>=0.7.3`) — structured logging throughout (configured in `main.py`)

**Rendering:**
- Pygame — human-mode rendering of the wargame board via `HumanRender` in `wargame_rl/wargame/envs/renders/human.py`

**Experiment Tracking:**
- Wandb (`>=0.21.0`) — experiment tracking, video recording, model logging; init/context manager in `wargame_rl/wargame/model/common/wandb.py`; supports disabled mode with `CSVLogger` fallback

**Testing:**
- Pytest (`>=8.3.4`) — test runner
- pytest-xdist (`>=3.8.0`) — parallel test execution (`-n auto`)
- pytest-cov (`>=7.0.0`) — coverage reporting (term-missing + XML)
- pytest-rerunfailures (`>=14.0`) — flaky test re-runs

**Build/Dev:**
- Ruff (`>=0.9.5`) — linter and formatter (config: `ruff.toml`)
- Mypy (`>=1.18.2`) — strict static type checking (`disallow_untyped_defs`, `no_implicit_optional`, `warn_return_any` in `pyproject.toml` `[tool.mypy]`)
- Pre-commit (`>=4.1.0`) — git hooks (config: `.pre-commit-config.yaml`)
- autoflake (`v2.3.1` via pre-commit) — removes unused imports/variables
- isort (`5.13.2` via pre-commit) — import sorting (Black profile)

**Profiling:**
- pyinstrument (`>=5.1.2`) — CPU profiling for training runs; invoked via `just profile` recipe, generates `profile.html`

**Video/Image:**
- imageio with ffmpeg (`>=2.34.0`) — MP4 episode recording during training; used in `wargame_rl/wargame/model/common/record_episode_callback.py`

## Key Dependencies

**Critical (training cannot run without):**
- `torch` — neural network forward/backward passes, optimizer, checkpoints
- `pytorch-lightning` — training loop, checkpointing, logging callbacks
- `gymnasium` — environment API, action/observation spaces, `env.step()`/`env.reset()`
- `numpy` — array operations throughout observation building, distance computation, action handling
- `pydantic` + `pydantic-yaml` — all configuration models; validation at construction time

**Infrastructure:**
- `wandb` — experiment tracking (can be disabled with `--no-wandb`)
- `tqdm` — progress bars for PPO rollout collection and minibatch updates
- `pygame` — rendering (only used with `render_mode="human"`)
- `imageio[ffmpeg]` — async episode recording as MP4 for wandb uploads
- `typer` — CLI argument parsing for training and simulation entry points

**Dev-only:**
- `ruff`, `mypy`, `pre-commit`, `pytest`, `pytest-xdist`, `pytest-cov`, `pytest-rerunfailures` — formatting, linting, type checking, testing

## Neural Network Architecture

**Transformer (primary, actively developed):**
- NanoGPT-style transformer adapted for RL — `TransformerNetwork` in `wargame_rl/wargame/model/net.py`
- Custom `Block` and `LayerNorm` layers from NanoGPT in `wargame_rl/wargame/model/dqn/layers.py`
- Config: `TransformerConfig` in `wargame_rl/wargame/model/common/config.py` (8 layers, 8 heads, 256 embedding, non-causal attention)
- Shared backbone option for PPO actor-critic via `share_backbone_with()` method

**MLP (legacy):**
- `MLPNetwork` in `wargame_rl/wargame/model/net.py` — 2-layer GELU MLP, 128 hidden dim
- PPO does not support MLP (`raise NotImplementedError` in `train.py`)

**Device handling:**
- Auto-detection: CUDA → MPS → CPU in `wargame_rl/wargame/model/common/device.py`
- Lightning `accelerator="auto"` for trainer

## Configuration

**Environment:**
- YAML config files in `examples/env_config/` deserialized into `WargameEnvConfig` (Pydantic)
- All config fields have defaults — environment works with zero configuration
- Reward curriculum defined as `reward_phases` list with calculators and success criteria (registry pattern with string identifiers)

**Algorithm:**
- `DQNConfig` in `wargame_rl/wargame/model/dqn/config.py`
- `PPOConfig` in `wargame_rl/wargame/model/ppo/config.py`
- `DQNTrainingConfig` / `PPOTrainingConfig` — epoch limits, checkpoint recording frequency
- `TransformerConfig` in `wargame_rl/wargame/model/common/config.py`

**Tooling:**
- `ruff.toml` — line-length 88, indent 4, double quotes, pyflakes + pycodestyle rules
- `pyproject.toml` `[tool.mypy]` — strict: `disallow_untyped_defs`, `no_implicit_optional`, `warn_return_any`
- `.pre-commit-config.yaml` — trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, check-docstring-first, autoflake, isort, ruff-check, ruff-format, mypy

## Platform Requirements

**Development:**
- Python 3.13+ (UV manages venv)
- CUDA-capable GPU recommended (auto-detected; falls back to MPS or CPU)
- UV package manager installed (`just setup` bootstraps everything)
- Just command runner

**Production (Docker):**
- Base image: `python:3.12-slim-bookworm`
- Builder image: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
- No GPU support in current Dockerfile (CPU only)

**Artifacts:**
- Checkpoints: `checkpoints/<run-name>/` (`.ckpt` files, Lightning format)
- Wandb logs: `wandb/` directory (local, gitignored)
- Coverage: `coverage.xml` (generated by `just test`)
- Profile: `profile.html` (generated by `just profile`)

---

*Stack analysis: 2026-04-06*
