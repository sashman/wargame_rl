# Technology Stack

**Analysis Date:** 2026-04-02

## Languages

**Primary:**

- Python 3.12+ — entire application (`pyproject.toml` `requires-python = ">=3.12"`). Local dev recipes target Python 3.13 via `Justfile` (`uv venv --python 3.13`).

**Secondary:**

- YAML — environment and scenario configuration loaded with Pydantic models (`examples/env_config/*.yaml`, parsed in `train.py` and `simulate.py` via `pydantic_yaml.parse_yaml_raw_as`).

## Runtime

**Environment:**

- CPython (3.12 in `Dockerfile` base image `python:3.12-slim-bookworm`; 3.13 common for dev per `Justfile`).

**Package Manager:**

- UV — dependency sync, venv, and `uv run` for CLI entry points (`Justfile`, `Dockerfile` builder stage `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`).

- Lockfile: `uv.lock` present at repo root; `uv sync --frozen` used in `Dockerfile` for reproducible installs.

## Frameworks

**Core:**

- Gymnasium 1.x — RL environment API; custom env registered in `wargame_rl/__init__.py` as `gymnasium_env/Wargame-v0` → `wargame_rl.wargame.envs.wargame:WargameEnv`.

- PyTorch — tensors, neural networks (`torch` in `pyproject.toml`; implementations in `wargame_rl/wargame/model/net.py`, algorithm modules under `wargame_rl/wargame/model/dqn/` and `ppo/`).

- PyTorch Lightning 2.5+ — training loops, callbacks, loggers (`pytorch-lightning`); modules such as `wargame_rl/wargame/model/dqn/lightning.py`, `wargame_rl/wargame/model/ppo/lightning.py`.

**CLI:**

- Typer — `train.py`, `simulate.py` application CLIs.

**Config / validation:**

- Pydantic (pulled via `pydantic-yaml`) — `WargameEnvConfig` and related models under `wargame_rl/wargame/envs/types/` and model configs in `wargame_rl/wargame/model/*/config.py`.

**Rendering:**

- Pygame — human rendering path (`wargame_rl/wargame/envs/renders/`).

**Testing:**

- Pytest 8.x with `pytest-cov` and `pytest-rerunfailures` — `Justfile` recipe `test`, tests under `tests/`.

**Build / Dev:**

- Just — task runner (`Justfile`).

- Pre-commit — hooks in `.pre-commit-config.yaml` (YAML checks, autoflake, isort, ruff, mypy mirror).

- Ruff — format and lint; config `ruff.toml` (line length 88, double quotes, `target-version = "py39"` for lint baseline).

- Mypy — strict-style settings in `pyproject.toml` `[tool.mypy]`; `Justfile` `lint` runs mypy over `wargame_rl/` and `tests/`.

- isort — Black profile via pre-commit.

- autoflake — unused import removal via pre-commit.

**Profiling / notebooks:**

- pyinstrument — HTML profiling wrapper in `Justfile` `profile` recipe.

- Jupyter — declared in `pyproject.toml` dependencies for notebook workflows.

**Visualization (offline):**

- Plotly (pinned `5.24.1`), pandas, matplotlib — plotting utilities e.g. `wargame_rl/plotting/training.py`.

**Video:**

- imageio with ffmpeg extra — episode recording to MP4 in `wargame_rl/wargame/model/common/record_episode_callback.py`.

## Key Dependencies

**Critical:**

- `gymnasium` — environment contract and registration.

- `torch` + `pytorch-lightning` — training and inference for DQN/PPO.

- `pydantic-yaml` — typed load/save of YAML configs.

- `wandb` — default experiment logging (optional disable via CLI); wiring in `wargame_rl/wargame/model/common/wandb.py`.

- `loguru` — structured logging (`main.py`, widespread in package).

**Infrastructure / UX:**

- `tqdm` — progress display where used in training pipelines.

- `numpy` — numerical arrays in env and model code.

## Configuration

**Environment:**

- Optional `CUDA_VISIBLE_DEVICES` — documented in project rules / `train.py` comment for CPU-only runs when CUDA is broken.

- `PATH_DATASETS` — optional override for dataset root; default `./datasets` in `wargame_rl/wargame/model/common/dataset.py` via `os.environ.get`.

- Wandb uses standard SDK auth (API key / login); not read from repo files. See `INTEGRATIONS.md`.

- `SDL_VIDEODRIVER=dummy` set during async episode recording in `wargame_rl/wargame/model/common/record_episode_callback.py` to avoid display requirement.

**Build:**

- `pyproject.toml` — project metadata, runtime and dev dependency groups, mypy settings.

- `ruff.toml` — Ruff lint/format.

- `.pre-commit-config.yaml` — git hook toolchain versions and args.

- `Dockerfile` — multistage UV build; final image runs `main.py` (paths in Dockerfile should be verified against actual package layout).

## Platform Requirements

**Development:**

- UV-installed venv, Python ≥3.12; optional GPU with working PyTorch CUDA build.

- FFmpeg available on PATH when using imageio video encoding (train with recording).

**Production:**

- Not a deployed web service; primary “production” shape is local/CI training and simulation. Container image defined in `Dockerfile` for runnable environment (example CMD `python main.py --number 10`).

---

*Stack analysis: 2026-04-02*
