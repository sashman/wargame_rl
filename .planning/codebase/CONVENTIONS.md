# Coding Conventions

**Analysis Date:** 2026-04-02

## Naming Patterns

**Files:**

- Python modules and packages use `snake_case` (e.g. `wargame.py`, `phase_manager.py`, `experience_replay.py`).
- Test modules use `test_<area>.py` under `tests/` (e.g. `test_env.py`, `test_reward_phases.py`). Slow or ordering-sensitive suites may use a prefix such as `test_z_e2e_training.py` so default collection order runs faster tests first.

**Functions and methods:**

- Use `snake_case` for functions and methods (e.g. `get_env_config` in `train.py`, `build_calculator` in `wargame_rl/wargame/envs/reward/calculators/registry.py`).

**Variables:**

- Use `snake_case` for locals and instance attributes. Underscore-prefixed dummies are allowed by Ruff (`dummy-variable-rgx` in `ruff.toml`).

**Types and classes:**

- Pydantic models and domain types use `PascalCase` (e.g. `WargameEnvConfig`, `ModelConfig` in `wargame_rl/wargame/envs/types/config.py`).
- Gym env and Lightning modules use `PascalCase` (e.g. `WargameEnv` in `wargame_rl/wargame/envs/wargame.py`, `DQNLightning`, `PPOLightning`).
- Some RL network symbols retain historical mixed style (e.g. `PPO_Transformer`, `RL_Network` in `wargame_rl/wargame/model/net.py` and `wargame_rl/wargame/model/ppo/ppo.py`). Prefer matching existing symbols in the same module when extending code.

**Registry / factory helpers:**

- Use descriptive `build_*` names for YAML-driven factories (e.g. `build_calculator`, `build_criteria`, `build_opponent_policy` under `wargame_rl/wargame/envs/reward/` and `wargame_rl/wargame/envs/opponent/`).

## Code Style

**Formatting:**

- **Ruff format** is the source of truth (`just format` → `uv run ruff format`). Config: `ruff.toml` — line length **88**, **double-quoted** strings, spaces, magic trailing commas preserved, `target-version = "py39"` (project `requires-python` is `>=3.12` in `pyproject.toml`; align new syntax with what Ruff/mypy accept).

**Linting:**

- **Ruff check** with default-ish selection: `select = ["E4", "E7", "E9", "F"]` in `ruff.toml` (`just lint` runs `uv run ruff check --fix` then mypy).
- **Mypy** (strict lean): `pyproject.toml` `[tool.mypy]` — `disallow_untyped_defs = true`, `no_implicit_optional = true`, `warn_return_any = true`, `show_error_codes = true`. CI/local lint uses `uv run mypy --ignore-missing-imports --install-types --non-interactive wargame_rl/ tests/` per `Justfile`.

**Pre-commit** (`.pre-commit-config.yaml`): trailing whitespace, YAML checks, **autoflake** (remove unused imports/variables), **isort** (`--profile black`), **ruff-check** + **ruff-format**, **mypy** with `--ignore-missing-imports`.

## Import Organization

**Order (observed in production code):**

1. `from __future__ import annotations` when used (e.g. `wargame_rl/wargame/envs/wargame.py`, `tests/test_reward_phases.py`).
2. Standard library (`typing`, `os`, `pathlib`, etc.).
3. Third-party (`gymnasium`, `torch`, `pydantic_yaml`, `typer`, `pytest` in tests).
4. First-party: `from wargame_rl....` — absolute imports from the installed package root.

**Path hacks in tests:**

- When importing repo-root modules (`train.py`, `simulate.py`), tests insert `sys.path` and may use `# noqa: E402` after the import (see `tests/test_simulate.py`).

**Barrel / re-exports:**

- `WargameEnv` module exposes `__all__` for backward compatibility (`wargame_rl/wargame/envs/wargame.py`).

## Error Handling

**Patterns:**

- **Pydantic** validates config at parse/load time; model validators in `wargame_rl/wargame/envs/types/config.py` raise `ValueError` with explicit messages for inconsistent placement (x/y rules, mixed explicit vs implicit coordinates).
- **Runtime guards** in env/model code raise `ValueError` for invalid arguments or registry misses (e.g. `wargame_rl/wargame/envs/reward/phase_manager.py`, `wargame_rl/wargame/envs/reward/calculators/registry.py`, `wargame_rl/wargame/model/ppo/lightning.py`).
- **Placement / domain** may raise `RuntimeError` when configuration cannot be satisfied (`wargame_rl/wargame/envs/domain/placement.py`).
- **Domain-specific errors** exist where appropriate (e.g. `GameClockError` in `wargame_rl/wargame/envs/domain/game_clock.py`).
- **CLI / IO**: `FileNotFoundError` for missing env config paths (`train.py` `get_env_config`).
- **External services**: `RuntimeError` when wandb init fails (`wargame_rl/wargame/model/common/wandb.py`).
- **Renderer**: `ValueError` for uninitialized pygame state; `QuitRequested` for user exit (`wargame_rl/wargame/envs/renders/human.py`).

**Prescription:** Prefer explicit exceptions with clear messages; avoid bare `except`. Use the same exception type as sibling code in the module.

## Logging

**Framework:** Loguru (`loguru.logger`) in selected subsystems.

**Patterns:**

- Import `from loguru import logger` and use structured placeholders: `logger.info("...", {})` (see `wargame_rl/wargame/model/common/record_episode_callback.py`, `wargame_rl/wargame/envs/reward/phase_manager.py`).
- Training entrypoints use shared helpers such as `get_logger` from `wargame_rl.wargame.model.common` (`train.py`).

## Comments

**When to comment:**

- Section banners in large test files (`# --- Step tests ---` in `tests/test_env.py`; `# ---------------------------------------------------------------------------` blocks in `tests/test_reward_phases.py`).
- Explain non-obvious **why** (e.g. cache clearing around device selection in tests).

**Docstrings:**

- Public tests often use one-line docstrings describing behavior (`tests/test_env.py`).
- Module docstrings summarize scope (`tests/test_integration.py`, `tests/test_z_e2e_training.py`).
- Typer CLI and shared types use docstrings where they clarify CLI or algorithm choice (`train.py`).

## Function Design

**Size:** Prefer focused functions; env orchestration is split across `wargame_rl/wargame/envs/domain/` and `wargame_rl/wargame/envs/env_components/` rather than one mega-function.

**Parameters:** Pass config and dependencies explicitly (`WargameEnv.__init__(config=..., renderer=...)`). Factory functions take parsed `WargameEnvConfig` or registry keys plus params.

**Return values:** Typed returns everywhere mypy enforces; use `typing.cast` when narrowing PyTorch or fixture types (e.g. `tests/conftest.py` for `PPO_Transformer`).

## Module Design

**Exports:** Use `__all__` where backward-compatible re-exports matter (`wargame_rl/wargame/envs/wargame.py`).

**Configuration:** Single source in Pydantic models loaded from YAML via `pydantic_yaml` (`parse_yaml_raw_as` in `train.py`; `WargameEnvConfig` in `wargame_rl/wargame/envs/types/`).

**Gym registration:** Package init registers the env with Gymnasium (`wargame_rl/__init__.py` — follow the same pattern when adding new env ids).

---

*Convention analysis: 2026-04-02*
