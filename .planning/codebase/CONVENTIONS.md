# Coding Conventions

**Analysis Date:** 2026-04-06

## Naming Patterns

**Files:**

- Python modules and packages use `snake_case` (e.g. `wargame.py`, `phase_manager.py`, `experience_replay.py`, `battle_factory.py`).
- Test modules use `test_<area>.py` under `tests/` (e.g. `test_env.py`, `test_reward_phases.py`, `test_mission_vp.py`). Slow or ordering-sensitive suites use a `test_z_` prefix so default collection order runs faster tests first (e.g. `tests/test_z_e2e_training.py`).

**Functions and methods:**

- Use `snake_case` for all functions and methods (e.g. `get_env_config` in `train.py`, `build_calculator` in `wargame_rl/wargame/envs/reward/calculators/registry.py`, `compute_distances` in `wargame_rl/wargame/envs/env_components/distance_cache.py`).
- Private helpers use a leading underscore (e.g. `_bresenham_line` in `wargame_rl/wargame/envs/domain/los.py`, `_models_to_obs` in `wargame_rl/wargame/envs/env_components/observation_builder.py`, `_build_models` in `wargame_rl/wargame/envs/domain/battle_factory.py`).
- Registry / factory helpers use descriptive `build_*` names for YAML-driven factories (e.g. `build_calculator`, `build_criteria`, `build_opponent_policy`, `build_vp_calculator`).

**Variables:**

- Use `snake_case` for locals and instance attributes.
- Underscore-prefixed dummies are allowed by Ruff (`dummy-variable-rgx` in `ruff.toml`).
- Private instance attributes use single leading underscore (e.g. `self._current_idx`, `self._battle`, `self._registry`).

**Types and classes:**

- Pydantic models use `PascalCase` (e.g. `WargameEnvConfig`, `ModelConfig`, `WeaponProfile`, `RewardPhaseConfig` in `wargame_rl/wargame/envs/types/config.py` and `wargame_rl/wargame/envs/reward/phase.py`).
- Domain entities use `PascalCase` (e.g. `WargameModel`, `WargameObjective` in `wargame_rl/wargame/envs/domain/entities.py`, `Battle` in `wargame_rl/wargame/envs/domain/battle.py`).
- Value objects use `PascalCase` frozen dataclasses (e.g. `BoardDimensions`, `DeploymentZone` in `wargame_rl/wargame/envs/domain/value_objects.py`).
- Protocols use `PascalCase` (e.g. `BattleView` in `wargame_rl/wargame/envs/domain/battle_view.py`).
- Enums use `PascalCase` class name with `snake_case` members (e.g. `BattlePhase.movement`, `PlayerSide.player_1`, `TurnOrder.random` in `wargame_rl/wargame/envs/types/game_timing.py`).
- Some RL network symbols retain historical mixed style: `PPO_Transformer`, `RL_Network` in `wargame_rl/wargame/model/net.py` and `wargame_rl/wargame/model/ppo/ppo.py`. Match existing symbols when extending these modules.
- Gym env and Lightning modules use `PascalCase` (e.g. `WargameEnv` in `wargame_rl/wargame/envs/wargame.py`, `DQNLightning` in `wargame_rl/wargame/model/dqn/lightning.py`, `PPOLightning` in `wargame_rl/wargame/model/ppo/lightning.py`).

**Constants:**

- Module-level constants use `UPPER_SNAKE_CASE` (e.g. `STAY_ACTION = 0` in `wargame_rl/wargame/envs/env_components/actions.py`, `ALL_BATTLE_PHASES` in the same file, `CALCULATOR_REGISTRY` in `wargame_rl/wargame/envs/reward/calculators/registry.py`, `NON_MOVEMENT_PHASES` in `wargame_rl/wargame/envs/types/game_timing.py`).
- Exception: module-level private registries use lower `_REGISTRY` (e.g. `wargame_rl/wargame/envs/opponent/registry.py`).

**Scripted behaviours:**

- Use descriptive `Scripted` prefix (e.g. `ScriptedAdvanceToObjectivePolicy` in `wargame_rl/wargame/envs/opponent/scripted_advance_to_objective_policy.py`).

## Code Style

**Formatting:**

- **Ruff format** is the source of truth (`just format` → `uv run ruff format`).
- Config in `ruff.toml`: line length **88**, indent width **4**, **double-quoted** strings, space indentation, magic trailing commas preserved.
- Target version: `py39` in `ruff.toml` (project `requires-python >= 3.12` in `pyproject.toml`; Ruff/mypy accept modern syntax).

**Linting:**

- **Ruff check**: `select = ["E4", "E7", "E9", "F"]` in `ruff.toml`. Run via `just lint` → `uv run ruff check --fix`.
- **Mypy** (strict lean): `pyproject.toml` `[tool.mypy]` — `disallow_untyped_defs = true`, `no_implicit_optional = true`, `warn_return_any = true`, `show_error_codes = true`. CI/local: `uv run mypy --ignore-missing-imports --install-types --non-interactive wargame_rl/ tests/`.

**Pre-commit** (`.pre-commit-config.yaml`): trailing whitespace, end-of-file fixer, YAML checks, check-added-large-files, check-docstring-first, **autoflake** (remove unused imports/variables), **isort** (`--profile black`), **ruff-check** + **ruff-format**, **mypy** with `--ignore-missing-imports`.

**Validation pipeline:**
- `just validate` = `just format` + `just lint` + `just test`
- Quick iteration: `just format && just lint`

## Import Organization

**Order (enforced by isort `--profile black` + autoflake):**

1. `from __future__ import annotations` (used pervasively, e.g. `wargame_rl/wargame/envs/wargame.py`, `wargame_rl/wargame/envs/domain/battle.py`, `wargame_rl/wargame/envs/reward/calculators/base.py`).
2. Standard library (`os`, `math`, `enum`, `abc`, `dataclasses`, `functools`, `collections.abc`, `typing`, `contextlib`, `datetime`, `types`).
3. Third-party (`gymnasium`, `numpy`, `torch`, `pytorch_lightning`, `pydantic`, `pydantic_yaml`, `typer`, `loguru`, `wandb`, `pygame`, `pytest` in tests).
4. First-party: `from wargame_rl.wargame....` — always absolute imports from the installed package root.

**TYPE_CHECKING pattern:**

- Heavy imports used only for type annotations are guarded behind `if TYPE_CHECKING:` to avoid circular imports and reduce runtime overhead. This pattern is used consistently (e.g. `wargame_rl/wargame/envs/reward/calculators/base.py`, `wargame_rl/wargame/envs/reward/calculators/closest_objective.py`, `wargame_rl/wargame/envs/opponent/policy.py`, `wargame_rl/wargame/model/ppo/agent.py`).

**Barrel / re-exports:**

- Packages expose public API via `__init__.py` with explicit `__all__` lists (e.g. `wargame_rl/wargame/envs/types/__init__.py`, `wargame_rl/wargame/envs/domain/__init__.py`, `wargame_rl/wargame/envs/env_components/__init__.py`).
- Private re-imports with underscore aliases where the factory name would shadow local symbols (e.g. `from ... import create_wargame_models as _create_wargame_models` in `wargame_rl/wargame/envs/wargame.py`).

**Path hacks in tests:**

- When importing repo-root modules (`train.py`, `simulate.py`), tests insert `sys.path` and may use `# noqa: E402` after the import (see `tests/test_simulate.py`).

## Type Hints

**Policy:** All public functions and methods have type hints (enforced by mypy `disallow_untyped_defs = true`).

**Style:**
- Use built-in generics: `list[str]`, `dict[str, int]`, `tuple[int, ...]` — not `typing.List`, `typing.Dict`.
- Use `X | None` (union syntax via `from __future__ import annotations`) — not `Optional[X]`.
- Use `typing.TypeAlias` for type aliases (e.g. `Device: TypeAlias = str | None | torch.device` in `wargame_rl/wargame/model/common/device.py`).
- Use `typing.Self` for classmethods returning the class (e.g. `RL_Network.from_env` in `wargame_rl/wargame/model/net.py`).
- Use `typing.NamedTuple` for typed tuples (e.g. `Experience`, `ExperienceBatch` in `wargame_rl/wargame/types.py`).
- Use `typing.Protocol` with `runtime_checkable` for structural subtyping (e.g. `BattleView` in `wargame_rl/wargame/envs/domain/battle_view.py`).
- Use `collections.abc.Callable` — not `typing.Callable` (e.g. `wargame_rl/wargame/envs/wargame.py`, `wargame_rl/wargame/envs/domain/turn_execution.py`).
- Use `typing.cast` when narrowing types that mypy cannot infer (e.g. PyTorch tensors, fixture params in `tests/conftest.py`).
- Apply `# type: ignore[...]` with specific error codes only where unavoidable (e.g. `# type: ignore[arg-type]` in `wargame_rl/wargame/envs/opponent/registry.py`).

## Data Modeling Patterns

**Pydantic BaseModel for config and serializable types:**

- All YAML-loadable config uses Pydantic `BaseModel` with `Field(...)` for descriptions, defaults, and constraints (e.g. `WargameEnvConfig`, `ModelConfig`, `RewardPhaseConfig`, `PPOConfig`, `DQNConfig`).
- Load from YAML via `pydantic_yaml.parse_yaml_raw_as(WargameEnvConfig, raw_str)` then reconstruct with `WargameEnvConfig(**config.model_dump())` for full validation in `train.py`.
- Use `model_validator(mode="after")` for cross-field validation (e.g. entity count vs list length in `wargame_rl/wargame/envs/types/config.py`).
- Use `field_validator` for input normalization (e.g. `normalize_blocking_mask` accepting 0/1 integers as bools in `wargame_rl/wargame/envs/types/config.py`).
- Use `model_dump()` for serialization to dict (e.g. `info.model_dump()` returned from env step/reset).
- Use `ConfigDict(arbitrary_types_allowed=True)` when model contains numpy arrays (e.g. `WargameEnvInfo` in `wargame_rl/wargame/envs/types/env_info.py`).

**Dataclasses for domain-internal objects:**

- Use `@dataclass` for mutable internal state (e.g. `WargameEnvObservation` in `wargame_rl/wargame/envs/types/env_observation.py`, `RewardPhase` and `RewardPhaseManager` in `wargame_rl/wargame/envs/reward/phase_manager.py`).
- Use `@dataclass(frozen=True, slots=True)` for value objects (e.g. `BoardDimensions`, `DeploymentZone` in `wargame_rl/wargame/envs/domain/value_objects.py`, `ActionSlice` in `wargame_rl/wargame/envs/env_components/actions.py`, `GameState` in `wargame_rl/wargame/envs/types/game_timing.py`, `StepContext` in `wargame_rl/wargame/envs/reward/step_context.py`).

**Plain classes for domain entities with mutable state:**

- `WargameModel` and `WargameObjective` in `wargame_rl/wargame/envs/domain/entities.py` are plain classes (no dataclass) with `__init__` — they carry mutable numpy arrays and per-episode state.

**NamedTuple for immutable data carriers:**

- `Experience` and `ExperienceBatch` in `wargame_rl/wargame/types.py` use `typing.NamedTuple`.

## Registry Pattern

Use string-keyed registries for YAML-configurable subsystems. Pattern:

1. Define an ABC (e.g. `PerModelRewardCalculator` in `wargame_rl/wargame/envs/reward/calculators/base.py`, `OpponentPolicy` in `wargame_rl/wargame/envs/opponent/policy.py`, `VPCalculator` in `wargame_rl/wargame/envs/mission/vp_calculator.py`).
2. Maintain a `dict[str, type[...]]` registry (e.g. `CALCULATOR_REGISTRY` in `wargame_rl/wargame/envs/reward/calculators/registry.py`, `_REGISTRY` in `wargame_rl/wargame/envs/opponent/registry.py`, `VP_CALCULATOR_REGISTRY` in `wargame_rl/wargame/envs/mission/registry.py`).
3. Provide a `build_*` factory that looks up the class by string key, raises `ValueError` listing available keys on miss, and instantiates with `(**params)`.
4. Concrete implementations self-register at import time (e.g. `register_policy(...)` at module bottom in `wargame_rl/wargame/envs/opponent/scripted_advance_to_objective_policy.py`).
5. For lazy registration, use `_auto_register()` that imports all known modules (e.g. `wargame_rl/wargame/envs/opponent/registry.py`).

## Error Handling

**Patterns:**

- **Pydantic validators** raise `ValueError` with explicit messages at parse/load time for invalid config (mixed coordinates, out-of-bounds positions, missing opponent policy when opponents configured). See `wargame_rl/wargame/envs/types/config.py`.
- **Value object `__post_init__`** raises `ValueError` for invariant violations (e.g. `BoardDimensions` checking positive width/height, `DeploymentZone` checking max > min in `wargame_rl/wargame/envs/domain/value_objects.py`).
- **Registry misses** raise `ValueError` with "Unknown type '...' Available: ..." message listing valid keys (e.g. `wargame_rl/wargame/envs/reward/calculators/registry.py`, `wargame_rl/wargame/envs/opponent/registry.py`, `wargame_rl/wargame/envs/mission/registry.py`).
- **Domain-specific errors** (e.g. `GameClockError` in `wargame_rl/wargame/envs/domain/game_clock.py`) for game state invariant violations.
- **`RuntimeError`** for integration failures (e.g. wandb init failure in `wargame_rl/wargame/model/common/wandb.py`).
- **`FileNotFoundError`** for missing config files (`train.py` `get_env_config`, `_validate_checkpoint_mode`).
- **`NotImplementedError`** for unsupported combinations (e.g. PPO + MLP in `train.py`).
- **Network heads** raise `ValueError` when requesting the wrong head (e.g. policy head on value network in `wargame_rl/wargame/model/net.py`).

**Prescription:** Prefer explicit exceptions with clear messages. Avoid bare `except`. Match the exception type used by sibling code in the same module. Validate at construction/initialization — keep runtime simple.

## Logging

**Framework:** Loguru (`from loguru import logger`).

**Patterns:**

- Use structured placeholders: `logger.info("Reward phase advanced: '{}' -> '{}' (success_rate={:.2f}, epoch={})", old_name, new_name, sr, epoch)` — see `wargame_rl/wargame/envs/reward/phase_manager.py`.
- Some modules still use `print()` for quick diagnostics (e.g. network parameter counts in `wargame_rl/wargame/model/net.py`, wandb warnings in `wargame_rl/wargame/model/common/wandb.py`). Prefer `logger` for new code.
- Training metrics logged via PyTorch Lightning's `self.log(...)` and Wandb's `wandb.log(...)` (see `wargame_rl/wargame/model/common/lightning_base.py`).

## Comments

**When to comment:**

- Section banners in large test files (`# --- Step tests ---` in `tests/test_env.py`; `# -- Properties ---` / `# -- Core methods ---` in `wargame_rl/wargame/envs/reward/phase_manager.py`).
- Explain non-obvious **why** (e.g. "Exclude the action mask tensor (last element) — it's used externally for action selection, not as network input" in `wargame_rl/wargame/model/net.py`).
- Attribution for adapted algorithms (e.g. "Transformer adapted from the NanoGPT implementation" in `wargame_rl/wargame/model/net.py`).

**Docstrings:**

- Module docstrings summarize scope (e.g. `"""Domain entities: WargameModel (unit) and WargameObjective (capture target)."""` in `wargame_rl/wargame/envs/domain/entities.py`).
- Public classes have multi-line docstrings describing purpose and key args (e.g. `WargameModel`, `Battle`, `ActionHandler`, `RewardPhaseManager`).
- Public methods have one-line or short docstrings (e.g. properties, `calculate`, `select_action`).
- Tests use one-line docstrings describing expected behavior (e.g. `"""reset() returns (observation, info) where info is a dict."""` in `tests/test_env.py`).
- Use `Field(description="...")` on Pydantic fields rather than separate docstrings for individual fields.

## Function Design

**Size:** Prefer focused functions under ~40 lines. Env orchestration is decomposed across `wargame_rl/wargame/envs/domain/` services (`turn_execution.py`, `placement.py`, `termination.py`, `los.py`) and `wargame_rl/wargame/envs/env_components/` rather than monolithic methods.

**Parameters:**
- Pass config and dependencies explicitly (`WargameEnv.__init__(config=..., renderer=...)`).
- Use keyword-only args (after `*`) for optional parameters (e.g. `ActionHandler.__init__(self, config, *, n_models=None, n_shoot_targets=0)` in `wargame_rl/wargame/envs/env_components/actions.py`).
- Factory functions take parsed config or registry keys plus params dict.

**Return values:**
- Typed returns everywhere (mypy enforced).
- Use `typing.cast` when narrowing types mypy cannot infer.
- Use explicit `-> None` for void methods.

## Module Design

**Exports:**

- Use `__all__` where backward-compatible re-exports matter. All major `__init__.py` files define `__all__` (e.g. `wargame_rl/wargame/envs/types/__init__.py`, `wargame_rl/wargame/envs/domain/__init__.py`, `wargame_rl/wargame/envs/env_components/__init__.py`).

**Configuration:**

- Single source of truth in Pydantic models loaded from YAML via `pydantic_yaml` (`parse_yaml_raw_as` in `train.py`).
- Config classes have sensible defaults so existing YAML configs keep working when new fields are added (backward compatibility through `Field(default=...)`).
- Deprecated fields use `Field(description="Deprecated: use ...")` and backfill validators (e.g. `apply_legacy_terminal_bonus_defaults` in `wargame_rl/wargame/envs/types/config.py`).

**Dependency direction:**

- Domain layer (`wargame_rl/wargame/envs/domain/`) depends only on types.
- Reward and renders depend on `BattleView` protocol (not `WargameEnv` directly).
- Env components depend on domain + types.
- `WargameEnv` composes all layers.
- Model layer depends on env types and env.
- Follow the pattern documented in `docs/ddd-envs.md`.

**Gym registration:**

- Package init (`wargame_rl/__init__.py`) registers the env with Gymnasium. Follow the same pattern when adding new env IDs.

**Enums:**

- Use `str, Enum` (or `StrEnum`) so enum values are YAML-serializable strings (e.g. `BattlePhase`, `PlayerSide`, `TurnOrder`, `NetworkType`).

## Protocol / Interface Pattern

- Use `typing.Protocol` with `@runtime_checkable` for decoupling (e.g. `BattleView` in `wargame_rl/wargame/envs/domain/battle_view.py`).
- Use ABC for subsystem extension points (e.g. `PerModelRewardCalculator`, `GlobalRewardCalculator` in `wargame_rl/wargame/envs/reward/calculators/base.py`, `SuccessCriteria` in `wargame_rl/wargame/envs/reward/criteria/base.py`, `OpponentPolicy` in `wargame_rl/wargame/envs/opponent/policy.py`, `VPCalculator` in `wargame_rl/wargame/envs/mission/vp_calculator.py`, `BaseAgent` in `wargame_rl/wargame/model/common/agent_base.py`, `RL_Network` in `wargame_rl/wargame/model/net.py`, `WargameLightningBase` in `wargame_rl/wargame/model/common/lightning_base.py`).

## Context Object Pattern

- `StepContext` (`wargame_rl/wargame/envs/reward/step_context.py`) is an extensible `@dataclass(slots=True)` data carrier assembled each step. Pass it to all reward calculators and success criteria so their signatures stay stable as new fields are added.

---

*Convention analysis: 2026-04-06*
