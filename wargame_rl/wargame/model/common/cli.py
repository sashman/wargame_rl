"""What the training CLIs share: the Typer-default unwrappers and the config loader.

Typer only substitutes real values when it parses argv. Called as a plain
function -- from tests, or any other Python caller -- a command's parameter
still holds the `OptionInfo` sentinel, which is truthy and is not `None`. Both
`train.py` and `train_per_model.py` unwrap through these, so the two drivers
share one rule and neither imports the other.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic_yaml import parse_yaml_raw_as
from typer.models import OptionInfo

from wargame_rl.wargame.envs.types import WargameEnvConfig


def get_env_config(
    env_config_path: str | None, render_mode: str | None
) -> WargameEnvConfig:
    """Load a scenario config, overriding `render_mode` from the CLI."""
    if env_config_path is None:
        return WargameEnvConfig(render_mode=render_mode)

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Environment config file not found: {env_config_path}")

    with open(env_config_path) as f:
        env_config = parse_yaml_raw_as(WargameEnvConfig, f.read())  # pyright: ignore[reportUndefinedVariable]

    # Override render_mode with CLI argument (including None)
    env_config.render_mode = render_mode

    return WargameEnvConfig(**env_config.model_dump())


def resolve_optional_str(value: str | OptionInfo | None) -> str | None:
    """Unwrap a Typer default; None means the option was not given."""
    if isinstance(value, OptionInfo):
        return None
    return value


def resolve_optional_float(value: float | OptionInfo | None) -> float | None:
    """Unwrap a Typer default, as `resolve_optional_int` does for ints."""
    if isinstance(value, OptionInfo):
        return None
    return value


def resolve_optional_int(value: int | OptionInfo | None) -> int | None:
    """Unwrap a Typer default; None means the option was not given."""
    if isinstance(value, OptionInfo):
        return None
    return value


def resolve_default(value: Any, default: Any) -> Any:
    """Unwrap a Typer default to a concrete fallback, for direct callers.

    `resolve_optional_int` returns `None`, which is right for options whose
    absence means "unset". These have real defaults, and a config handed an
    `OptionInfo` fails validation rather than falling back.
    """
    return default if isinstance(value, OptionInfo) else value


__all__ = [
    "get_env_config",
    "resolve_default",
    "resolve_optional_float",
    "resolve_optional_int",
    "resolve_optional_str",
]
