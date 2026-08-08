"""CLI overrides must reach `PPOConfig`, and must not leak Typer sentinels.

`train()` is a Typer command, so its parameters only hold real values when Typer
parses argv. Called as a plain function — from a test, a script, or any other
Python caller — each one still holds an `OptionInfo` sentinel, which is **not
None and is truthy**. An override guarded by `if value is not None` therefore
assigns the sentinel straight into the config, where it survives until something
tries arithmetic on it.
"""

from __future__ import annotations

import typer

from train import _resolve_optional_float, _resolve_optional_int
from wargame_rl.wargame.model.ppo.config import PPOConfig


def test_a_typer_sentinel_resolves_to_none() -> None:
    """The whole point: the sentinel must not reach the config."""
    sentinel = typer.Option(None, help="whatever")

    assert sentinel is not None
    assert _resolve_optional_float(sentinel) is None
    assert _resolve_optional_int(sentinel) is None


def test_real_values_pass_through_unchanged() -> None:
    assert _resolve_optional_float(1e-3) == 1e-3
    assert _resolve_optional_float(0.0) == 0.0
    assert _resolve_optional_int(0) == 0
    assert _resolve_optional_float(None) is None


def test_the_overridden_fields_exist_with_the_documented_defaults() -> None:
    """Guards the flags against a silent rename of the config field."""
    config = PPOConfig()

    assert config.lr == 3e-4
    assert config.max_grad_norm == 0.5
