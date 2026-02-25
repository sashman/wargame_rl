"""Registry mapping policy type strings to OpponentPolicy classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.types import OpponentPolicyConfig
    from wargame_rl.wargame.envs.wargame import WargameEnv

_REGISTRY: dict[str, type[OpponentPolicy]] = {}


def register_policy(name: str, cls: type[OpponentPolicy]) -> None:
    _REGISTRY[name] = cls


def build_opponent_policy(
    config: OpponentPolicyConfig,
    env: WargameEnv,
) -> OpponentPolicy:
    """Instantiate an OpponentPolicy from its YAML config."""
    cls = _REGISTRY.get(config.type)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown opponent policy type '{config.type}'. Available: {available}"
        )
    return cls(env=env, **config.params)  # type: ignore[call-arg]


def _auto_register() -> None:
    """Import built-in policy modules so they register themselves."""
    import importlib

    for mod in (
        "wargame_rl.wargame.envs.opponent.random_policy",
        "wargame_rl.wargame.envs.opponent.scripted_advance_to_objective_policy",
    ):
        importlib.import_module(mod)


def get_registry() -> dict[str, type[OpponentPolicy]]:
    """Return a copy of the current registry (useful for tests)."""
    return dict(_REGISTRY)
