"""Registry mapping baseline names to BaselinePolicy classes."""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy

_REGISTRY: dict[str, type[BaselinePolicy]] = {}


def register_baseline(name: str, cls: type[BaselinePolicy]) -> None:
    """Register a baseline policy class under a string identifier."""
    _REGISTRY[name] = cls


def build_baseline_policy(name: str, **params: object) -> BaselinePolicy:
    """Instantiate a baseline policy by name."""
    _auto_register()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown baseline policy '{name}'. Available: {available}")
    return cls(**params)  # type: ignore[call-arg]


def _auto_register() -> None:
    """Import built-in baseline modules so they register themselves."""
    import importlib

    for module in (
        "wargame_rl.wargame.envs.baseline.random_baseline",
        "wargame_rl.wargame.envs.baseline.scripted_greedy_nearest",
        "wargame_rl.wargame.envs.baseline.scripted_split_evenly",
        "wargame_rl.wargame.envs.baseline.scripted_squad_march",
        "wargame_rl.wargame.envs.baseline.scripted_squad_march_shoot",
    ):
        importlib.import_module(module)


def get_registry() -> dict[str, type[BaselinePolicy]]:
    """Return a copy of the current registry (useful for tests)."""
    _auto_register()
    return dict(_REGISTRY)
