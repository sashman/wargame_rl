"""The per-model facade's dependency direction, AST-walked as `test_board_layer_is_a_leaf` is.

Three arrows: the new `domain/` modules import only `domain/` and `types/`
(and numpy); `envs/per_model/` never imports the phase facade; and nothing
outside `envs/per_model/` imports it -- until stage 2 seats a network on it,
the facade is a leaf of the application.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ENVS = Path(__file__).resolve().parent.parent / "wargame_rl" / "wargame" / "envs"
PACKAGE = ENVS.parent.parent
NEW_DOMAIN_MODULES = (
    "dice.py",
    "activation.py",
    "fight_sequence.py",
    "unit_referees.py",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("name", NEW_DOMAIN_MODULES)
def test_the_new_domain_modules_import_only_the_domain_and_the_kernel(
    name: str,
) -> None:
    """Arrange a new domain module; act by walking its imports; assert every
    project import stays inside `domain/` or `types/`."""
    offenders = {
        module
        for module in _imported_modules(ENVS / "domain" / name)
        if module.startswith("wargame_rl")
        and not module.startswith("wargame_rl.wargame.envs.domain")
        and not module.startswith("wargame_rl.wargame.envs.types")
    }
    assert not offenders, f"{name} imports {offenders}"


def test_the_per_model_facade_never_imports_the_phase_facade() -> None:
    """The two facades share the domain and nothing else."""
    offenders = {
        path.name
        for path in _python_files(ENVS / "per_model")
        if "wargame_rl.wargame.envs.wargame" in _imported_modules(path)
    }
    assert not offenders, f"per_model modules importing wargame.py: {offenders}"


def test_nothing_outside_the_facade_imports_it_yet() -> None:
    """Stage 1 ships the facade as a leaf; stage 2's network is its first client."""
    offenders = {
        path.relative_to(PACKAGE).as_posix()
        for path in _python_files(PACKAGE)
        if "per_model" not in path.parts
        and any(
            module.startswith("wargame_rl.wargame.envs.per_model")
            for module in _imported_modules(path)
        )
    }
    assert not offenders, f"per_model is imported by {offenders}"
