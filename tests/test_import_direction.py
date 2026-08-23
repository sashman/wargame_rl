"""The dependency direction is pinned, not merely documented.

`docs/ddd-envs.md` § Dependency direction says the env layer does not reach
upward. For `model/` that is not only a style rule: `model/net.py` imports
`envs.wargame`, so an `envs -> model` import would be a genuine circular import
rather than an untidy one.

The rating subsystem adds a second arrow to protect. `rating/` never imports
`model/`; `model/` may import `rating/` -- which is what will let training log
Elo against the frozen scripted anchors without a cycle.

Walked with `ast` rather than grepped, so a module that merely *mentions* torch
in a docstring or a deferred-import comment does not fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "wargame_rl" / "wargame"

ENVS = PACKAGE_ROOT / "envs"
MODEL = PACKAGE_ROOT / "model"
RATING = PACKAGE_ROOT / "rating"


def _imported_modules(path: Path) -> set[str]:
    """Every module name this file imports, at any depth of the tree."""
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


def _forbidden(path: Path, prefixes: tuple[str, ...]) -> set[str]:
    return {
        name
        for name in _imported_modules(path)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }


@pytest.mark.parametrize("path", _python_files(ENVS), ids=lambda p: p.name)
def test_the_env_layer_never_reaches_upward(path: Path) -> None:
    """`envs/` may not import `model/`, `rating/` or torch.

    `model/net.py` imports `envs.wargame`, so this is a real cycle and not a
    matter of taste. The torch half is what keeps `envs/baseline/evaluate.py`
    importable without paying for a torch import -- which `debug.py` and
    `wargame_rl/wargame/selectors.py` both depend on.
    """
    forbidden = _forbidden(
        path, ("wargame_rl.wargame.model", "wargame_rl.wargame.rating", "torch")
    )

    assert not forbidden, f"{path.relative_to(PACKAGE_ROOT)} imports {forbidden}"


@pytest.mark.parametrize("path", _python_files(RATING), ids=lambda p: p.name)
def test_rating_never_imports_the_model_layer(path: Path) -> None:
    """One-way: `model/` may import `rating/`, never the reverse.

    Loading a checkpoint is `wargame_rl/wargame/selectors.py`'s job, and the
    arena receives an already-resolved selector. Keeping the arrow one-way is
    what lets training read a rating table later without a cycle.
    """
    forbidden = _forbidden(path, ("wargame_rl.wargame.model", "torch"))

    assert not forbidden, f"{path.relative_to(PACKAGE_ROOT)} imports {forbidden}"


@pytest.mark.parametrize(
    "path", [RATING / "score.py", RATING / "elo.py"], ids=lambda p: p.name
)
def test_the_rating_mathematics_is_free_of_the_project(path: Path) -> None:
    """The fit knows nothing about wargames.

    That is the whole reason it can be tested against hand-computed cases and
    synthetic arrays with no environment, no torch and no I/O -- and it is what
    keeps a rating bug distinguishable from an environment bug.
    """
    project = {
        name for name in _imported_modules(path) if name.startswith("wargame_rl")
    }

    assert not project, f"{path.name} imports {project}"


def test_the_model_layer_is_allowed_to_import_rating() -> None:
    """The permissive half of the rule, asserted so the direction is not read as
    'these two packages must not know about each other'."""
    assert MODEL.is_dir() and RATING.is_dir()
