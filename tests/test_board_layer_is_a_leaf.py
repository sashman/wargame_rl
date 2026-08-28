"""`envs/board/` may depend on the domain and on nothing else.

The package exists so that a heatmap, a scripted policy and the renderer can all
read the *same* board arithmetic. That only holds while the arrow points one
way: `renders/` imports `board/`, and `board/` imports neither `renders/` nor
`env_components/` nor the env facade.

Without this test the rule is a comment. The first person who wants a `Ring` will
import `renders/v2/control.py`, which imports `board/grid.py`, and the cycle
arrives with no error message. `tests/test_import_direction.py` catches the
model/rating/torch half automatically because it walks `envs/**`; this catches
the half that is specific to the new layer.

Walked with `ast` rather than grepped, so a module that merely *names* a
forbidden package in a docstring does not fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BOARD = (
    Path(__file__).resolve().parent.parent / "wargame_rl" / "wargame" / "envs" / "board"
)

FORBIDDEN = (
    "wargame_rl.wargame.envs.renders",
    "wargame_rl.wargame.envs.env_components",
    "wargame_rl.wargame.envs.reward",
    "wargame_rl.wargame.envs.baseline",
    "wargame_rl.wargame.envs.opponent",
    "wargame_rl.wargame.envs.state",
    "wargame_rl.wargame.envs.wargame",
    "wargame_rl.wargame.model",
    "wargame_rl.wargame.rating",
    "torch",
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


@pytest.mark.parametrize("path", _python_files(BOARD), ids=lambda p: p.name)
def test_the_board_layer_imports_only_the_domain(path: Path) -> None:
    """Arrange a module, act by walking its imports, assert none are forbidden."""
    forbidden = {
        name
        for name in _imported_modules(path)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN)
    }

    assert not forbidden, f"{path.name} imports {forbidden}"


def test_the_renderer_reaches_the_board_layer_through_control_only() -> None:
    """Only `control.py` may import `board/`, keeping v2's single domain seam.

    `control.py`'s own docstring calls itself "the one place v2 touches the
    domain", and `sight_matrix_from_terrain` lives there rather than in
    `replay.py` explicitly so a second seam is not opened. `board/` is a domain
    read like any other, so it goes through the same door.
    """
    renders = BOARD.parent / "renders"
    offenders = {
        path.relative_to(renders).as_posix()
        for path in _python_files(renders)
        if path.name != "control.py"
        and any(
            name.startswith("wargame_rl.wargame.envs.board")
            for name in _imported_modules(path)
        )
    }

    assert not offenders, (
        f"these reach board/ directly instead of via control.py: {offenders}"
    )
