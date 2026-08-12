"""Interactive debugging of a live match: stepping, and taking a step back.

`just debug` is the entry point. The presenter half lives in
`renders/v2/presenters/debug.py`; this package is the half that drives the env,
which is why it sits beside `baseline/` rather than under `renders/` — renders
may not import the env class.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.debug.session import run_session
from wargame_rl.wargame.envs.debug.undo import DEFAULT_DEPTH, UndoStack, capture_state

__all__ = [
    "DEFAULT_DEPTH",
    "UndoStack",
    "capture_state",
    "run_session",
]
