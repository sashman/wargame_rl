"""Renderer v2: a Scene → Backend → Presenter subsystem.

Public surface is the factory plus the presenters; the legacy `HumanRender` at
`renders/human.py` stays the default and is untouched.
"""

from wargame_rl.wargame.envs.renders.v2.factory import build_renderer
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
    InteractiveRenderer,
)
from wargame_rl.wargame.envs.renders.v2.presenters.recording import RecordingRenderer
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme

__all__ = [
    "build_renderer",
    "InteractiveRenderer",
    "RecordingRenderer",
    "DEFAULT_THEME",
    "Theme",
]
