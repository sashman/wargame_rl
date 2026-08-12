"""A presenter whose stepping the *caller* owns, for debugging a live match.

`InteractiveRenderer` pauses by spinning inside its own `render()`, so the loop
driving the match never learns a pause happened and cannot be asked to advance
one step. Stepping is therefore not a keybinding but a change of ownership: the
driver owns the pause, and the presenter only reports what was pressed.

`Renderer.render` is typed `-> None` and its ABC is deliberately minimal — the
comment on `FrameSource` warns against adding abstract methods — so the report
travels through a shared mutable `DebugControls` rather than a return value.

Everything drawn is inherited unchanged: board, both HUD panels, the tooltip and
the `[Tab]` key map all come from `InteractiveRenderer`. Only the control flow
differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.renders.v2.backend import RenderBackend
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
    InteractiveRenderer,
)
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme

# Paused, the window polls far faster than the match plays so a key press lands
# within a frame or two instead of waiting out a 4 fps tick.
PAUSED_POLL_FPS = 30
MIN_FPS = 1
MAX_FPS = 60


@dataclass
class DebugControls:
    """What the user has asked the match to do next.

    The one-shot flags are set here and cleared by the driver once acted on.
    Only the driver clears them, so a key pressed between two frames cannot be
    silently dropped.
    """

    # Starts paused: a session opens on the first frame and waits, rather than
    # running an episode away before the window has even been looked at.
    paused: bool = True
    step_once: bool = False
    step_back: bool = False


class DebugPresenter(InteractiveRenderer):
    """Drives a live window that never blocks, however long it stays paused."""

    def __init__(
        self,
        backend: RenderBackend,
        controls: DebugControls,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        super().__init__(backend, theme)
        self._controls = controls

    def render(self, view: BattleView) -> None:
        """Advance exactly one frame and return."""
        self._pump_and_present(view)

    def _is_paused(self) -> bool:
        # Read through to the controls, so the south panel's PAUSED chip still
        # reports the real state now that the driver owns it.
        return self._controls.paused

    def _present_fps(self) -> int:
        return max(self._fps, PAUSED_POLL_FPS) if self._controls.paused else self._fps

    def key_map(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Space", "play / pause"),
            (".", "step one forward"),
            (",", "step one back"),
            ("[", "slower"),
            ("]", "faster"),
            ("L", "line-of-sight debug ray"),
            ("Click", "pin a model's tooltip"),
            ("Esc", "quit"),
        )

    def _handle_key(self, event: pygame.event.Event, view: BattleView) -> None:
        controls = self._controls
        if event.key == pygame.K_SPACE:
            controls.paused = not controls.paused
        elif event.key in (pygame.K_PERIOD, pygame.K_RIGHT):
            # Stepping implies pausing: you asked for one step, not for the match
            # to resume from here.
            controls.step_once = True
            controls.paused = True
        elif event.key in (pygame.K_COMMA, pygame.K_LEFT):
            controls.step_back = True
            controls.paused = True
        elif event.key == pygame.K_LEFTBRACKET:
            self._fps = max(MIN_FPS, self._fps - 1)
        elif event.key == pygame.K_RIGHTBRACKET:
            self._fps = min(MAX_FPS, self._fps + 1)
        else:
            super()._handle_key(event, view)
