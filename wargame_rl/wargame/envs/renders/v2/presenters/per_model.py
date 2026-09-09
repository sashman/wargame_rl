"""Presenters for the per-model facade: one frame per decision, with the decision drawn.

Both wrap the shared scene with the decision overlay
(`renders/v2/decision.py`) and put the pending decision's caption in the
HUD's context slot. `PerModelPresenter` is the window, opening paused so the
first decision can be looked at before it is taken, with `.` stepping one
decision; `PerModelRecorder` collects frames headlessly for an MP4.

The presenters read no per-model type: the driver hands them a
`DecisionHighlight`, so the facade stays a leaf of the application.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pygame

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.renders.v2.backend import Canvas
from wargame_rl.wargame.envs.renders.v2.decision import (
    DecisionHighlight,
    decision_primitives,
)
from wargame_rl.wargame.envs.renders.v2.presenters.base import _CHIP_SIZE
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
    InteractiveRenderer,
)
from wargame_rl.wargame.envs.renders.v2.presenters.recording import RecordingRenderer
from wargame_rl.wargame.envs.renders.v2.scene import Scene
from wargame_rl.wargame.envs.renders.v2.theme import Theme


@dataclass
class PlayControls:
    """What the viewer has asked for, and what the driver is about to ask.

    `step_once` is set by a key and cleared by the driver once acted on, so a
    press between two frames is never dropped. `decision` is set by the driver
    before each frame; None draws the plain scene.
    """

    paused: bool = True
    step_once: bool = False
    decision: DecisionHighlight | None = None

    def take_step(self) -> bool:
        """Consume a one-shot step request."""
        if self.step_once:
            self.step_once = False
            return True
        return False


class _DecisionScene:
    """The overlay and the caption, shared by the window and the recorder."""

    controls: PlayControls
    _theme: Theme

    def _scene_for(self, view: BattleView) -> Scene:
        scene: Scene = super()._scene_for(view)  # type: ignore[misc]
        decision = self.controls.decision
        if decision is None:
            return scene
        extra = decision_primitives(view, decision, self._theme)
        return replace(scene, primitives=scene.primitives + extra)

    def _south_context(self, frame: Canvas, x: int, y: int) -> None:
        decision = self.controls.decision
        if decision is None:
            super()._south_context(frame, x, y)  # type: ignore[misc]
            return
        text = decision.caption
        if self._is_paused():  # type: ignore[attr-defined]
            text = f"{text}   PAUSED"
        self._text(frame, text, (x, y), _CHIP_SIZE, self._dim(), "midright")  # type: ignore[attr-defined]


class PerModelPresenter(_DecisionScene, InteractiveRenderer):
    """The window: opens paused; [Space] plays, [.] steps one decision."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.controls = PlayControls()

    def _is_paused(self) -> bool:
        return self.controls.paused

    def key_map(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Space", "play / pause"),
            (".", "step one decision"),
            *tuple(k for k in super().key_map() if k[0] != "Space"),
        )

    def _handle_key(self, event: pygame.event.Event, view: BattleView) -> None:
        if event.key == pygame.K_SPACE:
            self.controls.paused = not self.controls.paused
        elif event.key in (pygame.K_PERIOD, pygame.K_RIGHT, pygame.K_RETURN):
            self.controls.step_once = True
            self.controls.paused = True
        else:
            super()._handle_key(event, view)

    @property
    def wants_quit(self) -> bool:
        return self._should_quit


class PerModelRecorder(_DecisionScene, RecordingRenderer):
    """Headless: every rendered frame is kept, for `export_mp4`."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.controls = PlayControls(paused=False)
        self.frames: list[np.ndarray] = []

    def render(self, view: BattleView) -> None:
        super().render(view)
        self.frames.append(self.get_frame_array())

    def export_mp4(self, path: str, fps: int = 4) -> None:
        """Write the collected frames to an MP4."""
        import imageio  # type: ignore[import-untyped]

        writer = imageio.get_writer(
            path,
            format="FFMPEG",  # type: ignore[arg-type]
            mode="I",
            fps=fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for frame in self.frames:
                writer.append_data(frame)
        finally:
            writer.close()
