"""Headless presenter for video capture.

Holds the last composed frame and hands it back via `get_frame_array`, so it
satisfies the `FrameSource` protocol the recording callback depends on. The
caller sets `SDL_VIDEODRIVER=dummy` before import for headless pygame, exactly as
the training recording path already does.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.renders.v2.backend import Canvas
from wargame_rl.wargame.envs.renders.v2.presenters.base import BasePresenter


class RecordingRenderer(BasePresenter):
    """Renders frames to memory for MP4 export; no window."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._last_frame: Canvas | None = None

    def _hotkey_hint(self) -> str | None:
        """No keys in a video — the context slot uses the room instead."""
        return None

    def _present(self, frame: Canvas) -> None:
        self._last_frame = frame

    def get_frame_array(self) -> np.ndarray:
        if self._last_frame is None:
            raise ValueError("No frame rendered yet; call render() first")
        return self._backend.to_rgb_array(self._last_frame)
