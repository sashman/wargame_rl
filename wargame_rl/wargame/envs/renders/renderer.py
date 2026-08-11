from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView


class Renderer(ABC):
    @abstractmethod
    def setup(self, view: BattleView) -> None:
        pass

    @abstractmethod
    def render(self, view: BattleView) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


@runtime_checkable
class FrameSource(Protocol):
    """A renderer that can hand back the last frame as an RGB array.

    Not on the `Renderer` ABC on purpose: only the recording path needs frames,
    and making it abstract would force it onto every renderer. The recording
    callback depends on this protocol instead of a concrete class, so both the
    legacy `HumanRender` (which already has `epoch` and `get_frame_array`) and the
    v2 recording presenter satisfy it structurally.
    """

    epoch: int | None

    def get_frame_array(self) -> np.ndarray:
        """The most recently rendered frame as ``(H, W, 3)`` uint8."""
        ...
