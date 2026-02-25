"""Abstract base class for opponent policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class OpponentPolicy(ABC):
    """Selects actions for opponent models each turn."""

    @abstractmethod
    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
    ) -> WargameEnvAction:
        """Return one action per opponent model."""
        ...
