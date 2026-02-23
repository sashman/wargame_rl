from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv


class SuccessCriteria(ABC):
    """Evaluates whether the agent achieved the phase goal this episode."""

    @abstractmethod
    def is_successful(self, env: WargameEnv, ctx: StepContext) -> bool: ...
