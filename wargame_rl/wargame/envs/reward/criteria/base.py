from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class SuccessCriteria(ABC):
    """Evaluates whether the agent achieved the phase goal this episode."""

    @abstractmethod
    def is_successful(self, view: BattleView, ctx: StepContext) -> bool: ...

    def vp_threshold_for_terminal_bonus(self, view: BattleView) -> int | None:
        """Return VP threshold for terminal bonus, or None if not applicable."""
        return None
