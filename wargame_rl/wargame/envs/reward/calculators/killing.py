from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class KillingReward(GlobalRewardCalculator):
    """Get a reward for killing an oponent."""

    def __init__(
        self,
        weight: float = 1.0,
        bonus_killing_opponent: float = 5.0,
    ) -> None:
        super().__init__(weight=weight)
        self.bonus_killing_opponent = bonus_killing_opponent
        self._previous_opponent_models_killed: int = 0

    def reset_episode(self) -> None:
        """Clear per-episode state (called by the env on reset)."""
        self._previous_opponent_models_killed = 0

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        diff = ctx.opponent_models_killed - self._previous_opponent_models_killed
        self._previous_opponent_models_killed = ctx.opponent_models_killed
        return self.bonus_killing_opponent * diff
