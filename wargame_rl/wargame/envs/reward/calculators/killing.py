from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class KillingReward(GlobalRewardCalculator):
    """Get a reward for killing an opponent."""

    def __init__(
        self,
        weight: float = 1.0,
        bonus_killing_opponent: float = 5.0,
    ) -> None:
        super().__init__(weight=weight)
        self.bonus_killing_opponent = bonus_killing_opponent
        self._previous_player_kills: int = 0

    def reset_episode(self) -> None:
        """Clear per-episode state (called by the env on reset)."""
        self._previous_player_kills = 0

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        # ``player_models_killed`` counts opponents the player killed
        # (env convention: ``player_*`` = performed by the player). Reading
        # ``opponent_models_killed`` here would reward losing our own models.
        diff = ctx.player_models_killed - self._previous_player_kills
        self._previous_player_kills = ctx.player_models_killed
        return self.bonus_killing_opponent * diff
