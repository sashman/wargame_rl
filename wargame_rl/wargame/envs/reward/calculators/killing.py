from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class KillingReward(GlobalRewardCalculator):
    """Reward for killing opponent models this step."""

    def __init__(
        self,
        weight: float = 1.0,
        bonus_killing_opponent: float = 5.0,
    ) -> None:
        super().__init__(weight=weight)
        self.bonus_killing_opponent = bonus_killing_opponent

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        """Return the bonus for each opponent model killed on this step.

        Reads `player_models_killed` — the `player_` prefix means "by the
        player", matching `player_damage_dealt`. Reading
        `opponent_models_killed` instead paid the agent for its *own* losses.

        The context field is already a per-step count, so it is used directly;
        subtracting a running total made the reward telescope to the final step
        and go negative on the step after any kill.
        """
        return self.bonus_killing_opponent * float(ctx.player_models_killed)
