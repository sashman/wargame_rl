from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class ModelsLostPenalty(GlobalRewardCalculator):
    """Penalty for player models lost this step -- the cost side of the trade.

    Moving into an enemy's line of sight buys a chance to shoot at the price of
    being shot. Without this term the reward only ever pays the first half:
    `model_kills` and `killing` reward kills made, and nothing charges for the
    models it cost. Under that reward, exposing a model is free, so declining a
    bad exchange and taking cover can never look better than charging.

    Pair it with `model_kills` at a matched magnitude so an even trade is
    roughly neutral and only *favourable* exchanges pay.
    """

    def __init__(
        self,
        weight: float = 1.0,
        penalty_per_loss: float = 1.0,
    ) -> None:
        super().__init__(weight=weight)
        self.penalty_per_loss = penalty_per_loss

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        """Return the (negative) penalty for each player model lost this step.

        Reads `opponent_models_killed` -- the `opponent_` prefix means "by the
        opponent", so this is the count of *player* models that died, which is
        exactly what a loss penalty must charge for. `player_models_killed` is
        the opposite quantity and would pay the agent for its own kills twice.

        Global rather than per-model on purpose: `RewardPhaseManager` runs
        per-model calculators over alive models only, so a model killed this
        step earns nothing this step. Every model here carries `max_wounds: 1`,
        so every model that takes damage dies -- a per-model damage penalty
        would be identically zero.
        """
        return -self.penalty_per_loss * float(ctx.opponent_models_killed)
