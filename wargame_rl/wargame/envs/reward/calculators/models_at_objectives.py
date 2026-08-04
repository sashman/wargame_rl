"""Dense reward for the fraction of models standing on objectives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class ModelsAtObjectivesCalculator(GlobalRewardCalculator):
    """Dense reward = fraction of alive models within some objective's radius.

    This is the step-wise counterpart of the ``fraction_at_objectives`` success
    criteria. ``objective_coverage`` pays for *controlling* objectives, which
    saturates as soon as the player out-counts the opponent — roughly two models
    per point — and ``closest_objective`` pays for *approaching* rather than
    staying. Neither rewards massing models on objectives and holding them, so a
    phase gated on a model fraction had no dense gradient behind its own
    criteria. Returns unweighted; the phase manager applies the configured
    weight.
    """

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        alive = alive_mask_for(view.player_models)
        return ctx.distance_cache.fraction_at_objectives(alive_mask=alive)
