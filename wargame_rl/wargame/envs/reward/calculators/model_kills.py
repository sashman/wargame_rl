"""Per-model reward for kills the model itself made."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel

DEFAULT_BONUS_PER_KILL = 2.0


class ModelKillsCalculator(PerModelRewardCalculator):
    """Reward each model for the opponents *it* killed this step.

    The per-model counterpart of the global ``killing`` calculator. Shooting is
    half the agent's decisions, and under a global term every model receives an
    identical reward whether it fired, missed, or stood still — so the shooting
    head has no credit path at all. This gives it one.

    Note the scale difference from ``killing``: that calculator's default bonus
    of 5.0 is paid to all N models for every kill, so a 25-model wipe is worth
    5 x 25 = 125 per model. Here a wipe is worth roughly one kill's bonus per
    model, because the kills are divided among the shooters rather than
    broadcast.
    """

    def __init__(
        self,
        weight: float = 1.0,
        bonus_per_kill: float = DEFAULT_BONUS_PER_KILL,
    ) -> None:
        super().__init__(weight=weight)
        self.bonus_per_kill = bonus_per_kill

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        """Return the bonus for kills this model made on this step."""
        kills = ctx.player_kills_by_model
        if kills is None or model_idx >= len(kills):
            return 0.0
        return self.bonus_per_kill * float(kills[model_idx])
