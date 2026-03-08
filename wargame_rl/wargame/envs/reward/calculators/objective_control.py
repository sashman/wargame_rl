"""Reward for controlling objectives at scoring time (primary mission style)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ObjectiveControlCalculator(GlobalRewardCalculator):
    """Reward equal to VP earned this step from controlling objectives.

    Uses the same rule as the primary mission: 5 VP per objective controlled
    (player LoC > opponent LoC), cap 15 per turn. The env scores VP at the
    scoring moment (first active phase from round 2, e.g. Movement when
    Command is skipped) and sets vp_gained_this_step_player; this calculator
    returns weight * vp_gained_this_step_player so the reward signal is
    aligned with VP.
    """

    def calculate(self, env: WargameEnv, ctx: StepContext) -> float:
        vp_gained = getattr(env, "vp_gained_this_step_player", 0)
        return self.weight * float(vp_gained)
