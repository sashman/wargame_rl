"""Success criteria: player VP at or above a minimum threshold.

Threshold is derived from mission, number of objectives, and number of
battle rounds (theoretical max VP), so it scales with episode length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


def _theoretical_max_vp(view: BattleView) -> int:
    """The most VP this mission can pay over the whole battle.

    ⚠ Reads the mission's own numbers rather than branching on its *name*. This
    used to `return 0` for any `mission.type != "default"`, which made the
    threshold collapse to `min_vp` and `success_rate` pin at 1.0 -- a curriculum
    advancing on epoch count with nothing logged to say the gate had stopped
    meaning anything. No config in the repo sets a mission, so nothing ever
    exercised it; the first one to would have hit it silently.
    """
    config = view.config
    mission = config.mission
    scoring_rounds = max(
        0, config.number_of_battle_rounds - mission.first_scoring_round + 1
    )
    max_per_round = min(
        len(view.objectives) * mission.points_per_objective, mission.per_round_cap
    )
    return scoring_rounds * max_per_round


class PlayerVPMinCriteria(SuccessCriteria):
    """Succeeds when player VP at episode end meets a minimum threshold.

    Threshold = max(min_vp, round(fraction_of_max * theoretical_max)).
    Theoretical max is computed from number_of_battle_rounds, objectives,
    and mission params (default mission: vp_per_objective, cap_per_turn,
    min_round), so the bar scales with episode length.
    """

    def __init__(self, fraction_of_max: float, min_vp: int = 0) -> None:
        self.fraction_of_max = fraction_of_max
        self.min_vp = min_vp

    def _threshold(self, view: BattleView) -> int:
        theoretical = _theoretical_max_vp(view)
        from_fraction = int(round(self.fraction_of_max * theoretical))
        return max(self.min_vp, from_fraction)

    def is_successful(self, view: BattleView, ctx: StepContext) -> bool:
        return view.player_vp >= self._threshold(view)

    def vp_threshold_for_terminal_bonus(self, view: BattleView) -> int | None:
        """The VP threshold whose being met at termination triggers the bonus.

        None only when the mission pays nothing at all, rather than whenever it
        is not the one named `"default"` -- under the old name test any mission
        silently disabled `terminal_vp_bonus`.
        """
        if view.config.mission.per_round_cap <= 0:
            return None
        return self._threshold(view)
