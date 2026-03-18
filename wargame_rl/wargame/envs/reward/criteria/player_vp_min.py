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
    """Compute theoretical max VP for default mission from config."""
    config = view.config
    if config.mission.type != "default":
        return 0
    params = config.mission.params
    vp_per_obj = int(params.get("vp_per_objective", 5))
    cap_per_turn = int(params.get("cap_per_turn", 15))
    min_round = int(params.get("min_round", 2))
    n_rounds = config.number_of_battle_rounds
    n_obj = len(view.objectives)
    scoring_rounds = max(0, n_rounds - min_round + 1)
    max_per_round = min(n_obj * vp_per_obj, cap_per_turn)
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
        """Return the VP threshold when met at termination can trigger terminal bonus."""
        if view.config.mission.type != "default":
            return None
        return self._threshold(view)
