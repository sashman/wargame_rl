"""VP calculator protocol and implementations.

Mission-specific logic for how many victory points to award a side
when scoring is triggered (e.g. at end of command phase from round 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.types.game_timing import PlayerSide


class VPCalculator(ABC):
    """Computes victory points to add for one side when scoring is triggered."""

    @abstractmethod
    def compute_vp(
        self,
        view: BattleView,
        scoring_side: PlayerSide,
        current_round: int,
        player_side: PlayerSide,
    ) -> int:
        """Return the number of VP to add for the given side this scoring moment.

        player_side is the side that owns view.player_models (the RL agent).
        """
        ...


def objective_control_from_caches(
    player_norms_offset: np.ndarray,
    opponent_norms_offset: np.ndarray,
    obj_radii: np.ndarray,
) -> tuple[int, int]:
    """Backwards-compatible objective control count helper.

    Returns:
        (player_controlled, opponent_controlled) where contested objectives
        count for neither.
    """
    from wargame_rl.wargame.envs.env_components.distance_cache import (
        objective_ownership_from_norms_offset,
    )

    player_controls, opponent_controls = objective_ownership_from_norms_offset(
        player_norms_offset,
        opponent_norms_offset,
        obj_radii,
    )
    return int(np.sum(player_controls)), int(np.sum(opponent_controls))


class DefaultVPCalculator(VPCalculator):
    """Default mission: VP per controlled objective, cap per turn, from min_round."""

    def __init__(
        self,
        vp_per_objective: int = 5,
        cap_per_turn: int = 15,
        min_round: int = 2,
    ) -> None:
        self.vp_per_objective = vp_per_objective
        self.cap_per_turn = cap_per_turn
        self.min_round = min_round

    def compute_vp(
        self,
        view: BattleView,
        scoring_side: PlayerSide,
        current_round: int,
        player_side: PlayerSide,
    ) -> int:
        if current_round < self.min_round:
            return 0
        from wargame_rl.wargame.envs.domain.entities import alive_mask_for
        from wargame_rl.wargame.envs.env_components.distance_cache import (
            compute_distances,
            objective_ownership_from_norms_offset,
        )

        player_alive = alive_mask_for(view.player_models)
        player_cache = compute_distances(
            view.player_models, view.objectives, alive_mask=player_alive
        )
        n_obj = len(view.objectives)
        if view.opponent_models:
            opp_alive = alive_mask_for(view.opponent_models)
            opponent_cache = compute_distances(
                view.opponent_models, view.objectives, alive_mask=opp_alive
            )
            opponent_norms = opponent_cache.model_obj_norms_offset
        else:
            opponent_norms = np.zeros((0, n_obj), dtype=np.float64)

        player_controls, opponent_controls = objective_ownership_from_norms_offset(
            player_cache.model_obj_norms_offset,
            opponent_norms,
            player_cache.obj_radii,
        )
        n_player = int(np.sum(player_controls))
        n_opponent = int(np.sum(opponent_controls))
        controlled = n_player if scoring_side == player_side else n_opponent
        raw = controlled * self.vp_per_objective
        return min(self.cap_per_turn, raw)


class NoneVPCalculator(VPCalculator):
    """No VP awarded (mission disabled)."""

    def compute_vp(
        self,
        view: BattleView,
        scoring_side: PlayerSide,
        current_round: int,
        player_side: PlayerSide,
    ) -> int:
        return 0
