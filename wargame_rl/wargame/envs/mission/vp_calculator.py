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


def _count_controlled_by_side(
    player_in_range: np.ndarray,
    opponent_in_range: np.ndarray,
) -> tuple[int, int]:
    """Count objectives controlled by player and by opponent.

    Control: at least one model in radius; contested (both in range) counts for neither.
    """
    n_player = int(np.sum(player_in_range & ~opponent_in_range))
    n_opponent = int(np.sum(opponent_in_range & ~player_in_range))
    return n_player, n_opponent


def objective_control_from_caches(
    player_norms_offset: np.ndarray,
    opponent_norms_offset: np.ndarray,
    obj_radii: np.ndarray,
) -> tuple[int, int]:
    """Compute how many objectives each side controls.

    Uses same in-range rule as distance cache: model within radius when
    norm_offset <= obj_radius. Returns (player_controlled, opponent_controlled).
    """
    player_any = np.any(player_norms_offset <= obj_radii[np.newaxis, :], axis=0)
    opponent_any = np.any(opponent_norms_offset <= obj_radii[np.newaxis, :], axis=0)
    return _count_controlled_by_side(player_any, opponent_any)


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
        from wargame_rl.wargame.envs.env_components.distance_cache import (
            compute_distances,
        )

        player_cache = compute_distances(view.player_models, view.objectives)
        n_obj = len(view.objectives)
        if view.opponent_models:
            opponent_cache = compute_distances(view.opponent_models, view.objectives)
            opponent_norms = opponent_cache.model_obj_norms_offset
        else:
            opponent_norms = np.zeros((0, n_obj), dtype=np.float64)
        n_player, n_opponent = objective_control_from_caches(
            player_cache.model_obj_norms_offset,
            opponent_norms,
            player_cache.obj_radii,
        )
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
