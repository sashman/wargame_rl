"""One per-step plan-following score for the soldiers (#384, amendment 19):
the share of the best possible step toward the squad's committed objective,
in [-1, 1], with a perfect action available at every decision."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class PlanFollowingCalculator(PerModelRewardCalculator):
    """Pays the acting soldier how well its move followed the plan, per step.

    With `d_t` the distance from the soldier's base edge to its committed
    objective's EDGE before the move, `d_{t+1}` after it, and `M` the move
    distance for a step (`max_move_distance`, the config's `max_move_speed`):

        walking (d_t > 0):            clip((d_t - d_{t+1}) / min(M, d_t), -1, 1)
        inside and stayed (0, 0):     +1
        walked off (d_t = 0 < d_{t+1}): -min(1, d_{t+1} / M)

    A full-speed step straight at the objective's edge (or the shorter one
    that lands exactly on it) scores 1, as does staying inside, so a perfect
    action exists at every decision; standing still off the objective scores
    0 and walking straight away at full speed -1. The score is the same per
    step for every plan and bounded by 1, so a re-commit can never make a
    step pay more than a normal step (the tail that broke the first
    two-stream build) and the soldiers are indifferent to the planner's
    churn. Nothing pays for any objective but the committed one: a soldier
    parked on the wrong objective earns 0 per step. A soldier whose squad
    holds no commitment earns nothing. The first decision of a soldier in an
    episode is unpaid (no before-move distance is on record yet) -- the same
    gap `closest_objective_v2` has on a target switch, once per soldier.

    Built from Sash's proposal (2026-09-26): the progress term paid 2.0 per
    completed commitment, so a far objective paid less per step than a near
    one and the planner's choice leaked into the soldiers' per-step pay.
    """

    def __init__(self, weight: float = 1.0, max_move_distance: float = 6.0) -> None:
        super().__init__(weight)
        if max_move_distance <= 0.0:
            raise ValueError("max_move_distance must be positive")
        self.max_move_distance = float(max_move_distance)
        # Edge distances to EVERY objective at each soldier's last decision,
        # so a re-commit mid-walk still has a before-move distance to the new
        # objective.
        self._previous_edge_distances: dict[int, np.ndarray] = {}
        self.last_score: dict[int, float] = {}

    def reset_episode(self) -> None:
        self._previous_edge_distances.clear()
        self.last_score.clear()

    @staticmethod
    def _edge_distances(ctx: StepContext, model_idx: int) -> np.ndarray:
        cache = ctx.distance_cache
        edge = np.maximum(
            0.0, cache.model_obj_norms_offset[model_idx] - cache.obj_radii
        )
        return np.asarray(edge, dtype=float)

    def score(self, before: float, after: float) -> float:
        """The plan-following score for one step, from the two edge distances."""
        move = self.max_move_distance
        if before > 0.0:
            return float(np.clip((before - after) / min(move, before), -1.0, 1.0))
        if after <= 0.0:
            return 1.0
        return -min(1.0, after / move)

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        now = self._edge_distances(ctx, model_idx)
        previous = self._previous_edge_distances.get(model_idx)
        self._previous_edge_distances[model_idx] = now
        committed = ctx.committed_objective
        if committed is None or previous is None:
            self.last_score[model_idx] = 0.0
            return 0.0
        target = int(committed[model_idx])
        if not 0 <= target < now.shape[0]:
            self.last_score[model_idx] = 0.0
            return 0.0
        value = self.score(float(previous[target]), float(now[target]))
        self.last_score[model_idx] = value
        return value
