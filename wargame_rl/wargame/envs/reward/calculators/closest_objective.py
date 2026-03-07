from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveCalculator(PerModelRewardCalculator):
    """Reward based on distance to the nearest objective.

    Inside the objective radius, reward increases linearly from 0 at the edge
    of the zone to REWARD_AT_OBJECTIVE at the centre. The per-step reward is
    the change in this potential, so staying at the same location yields 0.
    """

    REWARD_AT_OBJECTIVE = 25.0

    def _potential(self, distance_to_center: float, objective_radius: float) -> float:
        if objective_radius <= 0:
            return 0.0
        if distance_to_center >= objective_radius:
            return 0.0
        proximity = (objective_radius - distance_to_center) / objective_radius
        return float(self.REWARD_AT_OBJECTIVE) * proximity

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        env: WargameEnv,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        closest_obj_idx = int(cache.model_obj_norms[model_idx].argmin())
        distance_to_center = float(cache.model_obj_norms[model_idx, closest_obj_idx])
        objective_radius = float(cache.obj_radii[closest_obj_idx])

        previous = model.previous_closest_objective_distance
        # Store the raw distance-to-centre for the next step.
        model.set_previous_closest_objective_distance(distance_to_center)

        if previous is None:
            return 0.0

        prev_distance = float(previous)
        current_potential = self._potential(distance_to_center, objective_radius)
        previous_potential = self._potential(prev_distance, objective_radius)

        return current_potential - previous_potential
