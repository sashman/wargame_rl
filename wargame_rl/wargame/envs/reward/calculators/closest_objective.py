from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveCalculator(PerModelRewardCalculator):
    """Reward based on distance to the nearest objective.

    Uses the same \"at objective\" rule as the environment (offset-based):
    inside when norm((model - obj) + r/2) <= r. Potential increases linearly
    from 0 at the edge of that zone to REWARD_AT_OBJECTIVE at the capture
    centre (obj - r/2). The per-step reward is the change in this potential.
    """

    REWARD_AT_OBJECTIVE = 25.0

    def _potential(self, norm_offset: float, objective_radius: float) -> float:
        """Potential when model is inside the game's capture circle (norm_offset <= r)."""
        if objective_radius <= 0:
            return 0.0
        if norm_offset > objective_radius:
            return 0.0
        proximity = (objective_radius - norm_offset) / objective_radius
        return float(self.REWARD_AT_OBJECTIVE) * proximity

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        env: WargameEnv,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        # Closest objective by offset norm (same \"inside\" rule as all_models_at_objectives).
        closest_obj_idx = int(cache.model_obj_norms_offset[model_idx].argmin())
        norm_offset = float(cache.model_obj_norms_offset[model_idx, closest_obj_idx])
        objective_radius = float(cache.obj_radii[closest_obj_idx])

        previous = model.previous_closest_objective_distance
        # Store offset norm for closest objective for next step's delta.
        model.set_previous_closest_objective_distance(norm_offset)

        if previous is None:
            return 0.0

        prev_norm_offset = float(previous)
        current_potential = self._potential(norm_offset, objective_radius)
        previous_potential = self._potential(prev_norm_offset, objective_radius)

        return current_potential - previous_potential
