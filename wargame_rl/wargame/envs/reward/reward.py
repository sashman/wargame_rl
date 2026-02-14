from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs import wargame
from wargame_rl.wargame.envs.reward.types.model_rewards import ModelRewards
from wargame_rl.wargame.envs.wargame_model import WargameModel


class Reward:
    def _get_closest_objective_reward(
        self, previous_model_distance: float, distance_to_closest_objective: float
    ) -> float:
        # Wargame model has reached the objective, large positive reward
        if distance_to_closest_objective == 0:
            return float(1)

        distance_improvement = float(
            distance_to_closest_objective - previous_model_distance
        )

        # Wargame model has not moved closer to the objective, tiny negative reward
        if distance_improvement == 0:
            return float(-0.05)

        # Wargame model has moved closer to the objective, small positive reward
        if distance_improvement < 0:
            return float(0.5)

        # Wargame model has moved away from the objective, small negative reward
        if distance_improvement > 0:
            return float(-0.5)

        return float(0)

    def _get_model_closest_objective_reward(
        self,
        model: WargameModel,
        previous_closest_objective_reward: float | None,
        env: wargame.WargameEnv,
    ) -> tuple[float, float]:
        closest_objective = min(
            env.objectives,
            key=lambda obj: float(np.linalg.norm(model.location - obj.location, ord=2)),
        )

        distance_to_closest_objective = float(
            np.linalg.norm(
                model.location
                - closest_objective.location
                + closest_objective.radius_size / 2,
                ord=2,
            )
        )

        normalized_distance = distance_to_closest_objective / (np.sqrt(2) * env.size)

        if previous_closest_objective_reward is None:
            return float(0), normalized_distance

        previous_closest_objective_reward = float(previous_closest_objective_reward)  # type: ignore
        closest_objective_reward = self._get_closest_objective_reward(
            previous_closest_objective_reward, normalized_distance
        )

        return closest_objective_reward, normalized_distance

    def _is_within_group_distance(
        self, model: WargameModel, env: wargame.WargameEnv
    ) -> bool:
        """True if this model is within group_max_distance of at least one other model with the same group_id."""
        same_group = [
            other
            for other in env.wargame_models
            if other is not model and other.group_id == model.group_id
        ]
        if not same_group:
            return True  # No group constraint when alone in group
        min_dist = min(
            float(np.linalg.norm(model.location - other.location, ord=2))
            for other in same_group
        )
        return min_dist <= env.config.group_max_distance

    def calculate_model_reward(
        self, model: WargameModel, env: wargame.WargameEnv
    ) -> ModelRewards:
        closest_objective_reward, normalized_distance = (
            self._get_model_closest_objective_reward(
                model, model.previous_closest_objective_reward, env
            )
        )

        model.previous_closest_objective_reward = normalized_distance

        group_distance_violation_penalty = 0.0
        if env.config.group_cohesion_enabled and not self._is_within_group_distance(
            model, env
        ):
            group_distance_violation_penalty = env.config.group_violation_penalty

        return ModelRewards(
            closest_objective_reward=closest_objective_reward,
            group_distance_violation_penalty=group_distance_violation_penalty,
        )

    def calculate_reward(self, env: wargame.WargameEnv) -> float:
        total_reward = float(0)

        for i, model in enumerate(env.wargame_models):
            model_rewards = self.calculate_model_reward(model, env)
            total_reward += model_rewards.total_reward

        average_reward = float(total_reward / len(env.wargame_models))

        return average_reward
