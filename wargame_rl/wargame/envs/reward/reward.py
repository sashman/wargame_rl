from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs import wargame
from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
from wargame_rl.wargame.envs.reward.types.model_rewards import ModelRewards
from wargame_rl.wargame.envs.wargame_model import WargameModel


class Reward:
    def _get_closest_objective_reward(
        self, previous_model_distance: float, distance_to_closest_objective: float
    ) -> float:
        if distance_to_closest_objective == 0:
            return float(1)

        distance_improvement = float(
            distance_to_closest_objective - previous_model_distance
        )

        if distance_improvement == 0:
            return float(-0.05)

        if distance_improvement < 0:
            return float(0.5)

        if distance_improvement > 0:
            return float(-0.5)

        return float(0)

    def _get_model_closest_objective_reward_cached(
        self,
        model_idx: int,
        model: WargameModel,
        previous_closest_objective_distance: float | None,
        cache: DistanceCache,
        max_diagonal: float,
    ) -> tuple[float, float]:
        closest_obj_idx = int(cache.model_obj_norms[model_idx].argmin())
        distance_to_closest = float(
            cache.model_obj_norms_offset[model_idx, closest_obj_idx]
        )
        normalized_distance = distance_to_closest / max_diagonal

        if previous_closest_objective_distance is None:
            return float(0), normalized_distance

        previous_closest_objective_distance = float(previous_closest_objective_distance)
        closest_objective_reward = self._get_closest_objective_reward(
            previous_closest_objective_distance, normalized_distance
        )

        return closest_objective_reward, normalized_distance

    def _is_within_group_distance_cached(
        self,
        model_idx: int,
        model: WargameModel,
        env: wargame.WargameEnv,
        cache: DistanceCache,
    ) -> bool:
        """True if this model is within group_max_distance of at least one other model with the same group_id."""
        if cache.model_model_norms is None:
            return True
        same_group_mask = np.array(
            [
                i != model_idx and m.group_id == model.group_id
                for i, m in enumerate(env.wargame_models)
            ]
        )
        if not same_group_mask.any():
            return True
        min_dist = float(cache.model_model_norms[model_idx, same_group_mask].min())
        return min_dist <= env.config.group_max_distance

    def calculate_model_reward(
        self,
        model_idx: int,
        model: WargameModel,
        env: wargame.WargameEnv,
        cache: DistanceCache,
        max_diagonal: float,
    ) -> ModelRewards:
        closest_objective_reward, normalized_distance = (
            self._get_model_closest_objective_reward_cached(
                model_idx,
                model,
                model.previous_closest_objective_distance,
                cache,
                max_diagonal,
            )
        )

        model.set_previous_closest_objective_distance(normalized_distance)

        group_distance_violation_penalty = 0.0
        if (
            env.config.group_cohesion_enabled
            and not self._is_within_group_distance_cached(model_idx, model, env, cache)
        ):
            group_distance_violation_penalty = env.config.group_violation_penalty

        model_rewards = ModelRewards(
            closest_objective_reward=closest_objective_reward,
            group_distance_violation_penalty=group_distance_violation_penalty,
        )
        model.model_rewards_history.append(model_rewards)
        return model_rewards

    def calculate_reward(self, env: wargame.WargameEnv, cache: DistanceCache) -> float:
        total_reward = float(0)
        max_diagonal = float(np.sqrt(env.board_width**2 + env.board_height**2))

        for i, model in enumerate(env.wargame_models):
            model_rewards = self.calculate_model_reward(
                i, model, env, cache, max_diagonal
            )
            total_reward += model_rewards.total_reward

        average_reward = float(total_reward / len(env.wargame_models))

        return average_reward
