from pydantic import BaseModel


class ModelRewards(BaseModel):
    closest_objective_reward: float = 0.0
    group_distance_violation_penalty: float = 0.0

    @property
    def total_reward(self) -> float:
        return self.closest_objective_reward + self.group_distance_violation_penalty
