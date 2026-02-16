"""Initial placement of models and objectives at reset.

Extracted so deployment strategies (random, fixed, zones) can be swapped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from wargame_rl.wargame.envs.wargame_model import WargameModel
    from wargame_rl.wargame.envs.wargame_objective import WargameObjective


def wargame_model_placement(
    wargame_models: list[WargameModel],
    deployment_zone: np.ndarray,
    rng: Generator,
) -> None:
    """Place each model at a random cell inside the deployment zone. Mutates models."""
    for model in wargame_models:
        model_x = rng.integers(deployment_zone[0], deployment_zone[2], dtype=np.int32)
        model_y = rng.integers(deployment_zone[1], deployment_zone[3], dtype=np.int32)
        model.location = np.array([model_x, model_y], dtype=np.int32)
        model.stats["current_wounds"] = model.stats["max_wounds"]
        model.model_rewards_history.clear()


def objective_placement(
    objectives: list[WargameObjective],
    deployment_zone: np.ndarray,
    board_width: int,
    board_height: int,
    rng: Generator,
) -> None:
    """Place each objective at a random cell outside the deployment zone. Mutates objectives."""
    for objective in objectives:
        objective_x = rng.integers(deployment_zone[2], board_width, dtype=np.int32)
        objective_y = rng.integers(deployment_zone[1], board_height, dtype=np.int32)
        objective.location = np.array([objective_x, objective_y], dtype=np.int32)
