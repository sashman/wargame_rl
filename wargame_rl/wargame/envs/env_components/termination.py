"""Termination logic for the wargame environment.

Extracted so termination conditions (all-at-objective, max turns, custom rules)
can be extended or swapped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame_model import WargameModel
    from wargame_rl.wargame.envs.wargame_objective import WargameObjective


def model_at_objective(model: WargameModel, objective: WargameObjective) -> bool:
    """True if the model is within the objective's radius."""
    return bool(
        np.linalg.norm(
            model.location - objective.location + objective.radius_size / 2,
            ord=2,
        )
        <= objective.radius_size
    )


def get_termination(
    wargame_models: list,
    objectives: list,
    current_turn: int,
    max_turns: int,
) -> bool:
    """True if every model has reached at least one objective."""

    if check_max_turns_reached(current_turn, max_turns):
        return True

    terminated = [False] * len(wargame_models)
    for i, model in enumerate(wargame_models):
        for objective in objectives:
            if model_at_objective(model, objective):
                terminated[i] = True
                break
    return all(terminated)


def check_max_turns_reached(current_turn: int, max_turns: int) -> bool:
    """True if the turn limit has been reached."""
    return current_turn >= max_turns
