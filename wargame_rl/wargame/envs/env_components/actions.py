"""Action space and application for the wargame environment.

Extracted so movement schemes (e.g. 4-direction, 8-direction) can be swapped
without changing the core env.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig


class MovementPhaseActions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    none = 4


class ActionHandler:
    """Builds action space and applies movement actions to models."""

    def __init__(self, config: WargameEnvConfig) -> None:
        self._config = config
        self._action_to_direction = {
            MovementPhaseActions.right.value: np.array([1, 0]),
            MovementPhaseActions.up.value: np.array([0, 1]),
            MovementPhaseActions.left.value: np.array([-1, 0]),
            MovementPhaseActions.down.value: np.array([0, -1]),
            MovementPhaseActions.none.value: np.array([0, 0]),
        }

    @property
    def action_space(self) -> spaces.Tuple:
        action_count = len(MovementPhaseActions)
        return spaces.Tuple(
            [
                spaces.Discrete(action_count)
                for _ in range(self._config.number_of_wargame_models)
            ]
        )

    def apply(
        self,
        action: WargameEnvAction,
        wargame_models: list[Any],
        board_width: int,
        board_height: int,
        action_space: spaces.Tuple,
    ) -> None:
        """Apply the action tuple to the wargame models (mutates locations)."""
        for i, act in enumerate(action.actions):
            if not action_space[i].contains(act):  # type: ignore
                raise ValueError(
                    f"Action {act} for wargame model {i} is out of bounds."
                )
            model = wargame_models[i]
            direction = self._action_to_direction[act]
            model.location = np.clip(
                model.location + direction,
                [0, 0],
                [board_width - 1, board_height - 1],
            )
