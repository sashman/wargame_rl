"""Action space and application for the wargame environment.

Polar coordinate movement: each model picks an (angle, speed) pair or stays
still.  The continuous displacement is rounded to the nearest integer cell so
that locations remain on the discrete grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig

STAY_ACTION = 0


class ActionHandler:
    """Builds action space and applies polar movement actions to models.

    Actions are encoded as a single integer per model:
        0           -> stay (no movement)
        1 .. N*S    -> move, where the index encodes (angle_bin, speed_bin):
            angle_idx  = (action - 1) // n_speed_bins
            speed_idx  = (action - 1) %  n_speed_bins

    angle_idx selects from *n_movement_angles* evenly-spaced directions
    starting at 0 rad (east / +x) and going counter-clockwise.

    speed_idx selects a speed linearly spaced from
    max_move_speed / n_speed_bins  up to  max_move_speed.
    """

    def __init__(self, config: WargameEnvConfig) -> None:
        self._config = config
        n_angles = config.n_movement_angles
        n_speeds = config.n_speed_bins
        max_speed = config.max_move_speed

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        speeds = np.linspace(max_speed / n_speeds, max_speed, n_speeds)

        self._unit_directions = np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )  # (n_angles, 2)
        self._speeds = speeds  # (n_speeds,)

        # Pre-compute the integer displacement for every (angle, speed) pair.
        # _displacements[angle_idx, speed_idx] -> (dx, dy) as int
        raw = (
            self._unit_directions[:, np.newaxis, :]
            * self._speeds[np.newaxis, :, np.newaxis]
        )  # (n_angles, n_speeds, 2)
        self._displacements: np.ndarray = np.rint(raw).astype(int)

        self._n_move_actions = n_angles * n_speeds
        self._n_speed_bins = n_speeds

    @property
    def n_actions(self) -> int:
        """Total number of discrete actions (stay + all angle*speed combos)."""
        return 1 + self._n_move_actions

    @property
    def action_space(self) -> spaces.Tuple:
        return spaces.Tuple(
            [
                spaces.Discrete(self.n_actions)
                for _ in range(self._config.number_of_wargame_models)
            ]
        )

    def _decode_action(self, action: int) -> np.ndarray:
        """Return the integer (dx, dy) displacement for *action*."""
        if action == STAY_ACTION:
            return np.array([0, 0], dtype=int)
        move_idx = action - 1
        angle_idx = move_idx // self._n_speed_bins
        speed_idx = move_idx % self._n_speed_bins
        return self._displacements[angle_idx, speed_idx]

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
            model.previous_location = model.location.copy()
            displacement = self._decode_action(act)
            model.location = np.clip(
                model.location + displacement,
                [0, 0],
                [board_width - 1, board_height - 1],
            )
