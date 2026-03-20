"""Action space and application for the wargame environment.

Polar coordinate movement: each model picks an (angle, speed) pair or stays
still.  The continuous displacement is rounded to the nearest integer cell so
that locations remain on the discrete grid.

The ``ActionRegistry`` partitions the flat action space into contiguous slices
(stay, movement, and future phase-specific slices) and provides phase-aware
action masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

STAY_ACTION = 0

ALL_BATTLE_PHASES: frozenset[BattlePhase] = frozenset(BattlePhase)


@dataclass(frozen=True, slots=True)
class ActionSlice:
    """A contiguous range of action indices belonging to one action type."""

    name: str
    start: int
    end: int
    valid_phases: frozenset[BattlePhase]

    @property
    def size(self) -> int:
        return self.end - self.start


class ActionRegistry:
    """Tracks contiguous action slices and produces phase-aware masks."""

    def __init__(self) -> None:
        self._slices: list[ActionSlice] = []
        self._by_name: dict[str, ActionSlice] = {}
        self._offset: int = 0

    def register(
        self,
        name: str,
        n_actions: int,
        valid_phases: frozenset[BattlePhase],
    ) -> ActionSlice:
        """Append a new slice at the current offset and return it."""
        if name in self._by_name:
            raise ValueError(f"Action slice '{name}' already registered")
        s = ActionSlice(
            name=name,
            start=self._offset,
            end=self._offset + n_actions,
            valid_phases=valid_phases,
        )
        self._slices.append(s)
        self._by_name[name] = s
        self._offset += n_actions
        return s

    @property
    def n_actions(self) -> int:
        return self._offset

    @property
    def slices(self) -> list[ActionSlice]:
        return list(self._slices)

    def slice_for(self, name: str) -> ActionSlice:
        return self._by_name[name]

    def get_action_mask(self, phase: BattlePhase) -> np.ndarray:
        """Return a ``(n_actions,)`` bool mask — True for valid actions."""
        mask = np.zeros(self._offset, dtype=bool)
        for s in self._slices:
            if phase in s.valid_phases:
                mask[s.start : s.end] = True
        return mask

    def get_model_action_masks(self, phase: BattlePhase, n_models: int) -> np.ndarray:
        """Return ``(n_models, n_actions)`` masks, tiled per model."""
        single = self.get_action_mask(phase)
        return np.tile(single, (n_models, 1))


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

    def __init__(
        self, config: WargameEnvConfig, *, n_models: int | None = None
    ) -> None:
        self._enforce_group_coherency_legality: bool = (
            config.enforce_group_coherency_legality
        )
        self._group_max_distance: float = float(config.group_max_distance)

        self._n_models = (
            n_models if n_models is not None else config.number_of_wargame_models
        )
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
        self._n_angles = n_angles

        self._registry = ActionRegistry()
        self._registry.register("stay", 1, ALL_BATTLE_PHASES)
        self._registry.register(
            "movement",
            self._n_move_actions,
            frozenset({BattlePhase.movement}),
        )

    @property
    def registry(self) -> ActionRegistry:
        return self._registry

    @property
    def n_actions(self) -> int:
        """Total number of discrete actions (stay + all angle*speed combos)."""
        return self._registry.n_actions

    @property
    def action_space(self) -> spaces.Tuple:
        return spaces.Tuple(
            [spaces.Discrete(self.n_actions) for _ in range(self._n_models)]
        )

    def _decode_action(self, action: int) -> np.ndarray:
        """Return the integer (dx, dy) displacement for *action*."""
        if action == STAY_ACTION:
            return np.array([0, 0], dtype=int)
        move_idx = action - 1
        angle_idx = move_idx // self._n_speed_bins
        speed_idx = move_idx % self._n_speed_bins
        result: np.ndarray = self._displacements[angle_idx, speed_idx]
        return result

    def encode_action(self, angle_idx: int, speed_idx: int) -> int:
        """Encode an (angle_idx, speed_idx) pair into an action integer."""
        return 1 + angle_idx * self._n_speed_bins + speed_idx

    def best_action_toward(
        self, dx: float, dy: float, max_step_length: float | None = None
    ) -> int:
        """Return the action that moves closest to the direction (dx, dy).

        Picks the angle bin nearest to atan2(dy, dx). When max_step_length is
        None, uses maximum speed. When max_step_length is set, chooses the
        largest speed bin whose displacement norm does not exceed that length;
        if no bin fits, returns the minimum-speed action in that direction so
        the caller can still make progress (e.g. step into an objective).
        Returns STAY_ACTION only if dx == dy == 0.
        """
        if dx == 0.0 and dy == 0.0:
            return STAY_ACTION
        target_angle = np.arctan2(dy, dx) % (2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, self._n_angles, endpoint=False)
        diffs = np.abs(angles - target_angle)
        diffs = np.minimum(diffs, 2 * np.pi - diffs)
        angle_idx = int(np.argmin(diffs))

        if max_step_length is not None:
            speed_idx = self._n_speed_bins - 1
            for s in range(self._n_speed_bins - 1, -1, -1):
                disp = self._displacements[angle_idx, s]
                if np.linalg.norm(disp) <= max_step_length:
                    speed_idx = s
                    break
            else:
                speed_idx = 0
        else:
            speed_idx = self._n_speed_bins - 1

        return self.encode_action(angle_idx, speed_idx)

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

            # Optionally enforce "group coherency" as a hard movement legality
            # constraint: after each tentative move, verify the moved model's
            # `group_id` peers still have a same-group neighbor within
            # `group_max_distance`. If not, revert this model to its previous
            # location (effectively treating it like `stay`).
            if self._enforce_group_coherency_legality and not np.all(displacement == 0):
                if not self._is_group_coherent(wargame_models, model.group_id):
                    if model.previous_location is None:
                        raise RuntimeError(
                            "previous_location was not set before applying action"
                        )
                    model.location = model.previous_location.copy()

    def _is_group_coherent(self, wargame_models: list[Any], group_id: int) -> bool:
        """Check that every model in `group_id` has a same-group peer nearby.

        Rule implemented here matches the existing reward criteria semantics:
        - A group with 0-1 models is always coherent.
        - Otherwise, every model must have at least one same-group member at
          distance <= `group_max_distance` (Euclidean / L2).
        """
        group_models = [m for m in wargame_models if m.group_id == group_id]
        if len(group_models) <= 1:
            return True

        locs = np.array([m.location for m in group_models], dtype=np.float64)
        deltas = locs[:, np.newaxis, :] - locs[np.newaxis, :, :]
        dists = np.linalg.norm(deltas, axis=2, ord=2)  # (n, n)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        return bool((min_dists <= self._group_max_distance).all())
