"""Action space and application for the wargame environment.

Polar coordinate movement: each model picks an (angle, speed) pair or stays
still. The displacement is applied exactly — the board is continuous, so a
speed bin means the distance it says in every direction.

The ``ActionRegistry`` partitions the flat action space into contiguous slices
(stay, movement, and future phase-specific slices) and provides phase-aware
action masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.domain.movement import resolve_move
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.value_objects import (
    POSITION_DTYPE,
    position,
    zero_position,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

STAY_ACTION = 0


def _base_arrays(models: list[Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """Centres and radii of the *alive* models in a list, for collision tests.

    Dead models are excluded: a casualty is removed from the table, so its base
    is not ground anyone has to walk around.
    """
    alive = [m for m in (models or []) if m.is_alive]
    if not alive:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    return (
        np.array([m.location for m in alive], dtype=float),
        np.array([m.base_radius for m in alive], dtype=float),
    )


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

    def has_slice(self, name: str) -> bool:
        """True if a slice with this name has been registered."""
        return name in self._by_name

    def get_action_mask(self, phase: BattlePhase) -> np.ndarray:
        """Return a ``(n_actions,)`` bool mask — True for valid actions."""
        mask = np.zeros(self._offset, dtype=bool)
        for s in self._slices:
            if phase in s.valid_phases:
                mask[s.start : s.end] = True
        return mask

    def get_model_action_masks(
        self,
        phase: BattlePhase,
        n_models: int,
        alive_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``(n_models, n_actions)`` masks, tiled per model.

        Dead models (alive_mask False) are restricted to STAY_ACTION only.
        """
        single = self.get_action_mask(phase)
        masks = np.tile(single, (n_models, 1))
        if alive_mask is not None:
            for i in range(n_models):
                if not alive_mask[i]:
                    masks[i, :] = False
                    masks[i, STAY_ACTION] = True
        return masks


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
        self,
        config: WargameEnvConfig,
        *,
        n_models: int | None = None,
        n_shoot_targets: int = 0,
    ) -> None:
        self._n_models = (
            n_models if n_models is not None else config.number_of_wargame_models
        )
        n_angles = config.n_movement_angles
        n_speeds = config.n_speed_bins
        # Through the resolver rather than off the config, so that a board using
        # a scale other than 1 inch per unit moves models the right distance.
        quantities = resolve_rules_quantities(config)
        max_speed = quantities.max_move_speed
        # A model with a base cannot stand with half of it off the table.
        self._base_radius = quantities.base_radius

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        speeds = np.linspace(max_speed / n_speeds, max_speed, n_speeds)

        self._unit_directions = np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )  # (n_angles, 2)
        self._speeds = speeds  # (n_speeds,)

        # Pre-compute the exact displacement for every (angle, speed) pair.
        # _displacements[angle_idx, speed_idx] -> (dx, dy) as POSITION_DTYPE.
        #
        # This used to be `np.rint(raw)`, snapping each move to a whole cell,
        # which was destroying information rather than approximating it: on the
        # 25v25 action space the 96 movement actions collapsed to **80 distinct
        # outcomes**, and a "speed 1" diagonal travelled 1.414 against an
        # orthogonal one's 1.000 -- 41% further for the same nominal speed.
        #
        # The dtype still matters as much as the rounding did. A displacement is
        # added to a location, so a wider one silently widens every location on
        # the board.
        raw = (
            self._unit_directions[:, np.newaxis, :]
            * self._speeds[np.newaxis, :, np.newaxis]
        )  # (n_angles, n_speeds, 2)
        self._displacements: np.ndarray = raw.astype(POSITION_DTYPE)

        self._n_move_actions = n_angles * n_speeds
        self._n_speed_bins = n_speeds
        self._n_angles = n_angles
        # Built lazily rather than here: `n_actions` depends on the registry,
        # which is populated below.
        self._action_space: spaces.Tuple | None = None

        self._registry = ActionRegistry()
        self._registry.register("stay", 1, ALL_BATTLE_PHASES)
        self._registry.register(
            "movement",
            self._n_move_actions,
            frozenset({BattlePhase.movement}),
        )

        if n_shoot_targets > 0:
            self._shooting_slice: ActionSlice | None = self._registry.register(
                "shooting",
                n_shoot_targets,
                frozenset({BattlePhase.shooting}),
            )
        else:
            self._shooting_slice = None

    @property
    def shooting_slice(self) -> ActionSlice | None:
        """Shooting action slice, or None when no shoot targets are registered."""
        return self._shooting_slice

    def decode_shooting_targets(
        self, action: WargameEnvAction, n_attackers: int
    ) -> list[tuple[int, int]]:
        """Read the action tuple as ``(attacker_idx, target_idx)`` shot declarations.

        Only the action *encoding* is interpreted here: entries outside the
        shooting slice are not shots, and entries past the end of the army have
        no attacker. Whether a declared shot can actually resolve — the attacker
        is alive, the target is alive, the model carries a weapon — is a rule,
        and belongs to `domain.shooting.resolve_shooting_phase`.

        Returns an empty list when this handler registered no shooting slice.
        """
        if self._shooting_slice is None:
            return []
        start, end = self._shooting_slice.start, self._shooting_slice.end
        return [
            (i, act - start)
            for i, act in enumerate(action.actions)
            if i < n_attackers and start <= act < end
        ]

    @property
    def registry(self) -> ActionRegistry:
        return self._registry

    @property
    def n_actions(self) -> int:
        """Total number of discrete actions (stay + all angle*speed combos)."""
        return self._registry.n_actions

    @property
    def n_move_actions(self) -> int:
        """Number of movement actions (angle*speed combos, excluding stay)."""
        return self._n_move_actions

    @property
    def action_space(self) -> spaces.Tuple:
        """The per-model action space. Built once; the shape never changes.

        This used to construct ``n_models`` fresh ``Discrete`` spaces on every
        access, and `apply` reads it once per movement application for both
        armies — 25 `Discrete.__init__` calls per step purely to run
        `.contains()`. Handing out one shared instance is safe because nothing
        samples from it: `WargameEnv.action_space` is a separate object and is
        the only one `.sample()` is called on, so no RNG stream is shared.
        """
        if self._action_space is None:
            self._action_space = spaces.Tuple(
                [spaces.Discrete(self.n_actions) for _ in range(self._n_models)]
            )
        return self._action_space

    def decode_action(self, action: int) -> np.ndarray:
        """Return the (dx, dy) displacement for *action*."""
        if action == STAY_ACTION:
            return zero_position()
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
        *,
        phase: BattlePhase = BattlePhase.movement,
        enemy_models: list[Any] | None = None,
    ) -> None:
        """Apply the action tuple to the wargame models (mutates locations).

        Dead models are skipped — they do not move regardless of the action.
        Shooting-slice actions are no-ops (Phase 5 adds resolution).
        Models displace only in the movement phase: the action mask already
        enforces that for a learned policy, but scripted policies bypass the
        mask, so honouring `phase` here keeps them on the same footing.

        With bases, moves are resolved against the other models: `enemy_models`
        stop a move on contact, the moving army's own models may be passed
        through but not ended on. Resolution runs in model index order, which
        gives model 0 a documented right of way — the price of a deterministic
        board.
        """
        # Hoisted, and typed: python-list bounds become int64 arrays, so passing
        # them to `np.clip` would widen the result whatever the inputs are.
        lower = position(self._base_radius, self._base_radius)
        upper = position(
            board_width - self._base_radius, board_height - self._base_radius
        )
        collides = self._base_radius > 0.0
        blocker_centres, blocker_radii = _base_arrays(
            enemy_models if collides else None
        )
        n_models = len(wargame_models)
        friendly_buffer = np.zeros((n_models, 2), dtype=float)
        friendly_radius_buffer = np.array(
            [m.base_radius for m in wargame_models], dtype=float
        )
        friendly_alive = np.array([m.is_alive for m in wargame_models], dtype=bool)
        for i, act in enumerate(action.actions):
            model = wargame_models[i]
            if not model.is_alive:
                continue
            if not action_space[i].contains(act):  # type: ignore
                raise ValueError(
                    f"Action {act} for wargame model {i} is out of bounds."
                )
            if phase is not BattlePhase.movement:
                continue
            if (
                self._shooting_slice is not None
                and self._shooting_slice.start <= act < self._shooting_slice.end
            ):
                continue
            model.previous_location = model.location.copy()
            displacement = self.decode_action(act)
            if not collides:
                model.location = np.clip(model.location + displacement, lower, upper)
                continue
            # Read live each iteration: earlier models in this loop have already
            # moved, and a model must not end on ground another just took. The
            # arrays are rebuilt from a preallocated buffer rather than a fresh
            # list comprehension per model -- that shape is O(n^2) in python and
            # is what made two reward calculators 80% of a step once already.
            for j, other in enumerate(wargame_models):
                friendly_buffer[j] = other.location
            keep = friendly_alive.copy()
            keep[i] = False
            friendly_centres = friendly_buffer[keep]
            friendly_radii = friendly_radius_buffer[keep]
            # The board edge is clamped into the *displacement*, before
            # collisions are resolved. Clamping the resolved point afterwards
            # would slide a model along the edge and back into someone else --
            # producing exactly the overlap the whole resolution just avoided,
            # only near a board edge and only sometimes.
            in_bounds = np.clip(model.location + displacement, lower, upper)
            model.location = resolve_move(
                model.location,
                in_bounds - model.location,
                model.base_radius,
                blocker_centres,
                blocker_radii,
                friendly_centres,
                friendly_radii,
            )
