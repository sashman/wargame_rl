"""Action space and application for the wargame environment.

Polar coordinate movement: each model picks an (angle, speed) pair or stays
still. The displacement is applied exactly — the board is continuous, so a
speed bin means the distance it says in every direction.

The ``ActionRegistry`` partitions the flat action space into contiguous slices
(stay, movement, and future phase-specific slices) and provides phase-aware
action masks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.coherency_enforcement import (
    CoherencyEnforcement,
    enforce_after_move,
)
from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.domain.movement import back_off_to_unengaged, resolve_move
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.value_objects import (
    POSITION_DTYPE,
    position,
    zero_position,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

STAY_ACTION = 0

# The advance roll is one D6 (`wargame.py::_roll_advance`), so the most an
# advance can add to a model's Move is 6". The bin ladder is built from this,
# which is what keeps an action index meaning the same distance every turn.
ADVANCE_DIE_FACES = 6.0

# Every slice name `ActionHandler` can register. A name outside this set in
# `dark_action_slices` is a typo that would silently darken nothing, so it
# raises rather than doing nothing.
KNOWN_ACTION_SLICES = frozenset({"stay", "movement", "shooting", "advance"})

# The move-type slice, in declaration order. `normal` first so that a policy
# emitting the slice's first action declares the default, and STAY in the
# command phase means the same thing -- which is what keeps every policy that
# does not act in the command phase working unchanged.
MOVE_TYPE_NORMAL = 0
MOVE_TYPE_ADVANCE = 1
N_MOVE_TYPES = 2


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

# The two phases in which a model actually moves. A charge is an ordinary move
# except for where it is allowed to END -- see `apply`.
_DISPLACING_PHASES: frozenset[BattlePhase] = frozenset(
    {BattlePhase.movement, BattlePhase.charge}
)


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
        model_moves: Sequence[float | None] | None = None,
    ) -> None:
        self._n_models = (
            n_models if n_models is not None else config.number_of_wargame_models
        )
        n_angles = config.n_movement_angles
        n_speeds = config.n_speed_bins
        n_advance_bins = config.n_advance_speed_bins
        # Through the resolver rather than off the config, so that a board using
        # a scale other than 1 inch per unit moves models the right distance.
        quantities = resolve_rules_quantities(config)
        max_speed = quantities.max_move_speed
        self._engagement_range = float(quantities.engagement_range)
        self._melee_enabled = bool(config.melee.enabled)
        # A model with a base cannot stand with half of it off the table.
        self._base_radius = quantities.base_radius
        # Resolved once, like every other rules distance. The mode is read here
        # rather than passed per call so scripted policies and the learned one
        # go through exactly the same enforcement.
        self._coherency_mode = CoherencyEnforcement(config.coherency.enforce_move)
        self._coherency_nearest = quantities.scale.to_units(
            config.coherency.nearest_distance
        )
        self._coherency_furthest = quantities.scale.to_units(
            config.coherency.furthest_distance
        )
        self.models_reverted_last_move = 0
        # (units, units_coherent, models_out) for the move as PROPOSED, before
        # enforcement. `None` outside the movement phase.
        self.intended_coherency_last_move: tuple[int, int, int] | None = None

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

        # Per-model Move (the rules' M). Resolved here, once, like every other
        # rules distance -- and only materialised when a config actually asks
        # for differing speeds.
        #
        # The shared table above is kept verbatim for the uniform case rather
        # than being re-derived as fractions x M, because the two are NOT the
        # same float: at M = 6, `6.0 / 6` is exactly 1.0 while
        # `linspace(1/6, 1, 6)[0] * 6` is 0.9999999999999999. Rebuilding it
        # would move every model on the board by an ULP and fail the golden
        # gates, for no gain on the configs that do not use this.
        self._move_speeds = np.full(self._n_models, max_speed, dtype=float)
        for index, move in enumerate(model_moves or ()):
            if index >= self._n_models:
                break
            if move is not None:
                self._move_speeds[index] = quantities.scale.to_units(move)
        if np.all(self._move_speeds == max_speed):
            self._model_displacements: np.ndarray | None = None
        else:
            per_model = (
                self._unit_directions[np.newaxis, :, np.newaxis, :]
                * np.linspace(
                    self._move_speeds / n_speeds,
                    self._move_speeds,
                    n_speeds,
                    axis=-1,
                )[:, np.newaxis, :, np.newaxis]
            )  # (n_models, n_angles, n_speeds, 2)
            self._model_displacements = per_model.astype(POSITION_DTYPE)

        self._n_move_actions = n_angles * n_speeds
        self._n_speed_bins = n_speeds
        self._n_angles = n_angles
        # Built lazily rather than here: `n_actions` depends on the registry,
        # which is populated below.
        self._action_space: spaces.Tuple | None = None

        self._registry = ActionRegistry()
        # A darkened slice keeps its width and loses every phase, so its actions
        # are masked for the whole episode while the policy head stays the same
        # shape. See `WargameEnvConfig.dark_action_slices`.
        dark = frozenset(config.dark_action_slices)
        unknown = dark - KNOWN_ACTION_SLICES
        if unknown:
            raise ValueError(
                f"dark_action_slices names unknown slices: {sorted(unknown)}. "
                f"Known slices: {sorted(KNOWN_ACTION_SLICES)}"
            )

        def phases(name: str, valid: frozenset[BattlePhase]) -> frozenset[BattlePhase]:
            return frozenset() if name in dark else valid

        self._registry.register("stay", 1, phases("stay", ALL_BATTLE_PHASES))
        # ⚠ The charge MOVE reuses this slice rather than adding one. Reach is
        # not the constraint it looks like: `back_off_to_unengaged` parks every
        # mover 8.7 MICRO-inches outside contact, so the distance a charge has
        # to cover is one speed bin, not a 2D6 roll. A dedicated absolute-rung
        # ladder would be ~80 actions bought to cross a hundred-thousandth of an
        # inch -- and it would change the output head's shape, which is what
        # makes an action-space arm unpairable against its control.
        self._registry.register(
            "movement",
            self._n_move_actions,
            phases(
                "movement",
                _DISPLACING_PHASES
                if config.melee.enabled
                else frozenset({BattlePhase.movement}),
            ),
        )

        if n_shoot_targets > 0:
            self._shooting_slice: ActionSlice | None = self._registry.register(
                "shooting",
                n_shoot_targets,
                phases("shooting", frozenset({BattlePhase.shooting})),
            )
        else:
            self._shooting_slice = None

        # ⚠ Registered LAST, so every action index that existed before advance
        # still means what it meant. Widening `n_speed_bins` instead would
        # renumber the movement slice (`decode_action` is angle-major,
        # speed-minor), and `_apply_warm_start_weights` loads with
        # `strict=False`, so every checkpoint would load and be wrong.
        self._n_advance_bins = n_advance_bins
        if n_advance_bins > 0:
            self._advance_slice: ActionSlice | None = self._registry.register(
                "advance",
                n_angles * n_advance_bins,
                phases("advance", frozenset({BattlePhase.movement})),
            )
        else:
            self._advance_slice = None

        # The DECLARATION, and the reason it is a slice of its own rather than a
        # property of a movement action. A move type is a unit's choice in the
        # rules; folding it into the per-model movement action makes it five
        # choices resolved by an OR, so one model's exploration spends four
        # squadmates' shooting. Splitting it also decouples the type from the
        # DISTANCE: a unit that has declared an advance can still move members
        # short, which is what keeps it in coherency, and a leader-declares rule
        # would be unimplementable without that split.
        #
        # Darkening "advance" darkens this too, so the rungs become unreachable
        # by the same switch -- a rung is legal only where a declaration was made.
        if n_advance_bins > 0:
            self._move_type_slice: ActionSlice | None = self._registry.register(
                "move_type",
                N_MOVE_TYPES,
                phases("advance", frozenset({BattlePhase.command})),
            )
        else:
            self._move_type_slice = None

    @property
    def move_type_slice(self) -> ActionSlice | None:
        """Move-type declaration slice, or None when the scenario has no advance."""
        return self._move_type_slice

    def declare_move_types(
        self, action: WargameEnvAction, wargame_models: list[Any]
    ) -> None:
        """Record each unit's declared move type from its LEADER's action.

        The leader is the lowest-indexed alive model of the unit. One model
        decides and the whole unit is bound, which is what the rules mean by a
        unit choosing a move type -- and it is only safe because the declaration
        no longer carries a distance: every squadmate keeps the whole movement
        slice and can stop short to hold formation.

        ⚠ STAY declares `normal`, so any policy that does not act in the command
        phase behaves exactly as it did before the declaration existed.

        Declaring an advance spends the unit's shooting immediately, whether or
        not a member then uses a long rung. That is the rules' cost: it attaches
        to the move type, not to the distance travelled.
        """
        if self._move_type_slice is None:
            return
        start = self._move_type_slice.start
        leaders: dict[int, int] = {}
        for index, model in enumerate(wargame_models):
            if not model.is_alive:
                continue
            leaders.setdefault(int(model.group_id), index)
        advancing = set()
        for group, leader in leaders.items():
            if leader >= len(action.actions):
                continue
            if int(action.actions[leader]) == start + MOVE_TYPE_ADVANCE:
                advancing.add(group)
        for model in wargame_models:
            declared = int(model.group_id) in advancing
            model.declared_advance = declared
            if declared:
                model.advanced_this_turn = True

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

    @property
    def move_speeds(self) -> np.ndarray:
        """Per-model Move in board units — ``(n_models,)``.

        Every entry is the scenario's ``max_move_speed`` unless the model's
        config overrode it. Scripted policies read this rather than the config
        so they step at the speed the handler will actually give them.
        """
        speeds: np.ndarray = self._move_speeds
        return speeds

    def decode_action(
        self,
        action: int,
        model_idx: int | None = None,
        advance_roll: float = 0.0,
    ) -> np.ndarray:
        """Return the (dx, dy) displacement for *action*.

        ``model_idx`` selects that model's own speed bins when the scenario
        gives its models different Move characteristics. Omitting it uses the
        shared table, which is the whole table whenever every model is equally
        fast — every config shipped today.

        ``advance_roll`` is this model's unit's D6 for the turn, and is read
        only for actions in the advance slice. It defaults to 0 so that every
        existing caller — five scripted baselines, both opponent policies, the
        joint decoder and the debug session — is unchanged: with no advance
        slice registered, no action can reach the branch that reads it.
        """
        if action == STAY_ACTION:
            return zero_position()
        if (
            self._advance_slice is not None
            and self._advance_slice.start <= action < self._advance_slice.end
        ):
            return self._advance_displacement(action, model_idx, advance_roll)
        move_idx = action - 1
        angle_idx = move_idx // self._n_speed_bins
        speed_idx = move_idx % self._n_speed_bins
        if model_idx is not None and self._model_displacements is not None:
            per_model: np.ndarray = self._model_displacements[
                model_idx, angle_idx, speed_idx
            ]
            return per_model
        result: np.ndarray = self._displacements[angle_idx, speed_idx]
        return result

    def advance_distance(self, bin_idx: int, move: float) -> float:
        """Absolute distance of an advance bin, in inches. Always beyond `move`.

        The ladder is `M + (bin + 1) x (6 / bins)`, so at `M = 6` and three bins
        it is 8", 10", 12" -- fixed rungs above the model's Move, not fractions
        of `M + roll`.

        Two defects of the fractional encoding go with this. **Stationary
        semantics**: an action index means the same displacement every turn, so
        a policy does not have to read `advance_roll` to know what its own
        action does; the roll now decides only which rungs are LEGAL, which is
        what the mask is for. **No dominated actions**: every rung is beyond a
        normal move's reach, so no advance can spend the unit's shooting for a
        distance a normal move already delivers.

        ⚠ The reason previously recorded for admitting dominated bins -- that a
        unit which cannot stop short cannot advance and halt to keep coherency
        -- does not hold, and was verified against `env.step`. Only ONE model
        need choose an advance for the unit to advance; its squadmates keep the
        whole normal slice and stop wherever they like.
        """
        return move + (bin_idx + 1) * (ADVANCE_DIE_FACES / self._n_advance_bins)

    def _advance_displacement(
        self, action: int, model_idx: int | None, advance_roll: float
    ) -> np.ndarray:
        """Displacement for an advance action, at a fixed distance above Move.

        `advance_roll` is not read: the rung is absolute. The roll gates which
        rungs are legal, through `advance_legality`, and an over-long rung that
        reaches here despite the mask is clamped to `M + roll` so the rules'
        maximum still holds.
        """
        index = action - self._advance_slice.start  # type: ignore[union-attr]
        angle_idx = index // self._n_advance_bins
        bin_idx = index % self._n_advance_bins
        move = (
            float(self._move_speeds[model_idx])
            if model_idx is not None
            else float(self._speeds[-1])
        )
        distance = min(self.advance_distance(bin_idx, move), move + advance_roll)
        direction = self._unit_directions[angle_idx]
        displacement: np.ndarray = (direction * distance).astype(POSITION_DTYPE)
        return displacement

    def advance_legality(self, models: list[Any]) -> np.ndarray:
        """`(n_models, n_advance_actions)` -- which rungs this turn's rolls allow.

        Two gates, and both are the point. A rung is legal only for a model
        whose unit **declared an advance** in the command phase, and only when
        its absolute distance is within that model's `M + roll`.

        The declaration gate is what makes the move type a unit decision: no
        model can reach a long rung on its own, and every model of a unit that
        did declare can, so the unit still moves as a body. The roll gate is
        what lets the rungs be absolute -- with three bins a roll of 1 leaves
        none legal, which is a resolution limit rather than a bug, since the
        rules' 7" advance never repays a turn of fire.
        """
        if self._advance_slice is None:
            return np.zeros((len(models), 0), dtype=bool)
        n_bins = self._n_advance_bins
        legality = np.zeros((len(models), self._advance_slice.size), dtype=bool)
        for index, model in enumerate(models):
            if not getattr(model, "declared_advance", False):
                continue
            move = float(self._move_speeds[index])
            reach = move + float(model.advance_roll)
            allowed = np.array(
                [self.advance_distance(b, move) <= reach for b in range(n_bins)],
                dtype=bool,
            )
            legality[index] = np.tile(allowed, self._n_angles)
        return legality

    @property
    def movement_slice(self) -> ActionSlice:
        """The normal-move action slice. Always registered."""
        return self._registry.slice_for("movement")

    @property
    def advance_slice(self) -> ActionSlice | None:
        """Advance action slice, or None when the scenario has no advance bins."""
        return self._advance_slice

    def encode_action(self, angle_idx: int, speed_idx: int) -> int:
        """Encode an (angle_idx, speed_idx) pair into an action integer."""
        return 1 + angle_idx * self._n_speed_bins + speed_idx

    def best_action_toward(
        self,
        dx: float,
        dy: float,
        max_step_length: float | None = None,
        model_idx: int | None = None,
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
                disp = self.decode_action(
                    self.encode_action(angle_idx, s), model_idx=model_idx
                )
                if np.linalg.norm(disp) <= max_step_length:
                    speed_idx = s
                    break
            else:
                speed_idx = 0
        else:
            speed_idx = self._n_speed_bins - 1

        return self.encode_action(angle_idx, speed_idx)

    def best_advance_toward(
        self,
        dx: float,
        dy: float,
        advance_roll: float,
        max_step_length: float | None = None,
        model_idx: int | None = None,
    ) -> int | None:
        """The advance action moving closest to `(dx, dy)`, or None if unavailable.

        The advance counterpart of `best_action_toward`: same nearest-angle
        choice, then the largest bin whose distance does not exceed
        `max_step_length`. Returns None when the scenario registers no advance
        slice, so a caller can fall back to a normal move without branching on
        the config.

        `advance_roll` is the unit's own D6 for this turn (`model.advance_roll`).
        The rungs are absolute, so the roll no longer changes what an action
        means -- it decides which rungs are legal, and this returns None when it
        leaves none, so a caller falls back to a normal move without branching.
        """
        if self._advance_slice is None:
            return None
        # A DARKENED slice is registered but valid in no phase, so its actions
        # are masked. Returning one would hand the caller an illegal action --
        # and it is what makes a darkened config a true "advance off" control
        # for a scripted policy, rather than one that silently emits moves the
        # mask forbids.
        if BattlePhase.movement not in self._advance_slice.valid_phases:
            return None
        if dx == 0.0 and dy == 0.0:
            return None

        target_angle = np.arctan2(dy, dx) % (2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, self._n_angles, endpoint=False)
        diffs = np.abs(angles - target_angle)
        diffs = np.minimum(diffs, 2 * np.pi - diffs)
        angle_idx = int(np.argmin(diffs))

        move = (
            float(self._move_speeds[model_idx])
            if model_idx is not None
            else float(self._speeds[-1])
        )
        reach = move + advance_roll
        # The longest rung that is both legal for this roll and no further than
        # the caller asked to travel. Descending, so a squad marching a bounded
        # distance takes the biggest step it is allowed rather than the first.
        ceiling = reach if max_step_length is None else min(reach, max_step_length)
        bin_idx = None
        for candidate in range(self._n_advance_bins - 1, -1, -1):
            if self.advance_distance(candidate, move) <= ceiling:
                bin_idx = candidate
                break
        if bin_idx is None:
            return None

        return self._advance_slice.start + angle_idx * self._n_advance_bins + bin_idx

    def _displaces_in(self, phase: BattlePhase) -> bool:
        """Does a movement action actually move a model in this phase?

        ⚠ Not simply "is this a displacing phase". `apply` validates an action
        against the action SPACE, which is phase-independent, so a scripted
        policy emitting a movement action in the charge phase would displace
        even on a config where the charge phase is a stub. The authority is the
        movement slice's own `valid_phases`, which is what the mask is built
        from -- so the mask and the resolver cannot disagree.
        """
        if phase is BattlePhase.movement:
            return True
        movement = self.movement_slice
        return (
            phase in _DISPLACING_PHASES
            and movement is not None
            and phase in movement.valid_phases
        )

    def _engaged_before_moving(
        self,
        phase: BattlePhase,
        wargame_models: list[Any],
        enemy_models: list[Any] | None,
    ) -> np.ndarray | None:
        """Which models start this movement phase engaged, or None if it cannot matter.

        Read BEFORE anything displaces: the rules make eligibility a property of
        the unit's state at the start of its move, and by the end of the phase
        the unit has (by construction) left engagement, since
        `back_off_to_unengaged` guarantees every endpoint is clear.
        """
        if not self._melee_enabled or phase is not BattlePhase.movement:
            return None
        alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
        if not alive_enemies:
            return None
        return engaged_with_any(
            np.array([m.location for m in wargame_models], dtype=float),
            np.array([m.location for m in alive_enemies], dtype=float),
            np.ones(len(alive_enemies), dtype=bool),
            engagement_range=self._engagement_range,
            base_diameter=2.0 * self._base_radius,
        )

    def _mark_fall_backs(
        self, wargame_models: list[Any], began_engaged: np.ndarray | None
    ) -> None:
        """A unit that began the phase engaged and moved has fallen back.

        `docs/rules/09-movement-phase.md`: a Normal move is eligible only for an
        UNENGAGED unit, so the only move an engaged unit may make is a fall
        back — M distance, must end unengaged, and **until the end of the turn
        it cannot shoot or declare a charge**.

        ⚠ The geometry needed nothing: `back_off_to_unengaged` already forces
        every endpoint out of engagement, which IS the fall-back constraint.
        What was missing is the COST. Before this, an engaged model took an
        ordinary move, walked out for free, and shot in the same turn.

        v1 infers the declaration rather than asking for one: a unit that began
        engaged and moved has fallen back. The rules would have it *declare* a
        move type, but a declaration lives in the command phase, which most
        configs skip — and inferring it costs no action and no agent step.
        `DEFERRED: fallback.declared_move_type`.

        Unit-level, because the rule is: a model that did not move is still in a
        unit that withdrew, and the unit is what loses its shooting.
        """
        if began_engaged is None or not began_engaged.any():
            return
        moved_units: set[int] = set()
        for index, model in enumerate(wargame_models):
            if not began_engaged[index] or not model.is_alive:
                continue
            previous = getattr(model, "previous_location", None)
            if previous is not None and not np.array_equal(previous, model.location):
                moved_units.add(int(model.group_id))
        if not moved_units:
            return
        for model in wargame_models:
            if int(model.group_id) in moved_units:
                model.fell_back_this_turn = True

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
        # A move must END unengaged (`09-movement-phase.md`); passing through an
        # engagement range is explicitly legal (`03-moving.md`). So the rings are
        # applied to the endpoint only, never to the path.
        # ⚠ THE CHARGE EXEMPTION, and it is the whole mechanic. A charge is the
        # one move whose endpoint MAY lie inside an enemy's engagement range --
        # `11-charge-phase.md` requires it to. Dropping the rings here (and only
        # here) is what lets contact happen at all; `back_off_to_unengaged`
        # still receives the OCCUPIED bases, so an endpoint may be engaged but
        # never inside another model. That primitive already existed: the
        # function merges enemy rings and occupied bases into one walk, so empty
        # rings plus populated bases is exactly "may be engaged, may not
        # overlap" with no edit to it.
        charging = phase is BattlePhase.charge and self._displaces_in(phase)
        alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
        if not charging and self._engagement_range > 0.0 and alive_enemies:
            engagement_centres = np.array(
                [m.location for m in alive_enemies], dtype=float
            )
            engagement_reach = np.array(
                [
                    self._engagement_range + float(m.base_radius) + self._base_radius
                    for m in alive_enemies
                ],
                dtype=float,
            )
        else:
            engagement_centres = np.empty((0, 2), dtype=float)
            engagement_reach = np.empty(0, dtype=float)
        n_models = len(wargame_models)
        friendly_buffer = np.zeros((n_models, 2), dtype=float)
        friendly_radius_buffer = np.array(
            [m.base_radius for m in wargame_models], dtype=float
        )
        friendly_alive = np.array([m.is_alive for m in wargame_models], dtype=bool)
        began_engaged = self._engaged_before_moving(phase, wargame_models, enemy_models)
        if phase is BattlePhase.command:
            # No early return: the per-model loop below still validates every
            # action against its space before skipping non-movement phases, and
            # an unvalidated declaration would be a silent out-of-range action.
            self.declare_move_types(action, wargame_models)
        for i, act in enumerate(action.actions):
            model = wargame_models[i]
            if not model.is_alive:
                continue
            if not action_space[i].contains(act):  # type: ignore
                raise ValueError(
                    f"Action {act} for wargame model {i} is out of bounds."
                )
            if not self._displaces_in(phase):
                continue
            if (
                self._shooting_slice is not None
                and self._shooting_slice.start <= act < self._shooting_slice.end
            ):
                continue
            model.previous_location = model.location.copy()
            displacement = self.decode_action(
                act, model_idx=i, advance_roll=model.advance_roll
            )
            if not collides:
                model.location = back_off_to_unengaged(
                    model.location,
                    np.clip(model.location + displacement, lower, upper),
                    engagement_centres,
                    engagement_reach,
                )
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
            # The bases the endpoint may not land inside: enemies (which also
            # block the path) and the moving army's own models (which may be
            # crossed but not ended on). Passed in because backing off walks the
            # endpoint into ground `resolve_move` had already cleared -- without
            # them a model rescued from an engagement ring comes to rest inside
            # a friendly base, measured at 0.18% of pairs.
            occupied_centres = np.concatenate([blocker_centres, friendly_centres])
            occupied_reach = (
                np.concatenate([blocker_radii, friendly_radii]) + model.base_radius
            )
            model.location = back_off_to_unengaged(
                model.location,
                resolve_move(
                    model.location,
                    in_bounds - model.location,
                    model.base_radius,
                    blocker_centres,
                    blocker_radii,
                    friendly_centres,
                    friendly_radii,
                ),
                engagement_centres,
                engagement_reach,
                occupied_centres,
                occupied_reach,
            )

        self._mark_fall_backs(wargame_models, began_engaged)

        # Every model in the force has now moved, which is the earliest point a
        # unit-level property of the *completed* move can be judged. Nothing to
        # do outside the movement phase -- no model displaced.
        if phase is BattlePhase.movement:
            # Judge the move the policy ACTUALLY MADE, before the referee edits
            # it. Sampling only after enforcement is what let a whole
            # investigation report 1.000 compliance for a policy that intends
            # 0.630 -- the metric was measuring the wrapper. Costs one extra
            # `evaluate_coherency` per movement phase, which is the same call
            # the tracker already makes once.
            self.intended_coherency_last_move = _intent_counts(
                wargame_models, self._coherency_nearest, self._coherency_furthest
            )
            self.models_reverted_last_move = enforce_after_move(
                wargame_models,
                self._coherency_nearest,
                self._coherency_furthest,
                self._coherency_mode,
            )


def _intent_counts(
    models: list[Any], nearest: float, furthest: float
) -> tuple[int, int, int]:
    """(units, units coherent, models out) for the move as the policy made it.

    Evaluated on the moved-but-not-yet-corrected positions, so it reports what
    the policy *chose* rather than what the referee left behind. Under
    enforcement the two diverge completely -- a policy intending 0.630
    coherency reads 1.000 once the revert has run.
    """
    alive = np.array([m.is_alive for m in models], dtype=bool)
    if not alive.any():
        return (0, 0, 0)
    report = evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=alive,
        base_radii=np.array([m.base_radius for m in models], dtype=float),
        nearest_distance=nearest,
        furthest_distance=furthest,
    )
    coherent = sum(1 for unit in report.units if unit.coherent)
    return (len(report.units), coherent, report.n_models_out_of_coherency)
