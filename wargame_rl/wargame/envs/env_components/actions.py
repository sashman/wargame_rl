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
from enum import Enum
from typing import Any

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.coherency_enforcement import (
    CoherencyEnforcement,
    enforce_after_move,
)
from wargame_rl.wargame.envs.domain.engagement import (
    engaged_with_any,
    engagement_matrix,
)
from wargame_rl.wargame.envs.domain.movement import back_off_to_unengaged, resolve_move
from wargame_rl.wargame.envs.domain.pile_in import (
    SELECTION_RANGE_INCHES,
    agent_move_is_legal,
)
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
# The largest total the charge's two D6 can show. Used only to normalise the
# roll for the observation -- the legality gate reads the roll in inches.
CHARGE_DICE_MAX = 12.0
# Floating-point slack for the charge-distance re-check. `resolve_move` clamps
# and backs off, so a legal move's realised length can differ from its rung's
# nominal length in the last bits; a bare `>` would revert legal charges.
_CHARGE_REACH_EPSILON = 1e-6

# Every slice name `ActionHandler` can register. A name outside this set in
# `dark_action_slices` is a typo that would silently darken nothing, so it
# raises rather than doing nothing.
KNOWN_ACTION_SLICES = frozenset(
    {"stay", "movement", "shooting", "advance", "move_type", "fight_order"}
)

# How many activation priorities a unit can declare in the fight phase.
# ⚠ The rules let the controlling player pick ANY eligible unit each time it is
# their turn to select, which for k units is a k! ordering and cannot be a
# simultaneous per-model action. A coarse PRIORITY is the expressible form of
# the same decision: every unit commits a level up front, and the engine selects
# the highest still eligible. Four levels keeps the slice at four actions while
# giving a real ordering; ties fall back to the lowest unit index, which is
# exactly what the engine did before, so a policy that declares nothing behaves
# as it always did.
FIGHT_ORDER_LEVELS = 4

# The move-type slice, in declaration order. `normal` first so that a policy
# emitting the slice's first action declares the default, and STAY in the
# command phase means the same thing -- which is what keeps every policy that
# does not act in the command phase working unchanged.
# ⚠ These are OFFSETS INTO A SLICE WHOSE SIZE VARIES BY SCENARIO, not fixed
# action indices. `normal` is always 0; `advance` and `charge` take the next
# offsets only when the scenario actually has them. Sizing the slice to what
# exists is what keeps an advance-only config at exactly the action count it
# had before the charge declaration existed -- a fixed 3-wide slice would have
# widened every advance config's output head and orphaned its checkpoints for a
# value it can never declare. Resolve with `ActionHandler.move_type_offset`.
MOVE_TYPE_NORMAL = "normal"
MOVE_TYPE_ADVANCE = "advance"
MOVE_TYPE_CHARGE = "charge"


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

# The phases in which a model actually moves. A charge is an ordinary move
# except for where it is allowed to END -- see `apply`. Pile-in and consolidate
# are ordinary moves capped far shorter, with their own after-conditions.
_DISPLACING_PHASES: frozenset[BattlePhase] = frozenset(
    {
        BattlePhase.movement,
        BattlePhase.charge,
        BattlePhase.pile_in,
        BattlePhase.consolidate,
    }
)

# `12-fight-phase.md`: a pile-in and a consolidation are both "Maximum distance
# 3"". One ladder serves both.
SHORT_MOVE_MAX_INCHES = 3.0


class MoveLadder(str, Enum):
    """Which distance table the movement slice decodes to.

    ⚠ The same action index means a different DISTANCE in different phases, and
    that is deliberate: it is what lets the charge and the pile-in reuse the
    movement slice instead of buying new actions. It is safe here where it was
    not for the advance because phase is deterministic, observable and constant
    within a step, whereas the advance's old mapping moved with a per-turn die.
    """

    normal = "normal"
    charge = "charge"
    short = "short"


# Phases whose move is judged as a WHOLE UNIT and reverted entire when illegal.
_UNIT_REFEREED_PHASES: frozenset[BattlePhase] = frozenset(
    {BattlePhase.charge, BattlePhase.pile_in, BattlePhase.consolidate}
)

# Phases whose move MAY end inside an enemy's engagement range -- the charge that
# makes contact, and the two fight-phase moves that keep it.
_ENGAGING_ENDPOINT_PHASES: frozenset[BattlePhase] = frozenset(
    {BattlePhase.charge, BattlePhase.pile_in, BattlePhase.consolidate}
)

_LADDER_FOR_PHASE: dict[BattlePhase, MoveLadder] = {
    BattlePhase.charge: MoveLadder.charge,
    BattlePhase.pile_in: MoveLadder.short,
    BattlePhase.consolidate: MoveLadder.short,
}


def ladder_for_phase(phase: BattlePhase | None) -> MoveLadder:
    """Which distance table the movement slice decodes to in `phase`.

    Public because the joint decoder must build its forward model on the SAME
    table the env will apply. It was handed a bool for the charge and would have
    modelled every pile-in rung at Move -- the shape of the advance defect that
    manufactured 32-43% of the advances executed at play.
    """
    if phase is None:
        return MoveLadder.normal
    return _LADDER_FOR_PHASE.get(phase, MoveLadder.normal)


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
        self._charge_approach_mask = bool(
            config.melee.enabled and config.melee.charge_approach_mask
        )
        self._charge_range = float(quantities.scale.to_units(config.melee.charge_range))
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

        # ⚠ **THE CHARGE LADDER, and it is MOVE-INDEPENDENT on purpose.**
        # `docs/rules/11-charge-phase.md` § Charge move: *"Maximum distance |
        # The charge roll."* Move does not cap a charge in the rules, so this
        # grid spans the 2D6's own range -- rung `s` travels
        # `(s + 1) x (12 / n_speeds)`, i.e. 2"/4"/6"/8"/10"/12" at six bins.
        # The 2D6 mask in `charge_legality` decides which of them are LEGAL.
        #
        # ⚠ **This closes `DEFERRED: charge.beyond_move_ladder`**, which made
        # the charge reuse the *movement* ladder and so capped it at Move (6").
        # Measured on the shipped config: the roll exceeds Move on **57.5-59.1%**
        # of declarations and **13.8-15.6%** of eligible declarations were
        # blocked by the cap rather than by the dice. The rules' whole reason to
        # declare a charge is that 2D6 can carry a unit FURTHER than it could
        # walk; capped at Move the gamble had its upside removed.
        #
        # ⚠ **It makes the movement slice mean a different DISTANCE in the
        # charge phase**, which is the defect the advance re-encoding removed --
        # so the distinction matters. The advance ladder changed meaning with a
        # per-turn DIE, so a policy had to read `advance_roll` to know what its
        # own action did, and the mapping moved under it turn to turn. Phase is
        # deterministic, observable (`normalized_phase` is in the game tensor)
        # and constant within a step, and in the charge phase the movement slice
        # is legal ONLY for a declared unit -- so there is no state in which the
        # two meanings compete. It costs **zero** new actions, where a dedicated
        # ladder would add 96 and orphan nothing only because no melee
        # checkpoint exists.
        # ⚠ Through the SCALE, like every other rules distance. `CHARGE_DICE_MAX`
        # is 2D6's maximum in INCHES; the displacement grid is in board units.
        # An audit found `charge_roll` itself still compared raw against board
        # units -- a latent trap while `inches_per_unit` is 1 everywhere, and a
        # silent one the moment a board uses another unit. `_charge_reach`
        # converts on the other side of the same comparison.
        self._charge_scale = quantities.scale
        # The rules' *"end within 1 inch of a target if it can"* clause.
        self._charge_touch_range = float(quantities.scale.to_units(1.0))
        step = float(quantities.scale.to_units(CHARGE_DICE_MAX)) / n_speeds
        # The pile-in / consolidate ladder: `SHORT_MOVE_MAX_INCHES` split into
        # the same number of rungs, so rung `s` travels `(s + 1) x (3 / bins)`.
        # Move-independent, like the charge: the rules cap both at a distance,
        # not at the model's Move.
        short_step = float(quantities.scale.to_units(SHORT_MOVE_MAX_INCHES)) / n_speeds
        self._short_displacements = (
            self._unit_directions[:, np.newaxis, :]
            * (np.arange(1, n_speeds + 1) * short_step)[np.newaxis, :, np.newaxis]
        ).astype(POSITION_DTYPE)
        self._charge_displacements = (
            self._unit_directions[:, np.newaxis, :]
            * (np.arange(1, n_speeds + 1) * step)[np.newaxis, :, np.newaxis]
        ).astype(POSITION_DTYPE)

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
        # ⚠ Sized to the move types this scenario HAS. `normal` is free (it is
        # what STAY declares), so the slice carries one action per *optional*
        # type. An advance-only config therefore keeps exactly the 2-wide slice
        # it had before the charge declaration existed, and its checkpoints
        # still load.
        # ⚠ **`or "move_type" in dark` is what keeps the arm PAIRABLE**, and it
        # is the mechanism this repo already uses for the advance. The melee arm
        # and its dark control differ in exactly one scalar -- `melee.enabled`
        # -- so they share an init; registering the declaration on that scalar
        # alone would give them 104 and 102 actions, different output heads and
        # different weights at step 0. The control names `move_type` in
        # `dark_action_slices`, which registers the slice and makes it valid in
        # NO phase: same shape, permanently inert.
        #
        # Gating on "the charge phase is stepped" instead was tried and is
        # WRONG in the other direction: `skip_phases: []` is a documented
        # setting that five test modules use, and it would have quietly given
        # every one of them an extra action and a real choice in a command
        # phase that used to offer only STAY.
        carries_charge = self._melee_enabled or "move_type" in dark
        self._move_types: tuple[str, ...] = (
            MOVE_TYPE_NORMAL,
            *((MOVE_TYPE_ADVANCE,) if n_advance_bins > 0 else ()),
            *((MOVE_TYPE_CHARGE,) if carries_charge else ()),
        )
        if len(self._move_types) > 1:
            self._move_type_slice: ActionSlice | None = self._registry.register(
                "move_type",
                len(self._move_types),
                phases("move_type", frozenset({BattlePhase.command})),
            )
        else:
            self._move_type_slice = None

        # The squad's PLAN: which objective it is committed to, declared by its
        # leader in the command phase, binding and PERSISTENT (STAY keeps it).
        # Registered LAST so no existing index moves, and only under
        # `declare_objectives` so every existing config keeps its exact action
        # space. Size is the objective BUDGET -- the tensor width -- with
        # legality masking off indices beyond the episode's real objectives.
        budget = config.objective_budget
        self._objective_budget = int(budget) if budget is not None else 0
        # ⚠ `getattr`, not attribute access: a config PICKLED before this
        # field existed (a checkpoint's hparams, a recording subprocess's
        # snapshot from a live run) must construct under new code. The
        # 2026-08-27 incident -- new builder code meeting an old config object
        # in a recording subprocess -- recurred the moment this field landed
        # under live trainers; a plain read crashes every such consumer.
        if bool(getattr(config, "declare_objectives", False)):
            self._objective_target_slice: ActionSlice | None = self._registry.register(
                "objective_target",
                self._objective_budget,
                phases("objective_target", frozenset({BattlePhase.command})),
            )
        else:
            self._objective_target_slice = None

        # ⚠ **WHO SWINGS FIRST WAS THE ENGINE'S CHOICE, and the rules make it the
        # player's.** `resolve_fight_step` picked `min(pool)` -- the lowest-indexed
        # eligible unit -- where `12-fight-phase.md` says *"players alternate
        # selecting one friendly eligible unit"*. In a game where merely making
        # activation alternate halved a 25.0 vp seat asymmetry, the order units
        # swing in is not bookkeeping.
        #
        # Registered LAST, after `move_type`, so no existing action index moves.
        # Declared by the unit's LEADER and binding the unit, exactly as the move
        # type is: a per-model priority would be five votes on one unit-level fact.
        # ⚠ STAY declares level 0 and ties break on the lowest unit index, which
        # is what the engine already did -- so a policy that never acts in the
        # fight phase selects in exactly the order it always did.
        if carries_charge:
            self._fight_order_slice: ActionSlice | None = self._registry.register(
                "fight_order",
                FIGHT_ORDER_LEVELS,
                phases("fight_order", frozenset({BattlePhase.fight})),
            )
        else:
            self._fight_order_slice = None

    @property
    def fight_order_slice(self) -> ActionSlice | None:
        """Activation-priority slice, or None where nothing fights."""
        return self._fight_order_slice

    def declare_objectives(
        self, action: WargameEnvAction, wargame_models: list[Any]
    ) -> None:
        """Record each unit's declared OBJECTIVE from its leader's action.

        The learning form of `baseline/reallocation.py`: the squad's plan as a
        first-class, leader-declared, unit-binding action. ⚠ Unlike the move
        type it PERSISTS -- `begin_turn` does not clear it -- so STAY means
        "keep the plan", and a squad re-plans only by declaring again. That is
        what makes it a commitment the execution reward can price rather than a
        per-turn impulse.
        """
        if self._objective_target_slice is None:
            return
        start = self._objective_target_slice.start
        end = self._objective_target_slice.end
        leaders: dict[int, int] = {}
        for index, model in enumerate(wargame_models):
            if not model.is_alive:
                continue
            leaders.setdefault(int(model.group_id), index)
        declared: dict[int, int] = {}
        for group, leader in leaders.items():
            if leader >= len(action.actions):
                continue
            chosen = int(action.actions[leader])
            if start <= chosen < end:
                declared[group] = chosen - start
        if not declared:
            return
        for model in wargame_models:
            group = int(model.group_id)
            if group in declared:
                model.declared_objective = declared[group]

    def objective_target_legality(
        self, models: list[Any], n_objectives: int
    ) -> np.ndarray:
        """`(n_models, budget)` -- which objective declarations exist to make.

        The slice is sized to the BUDGET (the tensor width); an episode drawing
        fewer objectives masks the padding indices off, exactly as the
        observation zero-pads the same rows. Alive models only.
        """
        if self._objective_target_slice is None:
            return np.zeros((len(models), 0), dtype=bool)
        legality = np.zeros(
            (len(models), self._objective_target_slice.size), dtype=bool
        )
        limit = min(int(n_objectives), self._objective_target_slice.size)
        alive = np.array([bool(m.is_alive) for m in models], dtype=bool)
        legality[alive, :limit] = True
        return legality

    @property
    def objective_target_slice(self) -> ActionSlice | None:
        """The objective-declaration slice, or None when the flag is off."""
        return self._objective_target_slice

    def declare_fight_order(
        self, action: WargameEnvAction, wargame_models: list[Any]
    ) -> None:
        """Record each unit's activation priority from its LEADER's action.

        Higher goes first. ⚠ Level 0 is what STAY declares, so a policy that does
        not act in the fight phase leaves every unit level 0 and the engine falls
        back to the lowest-index order it used before this existed.
        """
        if self._fight_order_slice is None:
            return
        start = self._fight_order_slice.start
        end = self._fight_order_slice.end
        leaders: dict[int, int] = {}
        for index, model in enumerate(wargame_models):
            if not model.is_alive:
                continue
            leaders.setdefault(int(model.group_id), index)
        priorities: dict[int, int] = {}
        for group, leader in leaders.items():
            if leader >= len(action.actions):
                continue
            chosen = int(action.actions[leader])
            priorities[group] = chosen - start if start <= chosen < end else 0
        for model in wargame_models:
            model.fight_priority = priorities.get(int(model.group_id), 0)

    @property
    def move_type_slice(self) -> ActionSlice | None:
        """Move-type declaration slice, or None when nothing may be declared."""
        return self._move_type_slice

    def move_type_offset(self, kind: str) -> int | None:
        """This move type's offset into the slice, or None when unavailable.

        Returns None rather than raising, so a scripted policy can ask for a
        declaration the scenario does not carry and fall through to a normal
        move -- the same contract `best_advance_toward` uses.
        """
        if self._move_type_slice is None or kind not in self._move_types:
            return None
        return self._move_types.index(kind)

    def move_type_action(self, kind: str) -> int | None:
        """The action index that declares `kind`, or None when unavailable."""
        offset = self.move_type_offset(kind)
        if offset is None or self._move_type_slice is None:
            return None
        return self._move_type_slice.start + offset

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
        advance_action = self.move_type_action(MOVE_TYPE_ADVANCE)
        charge_action = self.move_type_action(MOVE_TYPE_CHARGE)
        leaders: dict[int, int] = {}
        for index, model in enumerate(wargame_models):
            if not model.is_alive:
                continue
            leaders.setdefault(int(model.group_id), index)
        advancing: set[int] = set()
        charging: set[int] = set()
        for group, leader in leaders.items():
            if leader >= len(action.actions):
                continue
            chosen = int(action.actions[leader])
            if advance_action is not None and chosen == advance_action:
                advancing.add(group)
            elif charge_action is not None and chosen == charge_action:
                charging.add(group)
        for model in wargame_models:
            group = int(model.group_id)
            declared = group in advancing
            model.declared_advance = declared
            if declared:
                model.advanced_this_turn = True
            # ⚠ The charge declaration BINDS THE WHOLE UNIT, and that is the
            # entire point of putting it here. Measured on three behaviour
            # clones of a rigid charging teacher: the teacher declares for
            # 100% of a unit's members every time, the clones for 54-62%, and
            # the WHOLE unit only 23-35% of the time. A charge fails because
            # half the unit charges and the rest stand still, so the unit
            # stretches and the referee reverts it however good the rung
            # choice was. One leader-level binary choice replaces a 1-against-48
            # argmax repeated per model.
            model.declared_charge = group in charging

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
        ladder: MoveLadder = MoveLadder.normal,
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
        # ⚠ In the charge phase the movement slice decodes to the CHARGE ladder,
        # which spans the 2D6 (up to 12") rather than Move (6"). It is shared
        # across models because the rules cap a charge at the roll alone, so a
        # slow model charges exactly as far as a fast one.
        if ladder is MoveLadder.charge:
            charge: np.ndarray = self._charge_displacements[angle_idx, speed_idx]
            return charge
        if ladder is MoveLadder.short:
            short: np.ndarray = self._short_displacements[angle_idx, speed_idx]
            return short
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

    def advance_legality(
        self, models: list[Any], enemy_models: list[Any] | None
    ) -> np.ndarray:
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
        # ⚠ **An ENGAGED unit may not advance**, and until 2026-08-26 it could.
        # `09-movement-phase.md` makes a Normal move eligible only for an
        # unengaged unit, so an engaged unit's only move is a FALL BACK -- which
        # `implementation-status.md` row 63 records as capped at M. Without this
        # an engaged model kept all 48 advance rungs (measured) and could
        # therefore withdraw `M + roll`, which is the rules violation that row
        # names as the observable difference.
        #
        # Behaviourally a no-op wherever melee is off: `back_off_to_unengaged`
        # runs on every mover, so engagement is 0.0000% of model-pairs without
        # the charge's exemption, and the seeded digest is unchanged. It bites
        # exactly where it should -- a melee scenario where units are locked.
        # ⚠ `enemy_models` is REQUIRED, not defaulted, and that is the point.
        # An optional argument that silently disables a rule is exactly the trap
        # this file has now been caught by twice in one day -- the declaration
        # and the 2D6 cap were enforced only in the mask because `apply` never
        # received one. Passing `None` here is legitimate (no opposing force is
        # a real state) but it has to be a decision the caller writes down.
        engaged = self._engaged_units(models, enemy_models)
        for index, model in enumerate(models):
            if not getattr(model, "declared_advance", False):
                continue
            # Guarded on `engaged` being non-empty so the common case -- no
            # enemies passed, which is every non-melee path -- reads nothing off
            # the model at all.
            if engaged and int(model.group_id) in engaged:
                continue
            move = float(self._move_speeds[index])
            reach = move + float(model.advance_roll)
            allowed = np.array(
                [self.advance_distance(b, move) <= reach for b in range(n_bins)],
                dtype=bool,
            )
            legality[index] = np.tile(allowed, self._n_angles)
        return legality

    def _engaged_units(
        self, models: list[Any], enemy_models: list[Any] | None
    ) -> set[int]:
        """Which of `models`' units are currently in engagement range.

        Unit-level, like every other engagement test in this file: reducing a
        per-model answer over the unit is the coupling bug `shooting_masks`'
        own docstring warns against, and it is the one an audit found live at
        `_engaged_shooters` -- one model locking an enemy while four squadmates
        fired.
        """
        # ⚠ The enemy check comes FIRST and returns before touching `models`.
        # Callers that pass no enemies at all -- every non-melee path, and the
        # handler-level tests -- must not need a model to answer "is anyone
        # engaged", and one of those tests uses a stub without `is_alive`.
        if not models or not enemy_models:
            return set()
        alive_enemies = [m for m in enemy_models if m.is_alive]
        if not alive_enemies:
            return set()
        contacts = engagement_matrix(
            np.array([m.location for m in models], dtype=float),
            np.array([m.location for m in alive_enemies], dtype=float),
            np.ones(len(alive_enemies), dtype=bool),
            np.array([m.is_alive for m in models], dtype=bool),
            engagement_range=self._engagement_range,
            base_diameter=2.0 * self._base_radius,
        )
        return {
            int(models[index].group_id)
            for index in np.nonzero(np.asarray(contacts).any(axis=1))[0]
        }

    def short_move_legality(
        self, models: list[Any], enemy_models: list[Any] | None, phase: BattlePhase
    ) -> np.ndarray:
        """`(n_models, n_move_actions)` -- who may pile in or consolidate at all.

        ⚠ **Without this the phases are unmasked**, exactly as the charge
        DECLARATION was: every alive model could move 3" toward anything, twice
        a turn, whether or not the rules make its unit eligible. Measured when
        it was missing -- a scripted policy that piled in unconditionally
        dragged its army off the objectives and the bar fell from +23.9 to
        -27.8 vp.

        `12-fight-phase.md` § Pile-in move: eligible if the unit is engaged, or
        made a charge move this turn. The consolidate step takes the units that
        *were eligible to fight*, which is the same set plus those that fought
        and are now disengaged -- `fought_this_phase` carries that.
        """
        width = self._n_move_actions
        legality = np.zeros((len(models), width), dtype=bool)
        if phase not in (BattlePhase.pile_in, BattlePhase.consolidate):
            return legality
        alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
        if not alive_enemies:
            return legality
        positions = np.array([m.location for m in models], dtype=float)
        enemy_positions = np.array([m.location for m in alive_enemies], dtype=float)
        engaged = engaged_with_any(
            positions,
            enemy_positions,
            np.ones(len(alive_enemies), dtype=bool),
            np.array([m.is_alive for m in models], dtype=bool),
            engagement_range=self._engagement_range,
            base_diameter=2.0 * self._base_radius,
        )
        eligible: set[int] = set()
        for index, model in enumerate(models):
            if not model.is_alive:
                continue
            if (
                engaged[index]
                or getattr(model, "charged_this_turn", False)
                or (
                    phase is BattlePhase.consolidate
                    and getattr(model, "fought_this_phase", False)
                )
            ):
                eligible.add(int(model.group_id))
        for index, model in enumerate(models):
            if model.is_alive and int(model.group_id) in eligible:
                legality[index] = True
        return legality

    def declaration_legality(
        self, models: list[Any], enemy_models: list[Any] | None
    ) -> np.ndarray:
        """`(n_models, n_move_types)` -- which move types each model may DECLARE.

        ⚠ **The declaration was UNMASKED until 2026-08-26, on both seats.**
        `ActionRegistry.get_action_mask` is purely phase-based and nothing
        refined the `move_type` slice, so any alive model could declare any type
        in any command phase. Eligibility was enforced only later, on the MOVE,
        where an ineligible unit simply found no legal rung.

        That is a rules divergence in both directions, and both cost something:

        - **Charge.** `docs/rules/11-charge-phase.md` makes a unit ineligible if
          it is not within 12" of an enemy unit, is engaged, or advanced or fell
          back this turn. Declaring anyway is a *bit-exact no-op* -- it changes
          no state and earns no reward difference -- so an unmasked declaration
          hands the policy a free action with no consequence to learn from.
          Measured on the first unshaped arm: **71.4% of declared model-steps
          held zero legal rungs**, and declarations landed on eligible units
          *below chance* (31.4%, z = −2.54) against the scripted teacher's 40/40.
        - **Advance.** Worse, because it is not free. `declare_move_types` sets
          `advanced_this_turn` immediately, which both shooting masks read, so an
          ENGAGED unit -- barred from every rung by `advance_legality` -- could
          declare an advance, forfeit its whole shooting phase, and then be
          unable to move at all.

        ⚠ The gap map rated charge eligibility **implemented**
        (`implementation-status.md:103`). It was implemented for the *move*, not
        the *declaration*, and the row was wrong.

        `normal` is always legal: it is what STAY declares, so masking it would
        make the command phase unsteppable for a unit with nothing else to do.
        """
        n_types = len(self._move_types)
        legality = np.zeros((len(models), n_types), dtype=bool)
        if self._move_type_slice is None or n_types == 0:
            return legality
        alive = np.array([bool(m.is_alive) for m in models], dtype=bool)
        legality[:, self._move_types.index(MOVE_TYPE_NORMAL)] = alive

        if MOVE_TYPE_ADVANCE in self._move_types:
            column = self._move_types.index(MOVE_TYPE_ADVANCE)
            engaged = self._engaged_units(models, enemy_models)
            for index, model in enumerate(models):
                if not model.is_alive:
                    continue
                if engaged and int(model.group_id) in engaged:
                    continue
                # ⚠ NOT a rules gate -- a REPRESENTATION one, and it diverges
                # deliberately. The rules would let a unit rolling 1 advance 7";
                # the absolute ladder cannot express that, so at three bins such
                # a unit has no legal rung. Permitting the declaration there
                # would spend its shooting for a move it cannot make, which is
                # strictly dominated -- the exact class of action the absolute
                # re-encoding removed (dominated advances went 3.5-13.8% to
                # 0.0%). Masking is the smaller divergence of the two.
                move = float(self._move_speeds[index])
                reach = move + float(model.advance_roll)
                if any(
                    self.advance_distance(b, move) <= reach
                    for b in range(self._n_advance_bins)
                ):
                    legality[index, column] = True

        if MOVE_TYPE_CHARGE in self._move_types:
            column = self._move_types.index(MOVE_TYPE_CHARGE)
            alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
            if alive_enemies:
                eligible = self._charge_eligible_units(models, alive_enemies)
                # ⚠ NOT a rules gate -- a REPRESENTATION one, the same divergence
                # the advance declaration makes above, in the same direction, for
                # the same reason. The rules roll AFTER the declaration, so a
                # doomed declaration is a legal play there; here the roll comes
                # first (`DEFERRED: charge.blind_declaration`), so a unit whose
                # roll cannot cover the gap to ANY enemy is declaring a charge
                # that no rung can land. That declaration binds the unit into
                # the charge phase for a move the referee must revert -- a
                # zero-gradient trap, and 13.8% of a trained arm's attempts.
                # The scripted comparator has always declined these
                # (`_reachable_charge_units` asks gap <= roll); masking gives
                # the learned policy the same information.
                reachable = self._roll_reachable_units(models, alive_enemies)
                for index, model in enumerate(models):
                    if (
                        model.is_alive
                        and int(model.group_id) in eligible
                        and int(model.group_id) in reachable
                    ):
                        legality[index, column] = True
        return legality

    def _roll_reachable_units(
        self, models: list[Any], alive_enemies: list[Any]
    ) -> set[int]:
        """Units whose charge roll can cover the gap to at least one enemy.

        Reach is the ROLL alone, through the scale, exactly as
        `_charge_reach` and the charge ladder read it; contact is the
        engagement ring plus both bases, exactly as `_charge_is_legal`
        judges it. A gate that asked a different question from the referee
        would declare charges the referee must then revert.
        """
        positions = np.array([m.location for m in models], dtype=float)
        enemy_positions = np.array([m.location for m in alive_enemies], dtype=float)
        enemy_radii = np.array(
            [float(m.base_radius) for m in alive_enemies], dtype=float
        )
        gaps = (
            np.linalg.norm(
                positions[:, np.newaxis, :] - enemy_positions[np.newaxis, :, :],
                axis=2,
            )
            - enemy_radii[np.newaxis, :]
            - self._base_radius
        )
        contact = self._engagement_range
        reachable: set[int] = set()
        for index, model in enumerate(models):
            if not model.is_alive:
                continue
            group = int(model.group_id)
            if group in reachable:
                continue
            if float(gaps[index].min()) - contact <= self._charge_reach(model):
                reachable.add(group)
        return reachable

    def charge_legality(
        self, models: list[Any], enemy_models: list[Any] | None
    ) -> np.ndarray:
        """`(n_models, n_move_actions)` -- which charge moves the rules allow.

        Two gates, mirroring `advance_legality`.

        **Eligibility**, per `docs/rules/11-charge-phase.md`: a unit may declare
        a charge only when it is within 12" of an enemy unit, is NOT already
        engaged, and has neither advanced nor fallen back this turn. Measured
        unit-to-unit like the shooting mask's range test -- reducing a per-model
        answer over the unit is the coupling bug that function's own docstring
        warns against.

        **Distance**: the 2D6 is the charge move's maximum, so any speed bin
        travelling further is masked out. This is why the roll is rolled at the
        start of the side's turn and observable: legality is gated on it, so a
        policy choosing before it could not know which of its actions exist.

        Note the roll is a UNIT roll but the mask is per model, and a model with
        a smaller Move has fewer legal bins for the same roll -- which is the
        rules' own behaviour, since the cap is a distance and not a bin index.
        """
        n_models = len(models)
        movement = self.movement_slice
        legality = np.zeros((n_models, movement.size), dtype=bool)
        if not self._melee_enabled or n_models == 0:
            return legality
        alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
        if not alive_enemies:
            return legality

        eligible_units = self._charge_eligible_units(models, alive_enemies)
        if not eligible_units:
            return legality

        for index, model in enumerate(models):
            if not model.is_alive or int(model.group_id) not in eligible_units:
                continue
            # ⚠ **Only a unit that DECLARED**, exactly as an advance rung is
            # legal only for a unit that declared one. Without this the charge
            # was declared implicitly by picking a rung, so "charge or not" was
            # decided independently by every model -- and measured, a whole unit
            # then committed on only 23-35% of its charges against a rigid
            # teacher's 100%. See `declare_move_types`.
            #
            # Scenarios with no move-type slice keep the old behaviour, so every
            # melee measurement taken before the declaration existed is still
            # reproducible on its own config.
            if self._move_type_slice is not None and not getattr(
                model, "declared_charge", False
            ):
                continue
            reach = self._charge_reach(model)
            if reach <= 0.0:
                continue
            # ⚠ The CHARGE ladder, not the movement one. The rules cap a charge
            # at the roll alone, so its rungs span 2D6 rather than Move.
            distances = np.linalg.norm(self._charge_displacements, axis=-1)
            legality[index] = (distances <= reach + _CHARGE_REACH_EPSILON).reshape(-1)

        if self._charge_approach_mask:
            self._apply_charge_approach_mask(legality, models, alive_enemies)
        return legality

    def _apply_charge_approach_mask(
        self,
        legality: np.ndarray,
        models: list[Any],
        alive_enemies: list[Any],
    ) -> None:
        """Keep only charge moves that END CLOSER to the unit's derived target.

        Mirrors `_charge_is_legal`'s *while moving* clause exactly -- centre
        distance to the target unit's members, strictly smaller after the move
        than before -- so every action this removes is one the referee would
        revert the whole charge for. STAY lives outside the movement slice and
        is untouched: the clause binds movers, and a stationary squadmate does
        not veto its unit's charge.

        ⚠ Per-model and target-DERIVED, deliberately. This is NOT the joint
        "if it can" mask that collapsed the bar 5.67 -> 1.67 -- that one
        forced ENGAGEMENT, a joint property no per-model mask can see. The
        target is the unit's nearest enemy unit at charge time, which also
        points every member at ONE unit, the referee's exactly-one condition.
        A unit left with no legal rung by this mask keeps whatever the distance
        gate offered -- a mask that empties a declared unit's action set would
        make the declaration unsteppable, the failure mode the advance's
        no-legal-rung validator exists to prevent.
        """
        enemy_positions = np.array([m.location for m in alive_enemies], dtype=float)
        enemy_groups = np.array([int(m.group_id) for m in alive_enemies], dtype=int)
        flat_displacements = self._charge_displacements.reshape(-1, 2)

        units: dict[int, list[int]] = {}
        for index, model in enumerate(models):
            if legality[index].any():
                units.setdefault(int(model.group_id), []).append(index)

        for _group, members in units.items():
            positions = np.array([models[i].location for i in members], dtype=float)
            gaps = np.linalg.norm(
                positions[:, np.newaxis, :] - enemy_positions[np.newaxis, :, :],
                axis=2,
            )
            # The derived target: the enemy UNIT nearest this unit, exactly the
            # pair the referee's derived-target test will find.
            nearest = int(np.argmin(gaps.min(axis=0)))
            target_rows = np.nonzero(enemy_groups == enemy_groups[nearest])[0]
            target_positions = enemy_positions[target_rows]
            for row, index in enumerate(members):
                before = float(
                    np.linalg.norm(target_positions - positions[row], axis=1).min()
                )
                endpoints = positions[row][np.newaxis, :] + flat_displacements
                after = np.linalg.norm(
                    endpoints[:, np.newaxis, :] - target_positions[np.newaxis, :, :],
                    axis=2,
                ).min(axis=1)
                approach = legality[index] & (after < before)
                # Never empty a declared unit's set -- see the docstring.
                if approach.any():
                    legality[index] = approach

    def _charge_reach(self, model: Any) -> float:
        """This model's 2D6, in BOARD UNITS -- the roll is authored in inches."""
        roll = float(getattr(model, "charge_roll", 0.0))
        if roll <= 0.0:
            return 0.0
        return float(self._charge_scale.to_units(roll))

    def _displacements_for(self, model_idx: int) -> np.ndarray:
        """This model's `(n_angles, n_speeds, 2)` displacement grid."""
        if self._model_displacements is not None:
            grid: np.ndarray = self._model_displacements[model_idx]
            return grid
        shared: np.ndarray = self._displacements
        return shared

    def charge_eligible_units(
        self, models: list[Any], enemy_models: list[Any] | None
    ) -> set[int]:
        """Units the rules allow to DECLARE a charge, before any declaration.

        Public because the declaration is made in the COMMAND phase, one phase
        before `charge_legality` becomes meaningful: that mask is now gated on
        a declaration, so at command time it is empty by construction and a
        scripted policy asking it "may I charge?" would always hear no.
        """
        if not self._melee_enabled:
            return set()
        alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
        if not alive_enemies:
            return set()
        return self._charge_eligible_units(models, alive_enemies)

    def _charge_eligible_units(
        self, models: list[Any], alive_enemies: list[Any]
    ) -> set[int]:
        """Units that may declare a charge this turn."""
        positions = np.array([m.location for m in models], dtype=float)
        enemy_positions = np.array([m.location for m in alive_enemies], dtype=float)
        enemy_alive = np.ones(len(alive_enemies), dtype=bool)
        engaged = engaged_with_any(
            positions,
            enemy_positions,
            enemy_alive,
            np.array([m.is_alive for m in models], dtype=bool),
            engagement_range=self._engagement_range,
            base_diameter=2.0 * self._base_radius,
        )
        gaps = (
            np.linalg.norm(
                positions[:, np.newaxis, :] - enemy_positions[np.newaxis, :, :], axis=2
            )
            - 2.0 * self._base_radius
        )
        within = (gaps <= self._charge_range).any(axis=1)

        blocked: set[int] = set()
        candidates: set[int] = set()
        for index, model in enumerate(models):
            group = int(model.group_id)
            if not model.is_alive:
                continue
            if (
                engaged[index]
                or getattr(model, "advanced_this_turn", False)
                or getattr(model, "fell_back_this_turn", False)
            ):
                blocked.add(group)
            if within[index]:
                candidates.add(group)
        return candidates - blocked

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
        ladder: MoveLadder = MoveLadder.normal,
    ) -> int:
        """Return the action that moves closest to the direction (dx, dy).

        Picks the angle bin nearest to atan2(dy, dx). When max_step_length is
        None, uses maximum speed. ``ladder`` picks the distance table, which
        spans 2D6 rather than Move -- a caller aiming a charge with the movement
        ladder would size its step against distances the env will not use.
        When max_step_length is set, chooses the
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
                    self.encode_action(angle_idx, s),
                    model_idx=model_idx,
                    ladder=ladder,
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

    def displaces_in(self, phase: BattlePhase) -> bool:
        """Public form of `_displaces_in` -- the facade records per-phase
        coherency for exactly the phases in which a model can move."""
        return self._displaces_in(phase)

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

    def _enforce_short_move(
        self,
        wargame_models: list[Any],
        enemy_models: list[Any] | None,
        start_positions: list[np.ndarray] | None,
        batch: list[int],
    ) -> None:
        """Referee one unit's pile-in or consolidation; revert it whole if illegal.

        ⚠ **All-or-nothing at the UNIT**, exactly as the charge is. A pile-in is
        one unit's move in the rules, and `03-moving.md` reverts a move whose
        after-conditions fail rather than repairing it -- which also keeps this
        off `coherency.enforce_move`, a referee that is `off` on every shipped
        config and would let an illegal shuffle simply stand.

        The predicate is `pile_in.agent_move_is_legal`, deliberately the SAME
        `_is_legal` the engine's constructive `pile_in` obeys. Two
        implementations of one rule is how three different answers to "on an
        objective" came to coexist here.
        """
        if start_positions is None:
            return
        members = [index for index in batch if wargame_models[index].is_alive]
        if not members or not any(
            not np.array_equal(start_positions[index], wargame_models[index].location)
            for index in members
        ):
            return
        alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
        before = np.array([start_positions[index] for index in members], dtype=float)
        if agent_move_is_legal(
            wargame_models,
            members,
            before,
            alive_enemies,
            selection_range=self._charge_scale.to_units(SELECTION_RANGE_INCHES),
            engagement_range=self._engagement_range,
            base_radius=self._base_radius,
            coherency_nearest=self._coherency_nearest,
            coherency_furthest=self._coherency_furthest,
        ):
            return
        for index in members:
            wargame_models[index].location = np.array(start_positions[index], copy=True)

    def _charge_batches(
        self,
        phase: BattlePhase,
        wargame_models: list[Any],
        action: WargameEnvAction,
        charge_start: list[np.ndarray] | None,
    ) -> list[list[int]]:
        """Model indices to resolve together, in order.

        One batch holding the whole force for every phase that is not refereed
        at unit level, which keeps the existing index order and therefore every
        existing result. The charge, the pile-in and the consolidation each
        resolve one UNIT at a time, in group order, so the referee can put a
        failed unit back before the next unit has moved.
        """
        indices = list(range(len(action.actions)))
        if charge_start is None or phase not in _UNIT_REFEREED_PHASES:
            return [indices]
        units: dict[int, list[int]] = {}
        for index in indices:
            units.setdefault(int(wargame_models[index].group_id), []).append(index)
        return [members for _group, members in sorted(units.items())]

    def _enforce_charge(
        self,
        wargame_models: list[Any],
        enemy_models: list[Any] | None,
        start_positions: list[np.ndarray] | None,
        batch: list[int],
    ) -> None:
        """A charge that does not end legally is not made at all.

        `docs/rules/03-moving.md`: if any after-moving condition fails, *return
        every model to where it started*. That is the rules' own all-or-nothing,
        not a referee setting -- a charge ending unengaged is not a charge that
        went badly, it is a charge that did not happen. So this reverts
        unconditionally and is deliberately NOT routed through
        `coherency.enforce_move`, which defaults to `off` on every shipped
        config and would therefore let an illegal charge simply stand.

        Three conditions per unit that moved:

        * **coherency** -- the unit must still be one body;
        * **engaged with exactly ONE enemy unit** -- which covers the two
          after-moving conditions while a charge has a single DERIVED target:
          engaged with all of them, and with no non-target. ⚠ It also refuses a
          charge that clips a second unit, which the rules would ALLOW if both
          were declared targets (`11-charge-phase.md` selects *one or more*).
          With targets derived rather than declared a unit ending on two enemy
          units has by construction charged both, so this is a divergence of the
          derived-target model, not the rule it is named after;
        * **every model that moved ended CLOSER to the target unit** --
          `11-charge-phase.md` § Charge move, *While moving*: *"Each model must
          end its move closer to one or more charge targets."*

        ⚠ **That third condition did not exist until 2026-08-25**, and no gap-map
        row recorded its absence. Without it a charge was satisfied by ONE model
        reaching contact while its squadmates moved anywhere coherency allowed --
        a materially easier charge than the rules', and the half of the mechanic
        a learned policy is most likely to exploit. Measured before it landed: a
        two-model unit whose second model walked 1" directly AWAY from the only
        enemy still had its charge stand, and 2.4% of a rigid script's charging
        models (5.3% of its standing charges) already violated it by accident.
        Found by a rules-lawyer audit.

        ⚠ **The two *"if it can"* clauses remain DEFERRED, and the reason is
        now measured rather than assumed.** `DEFERRED: charge.while_moving_best_effort`.

        They are *end within 1" of a target if it can* and *end ENGAGED with one
        if it can*. Both were implemented as a per-model mask on 2026-08-26 --
        keep only the rungs that satisfy the clause, when any rung does -- and
        the result collapsed the mechanic: the scripted bar's attempts fell
        **5.67 -> 1.67** per episode and its standing charges **4.56 -> 1.11**.

        The mask is not too aggressive by accident; *"if it can"* is a **joint**
        property and a per-model mask cannot see it. A model can end engaged
        only if a legal UNIT move exists in which it does -- and forcing all
        five members onto contact scatters the squad, breaks the 2" chain, and
        the referee then reverts the whole charge. So the mask forbade the very
        moves that were legal.

        Expressing it needs the joint candidate set, which exists only in the
        play-time decoder (`model/common/decoding.py`), and folding decoding
        into training measured **-51.8 vp**. Until then the referee's coherency
        and engagement conditions carry it: a charge that could have reached
        contact and did not is not punished, which is a divergence in the
        permissive direction.

        ⚠ **The target is DERIVED from where the unit ends, not declared, and
        that is a measured decision rather than a shortcut.** A declaration
        would have to be an action, one model would have to spend its action on
        it, and that model could not then move -- and a model left behind while
        its squadmates charge breaks the 2" chain at any distance beyond about
        three inches, which reverts the whole charge. Measured directly against
        `evaluate_coherency`: a five-model unit whose declarer stays put is
        coherent at 2" and incoherent at 4", 8" and 12". So the declaration
        slice would have made almost every charge fail.
        `DEFERRED: charge.target_declaration`.
        """
        if start_positions is None:
            return
        members = [index for index in batch if wargame_models[index].is_alive]
        if not members or not any(
            not np.array_equal(start_positions[index], wargame_models[index].location)
            for index in members
        ):
            return
        alive_enemies = [m for m in (enemy_models or []) if m.is_alive]
        if self._charge_preconditions_hold(
            wargame_models, members, start_positions
        ) and self._charge_is_legal(
            wargame_models, members, alive_enemies, start_positions
        ):
            # Only a charge that STOOD earns the fight-order priority. A
            # reverted charge did not happen, so a unit that rolled short and
            # snapped back must not strike first for having tried.
            for index in members:
                wargame_models[index].charged_this_turn = True
            return
        for index in members:
            wargame_models[index].location = np.array(start_positions[index], copy=True)

    def _charge_preconditions_hold(
        self,
        wargame_models: list[Any],
        members: list[int],
        start_positions: list[np.ndarray],
    ) -> bool:
        """Was this unit allowed to make this charge at all?

        ⚠ **The mask was the ONLY thing enforcing either of these until
        2026-08-25.** `charge_legality` masks out an undeclared unit and every
        rung longer than the unit's 2D6 -- but `ActionHandler.apply` takes no
        mask, so a policy that bypasses one could charge without declaring, or
        travel 6" on a roll of 2, and the env accepted it. Measured: a declared
        model with `charge_roll = 2.0` is correctly masked out of the 6" rung
        and, handed that action anyway, travelled the full 6.0" and was granted
        `charged_this_turn`.

        That is the defect class this project has already paid for twice -- the
        joint decoder judged candidates against its own relaxation (+11.4 vp),
        and six unit tests missed a movement bug because none called `env.step`.
        Both times the constraint lived in one layer and the layer that actually
        moved models did not know about it. The advance has belt-and-braces
        already: `_advance_displacement` clamps to `min(distance, move + roll)`
        at resolution as well as masking. This gives the charge the same.

        Found by a rules-lawyer audit. It changes no shipped measurement --
        every current selector plays under the mask -- which is exactly why it
        needs a test rather than a score.
        """
        for index in members:
            model = wargame_models[index]
            if not getattr(model, "declared_charge", False):
                return False
            travelled = float(
                np.linalg.norm(
                    np.asarray(model.location, dtype=float)
                    - np.asarray(start_positions[index], dtype=float)
                )
            )
            # The rules cap the charge at the roll alone; this ladder also caps
            # it at Move (`DEFERRED: charge.beyond_move_ladder`), and the mask
            # already applies that half. Here only the roll is re-checked, so
            # the referee never rejects a move the ladder itself permitted.
            if travelled > self._charge_reach(model) + _CHARGE_REACH_EPSILON:
                return False
        return True

    def _charge_is_legal(
        self,
        wargame_models: list[Any],
        members: list[int],
        alive_enemies: list[Any],
        start_positions: list[np.ndarray],
    ) -> bool:
        """Did this unit's charge end in a legal position?

        `start_positions` is required, not optional: the *while moving* rule is
        a per-model comparison against where that model began, and a referee
        that cannot see the start cannot judge the move. Passing it is what
        closed the gap described in `_enforce_charge`.
        """
        if not alive_enemies:
            return False
        positions = np.array([wargame_models[i].location for i in members], dtype=float)
        contacts = engagement_matrix(
            positions,
            np.array([m.location for m in alive_enemies], dtype=float),
            np.ones(len(alive_enemies), dtype=bool),
            # `members` is already filtered to the unit's living models.
            np.ones(len(members), dtype=bool),
            engagement_range=self._engagement_range,
            base_diameter=2.0 * self._base_radius,
        )
        touched = {
            int(alive_enemies[j].group_id) for j in np.nonzero(contacts.any(axis=0))[0]
        }
        if len(touched) != 1:
            return False
        # ⚠ *While moving*: every model that MOVED must end closer to the target
        # unit. A model that stood still is not "moved", and the rules put the
        # condition on the model's own move -- so a stationary squadmate does
        # not veto its unit's charge, but a squadmate that walked away does.
        target = int(next(iter(touched)))
        target_positions = np.array(
            [m.location for m in alive_enemies if int(m.group_id) == target],
            dtype=float,
        )
        for row, index in enumerate(members):
            start = np.asarray(start_positions[index], dtype=float)
            if np.array_equal(start, positions[row]):
                continue
            before = float(np.linalg.norm(target_positions - start, axis=1).min())
            after = float(
                np.linalg.norm(target_positions - positions[row], axis=1).min()
            )
            if after >= before:
                return False
        report = evaluate_coherency(
            positions=positions,
            group_ids=np.zeros(len(members), dtype=np.intp),
            alive_mask=np.ones(len(members), dtype=bool),
            base_radii=np.array(
                [wargame_models[i].base_radius for i in members], dtype=float
            ),
            nearest_distance=self._coherency_nearest,
            furthest_distance=self._coherency_furthest,
        )
        return bool(report.all_coherent)

    def _is_displacement_action(self, action: int) -> bool:
        """Does this index mean a MOVE, rather than a declaration or a shot?

        ⚠ **Allow-list, not a deny-list, and that is the whole point.** This
        guarded only the shooting slice and let everything else through to
        `decode_action`, which reads an unrecognised index as `action - 1` into
        the angle table -- so the `move_type` declaration, registered last,
        crashed with `IndexError` whenever an unmasked sample put it in the
        movement phase. Every phase mask forbids that, and `apply` deliberately
        does not trust the mask; a deny-list has to be extended by whoever adds
        the next slice, and the next slice will be `fall_back` or `charge`.
        """
        if action == STAY_ACTION:
            return True
        if action <= self._n_move_actions:
            return True
        advance = self._advance_slice
        return advance is not None and advance.start <= action < advance.end

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
            np.array([m.is_alive for m in wargame_models], dtype=bool),
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
        # ⚠ **AND the two fight-phase moves**, added 2026-08-26. `pile_in` and
        # `consolidate` move models that are ALREADY inside an enemy's engagement
        # ring -- that is what makes them eligible -- so applying the rings to
        # their endpoint walked every one of them back to its start:
        # `back_off_to_unengaged` returns `start` for a model that begins inside a
        # ring it may not end in. Measured 87.3%/90.9% of ordered moves delivering
        # exactly 0.000". `12-fight-phase.md` requires these moves to END engaged,
        # exactly as the charge does. Occupied bases are still passed, so an
        # endpoint may be engaged and may never overlap.
        charging = phase in _ENGAGING_ENDPOINT_PHASES and self._displaces_in(phase)
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
        # ⚠ Every phase that is refereed at UNIT level needs its start
        # positions: the revert puts the whole unit back, and `previous_location`
        # is a single slot written once per movement phase, which would send a
        # non-mover back two phases.
        charge_start = (
            [np.array(m.location, copy=True) for m in wargame_models]
            if phase in _UNIT_REFEREED_PHASES and self._displaces_in(phase)
            else None
        )
        if phase is BattlePhase.command:
            # No early return: the per-model loop below still validates every
            # action against its space before skipping non-movement phases, and
            # an unvalidated declaration would be a silent out-of-range action.
            self.declare_move_types(action, wargame_models)
            self.declare_objectives(action, wargame_models)
        if phase is BattlePhase.fight:
            # Read BEFORE the fight resolves, which happens on the boundary
            # leaving this phase -- so the priorities the engine selects on are
            # the ones this step's action declared.
            self.declare_fight_order(action, wargame_models)

        # ⚠ A charge resolves ONE UNIT AT A TIME, and every other phase
        # resolves the whole force in index order exactly as before. The
        # all-or-nothing revert puts a failed unit back where it began, so if
        # a later unit has already advanced into that ground the board ends
        # up holding two models inside each other -- measured at 0.868in of
        # base overlap, 69% of a 32mm base, in a demo recording. Judging each
        # unit before the next one moves removes the window entirely, and
        # lets the next unit move against the RESTORED position rather than
        # against ground that was only briefly empty.
        def move_batch(indices: list[int]) -> None:
            for i in indices:
                act = action.actions[i]
                model = wargame_models[i]
                if not model.is_alive:
                    continue
                if not action_space[i].contains(act):  # type: ignore
                    raise ValueError(
                        f"Action {act} for wargame model {i} is out of bounds."
                    )
                if not self._displaces_in(phase):
                    continue
                if not self._is_displacement_action(act):
                    continue
                model.previous_location = model.location.copy()
                displacement = self.decode_action(
                    act,
                    model_idx=i,
                    advance_roll=model.advance_roll,
                    ladder=_LADDER_FOR_PHASE.get(phase, MoveLadder.normal),
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

        for batch in self._charge_batches(phase, wargame_models, action, charge_start):
            move_batch(batch)
            if charge_start is None:
                continue
            if phase is BattlePhase.charge:
                self._enforce_charge(wargame_models, enemy_models, charge_start, batch)
            else:
                self._enforce_short_move(
                    wargame_models, enemy_models, charge_start, batch
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
