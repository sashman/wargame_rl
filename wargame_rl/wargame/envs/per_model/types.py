"""The per-model facade's typed contract: one decision in, the next decision out.

A step is ONE decision: open a unit with a declaration, act with one model,
name a charge target, or close the turn. The env always states which kind it
expects next and exactly which values are legal -- the `DecisionPoint` -- so a
policy builds both of its factor masks from the observation and never has to
ask the env anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from wargame_rl.wargame.envs.domain.sequencing.activation import CHARGE_TARGET_DECLINE
from wargame_rl.wargame.envs.state.provenance import (
    PER_MODEL_FACADE_TAG,
    PHASE_FACADE_TAG,
    Cadence,
    PerModelProvenance,
    facade_of,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv

FACADE_TAG = PER_MODEL_FACADE_TAG
# The phase facade's artefacts carry no tag; that is what they are read as.

# Declaration masks are one width for every phase, so the observation's shape
# does not depend on which phase it is: the movement phase's four values.
N_DECLARATIONS = 4

# The commitment decision an `open` action may carry (#384, Stage 1): no
# decision on this step (the default; the writer is not the policy, or the
# unit committed earlier this turn), KEEP (the slot as it stands), or an
# objective index.
NO_COMMIT_DECISION = -2
COMMIT_KEEP = -1


class StepKind(str, Enum):
    """What one `step()` resolves."""

    open = "open"
    act = "act"
    target = "target"
    close_turn = "close_turn"


class PerModelAction(BaseModel):
    """One decision.

    `kind` says which factor `value` is: a declaration for `open`, an action
    index for `act`, an enemy unit (or `CHARGE_TARGET_DECLINE`) for `target`.
    `close_turn` carries neither a model nor a value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: StepKind
    model: int = -1
    value: int = -1
    # Only on `open`: the unit's ground commitment for this turn (#384 Stage
    # 1). `NO_COMMIT_DECISION` when the step carries none.
    commitment: int = NO_COMMIT_DECISION

    @model_validator(mode="after")
    def _shape(self) -> PerModelAction:
        if self.commitment != NO_COMMIT_DECISION and self.kind is not StepKind.open:
            raise ValueError("only an open step carries a commitment decision")
        if self.commitment < NO_COMMIT_DECISION:
            raise ValueError(f"commitment {self.commitment} names nothing")
        if self.kind is StepKind.close_turn:
            if self.model != -1 or self.value != -1:
                raise ValueError("close_turn carries no model and no value")
            return self
        if self.model < 0:
            raise ValueError(f"{self.kind.value} needs a model index, got {self.model}")
        if self.value < 0 and not (
            self.kind is StepKind.target and self.value == CHARGE_TARGET_DECLINE
        ):
            raise ValueError(f"{self.kind.value} needs a value, got {self.value}")
        return self

    @classmethod
    def open(
        cls, model: int, declaration: int, commitment: int = NO_COMMIT_DECISION
    ) -> PerModelAction:
        """Open `model`'s unit with `declaration`, and with `commitment` when
        the point offers the decision (`COMMIT_KEEP` or an objective index)."""
        return cls(
            kind=StepKind.open,
            model=model,
            value=int(declaration),
            commitment=int(commitment),
        )

    @classmethod
    def act(cls, model: int, action: int) -> PerModelAction:
        """`model` takes action index `action`."""
        return cls(kind=StepKind.act, model=model, value=int(action))

    @classmethod
    def target(cls, model: int, enemy_unit: int) -> PerModelAction:
        """Name the charge target for `model`'s open unit (or decline)."""
        return cls(kind=StepKind.target, model=model, value=int(enemy_unit))

    @classmethod
    def close_turn(cls) -> PerModelAction:
        """End our turn cycle; the only legal action at a closing point."""
        return cls(kind=StepKind.close_turn)


@dataclass(frozen=True)
class DecisionPoint:
    """The complete legality of the NEXT step.

    `selector_mask` says which models may be named. The value mask for the
    expected kind says which values each selectable model may take:
    `declaration_mask` for `open`, `action_mask` for `act`, `target_mask` for
    `target` (a decline is always allowed there). The other two are all False.
    """

    kind: StepKind
    seat_is_player: bool
    phase: BattlePhase | None
    selector_mask: np.ndarray
    declaration_mask: np.ndarray
    action_mask: np.ndarray
    target_mask: np.ndarray
    acted: np.ndarray
    open_unit: int | None
    forced_model: int | None
    # On an `open` point when the policy is the commitment writer (#384 Stage
    # 1): `(P, 1 + K)` -- column 0 KEEP, column 1 + k objective k -- True on
    # the rows of models whose unit has not committed this turn. None (or
    # all False) when the step offers no commitment decision.
    commit_mask: np.ndarray | None = None

    def offers_commitment(self, model: int) -> bool:
        """True when opening with `model` carries a commitment decision."""
        mask = self.commit_mask
        return mask is not None and model < mask.shape[0] and bool(mask[model].any())

    def why_illegal(self, action: PerModelAction) -> str | None:
        """None when `action` is legal here, else the reason."""
        if action.kind is not self.kind:
            return f"expected a {self.kind.value} step, got {action.kind.value}"
        if self.kind is StepKind.close_turn:
            return None
        if (
            action.model >= self.selector_mask.shape[0]
            or not self.selector_mask[action.model]
        ):
            return f"model {action.model} is not selectable"
        if action.commitment != NO_COMMIT_DECISION:
            if not self.offers_commitment(action.model):
                return f"model {action.model}'s unit takes no commitment decision here"
            assert self.commit_mask is not None
            column = 0 if action.commitment == COMMIT_KEEP else 1 + action.commitment
            if (
                column >= self.commit_mask.shape[1]
                or not self.commit_mask[action.model, column]
            ):
                return f"commitment {action.commitment} is not legal for model {action.model}"
        if self.kind is StepKind.open:
            mask = self.declaration_mask[action.model]
            what = "declaration"
        elif self.kind is StepKind.act:
            mask = self.action_mask[action.model]
            what = "action"
        else:
            if action.value == CHARGE_TARGET_DECLINE:
                return None
            mask = self.target_mask[action.model]
            what = "target"
        if action.value >= mask.shape[0] or not mask[action.value]:
            return f"{what} {action.value} is not legal for model {action.model}"
        return None

    @classmethod
    def closing(
        cls, n_models: int, n_actions: int, n_enemy_units: int
    ) -> DecisionPoint:
        """The closing point: nothing selectable, only `close_turn` legal."""
        return cls(
            kind=StepKind.close_turn,
            seat_is_player=True,
            phase=None,
            selector_mask=np.zeros(n_models, dtype=bool),
            declaration_mask=np.zeros((n_models, N_DECLARATIONS), dtype=bool),
            action_mask=np.zeros((n_models, n_actions), dtype=bool),
            target_mask=np.zeros((n_models, n_enemy_units), dtype=bool),
            acted=np.zeros(n_models, dtype=bool),
            open_unit=None,
            forced_model=None,
        )


@dataclass(frozen=True)
class PerModelObservation:
    """What the policy sees: the next decision, plus the turn's revealed state.

    The board itself is read through the env's `BattleView`; the token and
    relation observation is stage 2's. This carries what the decision needs
    and what the whole-army observation cannot express: which model has acted,
    and a unit's dice only once it has declared.
    """

    decision: DecisionPoint
    battle_round: int
    current_turn: int
    sub_step: int
    episode_step: int
    active_seat_is_player: bool
    revealed_advance_roll: np.ndarray
    revealed_charge_roll: np.ndarray
    consolidate_mode: np.ndarray
    advanced: np.ndarray
    fell_back: np.ndarray
    declared_charge: np.ndarray
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int


# The other side of the decision contract: whatever drives a seat. A chooser
# answers one env's pending decision; a batch chooser answers one decision per
# env in one call, which is what a network seat does for a whole wave and
# what a scripted or random seat does env by env. The per-model counterpart
# of the phase facade's `ActionSelector`.
Chooser: TypeAlias = "Callable[[PerModelEnv, PerModelObservation], PerModelAction]"
BatchChooser: TypeAlias = (
    "Callable[[Sequence[PerModelEnv], Sequence[PerModelObservation]], "
    "list[PerModelAction]]"
)


@dataclass(frozen=True)
class StepEffect:
    """What one `step` did to the player seat, for a reward paid per decision.

    `actor_set` is every player model the step newly marked acted -- the
    actor on an `act`, the whole unit on an `open` whose declaration closed it
    with no member taking a step (and on a charge-target decline, which does
    the same), nobody on a `target` that names a unit or on a `close_turn`.
    It is the DELTA of `DecisionPoint.acted`, which stays the cumulative mask.
    `kills_by_model` counts the player kills resolved DURING the step, by
    attacker -- a shooting unit's volley resolves at its close, on its last
    member's step, so the attackers a step's kills name are not its actor set.
    """

    actor_set: tuple[int, ...]
    kills_by_model: dict[int, int]

    @classmethod
    def none(cls) -> StepEffect:
        return cls(actor_set=(), kills_by_model={})


def require_per_model(provenance: dict[str, Any] | BaseModel) -> None:
    """Refuse an artefact the phase facade produced."""
    tag = facade_of(provenance)
    if tag != FACADE_TAG:
        raise ValueError(
            f"this artefact belongs to the '{tag}' facade and cannot be seated "
            f"in the per-model one"
        )


@dataclass(frozen=True)
class FacadeDivergence:
    """A point at which this facade's game parted from the phase facade's,
    because this facade applied a rule the phase facade cannot.

    Not a rules violation: every entry is a moment this facade was CORRECT
    where the whole-army step is not -- a unit selecting targets after an
    earlier unit's casualties are removed, attrition on the side whose turn
    it is not, a battle that continues after a wipe. Each is recorded the
    first time it makes a difference, so the bridge against the phase facade
    can say how far bit-identity was owed, and a reader of an episode can see
    which rule separated the two games. Each `rule` names a filed defect in
    the shared code; when that lands, the category cannot occur any more.
    """

    episode_step: int
    battle_round: int | None
    phase: BattlePhase | None
    rule: str


__all__ = [
    "BatchChooser",
    "CHARGE_TARGET_DECLINE",
    "Chooser",
    "Cadence",
    "DecisionPoint",
    "FACADE_TAG",
    "FacadeDivergence",
    "PER_MODEL_FACADE_TAG",
    "PHASE_FACADE_TAG",
    "PerModelAction",
    "PerModelObservation",
    "PerModelProvenance",
    "StepEffect",
    "StepKind",
    "facade_of",
    "require_per_model",
]
