"""The per-model facade's typed contract: one decision in, the next decision out.

A step is ONE decision: open a unit with a declaration, act with one model,
name a charge target, or close the turn. The env always states which kind it
expects next and exactly which values are legal -- the `DecisionPoint` -- so a
policy builds both of its factor masks from the observation and never has to
ask the env anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from wargame_rl.wargame.envs.domain.activation import CHARGE_TARGET_DECLINE
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

FACADE_TAG = "per_model"
# The phase facade's artefacts carry no tag; that is what they are read as.
PHASE_FACADE_TAG = "phase"

# Declaration masks are one width for every phase, so the observation's shape
# does not depend on which phase it is: the movement phase's four values.
N_DECLARATIONS = 4


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

    @model_validator(mode="after")
    def _shape(self) -> PerModelAction:
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
    def open(cls, model: int, declaration: int) -> PerModelAction:
        """Open `model`'s unit with `declaration`."""
        return cls(kind=StepKind.open, model=model, value=int(declaration))

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


class PerModelProvenance(BaseModel):
    """How to boot this episode again, stamped with the facade that played it."""

    model_config = ConfigDict(extra="forbid")

    facade: Literal["per_model"] = "per_model"
    config: dict[str, Any]
    rng_state: dict[str, Any]
    combat_seed: int
    seed: int | None = None
    driver: str | None = None


def facade_of(provenance: dict[str, Any] | BaseModel) -> str:
    """Which facade an artefact belongs to; an untagged one is the phase facade's."""
    data = provenance.model_dump() if isinstance(provenance, BaseModel) else provenance
    tag = data.get("facade")
    return PHASE_FACADE_TAG if tag is None else str(tag)


def require_per_model(provenance: dict[str, Any] | BaseModel) -> None:
    """Refuse an artefact the phase facade produced."""
    tag = facade_of(provenance)
    if tag != FACADE_TAG:
        raise ValueError(
            f"this artefact belongs to the '{tag}' facade and cannot be seated "
            f"in the per-model one"
        )
