"""Step types for the per-model facade.

A step is ONE model's action (Principle 1 of the next-architecture design,
issue #283): the policy chooses which model acts through a selector mask, the
env resolves that model's action before the next model acts, and one extra
step per turn cycle — the turn-closing step — carries no model action and is
where the turn's victory points are paid.

Unit-level choices are DECLARATIONS made on a unit's opening step in the phase
the rules put them in (move type in movement, shoot-or-not in shooting, the
charge in the charge phase, activation priority in the fight phase), replacing
the old facade's command-phase leader workaround, which existed only because
all 25 actions shared one step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


class StepKind(str, Enum):
    """What the env owes the policy at this step."""

    model_action = "model_action"
    turn_close = "turn_close"


class MoveDeclaration(IntEnum):
    """A unit's move type, declared on its opening step in the movement phase.

    ``remain_stationary`` skips the unit's remaining members — they cost no
    agent step, exactly as the rules make "remain stationary" a unit-level
    choice. ``advance`` exists only where the scenario registers advance bins;
    a fall back is inferred by the engine from a unit that began the phase
    engaged and moved, as it is in the whole-phase facade
    (`DEFERRED: fallback.declared_move_type`).
    """

    normal = 0
    remain_stationary = 1
    advance = 2


class ShootDeclaration(IntEnum):
    """Whether the unit shoots at all, declared on its opening shooting step.

    ``hold_fire`` skips the unit's remaining members. A unit that declares
    ``shoot`` still lets each member decline its own shot (STAY), which is the
    old facade's behaviour for every model.
    """

    shoot = 0
    hold_fire = 1


class ChargeDeclaration(IntEnum):
    """Whether the unit charges, declared on its opening charge-phase step.

    ``decline`` skips the unit's remaining members (a non-charging unit's only
    legal action in the charge phase is to stand still). ``charge`` binds the
    whole unit, exactly as the command-phase declaration did.
    """

    decline = 0
    charge = 1


@dataclass(frozen=True, slots=True)
class PerModelAction:
    """One step's action: which model acts, what it does, what its unit declares.

    ``model_index`` is None only on a turn-closing step, which carries no model
    action. ``declaration`` is read only on a unit's opening step — the phase
    decides which declaration enum it indexes — and ``None`` there means the
    phase's default (normal / shoot / decline / priority 0).
    """

    model_index: int | None = None
    action: int = STAY_ACTION
    declaration: int | None = None


@dataclass(slots=True)
class PerModelObservation:
    """What the per-model facade returns from ``reset`` and ``step``.

    Deliberately light: the build issue for the set network (#285) replaces
    this with the token/relation observation. What must exist already is the
    selection state — who may act, who has acted, which unit is open — and the
    two clocks' counters.
    """

    kind: StepKind
    phase: BattlePhase | None
    battle_round: int | None
    model_steps: int
    phase_model_steps: int
    selection_mask: np.ndarray
    acted_mask: np.ndarray
    open_unit: int | None
