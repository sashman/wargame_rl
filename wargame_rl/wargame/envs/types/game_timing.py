"""Game timing types: phases, sides, and state snapshots.

Models the full tabletop game timing structure — pre-game setup stages
and in-battle rounds/turns/phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SetupPhase(str, Enum):
    """Pre-game setup stages, resolved in order before battle begins."""

    muster_armies = "muster_armies"
    select_mission = "select_mission"
    create_battlefield = "create_battlefield"
    determine_attacker_defender = "determine_attacker_defender"
    declare_battle_formations = "declare_battle_formations"
    deploy_armies = "deploy_armies"
    determine_first_turn = "determine_first_turn"
    resolve_pre_battle_rules = "resolve_pre_battle_rules"


class BattlePhase(str, Enum):
    """Phases within a single player turn, executed in strict order.

    ⚠ **`pile_in` and `consolidate` are the fight phase's own steps**, promoted
    to phases of their own in 2026-08-26. `12-fight-phase.md` runs pile-in, then
    the fight, then consolidate, and each is a separate player decision -- but a
    model gets ONE action per step, and the fight phase's action is already the
    activation priority. Three decisions therefore need three decision points.

    ⚠ **Declaration order is EXECUTION order** (`BATTLE_PHASE_ORDER` is
    `tuple(BattlePhase)`), so these had to be inserted rather than appended, and
    two things move with them: `fight` is no longer the last phase, and
    `BATTLE_PHASE_ORDER[-1]` -- which stands in for end of turn -- is now
    `consolidate`. That is more correct: coherency should be regained after the
    survivors have finished moving, not before they consolidate.

    ⚠ Both are auto-skipped where melee is off, so every non-melee config keeps
    the `max_turns` and the stepped-phase set it always had.
    """

    command = "command"
    movement = "movement"
    shooting = "shooting"
    charge = "charge"
    pile_in = "pile_in"
    fight = "fight"
    consolidate = "consolidate"


class GamePhase(str, Enum):
    """Top-level game stage."""

    setup = "setup"
    battle = "battle"
    complete = "complete"


class PlayerSide(str, Enum):
    """Generic player identifier.

    The environment maps these to concrete roles (e.g. RL agent vs opponent,
    attacker vs defender) when integrating with the clock.
    """

    player_1 = "player_1"
    player_2 = "player_2"


SETUP_PHASE_ORDER: tuple[SetupPhase, ...] = tuple(SetupPhase)

BATTLE_PHASE_ORDER: tuple[BattlePhase, ...] = tuple(BattlePhase)

# The fight phase's own steps, which exist only where melee is enabled and are
# auto-skipped otherwise so that no non-melee config changes length. Named
# rather than counted: a bare `- 2` in a test is a magic number that stops
# meaning anything the next time a phase is added.
MELEE_ONLY_PHASES: tuple[BattlePhase, ...] = (
    BattlePhase.pile_in,
    BattlePhase.consolidate,
)

NON_MOVEMENT_PHASES: list[BattlePhase] = [
    p for p in BattlePhase if p != BattlePhase.movement
]


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable snapshot of the current game timing position.

    During setup, only ``game_phase`` and ``setup_phase`` are meaningful.
    During battle, ``battle_round``, ``active_player``, and ``phase`` are set.
    When complete, all optional fields are None.
    """

    game_phase: GamePhase
    setup_phase: SetupPhase | None = None
    battle_round: int | None = None
    active_player: PlayerSide | None = None
    phase: BattlePhase | None = None
