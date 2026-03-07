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
    """Phases within a single player turn, executed in strict order."""

    command = "command"
    movement = "movement"
    shooting = "shooting"
    charge = "charge"
    fight = "fight"


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
