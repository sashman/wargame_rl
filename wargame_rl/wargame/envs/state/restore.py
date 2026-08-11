"""Rebuild live state from a snapshot: the read direction of `snapshot.py`.

`build_snapshot` turns the board into a `GameStateSnapshot`; these functions
turn one back. They are kept together with the schema rather than on the env
because they are the half that changes whenever the schema does -- coordinates
becoming floats, a base radius appearing on every model, a version bump -- and
a schema change should be one module's problem.

Everything here mutates the objects it is handed. Restoring is not building:
the entities already exist with their configured stats, and only the parts a
snapshot actually carries are overwritten.
"""

from __future__ import annotations

from collections.abc import Sequence

from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.game_clock import GameClock
from wargame_rl.wargame.envs.domain.shooting import PairedShootingResult, ShootingResult
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.state.snapshot import (
    ClockSnapshot,
    CombatResultSnapshot,
    ModelSnapshot,
    ObjectiveSnapshot,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, GamePhase, PlayerSide
from wargame_rl.wargame.envs.types.geometry import Polygon


def restore_clock(clock: GameClock, snapshot: ClockSnapshot, total_steps: int) -> None:
    """Move *clock* to the timing position the snapshot recorded."""
    clock.set_state(
        GamePhase(snapshot.game_phase),
        battle_round=snapshot.battle_round,
        active_player=(
            PlayerSide(snapshot.active_player) if snapshot.active_player else None
        ),
        phase=BattlePhase(snapshot.battle_phase) if snapshot.battle_phase else None,
        total_steps=total_steps,
    )


def restore_models(
    models: Sequence[WargameModel], snapshots: Sequence[ModelSnapshot]
) -> None:
    """Restore position, wounds and turn flags for each model.

    Reward-shaping memory (previous/best objective distance, the per-model
    reward history) is *cleared* rather than restored: none of it is in the
    snapshot, and carrying stale values over would shape the next step against
    distances from a different episode.
    """
    for model, snapshot in zip(models, snapshots):
        model.location = position(*snapshot.location)
        model.previous_location = (
            position(*snapshot.previous_location)
            if snapshot.previous_location is not None
            else None
        )
        model.stats["current_wounds"] = snapshot.current_wounds
        model.advanced_this_turn = snapshot.advanced_this_turn
        model.previous_closest_objective_distance = None
        model.best_closest_objective_distance = None
        model.model_rewards_history.clear()


def restore_objectives(
    objectives: Sequence[WargameObjective], snapshots: Sequence[ObjectiveSnapshot]
) -> None:
    """Restore each objective's position, and its outline when it has one.

    A marker's radius is configuration rather than state, but an *area* is not:
    with `objectives_on_terrain` the outline is drawn from the layout, so it
    varies per episode and has to come back from the snapshot or the restored
    board scores control against a radius of 0.
    """
    for objective, snapshot in zip(objectives, snapshots):
        if snapshot.area is not None:
            objective.set_area(Polygon.from_points([(x, y) for x, y in snapshot.area]))
            continue
        objective.location = position(*snapshot.location)


def restore_shooting_results(
    snapshots: Sequence[CombatResultSnapshot],
) -> list[PairedShootingResult]:
    """Rebuild the last phase's shooting results from the snapshot.

    `killed` has ridden along since schema 2.3 and defaults to False on older
    recordings, where a kill restores as an ordinary hit. Nothing downstream of a
    restore acts on the flag -- the reward calculators consumed it in the step
    that produced it -- but the renderer draws a killing shot differently.
    """
    return [
        PairedShootingResult(
            attacker_idx=snapshot.attacker_idx,
            target_idx=snapshot.target_idx,
            result=ShootingResult(
                hits=snapshot.hits,
                wounds=snapshot.wounds,
                unsaved=snapshot.unsaved,
                damage_dealt=snapshot.damage_dealt,
            ),
            killed=snapshot.killed,
        )
        for snapshot in snapshots
    ]
