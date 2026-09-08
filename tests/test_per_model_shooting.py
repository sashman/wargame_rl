"""Shooting through the per-model step, at the orderings the whole-army step
cannot honour.

`04-making-attacks.md` selects every target before any attack resolves, so a
member names its unit blind to a squadmate's dice and an attack at a unit a
squadmate has just wiped is lost (`05` § 4). Destroyed models are removed
"only after the attacking unit has resolved all of its attacks", so the NEXT
unit selects its targets, and has its cover judged, against the board its
predecessor's casualties left. Each is pinned with dice that kill on every
attack, on hand-placed geometry the test asserts before it relies on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.domain.activation import MoveDeclaration, ShootDeclaration
from wargame_rl.wargame.envs.domain.dice import DiceCall
from wargame_rl.wargame.envs.domain.sight import CLEAR, COVER, HIDDEN
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.per_model import (
    PerModelAction,
    PerModelEnv,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config.terrain import TerrainPieceConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, PlayerSide


@dataclass
class KillingDice:
    """Every attack hits, wounds and goes unsaved.

    A shot draws hit dice, then wound dice, then save dice, so the calls cycle
    6, 6, 1 -- a 1 always fails a save, cover or not."""

    calls: int = 0

    def draw(
        self,
        call: DiceCall,
        low: int,
        high: int | None,
        size: int | tuple[int, ...] | None,
    ) -> np.ndarray:
        face = (6, 6, 1)[self.calls % 3]
        self.calls += 1
        return np.full(size if size is not None else (), face, dtype=np.int64)

    def advance_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        return 4.0

    def charge_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        return 7.0


FAR = 50.0


def _env(terrain: list[TerrainPieceConfig] | None = None) -> PerModelEnv:
    config = small_config(opponent_x=22, rounds=1)
    if terrain is not None:
        config = WargameEnvConfig.model_validate(
            config.model_dump() | {"terrain": [t.model_dump() for t in terrain]}
        )
    env = PerModelEnv(config, dice_factory=lambda _seed, _env: KillingDice())
    env.reset(seed=1)
    return env


def _place(
    env: PerModelEnv, ours: list[tuple[float, float]], theirs: list[tuple[float, float]]
) -> None:
    for model, (x, y) in zip(env.wargame_models, ours, strict=True):
        model.location = position(x, y)
    for model, (x, y) in zip(env.opponent_models, theirs, strict=True):
        model.location = position(x, y)


def _to_shooting(env: PerModelEnv) -> PerModelObservation:
    """Both units remain stationary; the next decision is the shooting phase's."""
    observation = None
    for opener in (0, 3):
        observation, *_ = env.step(
            PerModelAction.open(opener, MoveDeclaration.stationary)
        )
    assert observation is not None
    assert observation.decision.phase is BattlePhase.shooting
    return observation


def _shooting_action(env: PerModelEnv, group: int) -> int:
    shooting = env.player_seat.handler.shooting_slice
    assert shooting is not None
    return shooting.start + group


# Unit A is models 0-2 at x=14, y=10/12/14; unit B is models 3-5, moved up to
# y=18/20/22 so both are within a 12" gun of an enemy at (22, 10). Enemy unit
# 1 is parked out of everyone's range.
OURS = [
    (14.0, 10.0),
    (14.0, 12.0),
    (14.0, 14.0),
    (14.0, 18.0),
    (14.0, 20.0),
    (14.0, 22.0),
]
THEIR_FAR_UNIT = [(FAR, 30.0), (FAR, 32.0), (FAR, 34.0)]


def test_a_units_targets_are_selected_blind_and_an_attack_at_a_wiped_unit_is_lost() -> (
    None
):
    """Arrange three shooters against a two-model unit; act by naming it three
    times; assert nothing dies until the unit closes, the third member is
    still offered the target, and the third attack is lost."""
    env = _env()
    _place(env, OURS, [(22.0, 10.0), (22.0, 12.0), (22.0, 14.0), *THEIR_FAR_UNIT])
    env.opponent_models[2].stats["current_wounds"] = 0
    observation = _to_shooting(env)
    target = _shooting_action(env, 0)

    observation, *_ = env.step(PerModelAction.open(0, ShootDeclaration.shoot))
    for shooter in (0, 1):
        observation, *_ = env.step(PerModelAction.act(shooter, target))
    point = observation.decision
    assert all(m.is_alive for m in env.opponent_models[:2]), "a shot resolved early"
    assert point.kind is StepKind.act and point.action_mask[2, target]

    env.step(PerModelAction.act(2, target))
    assert not any(m.is_alive for m in env.opponent_models[:3])
    results = env.last_player_shooting_results
    assert [r.attacker_idx for r in results] == [0, 1], "the third attack was not lost"


def test_the_next_unit_selects_targets_against_the_board_its_predecessor_left() -> None:
    """Arrange an enemy unit with one model in range and two out; act by having
    unit A kill the one; assert unit B, which could target the unit at phase
    open, is offered nothing, and the departure is recorded."""
    env = _env()
    _place(env, OURS, [(22.0, 10.0), (40.0, 12.0), (40.0, 14.0), *THEIR_FAR_UNIT])
    observation = _to_shooting(env)
    target = _shooting_action(env, 0)
    assert observation.decision.declaration_mask[3, ShootDeclaration.shoot], (
        "at phase open unit B could shoot the enemy unit"
    )

    env.step(PerModelAction.open(0, ShootDeclaration.shoot))
    env.step(PerModelAction.act(0, target))
    env.step(PerModelAction.act(1, 0))
    observation, *_ = env.step(PerModelAction.act(2, 0))
    assert not env.opponent_models[0].is_alive
    point = observation.decision
    assert point.kind is StepKind.close_turn or (
        point.phase is BattlePhase.shooting
        and not point.declaration_mask[3, ShootDeclaration.shoot]
    ), "unit B was still offered a target with no member in range"
    assert any(
        d.rule == "shooting.targets_judged_after_casualties" for d in env.departures
    )


def test_cover_is_judged_against_the_members_the_attacking_unit_faces() -> None:
    """Arrange a wall hiding two of three enemy models from unit B but none
    from unit A; act by having A kill the exposed one; assert B's shot at the
    survivors is in cover -- the corpse no longer denies it."""
    wall = TerrainPieceConfig(footprint=(18, 19, 19, 22))
    env = _env(terrain=[wall])
    theirs = [(22.0, 10.0), (25.0, 19.6), (25.0, 21.5), *THEIR_FAR_UNIT]
    _place(env, OURS, theirs)
    shooter = env.wargame_models[3]
    seen = env.visibility_between(
        np.array([shooter.location], dtype=float),
        np.array(theirs[:3], dtype=float),
        np.ones((1, 3), dtype=bool),
        origin_models=[shooter],
        target_models=env.opponent_models[:3],
    )[0]
    assert seen.tolist() == [CLEAR, COVER, HIDDEN], f"geometry: {seen.tolist()}"
    _to_shooting(env)
    target = _shooting_action(env, 0)

    env.step(PerModelAction.open(0, ShootDeclaration.shoot))
    env.step(PerModelAction.act(0, target))
    env.step(PerModelAction.act(1, 0))
    observation, *_ = env.step(PerModelAction.act(2, 0))
    assert not env.opponent_models[0].is_alive
    point = observation.decision
    assert point.phase is BattlePhase.shooting and point.kind is StepKind.open
    observation, *_ = env.step(PerModelAction.open(3, ShootDeclaration.shoot))
    assert observation.decision.action_mask[3, target], "the survivors are targetable"
    env.step(PerModelAction.act(3, target))
    env.step(PerModelAction.act(4, 0))
    env.step(PerModelAction.act(5, 0))
    last = env.last_player_shooting_results[-1]
    assert last.attacker_idx == 3 and last.in_cover
    assert any(
        d.rule == "shooting.cover_judged_after_casualties" for d in env.departures
    )
