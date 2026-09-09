"""Dice through the port: the seeded default reproduces the phase facade, and an
injected source changes the game.

The bridge already proves the combat stream matches over whole episodes; this
pins the two things the bridge cannot see -- that a unit's advance and charge
rolls are the phase facade's for the same seed however they are asked for, and
that a source other than the default actually reaches the attack sequence,
tagged with what it is rolling for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.domain.activation import MoveDeclaration, ShootDeclaration
from wargame_rl.wargame.envs.domain.dice import DiceCall, DicePurpose, DiceSource
from wargame_rl.wargame.envs.per_model import (
    PerModelAction,
    PerModelEnv,
    SeededDice,
    StepKind,
)
from wargame_rl.wargame.envs.per_model.env import default_dice
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, PlayerSide
from wargame_rl.wargame.envs.wargame import WargameEnv


@dataclass
class FixedDice:
    """Every die shows `face`; every unit rolls `advance` and `charge`."""

    face: int
    advance: float = 4.0
    charge: float = 7.0

    def draw(
        self,
        call: DiceCall,
        low: int,
        high: int | None,
        size: int | tuple[int, ...] | None,
    ) -> np.ndarray:
        return np.full(size if size is not None else (), self.face, dtype=np.int64)

    def advance_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        return self.advance

    def charge_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        return self.charge


@dataclass
class RecordingDice:
    """Forwards to an inner source and records every call's purpose."""

    inner: DiceSource
    calls: list[DiceCall] = field(default_factory=list)
    rolls: list[tuple[str, PlayerSide, int, int]] = field(default_factory=list)

    def draw(
        self,
        call: DiceCall,
        low: int,
        high: int | None,
        size: int | tuple[int, ...] | None,
    ) -> np.ndarray:
        self.calls.append(call)
        return self.inner.draw(call, low, high, size)

    def advance_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        self.rolls.append(("advance", side, battle_round, unit))
        return float(self.inner.advance_roll(side, battle_round, unit))

    def charge_roll(self, side: PlayerSide, battle_round: int, unit: int) -> float:
        self.rolls.append(("charge", side, battle_round, unit))
        return float(self.inner.charge_roll(side, battle_round, unit))


def _advance_config():  # type: ignore[no-untyped-def]
    return small_config(melee=True, opponent_x=22, rounds=2).model_copy(
        update={"n_advance_speed_bins": 3}
    )


def test_the_seeded_source_deals_the_phase_facades_rolls_per_unit() -> None:
    """Arrange both facades on one seed; assert every unit's advance D6 and
    charge 2D6 agree, whichever order the per-model facade asks in."""
    config = _advance_config()
    old = WargameEnv(config)
    old.reset(seed=21)
    env = PerModelEnv(config)
    env.reset(seed=21)
    side = env.player_side
    for unit in (1, 0):
        old_advance = old.wargame_models[3 * unit].advance_roll
        old_charge = old.wargame_models[3 * unit].charge_roll
        assert env.dice.advance_roll(side, 1, unit) == old_advance
        assert env.dice.charge_roll(side, 1, unit) == old_charge
    seeded = env.dice
    assert isinstance(seeded, SeededDice)
    assert seeded.state["combat"] == dict(old._combat_rng.bit_generator.state)


def _shoot_once(env: PerModelEnv) -> None:
    """Stand still, then have unit 0 fire at the first legal target."""
    observation, _ = env.reset(seed=8)
    for opener in (0, 3):
        observation, *_ = env.step(
            PerModelAction.open(opener, MoveDeclaration.stationary)
        )
    point = observation.decision
    assert point.phase is BattlePhase.shooting and point.kind is StepKind.open
    observation, *_ = env.step(PerModelAction.open(0, ShootDeclaration.shoot))
    point = observation.decision
    while point.kind is StepKind.act and point.phase is BattlePhase.shooting:
        model = int(np.flatnonzero(point.selector_mask)[0])
        legal = [a for a in np.flatnonzero(point.action_mask[model]) if a != 0]
        observation, *_ = env.step(PerModelAction.act(model, int(legal[0])))
        point = observation.decision


def test_an_injected_source_reaches_the_attack_sequence() -> None:
    """Sixes hit and wound every time; ones never do."""
    config = small_config(opponent_x=22, rounds=1)
    sixes = PerModelEnv(config, dice_factory=lambda seed, env: FixedDice(6))
    _shoot_once(sixes)
    shots = sixes.last_player_shooting_results
    assert shots and all(r.result.hits == 2 for r in shots)
    ones = PerModelEnv(config, dice_factory=lambda seed, env: FixedDice(1))
    _shoot_once(ones)
    assert ones.last_player_shooting_results
    assert all(r.result.hits == 0 for r in ones.last_player_shooting_results)


def test_every_roll_is_tagged_with_its_purpose_and_its_roller() -> None:
    """Shooting draws name the shooter; the agent's units are asked for their
    advance roll on their declaration step, the scripted seat's at turn start."""
    recorder: dict[str, RecordingDice] = {}

    def factory(seed: int, env: PerModelEnv) -> RecordingDice:
        recorder["dice"] = RecordingDice(default_dice(seed, env))
        return recorder["dice"]

    config = small_config(opponent_x=22, rounds=1).model_copy(
        update={
            "n_advance_speed_bins": 3,
            "skip_phases": [
                BattlePhase.charge,
                BattlePhase.pile_in,
                BattlePhase.fight,
                BattlePhase.consolidate,
            ],
        }
    )
    env = PerModelEnv(config, dice_factory=factory)
    observation, _ = env.reset(seed=8)
    dice = recorder["dice"]
    asked_before = [r for r in dice.rolls if r[1] == env.player_side]
    assert not asked_before, "the agent's units are not asked for rolls at turn start"
    env.step(PerModelAction.open(0, MoveDeclaration.advance))
    assert ("advance", env.player_side, 1, 0) in dice.rolls
    assert ("advance", env.player_side, 1, 1) not in dice.rolls
    _shoot_once(env)
    # `_shoot_once` resets, and a reset builds a fresh source through the factory.
    dice = recorder["dice"]
    # Unit 1 holds fire, which ends our turn and runs the scripted opponent's.
    env.step(PerModelAction.open(3, ShootDeclaration.hold_fire))
    shooting = [c for c in dice.calls if c.purpose is DicePurpose.shooting]
    assert shooting and all(c.model is not None and c.unit == 0 for c in shooting[:1])
    opponent_side = (
        PlayerSide.player_2
        if env.player_side is PlayerSide.player_1
        else PlayerSide.player_1
    )
    assert any(r[0] == "advance" and r[1] == opponent_side for r in dice.rolls)
