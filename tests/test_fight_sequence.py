"""The resumable fight sequence replays the closed loop, and a striker may choose.

`FightSequence` is a transcription of `resolve_fight_step`; the load-bearing
assertion is that driving it with the loop's own default choice produces the
same blows, in the same order, on the same dice, as the loop itself. Beside
it: `fight_one_model` with a chosen target strikes that unit rather than the
first in contact.
"""

from __future__ import annotations

import copy

import numpy as np

from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.fight import FightSide, resolve_fight_step
from wargame_rl.wargame.envs.domain.fight_sequence import (
    End,
    FightSequence,
    Overrun,
    Select,
    contact_groups,
    default_choice,
    fight_one_model,
)
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.types.config import MeleeWeaponProfile

ENGAGEMENT = 1.0


def _model(x: float, y: float, group: int) -> WargameModel:
    return WargameModel(
        location=position(x, y),
        stats={"max_wounds": 2, "current_wounds": 2, "toughness": 4, "save": 5},
        distances_to_objectives=np.zeros((0, 2)),
        group_id=group,
        base_radius=0.0,
    )


def _two_units_a_side() -> tuple[list[WargameModel], list[WargameModel]]:
    """Two units a side, each engaged with its opposite number."""
    ours = [
        _model(0.0, 0.0, 0),
        _model(1.0, 0.0, 0),
        _model(0.0, 10.0, 1),
        _model(1.0, 10.0, 1),
    ]
    theirs = [
        _model(0.5, 0.6, 0),
        _model(1.5, 0.6, 0),
        _model(0.5, 10.6, 1),
        _model(1.5, 10.6, 1),
    ]
    return ours, theirs


def _blows(results: list) -> list[tuple[int, int, int, bool]]:
    return [
        (r.attacker_idx, r.target_idx, r.result.damage_dealt, r.killed) for r in results
    ]


def _drive(sequence: FightSequence, rng: np.random.Generator) -> tuple[list, list]:
    results: tuple[list, list] = ([], [])
    while True:
        event = sequence.next()
        if isinstance(event, End):
            return results
        if isinstance(event, Overrun):
            results[event.seat].extend(sequence.resolve_overrun(rng))
            continue
        assert isinstance(event, Select)
        group = default_choice(sequence.sides[event.seat].models, event.pool)
        results[event.seat].extend(sequence.resolve_selected(group, rng))


def test_the_sequence_replays_the_closed_loop_blow_for_blow() -> None:
    """Arrange the same board twice; act with the loop and with the sequence on
    the same dice; assert identical blows and identical wounds."""
    ours_a, theirs_a = _two_units_a_side()
    ours_b, theirs_b = copy.deepcopy(ours_a), copy.deepcopy(theirs_a)
    weapons = [[MeleeWeaponProfile()] for _ in range(4)]

    loop = resolve_fight_step(
        (FightSide(ours_a, weapons), FightSide(theirs_a, weapons)),
        np.random.default_rng(7),
        engagement_range=ENGAGEMENT,
        base_diameter=0.0,
        pass_range=5.0,
    )
    sequence = FightSequence(
        (FightSide(ours_b, weapons), FightSide(theirs_b, weapons)),
        engagement_range=ENGAGEMENT,
        base_diameter=0.0,
        pass_range=5.0,
    )
    driven = _drive(sequence, np.random.default_rng(7))

    assert _blows(loop[0]) == _blows(driven[0])
    assert _blows(loop[1]) == _blows(driven[1])
    assert [m.stats["current_wounds"] for m in ours_a] == [
        m.stats["current_wounds"] for m in ours_b
    ]
    assert [m.stats["current_wounds"] for m in theirs_a] == [
        m.stats["current_wounds"] for m in theirs_b
    ]
    assert sequence.fought == ({0, 1}, {0, 1})


def test_a_striker_may_choose_which_unit_in_contact_it_swings_at() -> None:
    """A model touching two enemy units strikes the one it names; unnamed, the
    first in contact -- the closed loop's rule."""
    attacker = _model(0.0, 0.0, 0)
    defenders = [_model(0.5, 0.0, 0), _model(0.0, 0.5, 1)]
    weapons = [[MeleeWeaponProfile(attacks=6, melee_skill=2, strength=8)]]
    sequence = FightSequence(
        (FightSide([attacker], weapons), FightSide(defenders, weapons)),
        engagement_range=ENGAGEMENT,
        base_diameter=0.0,
        pass_range=5.0,
    )
    matrix = sequence.contact_matrix(0)
    assert contact_groups([attacker], defenders, 0, matrix) == {0, 1}

    chosen = fight_one_model(
        [attacker],
        copy.deepcopy(defenders),
        0,
        np.random.default_rng(1),
        matrix=matrix,
        attacker_weapons=weapons,
        target_group=1,
    )
    default = fight_one_model(
        [attacker],
        copy.deepcopy(defenders),
        0,
        np.random.default_rng(1),
        matrix=matrix,
        attacker_weapons=weapons,
    )
    assert chosen is not None and chosen.target_group == 1 and chosen.target_idx == 1
    assert default is not None and default.target_group == 0 and default.target_idx == 0
