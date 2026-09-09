"""Unit-close referees on hand-placed geometry: the declared target, and the modes."""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.activation import ConsolidationMode
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.unit_referees import (
    charge_stands,
    consolidation_mode,
    fall_back_stands,
    revert_unit,
)
from wargame_rl.wargame.envs.domain.value_objects import position

ENGAGEMENT = 1.0
NEAREST, FURTHEST = 2.0, 9.0


def _model(
    x: float, y: float, group: int, *, declared_charge: bool = False
) -> WargameModel:
    model = WargameModel(
        location=position(x, y),
        stats={"max_wounds": 1, "current_wounds": 1, "toughness": 4, "save": 5},
        distances_to_objectives=np.zeros((0, 2)),
        group_id=group,
        base_radius=0.0,
    )
    model.declared_charge = declared_charge
    return model


def _charge_kwargs() -> dict[str, float]:
    return {
        "reach": 6.0,
        "engagement_range": ENGAGEMENT,
        "base_diameter": 0.0,
        "coherency_nearest": NEAREST,
        "coherency_furthest": FURTHEST,
    }


def test_a_charge_stands_only_against_the_unit_it_declared() -> None:
    """Arrange a unit that ended engaged with enemy unit 0; assert the charge
    stands for target 0 and not for target 1."""
    unit = [
        _model(1.0, 0.0, 0, declared_charge=True),
        _model(2.0, 0.0, 0, declared_charge=True),
    ]
    starts = {0: np.array([-2.0, 0.0]), 1: np.array([-1.0, 0.0])}
    enemies = [_model(2.5, 0.0, 0), _model(30.0, 30.0, 1)]
    assert charge_stands(
        unit, [0, 1], enemies, starts, target_group=0, **_charge_kwargs()
    )
    assert not charge_stands(
        unit, [0, 1], enemies, starts, target_group=1, **_charge_kwargs()
    )


def test_a_charge_that_clips_a_second_unit_or_overran_its_roll_does_not_stand() -> None:
    """Engaging a non-target, or travelling past the roll, fails the whole charge."""
    unit = [
        _model(1.0, 0.0, 0, declared_charge=True),
        _model(2.0, 0.0, 0, declared_charge=True),
    ]
    starts = {0: np.array([-2.0, 0.0]), 1: np.array([-1.0, 0.0])}
    clipped = [_model(2.5, 0.0, 0), _model(2.5, 0.8, 1)]
    assert not charge_stands(
        unit, [0, 1], clipped, starts, target_group=0, **_charge_kwargs()
    )
    far_starts = {0: np.array([-20.0, 0.0]), 1: np.array([-19.0, 0.0])}
    enemies = [_model(2.5, 0.0, 0)]
    assert not charge_stands(
        unit, [0, 1], enemies, far_starts, target_group=0, **_charge_kwargs()
    )
    revert_unit(unit, [0, 1], far_starts)
    assert np.array_equal(unit[0].location, far_starts[0])


def test_a_fall_back_must_end_clear_and_whole() -> None:
    """Still engaged, or torn apart, a fall back did not happen."""
    clear = [_model(0.0, 0.0, 0), _model(1.0, 0.0, 0)]
    enemies = [_model(10.0, 0.0, 0)]
    kwargs = {
        "engagement_range": ENGAGEMENT,
        "base_diameter": 0.0,
        "coherency_nearest": NEAREST,
        "coherency_furthest": FURTHEST,
    }
    assert fall_back_stands(clear, [0, 1], enemies, **kwargs)
    still_engaged = [_model(9.5, 0.0, 0), _model(8.5, 0.0, 0)]
    assert not fall_back_stands(still_engaged, [0, 1], enemies, **kwargs)
    torn = [_model(0.0, 0.0, 0), _model(5.0, 0.0, 0)]
    assert not fall_back_stands(torn, [0, 1], enemies, **kwargs)


def test_the_consolidation_mode_is_the_first_that_applies() -> None:
    """Engaged is ongoing; within 3\" is engaging; near an objective is objective;
    none of those is none -- in that order."""
    unit = [_model(0.0, 0.0, 0)]
    kwargs = {
        "engagement_range": ENGAGEMENT,
        "base_diameter": 0.0,
        "consolidate_distance": 3.0,
    }
    assert (
        consolidation_mode(unit, [0], [_model(0.5, 0.0, 0)], None, **kwargs)
        is ConsolidationMode.ongoing
    )
    assert (
        consolidation_mode(unit, [0], [_model(2.5, 0.0, 0)], None, **kwargs)
        is ConsolidationMode.engaging
    )
    near_objective = np.array([[2.0]])
    assert (
        consolidation_mode(unit, [0], [_model(20.0, 0.0, 0)], near_objective, **kwargs)
        is ConsolidationMode.objective
    )
    assert (
        consolidation_mode(
            unit, [0], [_model(20.0, 0.0, 0)], np.array([[8.0]]), **kwargs
        )
        is ConsolidationMode.none
    )
