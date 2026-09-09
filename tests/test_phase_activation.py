"""The unit lock, as pure state: who may act next, and what may not happen twice."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.sequencing.activation import (
    ActivationError,
    PhaseActivation,
)

GROUPS = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)


def _alive(dead: tuple[int, ...] = ()) -> np.ndarray:
    alive = np.ones(6, dtype=bool)
    for index in dead:
        alive[index] = False
    return alive


def test_the_lock_narrows_to_the_open_unit_and_widens_when_it_closes() -> None:
    """Arrange two units; act by opening one and acting through it; assert the
    selectable set narrows to the unit, then to nothing, then to the other."""
    activation = PhaseActivation(6)
    assert activation.selectable(_alive(), GROUPS, {0, 1}).tolist() == [True] * 6

    activation.open(1, 3, declaration=1, force_opener=True)
    assert activation.selectable(_alive(), GROUPS, {0, 1}).tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    activation.mark_acted(3, 1)
    assert activation.selectable(_alive(), GROUPS, {0, 1}).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    activation.mark_acted(4, 1)
    activation.mark_acted(5, 1)
    assert not activation.selectable(_alive(), GROUPS, {0, 1}).any()
    activation.close()
    assert activation.selectable(_alive(), GROUPS, {0, 1}).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_the_invariants_raise_rather_than_drift() -> None:
    """A second open, a stranger acting, a reopen and a double act all raise."""
    activation = PhaseActivation(6)
    activation.open(0, 0, declaration=1, force_opener=False)
    with pytest.raises(ActivationError):
        activation.open(1, 3, declaration=1, force_opener=False)
    with pytest.raises(ActivationError):
        activation.mark_acted(3, 1)
    activation.mark_acted(0, 0)
    with pytest.raises(ActivationError):
        activation.mark_acted(0, 0)
    activation.mark_acted(1, 0)
    activation.mark_acted(2, 0)
    activation.close()
    with pytest.raises(ActivationError):
        activation.open(0, 0, declaration=1, force_opener=False)
    with pytest.raises(ActivationError):
        activation.close()


def test_the_dead_and_units_outside_the_phase_are_never_selectable() -> None:
    """A casualty and a unit the phase does not admit never appear."""
    activation = PhaseActivation(6)
    selectable = activation.selectable(_alive(dead=(1,)), GROUPS, {0})
    assert selectable.tolist() == [True, False, True, False, False, False]
    activation.close_without_steps(0, [0, 1, 2])
    assert np.flatnonzero(activation.selectable(_alive(), GROUPS, {0, 1})).tolist() == [
        3,
        4,
        5,
    ]
    assert activation.acted[:3].all()


def test_a_forced_model_is_the_only_selectable_one_until_it_acts() -> None:
    """After a declaration the opener is bound; after it acts the unit widens."""
    activation = PhaseActivation(6)
    activation.open(0, 2, declaration=1, force_opener=True)
    assert activation.forced_model == 2
    assert np.flatnonzero(activation.selectable(_alive(), GROUPS, {0, 1})).tolist() == [
        2
    ]
    activation.mark_acted(2, 0)
    assert activation.forced_model is None
    assert np.flatnonzero(activation.selectable(_alive(), GROUPS, {0, 1})).tolist() == [
        0,
        1,
    ]
