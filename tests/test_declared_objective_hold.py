"""The declared-hold pot — §34's v12-lite term, env-state-driven."""

from __future__ import annotations

from typing import Any

import numpy as np

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.declared_objective_hold import (
    DeclaredObjectiveHoldCalculator,
)


class _Ctx:
    def __init__(self, cache: Any) -> None:
        self.distance_cache = cache


class _Model:
    def __init__(self, loc: Any, group: int, declared: int, alive: bool = True) -> None:
        self.location = np.array(loc, dtype=float)
        self.group_id = group
        self.declared_objective = declared
        self.is_alive = alive
        self.base_radius = 0.0


class _Obj:
    def __init__(self, loc: Any, radius: float = 1.0) -> None:
        self.location = np.array(loc, dtype=float)
        self.radius_size = radius
        self.area = None


class _View:
    def __init__(self, models: Any, objectives: Any) -> None:
        self.player_models = models
        self.objectives = objectives


def _pay(models: Any, objectives: Any, pot: float = 0.25) -> list[float]:
    calc = DeclaredObjectiveHoldCalculator(pot=pot)
    cache = compute_distances(models, objectives)
    view = _View(models, objectives)
    ctx = _Ctx(cache)
    return [
        calc.calculate(i, m, view, ctx)  # type: ignore[arg-type]
        for i, m in enumerate(models)
    ]


def test_the_pot_is_split_among_declaring_holders_and_conserved() -> None:
    # Arrange: two declaring holders on objective 0, one declaring holder on 1.
    objectives = [_Obj([0.0, 0.0]), _Obj([20.0, 0.0])]
    models = [
        _Model([0.1, 0.0], 0, 0),
        _Model([0.0, 0.2], 0, 0),
        _Model([20.0, 0.1], 1, 1),
    ]
    # Act
    pay = _pay(models, objectives)
    # Assert: each objective pays its whole pot once — split on 0, whole on 1.
    assert abs(pay[0] - 0.125) < 1e-12 and abs(pay[1] - 0.125) < 1e-12
    assert abs(pay[2] - 0.25) < 1e-12
    assert abs(sum(pay) - 0.5) < 1e-12


def test_holding_without_declaring_and_declaring_without_holding_pay_nothing() -> None:
    objectives = [_Obj([0.0, 0.0]), _Obj([20.0, 0.0])]
    models = [
        _Model([0.1, 0.0], 0, -1),  # holds 0, never declared
        _Model([10.0, 0.0], 1, 0),  # declared 0, stands nowhere near it
    ]
    pay = _pay(models, objectives)
    assert pay == [0.0, 0.0]


def test_a_dead_declarer_neither_earns_nor_dilutes() -> None:
    objectives = [_Obj([0.0, 0.0])]
    models = [
        _Model([0.1, 0.0], 0, 0),
        _Model([0.0, 0.1], 0, 0, alive=False),
    ]
    pay = _pay(models, objectives)
    assert abs(pay[0] - 0.25) < 1e-12 and pay[1] == 0.0
