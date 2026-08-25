"""An engaged unit that withdraws pays for it.

`docs/rules/09-movement-phase.md`: a Normal move is eligible only for an
UNENGAGED unit, so the only move an engaged unit may make is a fall back — and
"until the end of the turn it cannot shoot or declare a charge".

⚠ The geometry was already right. `back_off_to_unengaged` forces every endpoint
out of engagement, which IS the fall-back constraint. What was missing is the
COST: before this an engaged model took an ordinary move, walked out for free,
and shot in the same turn.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame_model import WargameModel


def _handler(melee: bool) -> ActionHandler:
    """Melee on needs the charge and fight phases stepped; the validator says so."""
    skip = [BattlePhase.shooting]
    if not melee:
        skip += [BattlePhase.command, BattlePhase.charge, BattlePhase.fight]
    return ActionHandler(
        WargameEnvConfig(
            number_of_wargame_models=2,
            models=[ModelConfig() for _ in range(2)],
            melee=MeleeConfig(enabled=melee),
            skip_phases=skip,
        )
    )


def _model(x: float, group: int = 0) -> WargameModel:
    return WargameModel(
        location=np.array([x, 10.0]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=group,
        base_radius=0.0,
    )


def _move_east(
    handler: ActionHandler, movers: list[WargameModel], enemy: list[WargameModel]
) -> None:
    east = handler.best_action_toward(1.0, 0.0)
    handler.apply(
        WargameEnvAction(actions=[east] * len(movers)),
        movers,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.movement,
        enemy_models=enemy,
    )


def test_a_unit_that_withdraws_from_contact_forfeits_its_shooting() -> None:
    handler = _handler(melee=True)
    movers = [_model(10.0), _model(10.2)]
    enemy = [_model(10.5, group=1)]
    _move_east(handler, movers, enemy)
    assert all(m.fell_back_this_turn for m in movers)


def test_a_unit_that_was_never_engaged_pays_nothing() -> None:
    handler = _handler(melee=True)
    movers = [_model(10.0), _model(10.2)]
    enemy = [_model(40.0, group=1)]
    _move_east(handler, movers, enemy)
    assert not any(m.fell_back_this_turn for m in movers)


def test_the_whole_unit_pays_even_the_models_that_stood_still() -> None:
    """The rule costs the UNIT its shooting, not the models that walked."""
    handler = _handler(melee=True)
    movers = [_model(10.0), _model(10.2)]
    enemy = [_model(10.5, group=1)]
    east = handler.best_action_toward(1.0, 0.0)
    handler.apply(
        WargameEnvAction(actions=[east, 0]),  # model 1 stays
        movers,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.movement,
        enemy_models=enemy,
    )
    assert movers[1].fell_back_this_turn, "a stationary model of a withdrawing unit"


def test_melee_off_is_an_exact_no_op() -> None:
    """The defect that shipped: walk out of contact for free, and still shoot."""
    handler = _handler(melee=False)
    movers = [_model(10.0), _model(10.2)]
    enemy = [_model(10.5, group=1)]
    _move_east(handler, movers, enemy)
    assert not any(m.fell_back_this_turn for m in movers)
