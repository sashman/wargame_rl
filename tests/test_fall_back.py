"""An engaged unit that withdraws pays for it.

`docs/rules/09-movement-phase.md`: a Normal move is eligible only for an
UNENGAGED unit, so the only move an engaged unit may make is a fall back — and
"until the end of the turn it cannot shoot or declare a charge".

⚠ **RETRACTED 2026-08-29: "the geometry was already right."**
`back_off_to_unengaged` clears only the endpoints that MOVED, so a member that
stood still stayed engaged while its squadmates marched away — measured tearing
units to 16.7–20.0" chain gaps on the opponent seat. The after-move conditions
(unit unengaged, unit coherent) are now `ActionHandler._enforce_fall_back`,
tested through `env.step` in `test_fall_back_referee.py`. What this file pins
is the COST: a unit that legally withdraws forfeits its shooting.
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


def test_a_withdrawal_that_leaves_a_member_engaged_never_happened() -> None:
    """REPLACES `test_the_whole_unit_pays_even_the_models_that_stood_still`.

    That test pinned the pre-referee behaviour this file's docstring retracts:
    one model walked out while its engaged squadmate stood still, and the unit
    was marked as having fallen back. Under the rules that fall back is illegal
    (the unit must END unengaged), so it reverts — nobody moved, nobody pays.
    The unit-level cost the old test wanted is still pinned, by
    `test_a_unit_that_withdraws_from_contact_forfeits_its_shooting` above and
    by the env.step suite in `test_fall_back_referee.py`.
    """
    handler = _handler(melee=True)
    movers = [_model(10.0), _model(10.2)]
    enemy = [_model(10.5, group=1)]
    east = handler.best_action_toward(1.0, 0.0)
    handler.apply(
        WargameEnvAction(actions=[east, 0]),  # model 1 stays, still engaged
        movers,
        60,
        44,
        handler.action_space,
        phase=BattlePhase.movement,
        enemy_models=enemy,
    )
    assert np.allclose(movers[0].location, [10.0, 10.0]), "the walker reverts"
    assert not any(m.fell_back_this_turn for m in movers), "nothing was paid"


def test_melee_off_is_an_exact_no_op() -> None:
    """The defect that shipped: walk out of contact for free, and still shoot."""
    handler = _handler(melee=False)
    movers = [_model(10.0), _model(10.2)]
    enemy = [_model(10.5, group=1)]
    _move_east(handler, movers, enemy)
    assert not any(m.fell_back_this_turn for m in movers)
