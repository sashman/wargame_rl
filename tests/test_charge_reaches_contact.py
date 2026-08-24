"""A charge is the one move allowed to end inside an enemy's engagement range.

⚠ This is the mechanic. Everything else in melee is bookkeeping around it.

`back_off_to_unengaged` runs on every mover on both seats, so engagement is
0.0000% of model-pairs across 60,520 observations — and the minimum edge-to-edge
gap is **1.000008740"** against an engagement range of 1.0. The army is not far
from contact; it is parked 8.7 micro-inches outside it and always has been. So
the charge does not need to cross a gap. It needs the exemption.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame_model import WargameModel

ENGAGEMENT = 1.0


def _handler(melee: bool) -> ActionHandler:
    skip = [BattlePhase.command, BattlePhase.shooting, BattlePhase.fight]
    if not melee:
        skip.append(BattlePhase.charge)
    return ActionHandler(
        WargameEnvConfig(
            number_of_wargame_models=1,
            models=[ModelConfig()],
            melee=MeleeConfig(enabled=melee),
            engagement_range=ENGAGEMENT,
            base_radius=0.0,
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


def _walk_at(handler: ActionHandler, phase: BattlePhase) -> tuple[float, bool]:
    """Step one model 1" at an enemy 1.5" away; return its gap and whether engaged.

    Short deliberately: a full 6" move would carry it straight past the enemy
    and out the far side, which ends unengaged for a reason that has nothing to
    do with the rule under test.
    """
    mover = _model(5.0)
    enemy = _model(6.5, group=1)
    handler.apply(
        WargameEnvAction(
            actions=[handler.best_action_toward(1.0, 0.0, max_step_length=1.0)]
        ),
        [mover],
        60,
        44,
        handler.action_space,
        phase=phase,
        enemy_models=[enemy],
    )
    gap = float(np.linalg.norm(mover.location - enemy.location))
    engaged = bool(
        engaged_with_any(
            np.array([mover.location]),
            np.array([enemy.location]),
            np.array([True]),
            engagement_range=ENGAGEMENT,
        )[0]
    )
    return gap, engaged


def test_a_normal_move_parks_just_outside_contact() -> None:
    """The behaviour that made engagement unreachable — and it is microscopic."""
    gap, engaged = _walk_at(_handler(melee=True), BattlePhase.movement)
    assert not engaged
    assert gap > ENGAGEMENT
    assert gap - ENGAGEMENT < 1e-4, (
        f"parked {gap - ENGAGEMENT:.3e} outside contact — the gap a charge must cross"
    )


def test_a_charge_move_may_end_engaged() -> None:
    """The exemption: the same walk, in the charge phase, reaches contact."""
    _, engaged = _walk_at(_handler(melee=True), BattlePhase.charge)
    assert engaged, "a charge could not reach contact"


def test_with_melee_off_the_charge_phase_displaces_nobody() -> None:
    """The switch. Off, the movement slice is not legal in the charge phase."""
    handler = _handler(melee=False)
    mover = _model(5.0)
    start = mover.location.copy()
    handler.apply(
        WargameEnvAction(
            actions=[handler.best_action_toward(1.0, 0.0, max_step_length=1.0)]
        ),
        [mover],
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=[_model(6.5, group=1)],
    )
    assert np.array_equal(mover.location, start)


def test_a_charge_still_may_not_end_inside_another_model() -> None:
    """The exemption drops the engagement rings, never the occupied bases."""
    handler = _handler(melee=True)
    mover = WargameModel(
        location=np.array([5.0, 10.0]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=0,
        base_radius=0.5,
    )
    enemy = WargameModel(
        location=np.array([6.0, 10.0]),
        stats={"toughness": 3, "save": 4, "max_wounds": 1, "current_wounds": 1},
        distances_to_objectives=np.zeros(1),
        group_id=1,
        base_radius=0.5,
    )
    handler.apply(
        WargameEnvAction(actions=[handler.best_action_toward(1.0, 0.0)]),
        [mover],
        60,
        44,
        handler.action_space,
        phase=BattlePhase.charge,
        enemy_models=[enemy],
    )
    separation = float(np.linalg.norm(mover.location - enemy.location))
    assert separation >= 1.0 - 1e-6, f"models overlap: centres {separation:.4f} apart"
