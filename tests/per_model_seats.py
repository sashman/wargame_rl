"""Test-side fixtures for the per-model facade.

`random_legal_action` -- the fuzz seat -- lives in the package
(`per_model/random_seat.py`) since the play and record recipes use it too;
it is re-exported here for the tests that share it. `charging_driver` is the
hand-written seat that closes, charges the nearest unit and strikes whatever
is offered -- the shortest route to a standing charge and a fight step, which
the melee and the set-network tests both need.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.sequencing.activation import (
    CHARGE_TARGET_DECLINE,
    ChargeDeclaration,
    MoveDeclaration,
    ShootDeclaration,
    ShortMoveDeclaration,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION, MoveLadder
from wargame_rl.wargame.envs.per_model import (
    DecisionPoint,
    PerModelAction,
    PerModelEnv,
    StepKind,
)
from wargame_rl.wargame.envs.per_model.random_seat import random_legal_action
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    TurnOrder,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import MeleeWeaponProfile, WeaponProfile
from wargame_rl.wargame.envs.types.config.melee import MeleeConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


def small_config(
    *,
    melee: bool = False,
    skip_phases: list[BattlePhase] | None = None,
    baseline: str = "squad_march_take",
    player_x: int = 14,
    opponent_x: int = 56,
    rounds: int = 3,
    turn_order: TurnOrder = TurnOrder.player,
    n_models: int = 6,
    group_ids: list[int] | None = None,
    max_groups: int = 2,
) -> WargameEnvConfig:
    """Two units of three a side, four objectives, a 60x40 board.

    The geometry `test_scripted_baseline_opponent` uses, so a mirrored and an
    un-mirrored seat are distinguishable; `player_x`/`opponent_x` move the
    armies without moving the objectives, so a melee test can start the two
    within charge range while a movement test keeps them apart.

    `n_models`, `group_ids` and `max_groups` reshape both armies at once, for
    the size-independence tests: the defaults are the two units of three, and
    `group_ids` may leave gaps (a `{0, 2}` army needs `max_groups=3`, since
    the config validator refuses `group_id >= max_groups`).
    """
    weapons = [WeaponProfile(range=12, attacks=2)]
    melee_weapons = [MeleeWeaponProfile()] if melee else []
    groups = group_ids if group_ids is not None else [i // 3 for i in range(n_models)]
    if len(groups) != n_models:
        raise ValueError("group_ids must name one group per model")
    return WargameEnvConfig(
        render_mode=None,
        board_width=60,
        board_height=40,
        number_of_wargame_models=n_models,
        number_of_opponent_models=n_models,
        number_of_objectives=4,
        objective_radius_size=3,
        number_of_battle_rounds=rounds,
        max_groups=max_groups,
        turn_order=turn_order,
        skip_phases=(
            skip_phases
            if skip_phases is not None
            else (
                []
                if melee
                else [
                    BattlePhase.command,
                    BattlePhase.charge,
                    BattlePhase.pile_in,
                    BattlePhase.fight,
                    BattlePhase.consolidate,
                ]
            )
        ),
        melee=MeleeConfig(enabled=melee) if melee else MeleeConfig(),
        models=[
            ModelConfig(
                x=player_x,
                y=_row_y(i, groups),
                group_id=groups[i],
                weapons=weapons,
                melee_weapons=melee_weapons,
            )
            for i in range(n_models)
        ],
        opponent_models=[
            ModelConfig(
                x=opponent_x,
                y=_row_y(i, groups),
                group_id=groups[i],
                weapons=weapons,
                melee_weapons=melee_weapons,
            )
            for i in range(n_models)
        ],
        objectives=[
            ObjectiveConfig(x=14, y=10),
            ObjectiveConfig(x=14, y=30),
            ObjectiveConfig(x=46, y=10),
            ObjectiveConfig(x=46, y=30),
        ],
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": baseline}
        ),
        reward_phases=[
            RewardPhaseConfig(
                name="play_it_out",
                reward_calculators=[RewardCalculatorConfig(type="vp_gain")],
                success_criteria=SuccessCriteriaConfig(type="player_ahead_on_vp"),
                terminate_on_success=False,
            )
        ],
    )


def toward_nearest_enemy(
    env: PerModelEnv, index: int, legal: np.ndarray, ladder: MoveLadder
) -> int:
    """The legal movement action that ends closest to any living enemy."""
    handler = env.player_action_handler
    model = env.wargame_models[index]
    enemies = np.array(
        [m.location for m in env.opponent_models if m.is_alive], dtype=float
    )
    if enemies.size == 0:
        return STAY_ACTION
    best, best_gap = STAY_ACTION, np.inf
    for candidate in np.flatnonzero(legal):
        action = int(candidate)
        if action == STAY_ACTION:
            continue
        end = np.asarray(model.location, dtype=float) + handler.decode_action(
            action, model_idx=index, ladder=ladder
        )
        gap = float(np.linalg.norm(enemies - end, axis=1).min())
        if gap < best_gap:
            best, best_gap = action, gap
    return best


def charging_driver(env: PerModelEnv, point: DecisionPoint) -> PerModelAction:
    """Close, charge the nearest unit, strike whatever is offered."""
    if point.kind is StepKind.close_turn:
        return PerModelAction.close_turn()
    model = int(np.flatnonzero(point.selector_mask)[0])
    phase = point.phase
    if point.kind is StepKind.open:
        row = point.declaration_mask[model]
        if phase is BattlePhase.movement:
            return PerModelAction.open(
                model, MoveDeclaration.normal if row[1] else MoveDeclaration.stationary
            )
        if phase is BattlePhase.shooting:
            return PerModelAction.open(model, ShootDeclaration.hold_fire)
        if phase is BattlePhase.charge:
            return PerModelAction.open(
                model, ChargeDeclaration.charge if row[1] else ChargeDeclaration.decline
            )
        return PerModelAction.open(model, ShortMoveDeclaration.decline)
    if point.kind is StepKind.target:
        targets = np.flatnonzero(point.target_mask[model])
        return PerModelAction.target(
            model, int(targets[0]) if targets.size else CHARGE_TARGET_DECLINE
        )
    legal = point.action_mask[model]
    if phase is BattlePhase.movement:
        return PerModelAction.act(
            model, toward_nearest_enemy(env, model, legal, MoveLadder.normal)
        )
    if phase is BattlePhase.charge:
        return PerModelAction.act(
            model, toward_nearest_enemy(env, model, legal, MoveLadder.charge)
        )
    return PerModelAction.act(model, int(np.flatnonzero(legal)[0]))


def shooting_charging_driver(env: PerModelEnv, point: DecisionPoint) -> PerModelAction:
    """`charging_driver`, except that a unit with a target shoots at it."""
    if point.kind is StepKind.open and point.phase is BattlePhase.shooting:
        model = int(np.flatnonzero(point.selector_mask)[0])
        row = point.declaration_mask[model]
        return PerModelAction.open(
            model, ShootDeclaration.shoot if row[1] else ShootDeclaration.hold_fire
        )
    if point.kind is StepKind.act and point.phase is BattlePhase.shooting:
        model = int(np.flatnonzero(point.selector_mask)[0])
        legal = np.flatnonzero(point.action_mask[model])
        shots = [int(a) for a in legal if a != STAY_ACTION]
        return PerModelAction.act(model, shots[0] if shots else STAY_ACTION)
    return charging_driver(env, point)


def _row_y(index: int, groups: list[int]) -> float:
    """Squadmates 2" apart in a column per unit; units 20" apart at the default.

    The default six land exactly where they always did (10, 12, 14 / 30, 32,
    34); more units spread evenly across the board's height.
    """
    distinct = sorted(set(groups))
    unit = distinct.index(groups[index])
    member = sum(1 for j in range(index) if groups[j] == groups[index])
    if len(distinct) <= 2:
        return 10.0 + 20.0 * unit + 2.0 * member
    return 4.0 + (32.0 / len(distinct)) * unit + 2.0 * member


__all__ = [
    "charging_driver",
    "random_legal_action",
    "shooting_charging_driver",
    "small_config",
    "toward_nearest_enemy",
]
