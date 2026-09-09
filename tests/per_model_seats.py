"""Test-side drivers for the per-model facade.

`random_legal_action` draws a uniformly random LEGAL decision from a decision
point -- the fuzz seat. It exists here rather than in the package because it
is a way of exercising the env, not a way of playing the game; the play and
record recipes of the follow-up PR reuse it from here.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.sequencing.activation import CHARGE_TARGET_DECLINE
from wargame_rl.wargame.envs.per_model import DecisionPoint, PerModelAction, StepKind
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


def random_legal_action(
    point: DecisionPoint, rng: np.random.Generator
) -> PerModelAction:
    """A uniformly random legal decision at `point`."""
    if point.kind is StepKind.close_turn:
        return PerModelAction.close_turn()
    candidates = np.flatnonzero(point.selector_mask)
    if candidates.size == 0:
        raise AssertionError(f"{point.kind.value} point offers no selectable model")
    model = int(rng.choice(candidates))
    if point.kind is StepKind.open:
        values = np.flatnonzero(point.declaration_mask[model])
        return PerModelAction.open(model, int(rng.choice(values)))
    if point.kind is StepKind.act:
        values = np.flatnonzero(point.action_mask[model])
        return PerModelAction.act(model, int(rng.choice(values)))
    choices = [int(v) for v in np.flatnonzero(point.target_mask[model])] + [
        CHARGE_TARGET_DECLINE
    ]
    return PerModelAction.target(model, int(rng.choice(choices)))


def small_config(
    *,
    melee: bool = False,
    skip_phases: list[BattlePhase] | None = None,
    baseline: str = "squad_march_take",
    player_x: int = 14,
    opponent_x: int = 56,
    rounds: int = 3,
    turn_order: TurnOrder = TurnOrder.player,
) -> WargameEnvConfig:
    """Two units of three a side, four objectives, a 60x40 board.

    The geometry `test_scripted_baseline_opponent` uses, so a mirrored and an
    un-mirrored seat are distinguishable; `player_x`/`opponent_x` move the
    armies without moving the objectives, so a melee test can start the two
    within charge range while a movement test keeps them apart.
    """
    weapons = [WeaponProfile(range=12, attacks=2)]
    melee_weapons = [MeleeWeaponProfile()] if melee else []
    return WargameEnvConfig(
        render_mode=None,
        board_width=60,
        board_height=40,
        number_of_wargame_models=6,
        number_of_opponent_models=6,
        number_of_objectives=4,
        objective_radius_size=3,
        number_of_battle_rounds=rounds,
        max_groups=2,
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
                y=10 + 20 * (i // 3) + (i % 3) * 2,
                group_id=i // 3,
                weapons=weapons,
                melee_weapons=melee_weapons,
            )
            for i in range(6)
        ],
        opponent_models=[
            ModelConfig(
                x=opponent_x,
                y=10 + 20 * (i // 3) + (i % 3) * 2,
                group_id=i // 3,
                weapons=weapons,
                melee_weapons=melee_weapons,
            )
            for i in range(6)
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
