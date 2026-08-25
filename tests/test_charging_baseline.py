"""The bar can charge — the blocker that stopped melee being measurable at all.

⚠ `BaselinePolicy.select_action` returned STAY for every phase outside command,
movement and shooting, so **no scripted baseline and no scripted opponent could
charge**. An agent trained with melee on would have been scored against a policy
physically incapable of the mechanic under test — verbatim the Advance failure,
where the bar could not advance and the arm measured the bar.

These pin the two halves that make the instrument usable: the hook is a no-op
for every policy that does not override it, and the charging policy actually
reaches contact through `env.step`.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

ENGAGEMENT = 1.0


def _env(melee: bool, opponent: str = "hold_deployment") -> WargameEnv:
    """Two squads of two a side, melee optional, an opponent that holds still."""
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": opponent}
        ),
        models=[
            ModelConfig(group_id=i // 2, melee_weapons=[MeleeWeaponProfile()])
            for i in range(4)
        ],
        opponent_models=[
            ModelConfig(group_id=i // 2, melee_weapons=[MeleeWeaponProfile()])
            for i in range(4)
        ],
        melee=MeleeConfig(enabled=melee),
        engagement_range=ENGAGEMENT,
        base_radius=0.0,
        skip_phases=(
            [BattlePhase.command, BattlePhase.fight]
            if melee
            else [BattlePhase.command, BattlePhase.charge, BattlePhase.fight]
        ),
    )
    env = create_environment(config)
    env.reset(seed=3)
    return env


def _engaged(env: WargameEnv) -> int:
    """How many of the player's living models are in contact with a living foe."""
    mine = [m for m in env.wargame_models if m.is_alive]
    foes = [m for m in env.opponent_models if m.is_alive]
    if not mine or not foes:
        return 0
    return int(
        np.count_nonzero(
            engaged_with_any(
                np.array([m.location for m in mine], dtype=float),
                np.array([m.location for m in foes], dtype=float),
                np.ones(len(foes), dtype=bool),
                np.ones(len(mine), dtype=bool),
                engagement_range=ENGAGEMENT,
            )
        )
    )


@pytest.mark.parametrize(
    "name",
    ["squad_march_take", "squad_march_shoot", "squad_march_deny", "hold_deployment"],
)
def test_the_hook_is_a_no_op_for_every_policy_that_does_not_override_it(
    name: str,
) -> None:
    """Adding the branch must not change a single measured baseline number."""
    # Arrange
    env = _env(melee=True)
    try:
        while env.game_clock_state.phase is not BattlePhase.charge:
            env.step(build_baseline_policy(name).select_action(env.wargame_models, env))

        # Act
        action = build_baseline_policy(name).select_action(env.wargame_models, env)

        # Assert
        assert all(a == 0 for a in action.actions), (
            "a policy that does not override select_charge must decline"
        )
    finally:
        env.close()


def test_the_charging_policy_reaches_contact_through_env_step() -> None:
    """The instrument, end to end. Nothing else in this file proves it works."""
    # Arrange. ⚠ Placed AT the charge phase, not before it: the movement phase
    # runs first and marches everybody at an objective, so an arrangement made
    # at reset is gone by the time the charge is declared.
    env = _env(melee=True)
    policy = build_baseline_policy("squad_march_take_charge")
    while env.game_clock_state.phase is not BattlePhase.charge:
        env.step(policy.select_action(env.wargame_models, env))
    # ⚠ The two enemy UNITS have to be far apart. Packed a base apart they are
    # all inside each other's engagement range, so any charge touches both and
    # the referee reverts it for clipping a second unit — the rule working, but
    # it reads here as the policy failing.
    for index, model in enumerate(env.wargame_models):
        model.location = np.array([10.0, 10.0 + index], dtype=model.location.dtype)
    for index, model in enumerate(env.opponent_models):
        spot = (13.0, 10.0 + index) if index < 2 else (40.0, 40.0 + index)
        model.location = np.array(spot, dtype=model.location.dtype)
    for model in env.wargame_models:
        # The 2D6 is rolled at the start of the side's turn, before this
        # rearrangement; fix it so the test is about the policy, not the dice.
        model.charge_roll = 6.0
    assert _engaged(env) == 0, "the arrangement must start unengaged"
    before = [np.array(m.location, copy=True) for m in env.wargame_models]

    # Act
    env.step(policy.select_action(env.wargame_models, env))

    # Assert. ⚠ Displacement, NOT `charged_this_turn` and not engagement after
    # the step. With `fight` in `skip_phases` the fight resolves on the boundary
    # inside this same step: it clears the flag, and it can kill the very model
    # that was reached. A charge that STOOD is one the referee did not put back
    # where it started, and that is durable.
    assert any(
        not np.array_equal(before[i], m.location)
        for i, m in enumerate(env.wargame_models)
    ), "the charging bar's declaration was reverted, so it never reached contact"


def test_with_melee_off_the_charging_policy_IS_squad_march_take() -> None:
    """Gate 4 of the pre-registration, and what keeps the dark control valid.

    The dark control steps the charge phase with `melee.enabled` false, so a bar
    whose charging leaked into that config would make the paired comparison
    measure the bar rather than melee.
    """
    # Arrange
    trajectories = []
    for name in ("squad_march_take", "squad_march_take_charge"):
        env = _env(melee=False)
        policy = build_baseline_policy(name)
        seen = []
        try:
            for _ in range(12):
                action = policy.select_action(env.wargame_models, env)
                seen.append(list(action.actions))
                env.step(action)
                seen.append([m.location.tolist() for m in env.wargame_models])
        finally:
            env.close()
        trajectories.append(seen)

    # Assert
    assert trajectories[0] == trajectories[1]


def test_the_opponent_seat_can_charge_too() -> None:
    """No 2x2 without it — and a bar asymmetric between seats is not a bar.

    ⚠ This project has measured a 24.6 vp seat asymmetry from shooting alone, so
    a mechanic wired on one seat only is a rules difference between them.
    """
    # Arrange
    env = _env(melee=True, opponent="squad_march_take_charge")
    walker = build_baseline_policy("squad_march_take")
    for index, model in enumerate(env.wargame_models):
        model.location = np.array([10.0, 10.0 + index], dtype=model.location.dtype)
    for index, model in enumerate(env.opponent_models):
        model.location = np.array([13.0, 10.0 + index], dtype=model.location.dtype)

    # Act
    moved = False
    for _ in range(8):
        before = [np.array(m.location, copy=True) for m in env.opponent_models]
        phase = env.game_clock_state.phase
        env.step(walker.select_action(env.wargame_models, env))
        if phase is BattlePhase.charge:
            moved = any(
                not np.array_equal(before[i], m.location)
                for i, m in enumerate(env.opponent_models)
            )
            if moved:
                break

    # Assert
    assert moved, "the opponent seat never moved in the charge phase"
