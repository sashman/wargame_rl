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

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.env_components.observation_builder import build_observation
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
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
        # ⚠ A stepped charge phase needs a stepped COMMAND phase: the unit's
        # leader declares the charge there, and that declaration is what binds
        # the whole unit into making it.
        skip_phases=(
            [BattlePhase.fight]
            if melee
            else [BattlePhase.command, BattlePhase.charge, BattlePhase.fight]
        ),
    )
    env = create_environment(config)
    env.reset(seed=3)
    return env


def _stay(env: WargameEnv) -> WargameEnvAction:
    """Every model declines, in whatever phase the clock is in."""
    return WargameEnvAction(actions=[0] * len(env.wargame_models))


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
        # ⚠ And the DECLARATION is made in the command phase, two phases before
        # this arrangement exists — the movement phase sits between them and
        # would march the squad off these spots. So the declaration is set
        # here, and the leader's own decision is covered end to end by
        # `test_the_leader_declares_and_the_whole_unit_charges`.
        model.declared_charge = True
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


def test_BOTH_melee_configs_seat_an_opponent_THAT_CAN_CHARGE() -> None:
    """⚠ The blocker was closed on the bar and left open on the opponent.

    Both configs seated `squad_march_take`, whose `charge_when_it_lands` is
    False, so the arm would have trained in the UNILATERAL cell of a mechanic
    whose whole measured value is the asymmetry between the seats. Measured
    n=100 per cell, paired, argmax, vp_margin to the player:

        player walks   / opponent walks    +14.05
        player charges / opponent walks    +38.00
        player walks   / opponent charges  -51.55
        player charges / opponent charges   +0.95

    Reading the top row alone says melee is worth +24; the mechanic is worth
    about zero. This is the Advance failure — a bar that could not use a core
    rule — moved one seat over, and an audit panel found it inside the commit
    that claimed to have closed it.
    """
    # Arrange / Act
    seated = {}
    for path in (
        "configs/experiments/25v25_maps_melee.yaml",
        "configs/experiments/25v25_maps_melee_dark.yaml",
    ):
        opponent = load_env_config(path).opponent_policy
        assert opponent is not None
        seated[path] = build_baseline_policy((opponent.params or {})["baseline"])

    # Assert
    for path, policy in seated.items():
        assert getattr(policy, "charge_when_it_lands", False), (
            f"{path} seats an opponent that cannot charge, so the arm would "
            "train in the unilateral cell"
        )


def test_the_leader_declares_and_the_WHOLE_UNIT_charges() -> None:
    """The feature, end to end: one leader's choice binds every squadmate.

    ⚠ **This is what the declaration is for.** Before it, a charge was declared
    implicitly by picking a rung, so "charge or not" was decided independently
    by every model. Measured on three behaviour clones of a rigid charging
    teacher: the teacher declares for 100% of a unit's members every time, the
    clones for 54-62%, and the WHOLE unit only 23-35% of the time — because
    `P(any charge)` of 0.59 is spread over ~48 rungs at ~0.012 each while STAY
    is ONE action at 0.41, and an argmax over a spread loses. A charge then
    fails not because the rungs disagree but because half the unit stands still,
    the unit stretches, and the referee reverts the lot.
    """
    # Arrange: two two-model squads, melee on, command stepped.
    env = _env(melee=True)
    handler = env.player_action_handler
    declare = handler.move_type_action("charge")
    assert declare is not None, "the melee config must carry the declaration"
    while env.game_clock_state.phase is not BattlePhase.command:
        env.step(_stay(env))
    for index, model in enumerate(env.wargame_models):
        model.charge_roll = 6.0
    leader = 0
    unit = [i for i, m in enumerate(env.wargame_models) if m.group_id == 0]
    assert leader in unit and len(unit) > 1, "need a unit with a real follower"

    # Act: ONLY the leader acts.
    actions = [0] * len(env.wargame_models)
    actions[leader] = declare
    env.step(WargameEnvAction(actions=actions))

    # Assert: every member of the leader's unit is bound, and no other unit is.
    try:
        assert all(env.wargame_models[i].declared_charge for i in unit)
        assert not any(
            m.declared_charge for i, m in enumerate(env.wargame_models) if i not in unit
        )
    finally:
        env.close()


def test_a_declared_unit_may_NOT_stand_still() -> None:
    """The binding half — permitting a charge is not the same as compelling it.

    A declaration that only *allowed* the rungs would leave every model free to
    stay, which is exactly the state the measurement found: half the unit
    charges, the unit stretches, and the referee reverts the lot. STAY survives
    only for a model the 2D6 cannot carry into contact, because a mask must
    never empty a row.
    """
    # Arrange: one squad in reach, declared; the other far away and not.
    env = _env(melee=True)
    try:
        while env.game_clock_state.phase is not BattlePhase.charge:
            env.step(_stay(env))
        for index, model in enumerate(env.wargame_models):
            near = model.group_id == 0
            model.location = np.array(
                [10.0 if near else 45.0, 10.0 + index], dtype=model.location.dtype
            )
            model.charge_roll = 6.0
            model.declared_charge = bool(near)
        for index, model in enumerate(env.opponent_models):
            model.location = np.array([13.0, 10.0 + index], dtype=model.location.dtype)

        # Act
        # Through `build_observation`, which is what `step` calls -- reaching for
        # a private env method would test a path no policy takes.
        observation = build_observation(
            env, action_registry=env.player_action_handler.registry
        )
        mask = np.asarray(observation.action_mask, dtype=bool)

        # Assert
        declared = [i for i, m in enumerate(env.wargame_models) if m.declared_charge]
        others = [i for i, m in enumerate(env.wargame_models) if not m.declared_charge]
        assert declared and others
        for i in declared:
            assert not mask[i, 0], "a declared model kept the option to stand still"
            assert mask[i].any(), "the mask must never empty a row"
        for i in others:
            assert mask[i, 0], "an undeclared model must still be able to stay"
    finally:
        env.close()
