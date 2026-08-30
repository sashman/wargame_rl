"""A fall back that does not end legally is not made at all — env.step-driven.

`docs/rules/09-movement-phase.md` § Fall-back move: after moving, the unit must
be unengaged; and it is a move under `03-moving.md`, so the unit must end in
coherency. All-or-nothing at the unit, like the charge.

⚠ **The referee did not exist until 2026-08-29, and the on-file claim that
`back_off_to_unengaged` already enforced the rule was FALSE** — it clears only
the endpoints that MOVED, so a member that stood still (or was blocked back to
zero) stayed engaged while its free squadmates marched away. Measured with the
v12 agent as player: every zero-death coherency break on the opponent seat was
a `fell_back_this_turn` unit with 1–2 members still engaged, tearing to
16.7–20.0" chain gaps. Same defect class as the charge's missing third
condition, and — per that episode's lesson — every test here calls `env.step`.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.melee import MeleeConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


def _env(melee: bool = True) -> WargameEnv:
    skip = [BattlePhase.shooting]
    if not melee:
        skip += [BattlePhase.command, BattlePhase.charge, BattlePhase.fight]
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        number_of_objectives=1,
        melee=MeleeConfig(enabled=melee),
        base_radius=0.0,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(group_id=0) for _ in range(2)],
        opponent_models=[ModelConfig(group_id=0) for _ in range(2)],
        skip_phases=skip,
    )
    env = create_environment(config)
    env.reset(seed=3)
    return env


def _stay(env: WargameEnv) -> WargameEnvAction:
    return WargameEnvAction(actions=[0] * len(env.wargame_models))


def _to_movement(env: WargameEnv) -> None:
    while env.game_clock_state.phase is not BattlePhase.movement:
        env.step(_stay(env))


def _pin_positions(env: WargameEnv, enemy_x: float) -> None:
    """One player squad at (10,10)/(10,11); the nearer enemy `enemy_x` away.

    With `engagement_range` 1.0 and zero base radius, model 0 begins engaged
    (gap 0.5") and model 1 does not (gap ~1.12"). The second enemy is parked
    far away so it can neither engage nor block anything.
    """
    env.wargame_models[0].location = np.array([10.0, 10.0])
    env.wargame_models[1].location = np.array([10.0, 11.0])
    env.opponent_models[0].location = np.array([enemy_x, 10.0])
    env.opponent_models[1].location = np.array([50.0, 40.0])


def test_a_fall_back_that_leaves_a_member_engaged_is_reverted() -> None:
    # Arrange: model 0 engaged; order it to STAY while model 1 walks east.
    env = _env()
    try:
        _to_movement(env)
        _pin_positions(env, enemy_x=9.5)
        east = env.player_action_handler.best_action_toward(1.0, 0.0)

        # Act
        env.step(WargameEnvAction(actions=[0, east]))

        # Assert: the unit's move did not happen — the walker is back at its
        # start, and the unit did not fall back (it remained stationary), so
        # it keeps its shooting.
        assert np.allclose(env.wargame_models[1].location, [10.0, 11.0])
        assert np.allclose(env.wargame_models[0].location, [10.0, 10.0])
        assert not any(m.fell_back_this_turn for m in env.wargame_models)
    finally:
        env.close()


def test_a_fall_back_that_ends_clear_and_coherent_stands_and_costs() -> None:
    # Arrange: both members walk east, away from the enemy to the west.
    env = _env()
    try:
        _to_movement(env)
        _pin_positions(env, enemy_x=9.5)
        east = env.player_action_handler.best_action_toward(1.0, 0.0)

        # Act
        env.step(WargameEnvAction(actions=[east, east]))

        # Assert: the unit moved, ends unengaged and coherent, and pays.
        assert env.wargame_models[0].location[0] > 10.0
        assert env.wargame_models[1].location[0] > 10.0
        assert all(m.fell_back_this_turn for m in env.wargame_models)
    finally:
        env.close()


def test_melee_off_never_reaches_the_referee() -> None:
    """The gate: without melee there is no engagement, so no revert path."""
    env = _env(melee=False)
    try:
        _to_movement(env)
        _pin_positions(env, enemy_x=9.5)
        east = env.player_action_handler.best_action_toward(1.0, 0.0)

        # Act: the same split order that is reverted with melee on.
        env.step(WargameEnvAction(actions=[0, east]))

        # Assert: the walker walked — the referee only exists behind melee.
        assert env.wargame_models[1].location[0] > 10.0
    finally:
        env.close()
