"""`melee.observe_engaged` — the shooting shield becomes visible.

The mechanic's entire measured value on the shipped profile is the shield
(+62.5 vp with the target gate, −4.0 without), and a linear probe on trained
latents read AUC 0.75 against a 0.95 kill line — the network was not
representing engagement. This column is a pure function of current positions:
never stale, populated genuinely on both sides, and absent (bit-identical
observation) unless the flag is on.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.env_components.observation_builder import build_observation
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


def _env(observe: bool) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[
            ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
            for _ in range(2)
        ],
        opponent_models=[
            ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
            for _ in range(2)
        ],
        melee=MeleeConfig(enabled=True, observe_engaged=observe),
        engagement_range=1.0,
        base_radius=0.0,
        skip_phases=[],
    )
    env = create_environment(config)
    env.reset(seed=7)
    return env


def test_an_engaged_pair_reads_ONE_on_both_sides() -> None:
    """Both seats see the same fact, live, whatever the turn."""
    # Arrange — m0 within 1.0 of enemy 0; m1 and enemy 1 far apart.
    env = _env(observe=True)
    try:
        env.wargame_models[0].location = np.array([10.0, 10.0])
        env.wargame_models[1].location = np.array([10.0, 11.5])
        env.opponent_models[0].location = np.array([10.5, 10.0])
        env.opponent_models[1].location = np.array([30.0, 30.0])

        # Act
        obs = build_observation(env)

        # Assert
        assert obs.wargame_models[0].engaged == 1.0
        assert obs.wargame_models[1].engaged == 0.0
        assert obs.opponent_models[0].engaged == 1.0
        assert obs.opponent_models[1].engaged == 0.0
    finally:
        env.close()


def test_a_dead_enemy_engages_NOBODY() -> None:
    """The corpse-suppression rule, carried into the observation."""
    # Arrange
    env = _env(observe=True)
    try:
        env.wargame_models[0].location = np.array([10.0, 10.0])
        enemy = env.opponent_models[0]
        enemy.location = np.array([10.5, 10.0])
        enemy.stats["current_wounds"] = 0

        # Act
        obs = build_observation(env)

        # Assert
        assert obs.wargame_models[0].engaged == 0.0
    finally:
        env.close()


def test_the_column_is_ABSENT_when_the_flag_is_off() -> None:
    """Off is bit-identical: no field, no width change, nothing to regenerate."""
    # Arrange / Act
    env = _env(observe=False)
    try:
        obs = build_observation(env)

        # Assert
        assert obs.wargame_models[0].engaged is None
        assert obs.opponent_models[0].engaged is None
    finally:
        env.close()


def test_the_flag_widens_both_token_types_by_exactly_one() -> None:
    # Arrange / Act
    off = _env(observe=False)
    on = _env(observe=True)
    try:
        obs_off = build_observation(off)
        obs_on = build_observation(on)

        # Assert
        assert obs_on.wargame_models[0].size == obs_off.wargame_models[0].size + 1
        assert obs_on.opponent_models[0].size == obs_off.opponent_models[0].size + 1
    finally:
        off.close()
        on.close()
