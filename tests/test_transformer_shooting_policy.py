from __future__ import annotations

import torch

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import TransformerNetwork


def _shooting_env_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        board_width=30,
        board_height=30,
        models=[
            ModelConfig(x=5, y=5, weapons=[WeaponProfile(range=24)]),
            ModelConfig(x=10, y=10, weapons=[WeaponProfile(range=24)]),
        ],
        opponent_models=[
            ModelConfig(x=20, y=5),
            ModelConfig(x=25, y=10),
        ],
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        opponent_policy=OpponentPolicyConfig(type="random"),
        n_movement_angles=8,
        n_speed_bins=3,
    )


def _advance_to_shooting(env: WargameEnv) -> tuple:
    stay = WargameEnvAction(actions=[STAY_ACTION] * env.config.number_of_wargame_models)
    return env.step(stay)


def test_transformer_policy_dead_player_row_is_stay_only() -> None:
    env = WargameEnv(config=WargameEnvConfig())
    net = TransformerNetwork.policy_from_env(env)

    env.reset(seed=0)
    env.wargame_models[0].take_damage(env.wargame_models[0].stats["current_wounds"])
    stay = WargameEnvAction(actions=[STAY_ACTION] * env.config.number_of_wargame_models)
    obs, _, _, _, _ = env.step(stay)

    logits = net(observation_to_tensor(obs, net.device))
    assert torch.isfinite(logits[0, 0, STAY_ACTION])
    assert torch.isneginf(logits[0, 0, 1:]).all()


def test_transformer_policy_dead_opponent_shooting_column_is_neginf() -> None:
    env = WargameEnv(config=_shooting_env_config())
    net = TransformerNetwork.policy_from_env(env)

    env.reset(seed=42)
    env.opponent_models[0].take_damage(env.opponent_models[0].stats["current_wounds"])
    obs, _, _, _, _ = _advance_to_shooting(env)

    shooting_slice = env._action_handler.shooting_slice
    assert shooting_slice is not None

    tensors = observation_to_tensor(obs, net.device)
    logits = net(tensors)
    dead_target_col = shooting_slice.start
    assert not tensors[4][:, dead_target_col].any()
    assert torch.isneginf(logits[0, :, dead_target_col]).all()


def test_transformer_policy_without_shooting_keeps_shoot_head_disabled() -> None:
    env = WargameEnv(config=WargameEnvConfig(number_of_opponent_models=0))
    net = TransformerNetwork.policy_from_env(env)
    assert net.shoot_query_proj is None
    assert net.shoot_key_proj is None
