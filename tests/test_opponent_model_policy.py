from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wargame_rl.wargame.envs.opponent.model_policy import ModelPolicy
from wargame_rl.wargame.envs.opponent.perspective import build_opponent_observation
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.net import TransformerNetwork


def _env_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        number_of_objectives=2,
        objective_radius_size=2,
        number_of_battle_rounds=4,
        opponent_policy=OpponentPolicyConfig(type="random"),
    )


def test_build_opponent_observation_swaps_sides() -> None:
    env = WargameEnv(config=_env_config())
    env.reset(seed=42)
    env._battle.add_player_vp(11)
    env._battle.add_opponent_vp(7)

    obs = build_opponent_observation(env)

    assert obs.n_wargame_models == env.config.number_of_opponent_models
    assert obs.n_opponent_models == env.config.number_of_wargame_models
    assert obs.player_vp == env.opponent_vp
    assert obs.opponent_vp == env.player_vp
    assert obs.player_vp_delta == env.opponent_vp_delta


def test_model_policy_loads_checkpoint_and_respects_mask(tmp_path: Path) -> None:
    env = WargameEnv(config=_env_config())
    env.reset(seed=42)

    policy_net = TransformerNetwork.policy_from_env(env)
    checkpoint_path = str(tmp_path / "ppo_policy.pt")
    torch.save({"policy_state_dict": policy_net.state_dict()}, checkpoint_path)

    model_policy = ModelPolicy(env=env, checkpoint_path=checkpoint_path)
    mask = np.zeros(
        (
            env.config.number_of_opponent_models,
            env._opponent_action_handler.n_actions,
        ),
        dtype=bool,
    )
    mask[:, 0] = True

    action = model_policy.select_action(env.opponent_models, env, action_mask=mask)
    assert len(action.actions) == env.config.number_of_opponent_models
    assert all(a == 0 for a in action.actions)
