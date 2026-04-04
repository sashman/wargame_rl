from __future__ import annotations

from pathlib import Path

import torch
from pydantic_yaml import to_yaml_str

from evaluate_elo import evaluate_checkpoint_elo
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.net import TransformerNetwork


def _config() -> WargameEnvConfig:
    return WargameEnvConfig(
        config_name="elo_test_cfg",
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        number_of_objectives=2,
        objective_radius_size=2,
        number_of_battle_rounds=3,
        opponent_policy=OpponentPolicyConfig(type="random"),
    )


def test_evaluate_checkpoint_elo_includes_scripted_and_snapshot(tmp_path: Path) -> None:
    cfg = _config()
    env = WargameEnv(config=cfg)
    policy_net = TransformerNetwork.policy_from_env(env)
    env.close()

    checkpoint_path = tmp_path / "current.pt"
    torch.save({"policy_state_dict": policy_net.state_dict()}, checkpoint_path)

    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    torch.save(
        {"policy_state_dict": policy_net.state_dict()},
        snapshot_dir / "snapshot_a.pt",
    )

    env_cfg_path = tmp_path / "env_config.yaml"
    env_cfg_path.write_text(to_yaml_str(cfg))

    ratings = evaluate_checkpoint_elo(
        checkpoint_path=str(checkpoint_path),
        env_config_path=str(env_cfg_path),
        episodes_per_opponent=1,
        snapshot_dir=str(snapshot_dir),
    )

    assert "agent_current" in ratings
    assert "policy:random" in ratings
    assert "policy:scripted_advance_to_objective" in ratings
    assert "snapshot:snapshot_a" in ratings
