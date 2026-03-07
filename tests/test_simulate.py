"""Tests for simulate.py (simulate-latest flow)."""

import os
import sys
from pathlib import Path

import pytest
import torch
from pydantic_yaml import to_yaml_str

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.dqn.config import NetworkType
from wargame_rl.wargame.model.net import TransformerNetwork

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
import simulate as sim_module  # noqa: E402


def test_get_latest_checkpoint_raises_when_no_checkpoints_dir(tmp_path: Path) -> None:
    """get_latest_checkpoint raises when checkpoints/ does not exist (e.g. before first train)."""
    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Checkpoints directory not found"):
            sim_module.get_latest_checkpoint()
    finally:
        os.chdir(orig_cwd)


def test_simulate_latest_discovers_checkpoint_and_runs_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() with no args (just simulate-latest) finds latest checkpoint and runs one episode."""
    from wargame_rl.wargame.model.common import auto_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    auto_device.cache_clear()

    run_dir = tmp_path / "checkpoints" / "run1"
    run_dir.mkdir(parents=True)
    cfg = WargameEnvConfig(
        render_mode=None,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=3,
        number_of_battle_rounds=5,
    )
    env = create_environment(cfg, renderer=None)
    net = TransformerNetwork.policy_from_env(env)
    lightning_state = {
        f"policy_net._orig_mod.{k}": v.clone() for k, v in net.state_dict().items()
    }
    torch.save({"state_dict": lightning_state}, run_dir / "dqn-epoch0.ckpt")
    (run_dir / "env_config.yaml").write_text(to_yaml_str(cfg))
    env.close()

    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        sim_module.main(
            checkpoint_path=None,
            num_episodes=1,
            render=False,
            env_config_path=None,
            network_type=NetworkType.TRANSFORMER,
        )
    finally:
        os.chdir(orig_cwd)
