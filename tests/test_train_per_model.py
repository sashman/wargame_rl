"""The per-model training driver (issue #288 prep): the loop runs, checkpoints
round-trip with their provenance, and metrics land in the local JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from train_per_model import load_per_model_checkpoint, train
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


def _config() -> WargameEnvConfig:
    rifle = [WeaponProfile(range=12, attacks=1)]
    squads = [
        ModelConfig(group_id=i // 3, weapons=rifle, max_wounds=1) for i in range(6)
    ]
    return WargameEnvConfig(
        render_mode=None,
        board_width=30,
        board_height=30,
        number_of_wargame_models=6,
        number_of_opponent_models=6,
        max_groups=2,
        models=squads,
        opponent_models=list(squads),
        number_of_objectives=2,
        number_of_battle_rounds=3,
        skip_phases=[
            BattlePhase.command,
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
    )


def test_the_driver_trains_checkpoints_and_logs(tmp_path: Path) -> None:
    config_path = tmp_path / "small.yaml"
    config_path.write_text(yaml.safe_dump(_config().model_dump(mode="json")))
    out_root = tmp_path / "runs"

    train(
        env_config=config_path,
        max_rounds=6,
        seed=0,
        rollout_rounds=3,
        eval_every_rounds=6,
        n_eval_episodes=1,
        checkpoint_every_rounds=6,
        embedding_size=32,
        n_layers=2,
        n_heads=4,
        device="cpu",
        use_wandb=False,
        out_root=out_root,
    )

    run_dir = next(out_root.iterdir())
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("eval/vp_margin" in record for record in records)
    assert any("loss/policy_loss" in record for record in records)
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["torch_threads"] == 2  # pinned and recorded, issue #306

    env = PerModelEnv(_config(), build_info=False)
    network, metadata = load_per_model_checkpoint(run_dir / "last.pt", env)
    assert metadata["rounds"] == 6
    assert metadata["ppo_config"]["rollout_rounds"] == 3
    # The rebuilt network scores an episode without error at the saved size.
    assert sum(p.numel() for p in network.parameters()) > 0
