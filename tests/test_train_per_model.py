"""The per-model driver, called as a function: it trains for a few rounds on
the small fixture, writes what a run directory must hold, and its checkpoint
plays again; the configs it refuses, it refuses by name."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic_yaml import to_yaml_str

from tests.per_model_seats import small_config
from train_per_model import check_cadence, refuse_curriculum, train
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.model.per_model import SetAgent, load_checkpoint


@pytest.fixture
def small_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "small.yaml"
    path.write_text(to_yaml_str(small_config(opponent_x=22)))
    return path


def test_a_short_run_writes_a_run_directory_and_a_playable_checkpoint(
    small_yaml: Path, tmp_path: Path
) -> None:
    run_dir = train(
        env_config_path=str(small_yaml),
        rounds=4,
        rollout_rounds=1,
        num_rollout_envs=2,
        eval_every_rounds=2,
        checkpoint_every_rounds=2,
        n_eval_episodes=3,
        eval_wave_size=2,
        n_layers=2,
        embedding_size=32,
        seed=1,
        no_wandb=True,
        checkpoint_root=str(tmp_path / "ckpt"),
    )
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "pm-00000004.pt").exists()
    assert (run_dir / "env_config.yaml").read_text() == small_yaml.read_text()
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["driver"]["seed"] == 1
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[0]["rounds"] == 0 and "eval/baseline_random_vp_margin" in rows[0]
    assert "eval/baseline_squad_march_objectives_held" in rows[0]
    assert provenance["ppo"]["num_rollout_envs"] == 2
    updates = [row for row in rows if "loss/train_loss" in row]
    assert [row["rounds"] for row in updates] == [2, 4]
    assert all("eval/vp_margin" in row for row in updates)
    # The facade's unit is the decision; the phase facade's `mean_episode_steps`
    # key would read a decision count as a phase count.
    assert all("mean_episode_decisions" in row for row in updates)
    assert all("mean_episode_steps" not in row for row in updates)
    assert all("eval/coherency_rate" in row for row in updates)
    assert all(row["train/closes"] >= 2 for row in updates)

    loaded = load_checkpoint(run_dir / "last.pt")
    assert loaded.rounds == 4 and loaded.seed == 1
    env = PerModelEnv(small_config(opponent_x=22))
    agent = SetAgent(loaded.network, greedy=True)
    observation, _ = env.reset(seed=3)
    done = False
    with torch.no_grad():
        while not done:
            observation, _, done, _, _ = env.step(agent.act(env, observation).action)
    assert env.current_turn == env.max_turns


def test_a_curriculum_config_is_refused_by_name() -> None:
    from wargame_rl.wargame.model.common.cli import get_env_config

    config = get_env_config("configs/dev/4v4_two_phases.yaml", None)
    with pytest.raises(ValueError, match="single reward phase"):
        refuse_curriculum(config)


def test_a_budget_below_one_rollout_is_refused(
    small_yaml: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="below one rollout"):
        train(
            env_config_path=str(small_yaml),
            rounds=1,
            rollout_rounds=1,
            num_rollout_envs=2,
            eval_every_rounds=2,
            checkpoint_every_rounds=2,
            n_eval_episodes=1,
            n_layers=1,
            embedding_size=32,
            no_wandb=True,
            checkpoint_root=str(tmp_path / "ckpt"),
        )


def test_a_cadence_off_the_rollout_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="multiple of the rollout"):
        check_cadence(6, 4, "eval_every_rounds")
    with pytest.raises(ValueError, match="multiple of the rollout"):
        check_cadence(2, 4, "checkpoint_every_rounds")
    check_cadence(8, 4, "eval_every_rounds")
