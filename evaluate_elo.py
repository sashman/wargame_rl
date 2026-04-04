#!/usr/bin/env python3
"""Evaluate PPO checkpoints with an Elo ladder against opponent policies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import typer
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.model.common.elo import EloRatingSystem
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.policy_checkpoint import load_ppo_policy_state_dict
from wargame_rl.wargame.model.ppo.elo_evaluator import (
    OpponentEntry,
    evaluate_elo_ladder,
)
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

app = typer.Typer(pretty_exceptions_enable=False)


def load_env_config(env_config_path: str) -> WargameEnvConfig:
    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Environment config file not found: {env_config_path}")
    with open(env_config_path) as f:
        parsed = parse_yaml_raw_as(WargameEnvConfig, f.read())  # pyright: ignore[reportUndefinedVariable]
    return cast(WargameEnvConfig, parsed)


def default_env_config_path_from_checkpoint(checkpoint_path: str) -> str:
    return str(Path(checkpoint_path).resolve().parent / "env_config.yaml")


def build_opponent_ladder(
    snapshot_dir: str | None,
    include_scripted: bool = True,
) -> list[OpponentEntry]:
    ladder: list[OpponentEntry] = []
    if include_scripted:
        for policy_type in ("random", "scripted_advance_to_objective"):
            ladder.append(
                OpponentEntry(
                    name=f"policy:{policy_type}",
                    policy=OpponentPolicyConfig(type=policy_type),
                )
            )

    if snapshot_dir is None:
        return ladder
    path = Path(snapshot_dir)
    if not path.exists():
        return ladder

    for snapshot in sorted(path.glob("*.pt")):
        ladder.append(
            OpponentEntry(
                name=f"snapshot:{snapshot.stem}",
                policy=OpponentPolicyConfig(
                    type="model",
                    params={
                        "checkpoint_path": str(snapshot),
                        "deterministic": True,
                    },
                ),
            )
        )
    return ladder


def evaluate_checkpoint_elo(
    checkpoint_path: str,
    env_config_path: str,
    episodes_per_opponent: int,
    snapshot_dir: str | None = None,
) -> dict[str, float]:
    env_config = load_env_config(env_config_path)
    if env_config.number_of_opponent_models <= 0:
        raise ValueError(
            "Elo evaluation requires opponent models; set number_of_opponent_models > 0"
        )

    env = create_environment(env_config, renderer=None)
    try:
        ppo_model = PPO_Transformer.from_env(env)
        policy_state = load_ppo_policy_state_dict(checkpoint_path)
        ppo_model.policy_network.load_state_dict(policy_state)
    finally:
        env.close()

    ladder = build_opponent_ladder(snapshot_dir=snapshot_dir, include_scripted=True)
    if not ladder:
        raise ValueError("No opponents found for Elo ladder.")

    ratings = evaluate_elo_ladder(
        policy_model=ppo_model,
        env_config=env_config,
        opponents=ladder,
        n_episodes=episodes_per_opponent,
        ratings=EloRatingSystem(initial_rating=1000.0, k_factor=32.0),
        agent_name="agent_current",
    )
    return ratings


@app.command()
def main(
    checkpoint_path: str = typer.Option(..., help="Path to PPO checkpoint or snapshot"),
    env_config_path: str | None = typer.Option(
        None,
        help="Path to env config. Defaults to env_config.yaml next to checkpoint.",
    ),
    episodes_per_opponent: int = typer.Option(10, help="Episodes per ladder opponent"),
    snapshot_dir: str | None = typer.Option(
        None,
        help="Optional directory containing snapshot *.pt opponents",
    ),
    output_json_path: str | None = typer.Option(
        None, help="Optional path to write Elo ratings JSON"
    ),
) -> None:
    resolved_env_config = (
        env_config_path
        if env_config_path is not None
        else default_env_config_path_from_checkpoint(checkpoint_path)
    )
    ratings = evaluate_checkpoint_elo(
        checkpoint_path=checkpoint_path,
        env_config_path=resolved_env_config,
        episodes_per_opponent=episodes_per_opponent,
        snapshot_dir=snapshot_dir,
    )

    for name, rating in sorted(ratings.items(), key=lambda kv: kv[1], reverse=True):
        typer.echo(f"{name:50s} {rating:8.2f}")

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump({"ratings": ratings}, f, indent=2)


if __name__ == "__main__":
    app()
