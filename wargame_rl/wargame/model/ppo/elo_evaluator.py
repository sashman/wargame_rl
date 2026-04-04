"""PPO Elo evaluation utilities against pluggable opponents."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.model.common.elo import EloRatingSystem
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.ppo.agent import Agent
from wargame_rl.wargame.model.ppo.networks import PPOModel


@dataclass(frozen=True)
class OpponentEntry:
    name: str
    policy: OpponentPolicyConfig


def episode_score_from_vp(player_vp: int, opponent_vp: int) -> float:
    """Convert terminal VP into Elo score."""
    if player_vp > opponent_vp:
        return 1.0
    if player_vp < opponent_vp:
        return 0.0
    return 0.5


def evaluate_vs_opponent(
    policy_model: PPOModel,
    env_config: WargameEnvConfig,
    opponent_policy: OpponentPolicyConfig,
    n_episodes: int,
) -> list[float]:
    """Play episodes versus one opponent and return per-episode scores."""
    cfg = env_config.model_copy(deep=True)
    cfg.opponent_policy = opponent_policy

    env = create_environment(cfg, renderer=None)
    agent = Agent(env)
    scores: list[float] = []
    was_training = policy_model.training
    policy_model.eval()

    try:
        with torch.no_grad():
            for _ in range(n_episodes):
                agent.run_episode(
                    policy_model,
                    epsilon=0.0,
                    render=False,
                    save_steps=False,
                )
                scores.append(episode_score_from_vp(env.player_vp, env.opponent_vp))
    finally:
        if was_training:
            policy_model.train()
        env.close()

    return scores


def evaluate_elo_ladder(
    policy_model: PPOModel,
    env_config: WargameEnvConfig,
    opponents: list[OpponentEntry],
    n_episodes: int,
    ratings: EloRatingSystem,
    agent_name: str = "agent_current",
) -> dict[str, float]:
    """Run Elo updates for the given opponents and return current ratings map."""
    ratings.ensure_player(agent_name)
    for opponent in opponents:
        ratings.ensure_player(opponent.name)
        scores = evaluate_vs_opponent(
            policy_model=policy_model,
            env_config=env_config,
            opponent_policy=opponent.policy,
            n_episodes=n_episodes,
        )
        for score in scores:
            ratings.update(agent_name, opponent.name, score)
    return dict(ratings.ratings)
