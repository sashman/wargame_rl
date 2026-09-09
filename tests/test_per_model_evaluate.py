"""Greedy seeded evaluation over the per-model facade is deterministic and
reads the env the way the scripted bar does."""

from __future__ import annotations

import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.model.per_model import (
    SetAgent,
    SetNetwork,
    SetNetworkConfig,
    evaluate_per_model,
)

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)


def test_two_evaluations_of_the_same_seeds_are_identical() -> None:
    env = PerModelEnv(small_config(opponent_x=22))
    torch.manual_seed(0)
    agent = SetAgent(SetNetwork.from_env(env, SMALL_TRUNK))
    retimer = PerStepReward(env)
    seeds = [500000, 500001]
    first = evaluate_per_model(env, agent, retimer, seeds)
    second = evaluate_per_model(env, agent, retimer, seeds)
    assert first.episodes == second.episodes
    assert [e.seed for e in first.episodes] == seeds
    assert first.n_episodes == 2
    # Greedy evaluation leaves the agent as it found it.
    assert agent.greedy is False


def test_the_readouts_agree_with_the_env_at_the_end() -> None:
    env = PerModelEnv(small_config(opponent_x=22))
    torch.manual_seed(1)
    agent = SetAgent(SetNetwork.from_env(env, SMALL_TRUNK))
    retimer = PerStepReward(env)
    result = evaluate_per_model(env, agent, retimer, [500002])
    episode = result.episodes[0]
    assert episode.player_vp == env.player_vp
    assert episode.opponent_vp == env.opponent_vp
    assert episode.success == (env.player_vp > env.opponent_vp)
    assert episode.reward == retimer.episode_reward
    assert 0.0 <= episode.fraction_alive <= 1.0
    assert 0 <= episode.objectives_held <= len(env.objectives)
    assert 0.0 <= episode.coherency_rate <= 1.0
    assert result.vp_margin == episode.player_vp - episode.opponent_vp
    assert result.vp_margin_se is None
