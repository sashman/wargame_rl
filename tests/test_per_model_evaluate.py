"""Batched, seeded evaluation over the per-model facade: waves of envs of
different sizes, results in seed order, and the same answer as one env at a
time."""

from __future__ import annotations

import pytest
import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.per_model import (
    PerModelEnv,
    evaluate_per_model_chooser,
    random_chooser,
    scripted_chooser,
)
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.model.per_model import (
    EvalResult,
    SetAgent,
    SetNetwork,
    SetNetworkConfig,
    evaluate_per_model,
)

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)
SEEDS = [500000, 500001, 500002]


def _agent(env: PerModelEnv, seed: int = 0) -> SetAgent:
    torch.manual_seed(seed)
    return SetAgent(SetNetwork.from_env(env, SMALL_TRUNK))


def test_a_wave_scores_the_same_as_one_env_at_a_time() -> None:
    """Batching is a throughput device, not a different measurement."""
    config = small_config(opponent_x=22)
    single = [PerModelEnv(config)]
    wave = [PerModelEnv(config) for _ in range(2)]
    agent = _agent(single[0])
    one_at_a_time = evaluate_per_model(single, agent, [PerStepReward(single[0])], SEEDS)
    batched = evaluate_per_model(wave, agent, [PerStepReward(e) for e in wave], SEEDS)
    assert batched == one_at_a_time
    assert batched.n_episodes == 3
    assert batched.decisions_per_episode is not None
    assert batched.episode_rewards is not None
    assert batched.success_per_episode is not None
    # Greedy evaluation leaves the agent as it found it.
    assert agent.greedy is False


def test_envs_of_different_sizes_share_one_wave() -> None:
    """The network is size-independent, so a wave may mix armies; every step
    is legal (the env raises otherwise) and every seed lands in order."""
    small = PerModelEnv(small_config(opponent_x=22))
    large = PerModelEnv(small_config(opponent_x=22, n_models=9, max_groups=3))
    agent = _agent(small)
    result = evaluate_per_model(
        [small, large], agent, [PerStepReward(small), PerStepReward(large)], SEEDS
    )
    assert result.n_episodes == 3
    assert len(result.vp_margin_per_episode) == 3


def test_the_readouts_agree_with_the_env_at_the_end() -> None:
    env = PerModelEnv(small_config(opponent_x=22))
    agent = _agent(env, seed=1)
    retimer = PerStepReward(env)
    result = evaluate_per_model([env], agent, [retimer], [500002])
    assert result.player_vp == env.player_vp
    assert result.opponent_vp == env.opponent_vp
    assert result.success_per_episode == ((env.player_vp > env.opponent_vp),)
    assert result.episode_rewards == (retimer.episode_reward,)
    assert 0.0 <= result.final_fraction_alive <= 1.0
    assert 0 <= result.objectives_held <= len(env.objectives)
    assert result.coherency_rate is not None and 0.0 <= result.coherency_rate <= 1.0
    assert result.vp_margin == env.player_vp - env.opponent_vp
    assert result.vp_margin_se is None
    # What this facade does not measure stays unmeasured.
    assert result.exposure_rate is None and result.firepower_ratio is None


def test_without_retimers_the_per_model_readouts_are_unmeasured() -> None:
    env = PerModelEnv(small_config(opponent_x=22))
    result = evaluate_per_model_chooser(random_chooser(0), [env], SEEDS[:2], "random")
    assert isinstance(result, EvalResult)
    assert result.episode_rewards is None
    assert result.success_per_episode is None
    assert result.mean_reward is None and result.success_rate is None
    assert result.decisions_per_episode is not None
    assert all(d > 0 for d in result.decisions_per_episode)


def test_a_scripted_chooser_is_seated_before_reset_and_read_per_call() -> None:
    envs = [PerModelEnv(small_config(opponent_x=22)) for _ in range(2)]
    choose = scripted_chooser(build_baseline_policy("squad_march_take"), envs)
    assert all(env.player_seat.adapter is not None for env in envs)
    first = evaluate_per_model_chooser(choose, envs, SEEDS, "take")
    second = evaluate_per_model_chooser(choose, envs, SEEDS, "take")
    assert first == second


def test_mismatched_lengths_are_refused() -> None:
    env = PerModelEnv(small_config(opponent_x=22))
    with pytest.raises(ValueError, match="combat_seeds"):
        evaluate_per_model_chooser(
            random_chooser(0), [env], SEEDS, "random", combat_seeds=[1]
        )
    with pytest.raises(ValueError, match="retimers"):
        evaluate_per_model_chooser(
            random_chooser(0), [env], SEEDS, "random", retimers=[]
        )
