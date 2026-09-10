"""The PPO loop over decision steps: the per-round clock in GAE, the lockstep
rollout that continues its episodes, and an update that reproduces the
sampled log-probs and leaves closing rows out of the policy."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.tokens import Head
from wargame_rl.wargame.model.per_model import (
    PerModelPPOConfig,
    Rollout,
    SetAgent,
    SetNetwork,
    SetNetworkConfig,
    Transition,
    collect_rollout,
    compute_gae,
    evaluate_transitions,
    ppo_update,
)
from wargame_rl.wargame.model.per_model.ppo import check_trainable

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)


def _transition(
    reward: float, value: float, *, close: bool = False, done: bool = False
) -> Transition:
    return Transition(
        tokens=cast(Any, None),
        model=-1 if close else 0,
        column=-1 if close else 0,
        head=Head.none if close else Head.displacement,
        phase=None,
        value=value,
        log_prob=0.0,
        reward=reward,
        done=done,
        is_close=close,
        env_index=0,
    )


def test_gae_discounts_only_across_closing_steps() -> None:
    """Hand-computed at gamma 0.5, lambda 0.5, V = 0.5 everywhere:
    decision, decision, close, decision, close-done."""
    rollout = Rollout(
        transitions=[
            _transition(1.0, 0.5),
            _transition(0.0, 0.5),
            _transition(2.0, 0.5, close=True),
            _transition(1.0, 0.5),
            _transition(0.0, 0.5, close=True, done=True),
        ],
        n_envs=1,
        bootstrap=[123.0],
        observations=[],
        closes=2,
    )
    config = PerModelPPOConfig(gamma=0.5, gae_lambda=0.5)
    returns, advantages = compute_gae(rollout, config)
    expected = torch.tensor([2.875, 1.875, 1.875, 0.5, -0.5])
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected + 0.5)


def test_gae_bootstraps_a_mid_turn_cut_at_gamma_one() -> None:
    rollout = Rollout(
        transitions=[_transition(1.0, 0.5), _transition(0.0, 0.5)],
        n_envs=1,
        bootstrap=[2.0],
        observations=[],
        closes=0,
    )
    _, advantages = compute_gae(rollout, PerModelPPOConfig(gamma=0.1, gae_lambda=0.1))
    # t1: 0 + 2.0 - 0.5 = 1.5; t0: 1 + 0.5 - 0.5 + 1.5 = 2.5 (no discount inside a turn).
    assert torch.allclose(advantages, torch.tensor([2.5, 1.5]))


def _envs(rounds: int = 3) -> tuple[list[PerModelEnv], list[PerStepReward], list[Any]]:
    envs = [
        PerModelEnv(small_config(rounds=rounds)),
        PerModelEnv(
            small_config(
                n_models=10,
                group_ids=[i // 2 for i in range(10)],
                max_groups=5,
                rounds=rounds,
            )
        ),
    ]
    observations = [
        env.reset(seed=100 + i, options={"augment_start": True})[0]
        for i, env in enumerate(envs)
    ]
    retimers = [PerStepReward(env) for env in envs]
    for retimer in retimers:
        retimer.reset()
    return envs, retimers, observations


def _agent(env: PerModelEnv, seed: int = 0) -> SetAgent:
    torch.manual_seed(seed)
    return SetAgent(SetNetwork.from_env(env, SMALL_TRUNK))


def test_a_lockstep_rollout_over_two_sizes_runs_to_budget_and_continues() -> None:
    # Six-round episodes: the smaller env closes its turns faster, and two
    # one-round budgets must not reach the end of either episode.
    envs, retimers, observations = _envs(rounds=6)
    agent = _agent(envs[0])
    config = PerModelPPOConfig(rollout_rounds=1)
    generator = torch.Generator().manual_seed(0)

    first = collect_rollout(
        envs, agent, retimers, observations, config, generator=generator
    )
    assert first.n_envs == 2
    assert config.rollout_rounds * 2 <= first.closes < config.rollout_rounds * 2 + 2
    assert first.n_steps % 2 == 0
    assert all(t.env_index == i % 2 for i, t in enumerate(first.transitions))
    steps_after_first = [env.episode_step for env in envs]
    ids_after_first = [env.episode_id for env in envs]

    second = collect_rollout(
        envs, agent, retimers, first.observations, config, generator=generator
    )
    # Nothing terminated, and the second rollout continued where the first
    # stopped.
    assert not first.episodes and not second.episodes
    assert [env.episode_id for env in envs] == ids_after_first
    assert all(
        env.episode_step > before for env, before in zip(envs, steps_after_first)
    )
    assert sum(t.is_close for t in second.transitions) == second.closes


def test_a_terminating_env_is_reset_inline_and_reported() -> None:
    envs, retimers, observations = _envs()
    agent = _agent(envs[0])
    config = PerModelPPOConfig(rollout_rounds=6)
    rollout = collect_rollout(envs, agent, retimers, observations, config)
    assert len(rollout.episodes) >= 2
    assert {episode.env_index for episode in rollout.episodes} == {0, 1}
    assert all(episode.rounds == 3 for episode in rollout.episodes)
    assert any(t.done for t in rollout.transitions)
    assert all(env.episode_id >= 2 for env in envs)


def test_evaluation_reproduces_the_sampled_log_probs() -> None:
    envs, retimers, observations = _envs()
    agent = _agent(envs[0])
    rollout = collect_rollout(
        envs, agent, retimers, observations, PerModelPPOConfig(rollout_rounds=1)
    )
    with torch.no_grad():
        evaluated = evaluate_transitions(agent.network, rollout.transitions)
    sampled = torch.tensor([t.log_prob for t in rollout.transitions])
    assert torch.allclose(evaluated.log_probs, sampled, atol=1e-5)
    assert torch.isfinite(evaluated.head_entropy).all()
    assert torch.isfinite(evaluated.selector_entropy).all()
    closing = torch.tensor([t.is_close for t in rollout.transitions])
    assert (evaluated.log_probs[closing] == 0).all()
    assert (evaluated.head_entropy[closing] == 0).all()
    assert (evaluated.selector_entropy[closing] == 0).all()
    assert (evaluated.head_entropy[~closing] > 0).any()


def test_an_update_moves_the_weights_and_reports_finite_stats() -> None:
    envs, retimers, observations = _envs()
    agent = _agent(envs[0])
    config = PerModelPPOConfig(rollout_rounds=1, batch_size=16, n_epochs=2)
    rollout = collect_rollout(envs, agent, retimers, observations, config)
    returns, advantages = compute_gae(rollout, config)
    before = [p.detach().clone() for p in agent.network.parameters()]
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=config.lr)
    stats = ppo_update(
        agent.network,
        optimizer,
        rollout,
        returns,
        advantages,
        config,
        generator=torch.Generator().manual_seed(0),
    )
    after = list(agent.network.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert stats.n_minibatches == 2 * -(-rollout.n_steps // 16)
    for value in (
        stats.train_loss,
        stats.policy_loss,
        stats.value_loss,
        stats.entropy_loss,
        stats.clip_fraction,
        stats.approx_kl,
        stats.grad_norm,
    ):
        assert value == value  # not NaN


def test_a_network_with_dropout_is_refused() -> None:
    network = SetNetwork(
        SetNetworkConfig(embedding_size=32, n_layers=1, n_heads=4, dropout=0.1),
        n_displacements=5,
    )
    with pytest.raises(ValueError, match="dropout"):
        check_trainable(network)
