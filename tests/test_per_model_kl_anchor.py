"""The per-model KL anchor (#332): off by default it changes nothing, on it
holds the policy nearer its starting weights, and its controller follows the
phase facade's rule.

Driven on the small two-units-a-side scenario with a tiny trunk, one rollout,
one update, on the CPU.
"""

from __future__ import annotations

import copy

import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.model.per_model.agent import SetAgent
from wargame_rl.wargame.model.per_model.net import SetNetwork, SetNetworkConfig
from wargame_rl.wargame.model.per_model.ppo import (
    KL_COEF_MAX,
    KL_COEF_MIN,
    PerModelPPOConfig,
    adapt_kl_coef,
    collect_rollout,
    compute_gae,
    evaluate_transitions,
    ppo_update,
)


def test_the_controller_follows_the_phase_facades_rule() -> None:
    assert adapt_kl_coef(1.0, 0.5, 0.0) == 1.0  # no target: fixed
    assert adapt_kl_coef(1.0, 0.1, 1.0) == 0.5  # closer than target / 1.5
    assert adapt_kl_coef(1.0, 2.0, 1.0) == 2.0  # further than 1.5 x target
    assert adapt_kl_coef(1.0, 1.0, 1.0) == 1.0  # inside the band
    assert adapt_kl_coef(KL_COEF_MIN, 0.0, 1.0) == KL_COEF_MIN
    assert adapt_kl_coef(KL_COEF_MAX, 9.0, 1.0) == KL_COEF_MAX


def _rollout(seed: int):  # type: ignore[no-untyped-def]
    torch.manual_seed(seed)
    env = PerModelEnv(small_config(rounds=2))
    network = SetNetwork.from_env(env, SetNetworkConfig(n_layers=1, embedding_size=32))
    config = PerModelPPOConfig(
        rollout_rounds=2, num_rollout_envs=1, batch_size=32, n_epochs=2
    )
    agent = SetAgent(network)
    retimer = PerStepReward(env)
    observation, _ = env.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    rollout = collect_rollout(
        [env], agent, [retimer], [observation], config, generator=generator
    )
    returns, advantages = compute_gae(rollout, config)
    return network, rollout, returns, advantages, config


def _update(network, rollout, returns, advantages, config, **kwargs):  # type: ignore[no-untyped-def]
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    return ppo_update(
        network,
        optimizer,
        rollout,
        returns,
        advantages,
        config,
        generator=torch.Generator().manual_seed(1),
        **kwargs,
    )


def _drift(network: SetNetwork, reference: SetNetwork, rollout) -> float:  # type: ignore[no-untyped-def]
    with torch.no_grad():
        ours = evaluate_transitions(network, rollout.transitions)
        theirs = evaluate_transitions(reference, rollout.transitions)
        log_rho = (theirs.log_probs - ours.log_probs)[ours.has_policy]
        return float(((log_rho.exp() - 1.0) - log_rho).mean().item())


def test_off_by_default_the_anchor_changes_nothing() -> None:
    # Arrange: two identical networks, the same rollout, the same update seed.
    network, rollout, returns, advantages, config = _rollout(3)
    reference = copy.deepcopy(network)
    plain, anchored = copy.deepcopy(network), copy.deepcopy(network)
    # Act: one with no reference, one with a reference at coefficient 0.
    stats_plain = _update(plain, rollout, returns, advantages, config)
    stats_anchored = _update(
        anchored,
        rollout,
        returns,
        advantages,
        config,
        reference=reference,
        kl_ref_coef=0.0,
    )
    # Assert: bit-identical weights, and no drift term reported.
    for a, b in zip(plain.state_dict().values(), anchored.state_dict().values()):
        assert torch.equal(a, b)
    assert stats_plain.kl_ref == 0.0 and stats_anchored.kl_ref == 0.0
    assert stats_anchored.kl_ref_coef == 0.0


def test_the_anchor_holds_the_policy_nearer_its_start() -> None:
    # Arrange
    network, rollout, returns, advantages, config = _rollout(4)
    reference = copy.deepcopy(network)
    reference.eval()
    plain, anchored = copy.deepcopy(network), copy.deepcopy(network)
    # Act: the same update with and without a heavy anchor.
    _update(plain, rollout, returns, advantages, config)
    stats = _update(
        anchored,
        rollout,
        returns,
        advantages,
        config,
        reference=reference,
        kl_ref_coef=50.0,
    )
    # Assert: the anchored network drifted less from the start, and said so.
    assert _drift(anchored, reference, rollout) < _drift(plain, reference, rollout)
    assert stats.kl_ref_coef == 50.0
    assert stats.kl_ref >= 0.0
