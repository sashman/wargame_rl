"""The per-model PPO loop (issue #286): the per-round discount, the rollout
budget in rounds, and — the load-bearing one — that `evaluate_transitions`
under unchanged weights reproduces exactly the log-probs the agent sampled,
which is what makes the importance ratio start at 1."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.per_model.observation import build_token_observation
from wargame_rl.wargame.envs.per_model.types import PerModelAction
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.per_model import (
    SetAgent,
    SetNetwork,
    SetNetworkConfig,
    StepDecision,
)
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    Transition,
    collect_rollout,
    compute_gae,
    evaluate_transitions,
    ppo_update,
)

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)


def _shooting_config() -> WargameEnvConfig:
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


def _fake_transition(reward: float, value: float, is_close: bool) -> Transition:
    decision = StepDecision(
        action=PerModelAction(model_index=None),
        log_prob=0.0,
        value=value,
        entropy=0.0,
    )
    return Transition(
        observation=None,  # type: ignore[arg-type]  # GAE never reads it
        decision=decision,
        reward=reward,
        done=False,
        is_close=is_close,
    )


def test_gae_discounts_only_across_the_closing_step() -> None:
    """Model steps within a turn are the same instant: with zero values, the
    return at the turn's first step is the whole turn's reward undiscounted,
    plus the NEXT turn's discounted by one gamma — never by step count."""
    config = PerModelPPOConfig(gamma=0.5, gae_lambda=1.0)
    transitions = [
        _fake_transition(1.0, 0.0, is_close=False),
        _fake_transition(1.0, 0.0, is_close=False),
        _fake_transition(10.0, 0.0, is_close=True),  # round 1 closes
        _fake_transition(1.0, 0.0, is_close=False),
        _fake_transition(10.0, 0.0, is_close=True),  # round 2 closes
    ]
    advantages, returns = compute_gae(transitions, 0.0, config)
    # From the front: 1 + 1 + 10 (same instant) + 0.5 * (1 + 10) (one round away).
    assert returns[0] == pytest.approx(12.0 + 0.5 * 11.0)
    # Within the round nothing decays: the three steps differ only by the
    # rewards already banked between them.
    assert returns[1] == pytest.approx(returns[0] - 1.0)
    assert returns[2] == pytest.approx(returns[1] - 1.0)
    # After the close, one gamma has been applied to the future.
    assert returns[3] == pytest.approx(1.0 + 10.0)


def test_the_rollout_budget_is_counted_in_rounds() -> None:
    torch.manual_seed(0)
    env = PerModelEnv(_shooting_config(), build_info=False)
    agent = SetAgent(SetNetwork.from_env(env, SMALL_TRUNK))
    transitions, _bootstrap = collect_rollout(
        env, agent, n_rounds=4, generator=torch.Generator().manual_seed(1)
    )
    assert sum(1 for t in transitions if t.is_close) == 4
    # 3-round episodes: the budget crossed an episode boundary and kept going.
    assert any(t.done for t in transitions)


def test_evaluation_reproduces_the_sampled_log_probs() -> None:
    """Under unchanged weights the recomputed joint log-prob equals the one
    sampled at act time, for every step kind — so PPO's first ratio is 1."""
    torch.manual_seed(0)
    env = PerModelEnv(_shooting_config(), build_info=False)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    agent = SetAgent(network)
    transitions, _ = collect_rollout(
        env, agent, n_rounds=3, generator=torch.Generator().manual_seed(2)
    )
    network.eval()
    with torch.no_grad():
        evaluated = evaluate_transitions(network, transitions)
    sampled = np.array([t.decision.log_prob for t in transitions])
    np.testing.assert_allclose(
        evaluated.log_probs.numpy(), sampled, rtol=1e-4, atol=1e-5
    )
    closes = np.array([t.is_close for t in transitions])
    assert not evaluated.has_decision.numpy()[closes].any()
    assert evaluated.has_decision.numpy()[~closes].all()


def test_one_update_runs_and_moves_the_weights() -> None:
    torch.manual_seed(0)
    env = PerModelEnv(_shooting_config(), build_info=False)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    agent = SetAgent(network)
    config = PerModelPPOConfig(minibatch_size=16, n_update_epochs=1)
    transitions, bootstrap = collect_rollout(
        env, agent, n_rounds=3, generator=torch.Generator().manual_seed(3)
    )
    before = [p.detach().clone() for p in network.parameters()]
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    losses = ppo_update(
        network,
        optimizer,
        transitions,
        bootstrap,
        config,
        generator=torch.Generator().manual_seed(4),
    )
    assert all(np.isfinite(value) for value in losses.values())
    moved = any(
        not torch.equal(a, b) for a, b in zip(before, network.parameters(), strict=True)
    )
    assert moved, "an update that changes nothing trained nothing"


def test_token_observations_survive_the_buffer_round_trip() -> None:
    """The stored observation re-collates identically later — the buffer keeps
    numpy, not views into a live env."""
    env = PerModelEnv(_shooting_config(), build_info=False)
    observation, _ = env.reset(seed=5)
    tokens = build_token_observation(env, observation)
    snapshot = tokens.player_tokens.copy()
    env.step(
        PerModelAction(
            model_index=int(np.flatnonzero(observation.selection_mask)[0]), action=5
        )
    )
    np.testing.assert_array_equal(tokens.player_tokens, snapshot)
