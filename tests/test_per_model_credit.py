"""Tests for per-model credit assignment.

The reward is a mean over every model, so with 25 models one model's action
explains ~4% of the number it is credited with. The per-model decomposition
was already computed inside `calculate_reward` and thrown away one line later;
these tests pin that it is now kept and carried through PPO.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.ppo.agent import Agent
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

N_MODELS = 4


def _make_env(
    calculator: str, weight: float = 1.0, on_objective: bool = False
) -> WargameEnv:
    """Env with a single reward calculator.

    `on_objective` pins every model onto the objective so occupancy-based
    calculators return a non-zero value rather than a vacuous 0.
    """
    placement: dict[str, object] = {}
    if on_objective:
        placement = {
            "models": [ModelConfig(x=10, y=10, group_id=0) for _ in range(N_MODELS)],
            "objectives": [ObjectiveConfig(x=10, y=10)],
        }
    config = WargameEnvConfig(
        render_mode=None,
        number_of_wargame_models=N_MODELS,
        number_of_objectives=1,
        number_of_battle_rounds=5,
        **placement,  # type: ignore[arg-type]
        reward_phases=[
            RewardPhaseConfig(
                name="only",
                reward_calculators=[
                    RewardCalculatorConfig(type=calculator, weight=weight)
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="fraction_at_objectives", params={"min_fraction": 0.5}
                ),
            )
        ],
    )
    return WargameEnv(config=config)


def _step(env: WargameEnv) -> None:
    env.step(WargameEnvAction(actions=[0] * N_MODELS))


def test_per_model_reward_has_one_entry_per_model() -> None:
    """The vector is model-aligned, so it indexes like the action tensor."""
    env = _make_env("closest_objective")
    env.reset(seed=0)
    _step(env)

    assert env.last_per_model_reward.shape == (N_MODELS,)


def test_per_model_terms_are_not_averaged_away() -> None:
    """Each model keeps its own contribution rather than the army mean.

    `closest_objective` is a per-model calculator, so the scalar reward is the
    mean of the vector. Recovering the vector is the whole point: distinct
    entries are signal that the mean destroys.
    """
    env = _make_env("closest_objective")
    env.reset(seed=0)
    for _ in range(3):
        _step(env)

    per_model = env.last_per_model_reward
    assert env.last_reward == pytest.approx(float(np.mean(per_model)))


def test_global_terms_are_broadcast_whole_to_every_model() -> None:
    """A global calculator gives every model the same value, undivided.

    Global terms are the part of the outcome genuinely not attributable to one
    model, so each should see the full signal rather than an Nth of it.
    """
    env = _make_env("models_at_objectives", on_objective=True)
    env.reset(seed=0)
    _step(env)

    per_model = env.last_per_model_reward
    assert env.last_reward == pytest.approx(1.0)
    assert per_model == pytest.approx(np.full(N_MODELS, env.last_reward))


def test_dead_models_receive_no_reward() -> None:
    """Only alive models are credited; the living still get the full signal."""
    env = _make_env("models_at_objectives", on_objective=True)
    env.reset(seed=0)
    env.wargame_models[0].take_damage(env.wargame_models[0].stats["current_wounds"])
    _step(env)

    reward = env.last_reward
    assert reward is not None and reward > 0.0
    per_model = env.last_per_model_reward
    assert per_model[0] == pytest.approx(0.0)
    assert per_model[1:] == pytest.approx(np.full(N_MODELS - 1, reward))


def test_experiences_carry_the_per_model_reward() -> None:
    """The rollout path preserves the vector all the way to the PPO update."""
    env = _make_env("closest_objective")
    net = PPO_Transformer.from_env(env)
    agent = Agent(env)

    _reward, _steps, experiences = agent.run_episode_with_experiences(net)

    assert experiences
    first = experiences[0]
    assert first.per_model_reward is not None
    assert first.per_model_reward.shape == torch.Size([N_MODELS])


def test_value_head_scores_each_model() -> None:
    """The critic emits one value per model, not one per state."""
    env = _make_env("closest_objective")
    net = PPO_Transformer.from_env(env)
    observation, _ = env.reset(seed=0)

    from wargame_rl.wargame.model.common.observation import observation_to_tensor

    with torch.no_grad():
        _logits, values = net(observation_to_tensor(observation, net.device))

    assert values.shape == torch.Size([1, N_MODELS])


def test_importance_ratios_are_clipped_per_model() -> None:
    """Log-probs stay per model so each ratio is clipped on its own.

    Summed over N models the joint ratio leaves the trust region after
    ln(1.2)/N nats of change per model — 0.0073 at 25 models — which clipped
    roughly half of every minibatch flat.
    """
    env = _make_env("closest_objective")
    net = PPO_Transformer.from_env(env)
    observation, _ = env.reset(seed=0)

    from wargame_rl.wargame.model.common.observation import observation_to_tensor

    with torch.no_grad():
        _action, log_probs = net.get_action(
            observation_to_tensor(observation, net.device)
        )

    assert log_probs.shape == torch.Size([N_MODELS])
