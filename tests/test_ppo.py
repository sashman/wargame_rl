import pytest
import torch
from pytorch_lightning import Trainer

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.model.ppo.agent import Agent
from wargame_rl.wargame.model.ppo.config import PPOConfig
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.networks import PPOModel
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer
from wargame_rl.wargame.types import Experience

# ---------------------------------------------------------------------------
# PPOModel — forward
# ---------------------------------------------------------------------------


def test_ppo_model_forward_single_observation_returns_correct_shapes(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    # Arrange
    obs, _ = env.reset()
    state_tensors = observation_to_tensor(obs, ppo_net.device)
    n_models = env.config.number_of_wargame_models
    n_actions = env._action_handler.n_actions

    # Act
    logits, values = ppo_net(state_tensors)

    # Assert
    assert logits.shape == (1, n_models, n_actions)
    assert values.shape == (1,)


def test_ppo_model_forward_batch_returns_correct_shapes(
    env: WargameEnv, ppo_net: PPO_Transformer, experiences: list[Experience]
) -> None:
    # Arrange
    batch_size = 256
    state_tensors = observations_to_tensor_batch(
        [exp.state for exp in experiences[:batch_size]], device=ppo_net.device
    )
    n_models = env.config.number_of_wargame_models
    n_actions = env._action_handler.n_actions

    # Act
    logits, values = ppo_net(state_tensors)

    # Assert
    assert logits.shape == (batch_size, n_models, n_actions)
    assert values.shape == (batch_size,)


def test_ppo_model_forward_shared_transformer_returns_correct_shapes(
    env: WargameEnv, experiences: list[Experience]
) -> None:
    # Arrange
    batch_size = 256
    shared_net = PPO_Transformer.from_env(
        env=env, config=PPOConfig(share_transformer=True)
    )
    state_tensors = observations_to_tensor_batch(
        [exp.state for exp in experiences[:batch_size]], device=shared_net.device
    )
    n_models = env.config.number_of_wargame_models
    n_actions = env._action_handler.n_actions

    # Act
    logits, values = shared_net(state_tensors)

    # Assert
    assert logits.shape == (batch_size, n_models, n_actions)
    assert values.shape == (batch_size,)


def test_ppo_from_env_with_shared_transformer_shares_backbone_modules(
    env: WargameEnv,
) -> None:
    # Arrange
    shared_net = PPO_Transformer.from_env(
        env=env, config=PPOConfig(share_transformer=True)
    )

    # Assert
    assert isinstance(shared_net.policy_network, TransformerNetwork)
    assert isinstance(shared_net.value_network, TransformerNetwork)
    assert (
        shared_net.policy_network.game_embedding
        is shared_net.value_network.game_embedding
    )
    assert (
        shared_net.policy_network.objective_embedding
        is shared_net.value_network.objective_embedding
    )
    assert (
        shared_net.policy_network.wargame_model_embedding
        is shared_net.value_network.wargame_model_embedding
    )
    assert shared_net.policy_network.transformer is shared_net.value_network.transformer


def test_ppo_shared_transformer_reduces_total_parameter_count(env: WargameEnv) -> None:
    # Arrange
    separate = PPO_Transformer.from_env(
        env=env, config=PPOConfig(share_transformer=False)
    )
    shared = PPO_Transformer.from_env(env=env, config=PPOConfig(share_transformer=True))

    # Act
    separate_params = sum(p.numel() for p in separate.parameters())
    shared_params = sum(p.numel() for p in shared.parameters())

    # Assert
    assert shared_params < separate_params


def test_ppo_model_raises_when_policy_and_value_networks_are_swapped(
    env: WargameEnv,
) -> None:
    policy_net = TransformerNetwork.from_env(env, is_policy=True)
    value_net = TransformerNetwork.from_env(env, is_policy=False)

    with pytest.raises(ValueError):
        PPOModel(policy_network=value_net, value_network=policy_net)


# ---------------------------------------------------------------------------
# PPOModel — get_action
# ---------------------------------------------------------------------------


def test_ppo_get_action_returns_one_valid_action_per_model(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    # Arrange
    obs, _ = env.reset()
    state_tensors = observation_to_tensor(obs, ppo_net.device)
    n_models = env.config.number_of_wargame_models
    n_actions = env._action_handler.n_actions

    # Act
    env_action, log_prob = ppo_net.get_action(state_tensors)

    # Assert
    assert isinstance(env_action, WargameEnvAction)
    assert len(env_action.actions) == n_models
    assert all(0 <= a < n_actions for a in env_action.actions)
    assert log_prob.shape == torch.Size([])  # scalar


def test_ppo_get_action_deterministic_always_returns_same_action(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    # Arrange
    obs, _ = env.reset()
    state_tensors = observation_to_tensor(obs, ppo_net.device)

    # Act
    action1, _ = ppo_net.get_action(state_tensors, deterministic=True)
    action2, _ = ppo_net.get_action(state_tensors, deterministic=True)

    # Assert
    assert action1.actions == action2.actions


# ---------------------------------------------------------------------------
# PPOModel — evaluate_actions
# ---------------------------------------------------------------------------


def test_ppo_evaluate_actions_returns_correct_shapes(
    env: WargameEnv, ppo_net: PPO_Transformer, experiences: list[Experience]
) -> None:
    # Arrange
    batch_size = 256
    state_tensors = observations_to_tensor_batch(
        [exp.state for exp in experiences[:batch_size]], device=ppo_net.device
    )
    actions = torch.tensor(
        [exp.action.actions for exp in experiences[:batch_size]],
        dtype=torch.long,
        device=ppo_net.device,
    )
    n_models = env.config.number_of_wargame_models
    n_actions = env._action_handler.n_actions

    # Act
    logits, log_probs, entropy = ppo_net.evaluate_actions(state_tensors, actions)

    # Assert
    assert logits.shape == (batch_size, n_models, n_actions)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def test_ppo_agent_run_episode_returns_correct_types(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    # Arrange
    agent = Agent(env)

    # Act
    total_reward, steps, collected_experiences = agent.run_episode_with_experiences(
        ppo_net
    )

    # Assert
    assert isinstance(total_reward, float)
    assert isinstance(steps, int)
    assert steps > 0
    assert len(collected_experiences) > 0


def test_ppo_agent_experiences_contain_scalar_log_prob(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    # Arrange
    agent = Agent(env)

    # Act
    _, _, collected_experiences = agent.run_episode_with_experiences(ppo_net)
    first = collected_experiences[0]

    # Assert
    assert isinstance(first, Experience)
    assert first.log_prob is not None
    assert first.log_prob.shape == torch.Size([])  # scalar


def test_ppo_agent_run_episode_without_saving_steps_returns_empty_experiences(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    # Arrange
    agent = Agent(env)

    # Act
    total_reward, steps, collected_experiences = agent.run_episode_with_experiences(
        ppo_net, save_steps=False
    )

    # Assert
    assert isinstance(total_reward, float)
    assert steps > 0
    assert collected_experiences == []


# ---------------------------------------------------------------------------
# PPOLightning — compute_returns
# ---------------------------------------------------------------------------


def test_compute_returns_propagates_future_reward_backwards(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    """Returns at steps before the reward should be smaller, decaying with distance."""
    # Arrange
    model = PPOLightning(env=env, ppo_model=ppo_net, log=False)
    rewards = torch.tensor([0.0, 0.0, 1.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    values = torch.zeros(3)

    # Act
    returns = model.compute_returns(rewards, dones, values)

    # Assert: reward propagates back and decays with distance
    assert returns[2] > returns[1] > returns[0]


def test_compute_returns_terminal_state_blocks_reward_propagation(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    """A done=1 at step t should prevent rewards at t+1 from flowing back to t or earlier."""
    # Arrange
    model = PPOLightning(env=env, ppo_model=ppo_net, log=False)
    # Episode ends at t=1; the reward at t=2 belongs to a new episode
    rewards = torch.tensor([0.0, 0.0, 1.0])
    dones = torch.tensor([0.0, 1.0, 0.0])
    values = torch.zeros(3)

    # Act
    returns = model.compute_returns(rewards, dones, values)

    # Assert: the terminal at t=1 blocks the future reward from reaching t=0 and t=1
    assert returns[0].item() == pytest.approx(0.0, abs=1e-5)
    assert returns[1].item() == pytest.approx(0.0, abs=1e-5)
    assert returns[2].item() == pytest.approx(1.0, abs=1e-5)


def test_compute_returns_terminal_state_blocks_reward_propagation_per_env(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    """done[t, env]=1 should block future reward propagation for that env."""
    # Arrange
    model = PPOLightning(env=env, ppo_model=ppo_net, log=False)
    # rewards only at final step
    rewards = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    # Env0 terminates at t=1; Env1 continues.
    dones = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    values = torch.zeros_like(rewards)

    # Act
    returns = model.compute_returns(rewards, dones, values)

    # Assert: env0 is fully blocked by dones at t=1
    assert returns.shape == rewards.shape
    assert returns[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert returns[1, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert returns[2, 0].item() == pytest.approx(1.0, abs=1e-5)

    # Assert: env1 should receive some propagated value from the final reward
    assert returns[0, 1].item() > 0.0
    assert returns[2, 1].item() == pytest.approx(1.0, abs=1e-5)


def test_auto_detect_num_rollout_envs_uses_divisor_of_n_steps(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    """When `num_rollout_envs <= 0`, selection should never break rollout modulo."""
    model = PPOLightning(
        env=env,
        ppo_model=ppo_net,
        log=False,
        n_steps=60,
        num_rollout_envs=0,
    )
    assert model.num_rollout_envs >= 1
    assert 60 % model.num_rollout_envs == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_ppo_training_runs_without_error(
    env: WargameEnv, ppo_net: PPO_Transformer
) -> None:
    model = PPOLightning(env=env, ppo_model=ppo_net, log=False)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=2,
        val_check_interval=1,
        logger=None,
    )

    trainer.fit(model)
