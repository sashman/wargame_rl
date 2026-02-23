import torch

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observation_to_flat_tensor
from wargame_rl.wargame.model.ppo.agent import Agent
from wargame_rl.wargame.model.ppo.config import PPOConfig
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_MLP, PPO_Transformer


def test_ppo_imports() -> None:
    """Test that PPO modules can be imported correctly."""
    print("Testing PPO imports...")

    # Test that classes can be imported
    assert PPO_MLP is not None
    assert PPO_Transformer is not None
    assert PPOConfig is not None

    print("PPO imports test passed!")


def test_ppo_config_creation() -> None:
    """Test that PPOConfig can be instantiated."""
    print("Testing PPO config creation...")

    config = PPOConfig()
    assert config is not None
    assert config.batch_size == 64
    assert config.lr == 0.0003
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.eps_clip == 0.2
    assert config.vf_coef == 0.5
    assert config.ent_coef == 0.01
    assert config.max_grad_norm == 0.5
    assert config.n_epochs == 10
    assert config.n_steps == 2048
    assert config.n_episodes == 10
    assert config.hidden_size == 128
    assert config.num_layers == 2
    assert config.log is True

    print("PPO config creation test passed!")


def test_ppo_config_values() -> None:
    """Test PPOConfig values are set correctly."""
    print("Testing PPO config values...")

    # Test default values
    config = PPOConfig()

    # Check that all expected parameters are present and correct
    expected_values = {
        "batch_size": 64,
        "lr": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "eps_clip": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "n_epochs": 10,
        "n_steps": 2048,
        "n_episodes": 10,
        "hidden_size": 128,
        "num_layers": 2,
        "log": True,
    }

    for key, expected_value in expected_values.items():
        actual_value = getattr(config, key)
        assert actual_value == expected_value, (
            f"Config {key} should be {expected_value}, got {actual_value}"
        )

    print("PPO config values test passed!")


def test_ppo_agent_creation(env: WargameEnv) -> None:
    """Test PPO Agent creation."""
    print("Testing PPO Agent creation...")

    # Create agent
    agent = Agent(env=env)

    # Test that it can be initialized
    assert agent is not None
    assert hasattr(agent, "env")

    print("PPO Agent creation test passed!")


def test_ppo_module_structure() -> None:
    """Test that PPO modules have the expected structure."""
    print("Testing PPO module structure...")

    # Test that we can access the classes
    assert hasattr(PPO_MLP, "__init__")
    assert hasattr(PPO_Transformer, "__init__")
    assert hasattr(PPOConfig, "__init__")

    # Test that config can be instantiated
    config = PPOConfig()
    assert hasattr(config, "batch_size")
    assert hasattr(config, "lr")
    assert hasattr(config, "gamma")

    print("PPO module structure test passed!")


def test_ppo_agent_run_episode_returns_three_values(
    env: WargameEnv, ppo_mlp_net: PPO_MLP
) -> None:
    """Run_episode returns (total_reward, steps, experiences)."""
    agent = Agent(env)
    reward, steps, experiences = agent.run_episode(
        ppo_mlp_net, epsilon=0.0, save_steps=True
    )
    assert isinstance(reward, float)
    assert isinstance(steps, int)
    assert isinstance(experiences, list)
    assert steps >= 0
    assert len(experiences) == steps


def test_ppo_agent_run_episode_with_save_steps_false_returns_empty_experiences(
    env: WargameEnv, ppo_mlp_net: PPO_MLP
) -> None:
    """When save_steps is False, experiences list is empty."""
    agent = Agent(env)
    _reward, _steps, experiences = agent.run_episode(
        ppo_mlp_net, epsilon=0.0, save_steps=False
    )
    assert experiences == []


def test_ppo_networks_get_action_deterministic_returns_same_for_same_state(
    env: WargameEnv, ppo_mlp_net: PPO_MLP
) -> None:
    """Deterministic policy returns the same action for the same state."""
    observation, _ = env.reset()
    state_tensor = observation_to_flat_tensor(
        observation, device=ppo_mlp_net.device
    ).unsqueeze(0)
    action1, _ = ppo_mlp_net.get_action(state_tensor, deterministic=True)
    action2, _ = ppo_mlp_net.get_action(state_tensor, deterministic=True)
    assert action1 == action2


def test_ppo_observation_to_flat_tensor_shape_matches_observation_size(
    env: WargameEnv,
) -> None:
    """Flat tensor from observation has length equal to observation.size."""
    observation, _ = env.reset()
    flat = observation_to_flat_tensor(observation)
    assert flat.shape == (observation.size,)
    assert flat.dtype == torch.float32


def test_ppo_lightning_train_dataloader_yields_one_batch(
    env: WargameEnv, ppo_mlp_net: PPO_MLP
) -> None:
    """Train dataloader has length 1 so training_step is called once per epoch."""
    model = PPOLightning(env=env, policy_net=ppo_mlp_net, n_epochs=1)
    loader = model.train_dataloader()
    batches = list(loader)
    assert len(batches) == 1


def test_ppo_lightning_training_step_returns_scalar_loss(
    env: WargameEnv, ppo_mlp_net: PPO_MLP
) -> None:
    """Training step returns a scalar tensor loss."""
    torch.manual_seed(42)
    model = PPOLightning(
        env=env,
        policy_net=ppo_mlp_net,
        n_epochs=2,
        log=False,
    )
    # Batch is a dummy tensor from the dataloader; training_step ignores it
    loss = model.training_step(batch=torch.tensor([0.0]), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.dtype == torch.float32


if __name__ == "__main__":
    # This is just for manual testing - pytest will run the functions above
    print("Running comprehensive PPO tests...")
    print("Note: These tests are meant to be run with pytest, not directly.")
