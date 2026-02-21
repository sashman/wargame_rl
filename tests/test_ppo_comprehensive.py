from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.ppo.agent import Agent
from wargame_rl.wargame.model.ppo.config import PPOConfig
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


if __name__ == "__main__":
    # This is just for manual testing - pytest will run the functions above
    print("Running comprehensive PPO tests...")
    print("Note: These tests are meant to be run with pytest, not directly.")
