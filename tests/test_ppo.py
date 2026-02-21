#!/usr/bin/env python3
"""Test script to verify PPO implementation."""

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


def test_ppo_config() -> None:
    """Test that PPOConfig can be instantiated."""
    print("Testing PPO config...")

    config = PPOConfig()
    assert config is not None

    print("PPO config test passed!")


if __name__ == "__main__":
    test_ppo_imports()
    test_ppo_config()
    print("All PPO tests passed!")
