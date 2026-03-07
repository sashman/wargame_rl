#!/usr/bin/env python3
"""Simple test to verify PPO modules can be imported."""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports() -> bool:
    """Test that all PPO modules can be imported."""
    try:
        from wargame_rl.wargame.model.ppo.agent import Agent
        from wargame_rl.wargame.model.ppo.config import PPOConfig
        from wargame_rl.wargame.model.ppo.lightning import PPOLightning
        from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

        print("All PPO modules imported successfully!")
        print("PPO_Transformer:", PPO_Transformer)
        print("PPOConfig:", PPOConfig)
        print("PPOLightning:", PPOLightning)
        print("Agent:", Agent)

        # Test that we can create config
        config = PPOConfig()
        print("Config created successfully:", config)

        print("All tests passed!")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
