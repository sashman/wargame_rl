import gymnasium as gym

from wargame_rl.wargame.envs.env_types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def create_environment(render_mode: str | None = "human") -> gym.Env:
    """Create the Wargame environment.

    Args:
        render_mode: Rendering mode ("human" for visual, None for no rendering)

    Returns:
        Configured gymnasium environment
    """
    wargame_config = WargameEnvConfig(render_mode=render_mode)
    env = WargameEnv(wargame_config)
    env.reset()

    return env
