from wargame_rl.wargame.envs.env_types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def create_environment(env_config: WargameEnvConfig) -> WargameEnv:
    """Create the Wargame environment.

    Returns:
        Configured gymnasium environment
    """
    env = WargameEnv(env_config)
    env.reset()

    return env
