from wargame_rl.wargame.envs.renders import renderer
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def create_environment(
    env_config: WargameEnvConfig, renderer: renderer.Renderer | None = None
) -> WargameEnv:
    """Create the Wargame environment.

    Returns:
        Configured gymnasium environment
    """
    env = WargameEnv(env_config, renderer)

    return env
