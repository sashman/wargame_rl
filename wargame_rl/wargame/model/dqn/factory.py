import gymnasium as gym

from wargame_rl.wargame.model.dqn.config import WargameConfig


def create_environment(
    wargame_config: WargameConfig, render_mode: str | None = "human"
) -> gym.Env:
    """Create the Wargame environment.

    Args:
        render_mode: Rendering mode ("human" for visual, None for no rendering)

    Returns:
        Configured gymnasium environment
    """
    env = gym.make(
        id=wargame_config.env_id,
        render_mode=render_mode,
        **wargame_config.env_make_params,
    )
    return env
