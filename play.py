"""Watch a scripted policy play, in a window, with no checkpoint required.

`simulate.py` needs trained weights; this does not. It drives the player with a
scripted baseline so the HUD has real numbers moving through it — a way to look
at the renderer, try the keys, and sanity-check a config by eye before spending
GPU hours on it.

    just play                                   # golden 25v25, squad_march_shoot
    just play configs/dev/tiny.yaml random      # a small board, random actions
    just play configs/golden/25v25_curriculum.yaml squad_march tabletop

Keys are listed in the window itself — press [Tab].
"""

from __future__ import annotations

import typer
from loguru import logger
from pydantic_yaml import parse_yaml_file_as

from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.renders.human import QuitRequested
from wargame_rl.wargame.envs.renders.v2 import build_renderer
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

app = typer.Typer(add_completion=False)


@app.command()
def play(
    env_config_path: str = typer.Argument(
        "configs/golden/25v25_shooting_opponent.yaml",
        help="Environment config to play.",
    ),
    policy: str = typer.Argument(
        "squad_march_shoot", help="Scripted baseline driving the player."
    ),
    theme: str = typer.Argument("default", help="Renderer theme: default | tabletop."),
    episodes: int = typer.Option(0, help="Episodes to play; 0 plays until you quit."),
    seed: int = typer.Option(700000, help="Seed of the first episode."),
    fps: int = typer.Option(4, help="Frames per second."),
    backend: str = typer.Option("pillow", help="Drawing backend."),
) -> None:
    """Play episodes in a window until Esc (or `episodes` have finished)."""
    available = sorted(get_registry())
    if policy not in available:
        raise typer.BadParameter(f"unknown policy {policy!r}; try one of {available}")

    config = parse_yaml_file_as(WargameEnvConfig, env_config_path)
    renderer = build_renderer("v2", "interactive", backend=backend, theme=theme)
    env = WargameEnv(config=config, renderer=renderer)
    # The window's frame rate, which the interactive presenter reads off the
    # view's metadata in `setup` — so it has to be set before that call.
    env.metadata = {**WargameEnv.metadata, "render_fps": fps}
    # The context slot stamps whatever produced the frames; here, this run.
    renderer.run_label = f"{policy} · {env_config_path.split('/')[-1]}"  # type: ignore[attr-defined]
    scripted = build_baseline_policy(policy)

    logger.info(f"Playing {env_config_path} with {policy}. [Tab] lists the keys.")
    episode = 0
    try:
        while episodes == 0 or episode < episodes:
            observation, _info = env.reset(seed=seed + episode)
            renderer.setup(env)
            done = False
            while not done:
                action = scripted.select_action(
                    env.player_models, env, action_mask=observation.action_mask
                )
                observation, _reward, terminated, truncated, _info = env.step(action)
                env.render()
                done = terminated or truncated
            logger.info(
                f"episode {episode}: VP {env.player_vp}-{env.opponent_vp}, "
                f"reward {env.episode_reward:.2f}"
            )
            episode += 1
    except (QuitRequested, KeyboardInterrupt):
        logger.info("Closing.")
    finally:
        env.close()


if __name__ == "__main__":
    app()
