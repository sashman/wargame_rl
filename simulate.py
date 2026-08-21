#!/usr/bin/env python3
"""
Simulation script for running trained agents in the Wargame environment.

Usage:
    python simulate.py --checkpoint path/to/checkpoint.ckpt \
        [--episodes 10] [--render]
"""

import logging
import os
from pathlib import Path

import typer
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.renders.human import QuitRequested
from wargame_rl.wargame.envs.renders.v2 import build_renderer, threat_options
from wargame_rl.wargame.envs.renders.v2.control import THREAT_SMOOTHING, THREAT_SPACING
from wargame_rl.wargame.envs.state import EventLogExporter, JsonMatchCodec
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.argmax_agent import ArgmaxAgent
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.net import TransformerNetwork

app = typer.Typer(pretty_exceptions_enable=False)


def get_env_config(env_config_path: str | None, render: bool) -> WargameEnvConfig:
    if env_config_path is None:
        return WargameEnvConfig(render_mode="human" if render else None)

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Environment config file not found: {env_config_path}")

    with open(env_config_path) as f:
        env_config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, f.read())  # pyright: ignore[reportUndefinedVariable]

    # Override render_mode with CLI argument
    env_config.render_mode = "human" if render else None

    return env_config


def simulate(
    checkpoint_path: str,
    num_episodes: int = 10,
    render: bool = True,
    env_config_path: str | None = None,
    record_events: bool = False,
    seed: int | None = None,
    renderer_name: str = "legacy",
    backend: str = "pillow",
    theme: str = "default",
    show_threat_range: bool = False,
    show_engagement_range: bool = False,
    threat_grid: float = THREAT_SPACING,
    threat_smoothing: int = THREAT_SMOOTHING,
) -> None:
    """Run simulation with trained agent.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        record_events: Whether to record the last episode as a JSON event log
        seed: Seeds the run, making it reproducible. Every episode after the
            first continues the same generator stream, so one seed pins the
            whole run and each recorded episode still carries its own
            provenance. Without it the layout comes from process entropy and
            the run cannot be recreated.
        renderer_name: ``legacy`` (HumanRender) or ``v2`` (the new renderer)
        backend: v2 drawing backend (``pillow``, ``pygame`` or ``pygame_aa``)
        theme: v2 theme name (``default`` or ``tabletop``)
        show_threat_range: draw each side's range ∩ line-of-sight footprint
        show_engagement_range: draw each side's engagement zone
        threat_grid: sampling grid for the threat sweep, in inches
        threat_smoothing: Chaikin passes on the threat outline
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")

    env_config = get_env_config(env_config_path, render)
    renderer = (
        build_renderer(
            renderer_name,
            "interactive",
            backend=backend,
            theme=theme,
            threat_options=threat_options(
                show_threat_range, show_engagement_range, threat_grid, threat_smoothing
            ),
        )
        if render
        else None
    )

    event_exporter: EventLogExporter | None = None
    if record_events:
        event_exporter = EventLogExporter(anchor_interval=10)

    env = create_environment(
        env_config=env_config,
        renderer=renderer,
        state_exporters=[event_exporter] if event_exporter else None,
    )
    env.driver_label = checkpoint_path
    if seed is not None:
        # Once, before the loop. The agent resets per episode without a seed,
        # which continues this stream rather than restarting it -- so the run is
        # reproducible as a whole and each episode is individually recoverable
        # from the generator state its provenance records.
        env.reset(seed=seed)
    logging.info(f"Action space: {env.action_space}")
    logging.info(f"Observation space: {env.observation_space}")
    logging.info(f"Running {num_episodes} episodes...")

    agent = ArgmaxAgent(env)
    logging.info(f"Agent created: {agent}")

    try:
        policy_net = TransformerNetwork.from_checkpoint(env, checkpoint_path)
        logging.info(f"Loaded model from checkpoint: {checkpoint_path} successfully!")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            logging.error(f"Model size mismatch error: {e}")
            logging.error(
                "This checkpoint was trained with a different environment configuration."
            )
            logging.error("The current environment has:")
            obs, _ = env.reset()
            logging.error(f"  - Observation size: {obs.size}")
            logging.error(f"  - Number of wargame models: {obs.n_wargame_models}")
            logging.error(f"  - Number of actions: {env._action_handler.n_actions}")
            logging.error(
                "Please train a new model with the current environment configuration or use a compatible checkpoint."
            )
            raise
        else:
            raise

    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        try:
            reward, steps = agent.run_episode(
                policy_net, epsilon=0.0, render=render, save_steps=False
            )
        except QuitRequested:
            logging.info("Application stopped by user (Esc)")
            break
        episode_rewards.append(reward)
        episode_steps.append(steps)

        logging.info(
            f"Episode {episode + 1:3d}: Reward = {reward:8.3f}, Steps = {steps:3d}"
        )

        # Add small delay for human rendering
        if render:
            import time

            time.sleep(0.3)

    # Calculate and display statistics
    if not episode_rewards:
        logging.info("No episodes completed.")
        env.close()
        return
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = sum(episode_steps) / len(episode_steps)
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)

    logging.info("\n" + "=" * 50)
    logging.info("SIMULATION RESULTS:")
    logging.info(f"Average reward: {avg_reward:.3f}")
    logging.info(f"Average steps:  {avg_steps:.1f}")
    logging.info(f"Max reward:     {max_reward:.3f}")
    logging.info(f"Min reward:     {min_reward:.3f}")
    logging.info(
        f"Success rate:   {sum(1 for r in episode_rewards if r > -0.5) / len(episode_rewards) * 100:.1f}%"
    )
    logging.info("=" * 50)

    if event_exporter and len(event_exporter.log) > 0:
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        out_path = recordings_dir / "simulate_events.jsonl"
        codec = JsonMatchCodec()
        out_path.write_bytes(codec.encode(event_exporter.log))
        logging.info(
            f"Event log written to {out_path} ({len(event_exporter.log)} events)"
        )

    env.close()


def get_latest_checkpoint() -> str:
    if not os.path.exists("checkpoints"):
        raise FileNotFoundError(
            "Checkpoints directory not found, please run `just train` first to create it."
        )

    # Recursively find all .ckpt files in subdirectories
    checkpoint_files = []
    for root, dirs, files in os.walk("checkpoints"):
        for file in files:
            if file.endswith(".ckpt"):
                full_path = os.path.join(root, file)
                checkpoint_files.append(full_path)

    if len(checkpoint_files) == 0:
        raise FileNotFoundError("No checkpoint files found in checkpoints directory.")

    # Sort by modification time and return the latest
    latest_checkpoint = sorted(checkpoint_files, key=lambda x: os.path.getmtime(x))[-1]

    return latest_checkpoint


def get_env_config_path_for_checkpoint(checkpoint_path: str) -> str:
    """Get the basepath of the checkpoint and return the env config from the checkpoint directory."""
    basepath = os.path.dirname(checkpoint_path)
    env_config_path = os.path.join(basepath, "env_config.yaml")
    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Environment config file not found: {env_config_path}")

    return env_config_path


@app.command()
def main(
    checkpoint_path: str | None = typer.Option(
        None,
        help="Path to the trained model checkpoint, defaults to the latest checkpoint.",
    ),
    num_episodes: int = typer.Option(10, help="Number of episodes to run"),
    render: bool = typer.Option(True, help="Whether to render the environment"),
    env_config_path: str | None = typer.Option(
        None,
        help="Path to the environment config file, defaults to env_config.yaml from checkpoint directory.",
    ),
    seed: int | None = typer.Option(
        None,
        help="Seed the run so it can be reproduced. Without it the layout comes "
        "from process entropy and a recording cannot be recreated exactly.",
    ),
    record_events: bool = typer.Option(
        False,
        help="Record the last episode as a JSON event log (written to recordings/)",
    ),
    renderer: str = typer.Option(
        "legacy", help="Renderer to use: 'legacy' (HumanRender) or 'v2'"
    ),
    backend: str = typer.Option(
        "pillow", help="v2 drawing backend: 'pillow', 'pygame' or 'pygame_aa'"
    ),
    theme: str = typer.Option("default", help="v2 theme: 'default' or 'tabletop'"),
    show_threat_range: bool = typer.Option(
        False,
        "--threat-range",
        help="Outline the ground each side can shoot: weapon range intersected with line of sight.",
    ),
    show_engagement_range: bool = typer.Option(
        False,
        "--engagement-range",
        help="Shade each side's engagement range — inside it, a model may not shoot at all.",
    ),
    threat_grid: float = typer.Option(
        THREAT_SPACING,
        help="Sampling grid for the threat sweep, in inches. Coarser is cheaper and blurs small cover pockets.",
    ),
    threat_smoothing: int = typer.Option(
        THREAT_SMOOTHING,
        help="Chaikin smoothing passes on the threat outline. 0 draws the sampled shape exactly.",
    ),
) -> None:
    # Handle dynamic defaults inside the function
    # Typer fills a default only when *it* invokes the command. Called directly
    # -- as `tests/test_simulate.py` does -- an option nobody passed is still an
    # `OptionInfo`, and this is the first one that reaches the env, where
    # gymnasium rejects it as a seed.
    if not isinstance(seed, int):
        seed = None

    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()

    if env_config_path is None:
        env_config_path = get_env_config_path_for_checkpoint(checkpoint_path)

    simulate(
        checkpoint_path,
        num_episodes,
        render,
        env_config_path,
        record_events=record_events,
        seed=seed,
        renderer_name=renderer,
        backend=backend,
        theme=theme,
        show_threat_range=show_threat_range,
        show_engagement_range=show_engagement_range,
        threat_grid=threat_grid,
        threat_smoothing=threat_smoothing,
    )


if __name__ == "__main__":
    app()
