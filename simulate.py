#!/usr/bin/env python3
"""
Simulation script for running trained DQN agents in the Wargame environment.

Usage:
    python simulate.py --checkpoint path/to/checkpoint.ckpt \
        [--episodes 10] [--render]
"""

import logging
import os

import typer
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.renders.human import HumanRender
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.factory import create_environment

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
) -> None:
    """Run simulation with trained agent.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")

    env_config = get_env_config(env_config_path, render)
    renderer = HumanRender()
    env = create_environment(env_config=env_config, renderer=renderer)
    logging.info(f"Action space: {env.action_space}")
    logging.info(f"Observation space: {env.observation_space}")
    logging.info(f"Running {num_episodes} episodes...")

    agent = Agent(env)
    logging.info(f"Agent created: {agent}")

    try:
        policy_net = DQN.from_checkpoint(env, checkpoint_path)
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
            from wargame_rl.wargame.envs.wargame import MovementPhaseActions

            logging.error(f"  - Number of actions: {len(MovementPhaseActions)}")
            logging.error(
                "Please train a new model with the current environment configuration or use a compatible checkpoint."
            )
            raise
        else:
            raise

    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        reward, steps = agent.run_episode(
            policy_net, epsilon=0.0, render=render, save_steps=False
        )
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
            if file.startswith("dqn-") and file.endswith(".ckpt"):
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
    checkpoint_path: str = typer.Option(
        None,
        help="Path to the trained model checkpoint, defaults to the latest checkpoint.",
    ),
    num_episodes: int = typer.Option(10, help="Number of episodes to run"),
    render: bool = typer.Option(True, help="Whether to render the environment"),
    env_config_path: str = typer.Option(
        None,
        help="Path to the environment config file, defaults to env_config.yaml from checkpoint directory.",
    ),
) -> None:
    # Handle dynamic defaults inside the function
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()

    if env_config_path is None:
        env_config_path = get_env_config_path_for_checkpoint(checkpoint_path)

    simulate(checkpoint_path, num_episodes, render, env_config_path)


if __name__ == "__main__":
    app()
