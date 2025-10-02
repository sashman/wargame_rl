#!/usr/bin/env python3
"""
Simulation script for running trained DQN agents in the Wargame environment.

Usage:
    python simulate.py --checkpoint path/to/checkpoint.ckpt \
        [--episodes 10] [--render]
"""

import logging

import typer

from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.factory import create_environment

app = typer.Typer(pretty_exceptions_enable=False)


def simulate(checkpoint_path: str, num_episodes: int = 10, render: bool = True):
    """Run simulation with trained agent.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    env = create_environment(render_mode="human" if render else None)
    logging.info(f"Action space: {env.action_space}")
    logging.info(f"Observation space: {env.observation_space}")
    logging.info(f"Running {num_episodes} episodes...")

    agent = Agent(env)
    logging.info(f"Agent created: {agent}")

    policy_net = DQN.from_checkpoint(env, checkpoint_path)
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")

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


@app.command()
def main(
    checkpoint_path: str = typer.Option(
        ..., help="Path to the trained model checkpoint"
    ),
    num_episodes: int = typer.Option(10, help="Number of episodes to run"),
    render: bool = typer.Option(True, help="Whether to render the environment"),
):
    simulate(checkpoint_path, num_episodes, render)


if __name__ == "__main__":
    app()
