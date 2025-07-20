import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from torch import nn

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use("Agg")


def compute_policy_on_grid(
    target_state: tuple[int, int], box_agent: spaces.Box, policy_dqn: nn.Module
) -> tuple[np.ndarray, np.ndarray]:
    x_min, y_min = box_agent.low
    x_max, y_max = box_agent.high

    values_function = np.empty([x_max - x_min + 1, y_max - y_min + 1])
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            agent_state = np.array([x, y])
            state = {"agent": agent_state, "target": target_state}

            tensor_state = policy_dqn.to_tensor_state(state)
            values_function[x, y] = policy_dqn(tensor_state).max()

    return values_function, target_state


def save_graph(rewards_per_episode, epsilon_history, loss, env):
    # compute all states of the current policy dqn
    values_function, target_state = compute_policy_on_grid(env)

    # Save plots
    fig = plt.figure(1, figsize=(12, 8), dpi=100)

    # Plot average rewards (Y-axis) vs episodes (X-axis)
    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])
    plt.subplot(221)  # plot on a 1 row x 2 col grid, at cell 1
    # plt.xlabel('Episodes')
    plt.ylabel("Mean Rewards")
    plt.plot(mean_rewards)
    # Plot epsilon decay (Y-axis) vs episodes (X-axis)
    plt.subplot(222)  # plot on a 1 row x 2 col grid, at cell 2
    # plt.xlabel('Time Steps')
    plt.ylabel("Epsilon Decay")
    plt.plot(epsilon_history)

    # Plot loss (Y-axis) vs episodes (X-axis)
    plt.subplot(2, 2, 3)
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.yscale("log")

    # Plot policy on grid
    plt.subplot(2, 2, 4)
    plt.imshow(values_function.T, origin="upper")
    plt.plot(target_state[0], target_state[1], "or")
    plt.colorbar()

    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    # Save plots
    fig.savefig(self.GRAPH_FILE)
    plt.close(fig)
