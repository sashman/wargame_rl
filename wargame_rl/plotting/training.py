import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from wargame_rl.wargame.model.dqn.state import state_to_tensor
from wargame_rl.wargame.types import State

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use("Agg")


def compute_policy_on_grid(
    box_agent: spaces.Box,
    policy_dqn: nn.Module,
    target_state: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, y_min = box_agent.low
    x_max, y_max = box_agent.high

    if target_state is None:
        target_state = (x_min + x_max) // 2, (y_min + y_max) // 2

    values_function = np.empty([x_max - x_min + 1, y_max - y_min + 1])
    with torch.no_grad():
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                agent_state = np.array([x, y])
                state = State(agent=agent_state, target=target_state)
                tensor_state = state_to_tensor(state)
                values_function[x, y] = policy_dqn(tensor_state).max()

    return values_function, target_state


def plot_policy_on_grid(values_function, target_state) -> plt.Figure:
    fig = plt.figure()
    plt.imshow(values_function.T, origin="upper")
    plt.plot(target_state[0], target_state[1], "or")
    plt.colorbar()
    return fig
