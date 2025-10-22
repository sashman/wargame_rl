import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from wargame_rl.wargame.envs.types import WargameEnvObservation, WargameModelObservation
from wargame_rl.wargame.model.dqn.device import Device
from wargame_rl.wargame.model.dqn.observation import observation_to_tensor

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use("Agg")


def build_batch_tensor(
    observation: WargameEnvObservation, size: int, device: Device = None
) -> torch.Tensor:
    n_models = len(observation.wargame_models)

    x_min, y_min = 0, 0
    x_max, y_max = size, size
    batch_list = []
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            location = np.array([x, y])
            distances_to_objectives = np.array(
                [
                    np.linalg.norm(location - objective.location, ord=2)
                    for objective in observation.objectives
                ]
            )
            testing_state = WargameEnvObservation(
                current_turn=observation.current_turn,
                wargame_models=[
                    WargameModelObservation(
                        location=location,
                        distances_to_objectives=distances_to_objectives,
                    )
                    for _ in range(n_models)
                ],
                objectives=observation.objectives,
            )
            tensor_state = observation_to_tensor(testing_state, device=device)
            batch_list.append(tensor_state)
    batch_tensor = torch.cat(batch_list)
    return batch_tensor


def compute_values_function(
    observation: WargameEnvObservation, size: int, policy_net: nn.Module
) -> torch.Tensor:
    batch_tensor = build_batch_tensor(observation, size, policy_net.device)
    n_models = len(observation.wargame_models)

    with torch.no_grad():
        output: torch.Tensor = policy_net(batch_tensor)
        values_function = (
            output.max(-1)[0].transpose(0, 1).reshape(n_models, size, size)
        )
    return values_function


def plot_policy_on_grid(
    values_function: torch.Tensor, observation: WargameEnvObservation
) -> plt.Figure:
    n_models, nx, ny = values_function.shape
    p = max(math.ceil(math.sqrt(n_models)), 2)
    ratio = nx / ny
    fig, axs = plt.subplots(p, p, figsize=(10 * ratio, 10))

    # convert to numpy
    values_function_np = values_function.detach().cpu().numpy()
    vmin, vmax = values_function_np.min(), values_function_np.max()

    # Flatten axs to handle cases where p*p > n_models
    axs = axs.flatten()

    # Store the first image handle to use for the colorbar
    im = None
    for i in range(n_models):
        ax = axs[i]
        im = ax.imshow(values_function_np[i].T, origin="upper", vmin=vmin, vmax=vmax)
        for objective in observation.objectives:
            ax.plot(objective.location[0], objective.location[1], "or")

    # Hide unused subplots
    for j in range(n_models, len(axs)):
        axs[j].axis("off")

    # Add shared colorbar
    if im is not None:
        fig.colorbar(im, ax=axs.tolist(), shrink=0.6, orientation="vertical")

    return fig
