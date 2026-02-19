import numpy as np
import torch
from torch import Tensor

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.model.dqn.device import Device, get_device

HALF_GRID_SIZE = 25.0


def action_to_tensor(action: WargameEnvAction, device: Device | None = None) -> Tensor:
    device = get_device(device)
    _, action_tensor = torch.tensor(
        action.actions, dtype=torch.float32, device=device
    ).max(dim=-1)
    return action_tensor.unsqueeze(0)


def _normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - HALF_GRID_SIZE) / HALF_GRID_SIZE


def _group_ids_to_one_hot(group_ids: np.ndarray, max_groups: int) -> np.ndarray:
    """Vectorized one-hot encoding for an array of group IDs."""
    indices = np.clip(group_ids - 1, 0, max_groups - 1)
    one_hot = np.zeros((len(indices), max_groups), dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return one_hot


def _observation_to_numpy(
    state: WargameEnvObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single observation to three NumPy arrays (current_turn, objectives, models)."""
    models = state.wargame_models
    max_groups = models[0].max_groups

    locs = np.array([m.location for m in models], dtype=np.float32)
    dists = np.array(
        [m.distances_to_objectives.flatten() for m in models], dtype=np.float32
    )
    group_ids = np.array([m.group_id for m in models], dtype=np.int32)

    model_features = np.hstack(
        [
            _normalize(locs),
            _normalize(dists),
            _group_ids_to_one_hot(group_ids, max_groups),
        ]
    )

    obj_locs = np.array([o.location for o in state.objectives], dtype=np.float32)
    obj_features = _normalize(obj_locs)

    current_turn = np.array([0], dtype=np.float32)

    return current_turn, obj_features, model_features


def observation_to_tensor(
    state: WargameEnvObservation, device: Device | None = None
) -> list[torch.Tensor]:
    """Convert observation to tensors.

    Order of tensors
    ----------------

    The tensors are returned in the following order:
        1. tensor_current_turn: Tensor containing the current turn as a float32
        tensor of shape (1,).
        2. tensor_objectives: Tensor of all objectives; shape (num_objectives, location_dims),
        values normalized to [-1, 1].
        3. tensor_wargame_models: Tensor of all wargame models; shape (num_models, model_features),
        where model_features includes normalized location, distances to objectives (normalized to [-1, 1]),
        and group_id as one-hot of length max_groups.
    """
    device = get_device(device)
    current_turn, obj_features, model_features = _observation_to_numpy(state)

    return [
        torch.from_numpy(current_turn).to(device),
        torch.from_numpy(obj_features).to(device),
        torch.from_numpy(model_features).to(device),
    ]


def observations_to_tensor_batch(
    states: list[WargameEnvObservation], device: Device = None
) -> list[torch.Tensor]:
    """Batch-convert multiple observations to tensors without per-state tensor allocation."""
    assert len(states) > 0, "No states to convert to tensor"
    device = get_device(device)

    np_results = [_observation_to_numpy(s) for s in states]

    batch_turn = np.stack([r[0] for r in np_results])
    batch_obj = np.stack([r[1] for r in np_results])
    batch_models = np.stack([r[2] for r in np_results])

    return [
        torch.from_numpy(batch_turn).to(device),
        torch.from_numpy(batch_obj).to(device),
        torch.from_numpy(batch_models).to(device),
    ]
