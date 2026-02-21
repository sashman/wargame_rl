import numpy as np
import torch
from torch import Tensor

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.model.dqn.device import Device, get_device


def action_to_tensor(action: WargameEnvAction, device: Device | None = None) -> Tensor:
    device = get_device(device)
    _, action_tensor = torch.tensor(
        action.actions, dtype=torch.float32, device=device
    ).max(dim=-1)
    return action_tensor.unsqueeze(0)


def _normalize(arr: np.ndarray, half_board: np.ndarray) -> np.ndarray:
    """Normalize values to [-1, 1] using per-axis board half-sizes."""
    result: np.ndarray = (arr - half_board) / half_board
    return result


def _group_ids_to_one_hot(group_ids: np.ndarray, max_groups: int) -> np.ndarray:
    """Vectorized one-hot encoding for an array of group IDs."""
    indices = np.clip(group_ids - 1, 0, max_groups - 1)
    one_hot = np.zeros((len(indices), max_groups), dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return one_hot


def _same_group_closest_distance(
    locs: np.ndarray, group_ids: np.ndarray, max_dist: float
) -> np.ndarray:
    """For each model, compute the normalised distance to the nearest model in the same group.

    Returns shape (num_models, 1) with values in [0, 1].
    A model that is the sole member of its group receives 1.0 (maximum distance).
    """
    n = len(locs)

    diff = locs[:, np.newaxis, :] - locs[np.newaxis, :, :]
    pairwise = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(pairwise, np.inf)

    same_group = group_ids[:, np.newaxis] == group_ids[np.newaxis, :]
    pairwise = np.where(same_group, pairwise, np.inf)

    closest = pairwise.min(axis=1)
    closest = np.where(np.isinf(closest), max_dist, closest)
    closest = np.clip(closest, 0.0, max_dist) / max_dist
    return closest.astype(np.float32).reshape(n, 1)


def _observation_to_numpy(
    state: WargameEnvObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single observation to three NumPy arrays (current_turn, objectives, models)."""
    models = state.wargame_models
    max_groups = models[0].max_groups

    half_board = np.array(
        [state.board_width / 2.0, state.board_height / 2.0], dtype=np.float32
    )
    n_objectives = len(state.objectives)
    half_board_tiled = np.tile(half_board, n_objectives)
    max_dist = float(np.sqrt(state.board_width**2 + state.board_height**2))

    locs = np.array([m.location for m in models], dtype=np.float32)
    dists = np.array(
        [m.distances_to_objectives.flatten() for m in models], dtype=np.float32
    )
    group_ids = np.array([m.group_id for m in models], dtype=np.int32)

    model_features = np.hstack(
        [
            _normalize(locs, half_board),
            _normalize(dists, half_board_tiled),
            _group_ids_to_one_hot(group_ids, max_groups),
            _same_group_closest_distance(locs, group_ids, max_dist),
        ]
    )

    obj_locs = np.array([o.location for o in state.objectives], dtype=np.float32)
    obj_features = _normalize(obj_locs, half_board)

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
        group_id as one-hot of length max_groups, and the normalised distance to the closest
        model in the same group (single scalar in [0, 1]).

    Normalization uses the board dimensions stored on the observation
    (per-axis for coordinates/distances, board diagonal for Euclidean distances).
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
