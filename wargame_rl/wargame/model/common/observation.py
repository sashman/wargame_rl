import numpy as np
import torch
from torch import Tensor

from wargame_rl.wargame.envs.types import WargameEnvObservation
from wargame_rl.wargame.model.common import Device, get_device


def apply_action_mask(q_values: Tensor, mask: Tensor) -> Tensor:
    """Set Q-values for invalid actions to ``-inf``.

    Broadcasts the mask against *q_values* so it works whether *q_values*
    has a leading batch dim and *mask* does not, or both match.
    """
    if mask.numel() == 0:
        return q_values
    return q_values.masked_fill(~mask, float("-inf"))


def _normalize(arr: np.ndarray, half_board: np.ndarray) -> np.ndarray:
    """Normalize values to [-1, 1] using per-axis board half-sizes."""
    result: np.ndarray = (arr - half_board) / half_board
    return result


def _group_ids_to_one_hot(group_ids: np.ndarray, max_groups: int) -> np.ndarray:
    """Vectorized one-hot encoding for an array of group IDs."""
    indices = np.clip(group_ids, 0, max_groups - 1)
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


def _models_to_features(
    models: list,
    half_board: np.ndarray,
    half_board_tiled: np.ndarray,
    max_dist: float,
    max_groups: int,
    feature_dim: int,
) -> np.ndarray:
    """Convert a list of model observations to a feature matrix.

    Returns shape (n_models, feature_dim). When n_models == 0 returns
    a (0, feature_dim) array so the tensor always has a known width.
    """
    if not models:
        return np.zeros((0, feature_dim), dtype=np.float32)

    locs = np.array([m.location for m in models], dtype=np.float32)
    dists = np.array(
        [m.distances_to_objectives.flatten() for m in models], dtype=np.float32
    )
    group_ids = np.array([m.group_id for m in models], dtype=np.int32)

    return np.hstack(
        [
            _normalize(locs, half_board),
            _normalize(dists, half_board_tiled),
            _group_ids_to_one_hot(group_ids, max_groups),
            _same_group_closest_distance(locs, group_ids, max_dist),
        ]
    )


def _observation_to_numpy(
    state: WargameEnvObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Convert a single observation to NumPy arrays.

    Returns (current_turn, objectives, player_models, opponent_models, action_mask).
    ``action_mask`` is ``(n_models, n_actions)`` or None when not available.
    """
    models = state.wargame_models
    max_groups = models[0].max_groups

    half_board = np.array(
        [state.board_width / 2.0, state.board_height / 2.0], dtype=np.float32
    )
    n_objectives = len(state.objectives)
    half_board_tiled = np.tile(half_board, n_objectives)
    max_dist = float(np.sqrt(state.board_width**2 + state.board_height**2))

    # 2 (loc) + n_objectives*2 (dists) + max_groups (group one-hot) + 1 (closest)
    feature_dim = 2 + n_objectives * 2 + max_groups + 1

    model_features = _models_to_features(
        models, half_board, half_board_tiled, max_dist, max_groups, feature_dim
    )
    opponent_features = _models_to_features(
        state.opponent_models,
        half_board,
        half_board_tiled,
        max_dist,
        max_groups,
        feature_dim,
    )

    # Objectives: location (2) + player_loc (1) + opponent_loc (1) + radius (1) + closest_* (2)
    max_radius = 100.0
    max_dist_obj = (
        1500.0  # Match WargameObjective.MAX_DISTANCE_FOR_SPACE for normalization
    )
    obj_locs = np.array([o.location for o in state.objectives], dtype=np.float32)
    obj_loc_normalized = _normalize(obj_locs, half_board)
    obj_loc = np.array(
        [
            [
                getattr(o, "player_level_of_control", 0.0),
                getattr(o, "opponent_level_of_control", 0.0),
                getattr(o, "radius_size", 0) / max_radius,
                np.clip(
                    getattr(o, "closest_player_distance", 0.0) / max_dist_obj,
                    0.0,
                    1.0,
                ),
                np.clip(
                    getattr(o, "closest_opponent_distance", 0.0) / max_dist_obj,
                    0.0,
                    1.0,
                ),
            ]
            for o in state.objectives
        ],
        dtype=np.float32,
    )
    obj_features = np.hstack([obj_loc_normalized, obj_loc])  # (n_objectives, 7)

    n_phases = 5  # len(BattlePhase)
    normalized_round = state.battle_round / max(state.n_rounds, 1)
    normalized_phase = state.battle_phase_index / max(n_phases - 1, 1)
    max_vp = 100.0  # Normalize VP to [0, 1] for network input
    player_vp_norm = getattr(state, "player_vp", 0) / max_vp
    opponent_vp_norm = getattr(state, "opponent_vp", 0) / max_vp
    game_features = np.array(
        [
            0.0,
            normalized_round,
            normalized_phase,
            player_vp_norm,
            opponent_vp_norm,
        ],
        dtype=np.float32,
    )

    return (
        game_features,
        obj_features,
        model_features,
        opponent_features,
        state.action_mask,
    )


def _mask_to_tensor(
    mask: np.ndarray | None,
    n_models: int,
    n_actions: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert an action mask to a bool tensor, defaulting to all-True."""
    if mask is not None:
        return torch.from_numpy(mask.astype(np.bool_)).to(device)
    return torch.ones(n_models, n_actions, dtype=torch.bool, device=device)


def observation_to_tensor(
    state: WargameEnvObservation, device: Device | None = None
) -> list[torch.Tensor]:
    """Convert observation to tensors.

    Order of tensors
    ----------------

    The tensors are returned in the following order:
        1. game_features: shape (5,) — placeholder, round, phase, player_vp, opponent_vp
        2. tensor_objectives: shape (num_objectives, 7) — loc (2), LoC (2), radius (1), closest_player_dist (1), closest_opponent_dist (1)
        3. tensor_wargame_models: shape (num_models, model_features)
        4. tensor_opponent_models: shape (num_opponent_models, model_features)
           (0 rows when no opponents)
        5. tensor_action_mask: shape (n_models, n_actions), bool

    model_features includes normalized location, distances to objectives,
    group_id one-hot, and closest same-group distance.
    """
    device = get_device(device)
    current_turn, obj_features, model_features, opp_features, mask = (
        _observation_to_numpy(state)
    )

    n_models = model_features.shape[0]
    n_actions = mask.shape[1] if mask is not None else 0
    resolved_device = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )

    return [
        torch.from_numpy(current_turn).to(resolved_device),
        torch.from_numpy(obj_features).to(resolved_device),
        torch.from_numpy(model_features).to(resolved_device),
        torch.from_numpy(opp_features).to(resolved_device),
        _mask_to_tensor(mask, n_models, n_actions, resolved_device),
    ]


def observations_to_tensor_batch(
    states: list[WargameEnvObservation], device: Device = None
) -> list[torch.Tensor]:
    """Batch-convert multiple observations to tensors without per-state tensor allocation."""
    assert len(states) > 0, "No states to convert to tensor"
    device = get_device(device)
    resolved_device = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )

    np_results = [_observation_to_numpy(s) for s in states]

    batch_turn = np.stack([r[0] for r in np_results])
    batch_obj = np.stack([r[1] for r in np_results])
    batch_models = np.stack([r[2] for r in np_results])
    batch_opp = np.stack([r[3] for r in np_results])

    masks = [r[4] for r in np_results]
    n_models = batch_models.shape[1]
    if masks[0] is not None:
        batch_masks = np.stack(masks)  # type: ignore[arg-type]
        mask_tensor = torch.from_numpy(batch_masks.astype(np.bool_)).to(resolved_device)
    else:
        mask_tensor = torch.ones(
            len(states), n_models, 0, dtype=torch.bool, device=resolved_device
        )

    return [
        torch.from_numpy(batch_turn).to(resolved_device),
        torch.from_numpy(batch_obj).to(resolved_device),
        torch.from_numpy(batch_models).to(resolved_device),
        torch.from_numpy(batch_opp).to(resolved_device),
        mask_tensor,
    ]
