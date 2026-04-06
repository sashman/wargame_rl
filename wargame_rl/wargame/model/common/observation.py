from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from wargame_rl.wargame.envs.domain.shooting import DefenderStats, expected_damage
from wargame_rl.wargame.envs.types import WargameEnvObservation
from wargame_rl.wargame.model.common import Device, get_device

# ---------------------------------------------------------------------------
# Normalization constants — divisors chosen so typical values map to [0, 1].
# ---------------------------------------------------------------------------
NORM_ATTACKS = 10.0
NORM_BALLISTIC_SKILL = 6.0  # D6 max
NORM_STRENGTH = 10.0
NORM_AP = 6.0  # D6 max
NORM_DAMAGE = 10.0
NORM_TOUGHNESS = 10.0
NORM_SAVE = 7.0  # 7 = "no save" ceiling
NORM_MAX_WOUNDS = 100.0
NORM_EXPECTED_DAMAGE = 10.0

N_WOUND_FEATURES = 3  # alive, wound_ratio, max_wounds_norm
N_COMBAT_STATS = 7  # attacks, bs, strength, ap, damage, toughness, save
N_BATTLE_PHASES = 5  # command, movement, shooting, charge, fight


@dataclass(frozen=True, slots=True)
class _ObsWeaponStats:
    """Bridges observation's ``weapon_*`` fields to the ``WeaponStats`` protocol."""

    attacks: int
    ballistic_skill: int
    strength: int
    ap: int
    damage: int


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

    The last three columns are: ``alive`` (0–1), normalized current wounds,
    and normalized max wounds (÷ 100), per Phase 2 observation contract.
    """
    if not models:
        return np.zeros((0, feature_dim), dtype=np.float32)

    locs = np.array([m.location for m in models], dtype=np.float32)
    dists = np.array(
        [m.distances_to_objectives.flatten() for m in models], dtype=np.float32
    )
    group_ids = np.array([m.group_id for m in models], dtype=np.int32)

    core = np.hstack(
        [
            _normalize(locs, half_board),
            _normalize(dists, half_board_tiled),
            _group_ids_to_one_hot(group_ids, max_groups),
            _same_group_closest_distance(locs, group_ids, max_dist),
        ]
    )
    alive_col = np.array([[m.alive] for m in models], dtype=np.float32)
    cw = np.array([[float(m.current_wounds)] for m in models], dtype=np.float32)
    mw = np.array([[float(m.max_wounds)] for m in models], dtype=np.float32)
    mw_safe = np.maximum(mw, 1.0)
    wound_ratio = np.clip(cw / mw_safe, 0.0, 1.0)
    max_w_norm = np.clip(mw / NORM_MAX_WOUNDS, 0.0, 1.0)
    w_attacks = np.array(
        [[m.weapon_attacks / NORM_ATTACKS] for m in models], dtype=np.float32
    )
    w_bs = np.array(
        [[m.weapon_ballistic_skill / NORM_BALLISTIC_SKILL] for m in models],
        dtype=np.float32,
    )
    w_str = np.array(
        [[m.weapon_strength / NORM_STRENGTH] for m in models], dtype=np.float32
    )
    w_ap = np.array([[m.weapon_ap / NORM_AP] for m in models], dtype=np.float32)
    w_dmg = np.array(
        [[m.weapon_damage / NORM_DAMAGE] for m in models], dtype=np.float32
    )
    t_col = np.array([[m.toughness / NORM_TOUGHNESS] for m in models], dtype=np.float32)
    sv_col = np.array([[m.save_stat / NORM_SAVE] for m in models], dtype=np.float32)
    out = np.hstack(
        [
            core,
            alive_col,
            wound_ratio,
            max_w_norm,
            w_attacks,
            w_bs,
            w_str,
            w_ap,
            w_dmg,
            t_col,
            sv_col,
        ]
    )
    assert out.shape[1] == feature_dim, (out.shape[1], feature_dim)
    return out


def _observation_to_numpy(
    state: WargameEnvObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Convert a single observation to NumPy arrays.

    Returns (current_turn, objectives, player_models, opponent_models, action_mask).
    ``action_mask`` is ``(n_models, n_actions)`` or None when not available.
    """
    models = state.wargame_models
    max_groups = (
        models[0].max_groups
        if models
        else (state.opponent_models[0].max_groups if state.opponent_models else 1)
    )

    half_board = np.array(
        [state.board_width / 2.0, state.board_height / 2.0], dtype=np.float32
    )
    n_objectives = len(state.objectives)
    half_board_tiled = np.tile(half_board, n_objectives)
    max_dist = float(np.sqrt(state.board_width**2 + state.board_height**2))

    n_opponent = len(state.opponent_models)

    n_spatial = 2 + n_objectives * 2  # location + distances-to-objectives
    n_group = max_groups + 1  # one-hot group + closest same-group distance
    base_feature_dim = n_spatial + n_group + N_WOUND_FEATURES + N_COMBAT_STATS

    model_features = _models_to_features(
        models, half_board, half_board_tiled, max_dist, max_groups, base_feature_dim
    )
    opponent_features = _models_to_features(
        state.opponent_models,
        half_board,
        half_board_tiled,
        max_dist,
        max_groups,
        base_feature_dim,
    )

    n_player = len(models)
    if n_player > 0 and n_opponent > 0:
        ed_matrix = np.zeros((n_player, n_opponent), dtype=np.float32)
        for pi in range(n_player):
            pm = models[pi]
            if pm.weapon_attacks == 0:
                continue
            for oi in range(n_opponent):
                om = state.opponent_models[oi]
                if om.toughness == 0:
                    continue
                weapon = _ObsWeaponStats(
                    attacks=pm.weapon_attacks,
                    ballistic_skill=pm.weapon_ballistic_skill,
                    strength=pm.weapon_strength,
                    ap=pm.weapon_ap,
                    damage=pm.weapon_damage,
                )
                defender = DefenderStats(toughness=om.toughness, save=om.save_stat)
                ed_matrix[pi, oi] = expected_damage(weapon, defender)
        ed_normalized = np.clip(ed_matrix / NORM_EXPECTED_DAMAGE, 0.0, 1.0)
        model_features = np.hstack([model_features, ed_normalized])
        opp_padding = np.zeros((n_opponent, n_opponent), dtype=np.float32)
        opponent_features = np.hstack([opponent_features, opp_padding])
    elif n_opponent > 0:
        opp_padding = np.zeros((n_opponent, n_opponent), dtype=np.float32)
        opponent_features = np.hstack([opponent_features, opp_padding])

    obj_locs = np.array([o.location for o in state.objectives], dtype=np.float32)
    obj_features = _normalize(obj_locs, half_board)

    normalized_round = state.battle_round / max(state.n_rounds, 1)
    normalized_phase = state.battle_phase_index / max(N_BATTLE_PHASES - 1, 1)
    game_features = np.array(
        [
            0.0,
            normalized_round,
            normalized_phase,
            float(state.player_vp),
            float(state.opponent_vp),
            float(state.player_vp_delta),
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
        1. game_features: shape (6,) — placeholder, normalized_round, normalized_phase, player_vp, opponent_vp, player_vp_delta
        2. tensor_objectives: shape (num_objectives, 2), normalized to [-1, 1]
        3. tensor_wargame_models: shape (num_models, feature_dim)
        4. tensor_opponent_models: shape (num_opponent_models, feature_dim)
           (0 rows when no opponents)
        5. tensor_action_mask: shape (n_models, n_actions), bool

    feature_dim = base + n_opponent, where base includes normalized location,
    distances to objectives, group_id one-hot, closest same-group distance,
    wound features (alive, wound_ratio, max_wounds_norm), and combat stats
    (attacks, bs, strength, ap, damage, toughness, save — each divided by
    its NORM_* constant). The final n_opponent columns are expected damage
    per target (player models) or zero-padding (opponent models).
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
