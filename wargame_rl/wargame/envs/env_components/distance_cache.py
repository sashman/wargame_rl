from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame_model import WargameModel
    from wargame_rl.wargame.envs.wargame_objective import WargameObjective


@dataclass(slots=True)
class DistanceCache:
    model_obj_deltas: np.ndarray  # (n_models, n_objectives, 2)
    model_obj_norms: np.ndarray  # (n_models, n_objectives)
    # Back-compat field name: distance to objective *center* (Euclidean).
    model_obj_norms_offset: np.ndarray  # (n_models, n_objectives)
    obj_radii: np.ndarray  # (n_objectives,)
    model_model_norms: np.ndarray | None  # (n_models, n_models) or None

    def all_models_at_objectives(self, alive_mask: np.ndarray | None = None) -> bool:
        """True if every model is within the radius of at least one objective.

        Dead models (alive_mask False) are treated as satisfied so they don't
        block the all-at-objectives check.
        """
        at_objective = self.model_obj_norms_offset <= self.obj_radii
        per_model = at_objective.any(axis=1)
        if alive_mask is not None:
            per_model = per_model | ~alive_mask
        return bool(per_model.all())

    def min_distances_to_same_group(
        self,
        group_ids: np.ndarray,
        alive_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-model min distance to nearest same-group model.

        Dead models are excluded from same-group calculations. A live model
        whose group members are all dead gets 0 (alone in group = no penalty).
        Requires model_model_norms to be set.
        """
        if self.model_model_norms is None:
            raise ValueError("model_model_norms is required")
        n = len(group_ids)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            same = (np.arange(n) != i) & (group_ids == group_ids[i])
            if alive_mask is not None:
                same = same & alive_mask
            if not same.any():
                out[i] = 0.0
            else:
                out[i] = float(self.model_model_norms[i, same].min())
        return out

    def all_models_within_group_distance(
        self,
        group_ids: np.ndarray,
        max_distance: float,
        alive_mask: np.ndarray | None = None,
    ) -> bool:
        """True if every model is within max_distance of at least one same-group model (or alone in its group)."""
        min_dists = self.min_distances_to_same_group(group_ids, alive_mask=alive_mask)
        return bool((min_dists <= max_distance).all())


def compute_distances(
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    compute_model_model: bool = False,
    alive_mask: np.ndarray | None = None,
) -> DistanceCache:
    model_locs = np.array([m.location for m in wargame_models])  # (n_models, 2)
    obj_locs = np.array([o.location for o in objectives])  # (n_objectives, 2)
    obj_radii = np.array([o.radius_size for o in objectives], dtype=float)  # (n_obj,)

    # (n_models, n_objectives, 2)
    deltas = model_locs[:, np.newaxis, :] - obj_locs[np.newaxis, :, :]

    # (n_models, n_objectives)
    norms = np.linalg.norm(deltas, axis=2, ord=2)

    model_model = None
    if compute_model_model:
        mm_deltas = model_locs[:, np.newaxis, :] - model_locs[np.newaxis, :, :]
        model_model = np.linalg.norm(mm_deltas, axis=2, ord=2)

    if alive_mask is not None:
        dead = ~alive_mask
        norms = norms.copy()
        norms[dead] = np.inf
        if model_model is not None:
            model_model = model_model.copy()
            model_model[dead, :] = np.inf
            model_model[:, dead] = np.inf

    return DistanceCache(
        model_obj_deltas=deltas,
        model_obj_norms=norms,
        # "offset" is kept in the field name for backward-compat, but we now
        # define it as the straight Euclidean distance to the objective center.
        model_obj_norms_offset=norms,
        obj_radii=obj_radii,
        model_model_norms=model_model,
    )


def objective_ownership_from_norms_offset(
    player_norms_offset: np.ndarray,
    opponent_norms_offset: np.ndarray,
    obj_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-objective ownership flags from distance caches.

    Ownership rule (shared with VP scoring):
    - A side "controls" an objective if at least one model is within the objective
      radius using the same in-range test as the distance cache:
      `model_obj_norms_offset <= obj_radii`.
    - Contested objectives (both sides have at least one model in range) count as
      controlled by neither.

    Returns:
        (player_controls, opponent_controls), each a boolean array of shape
        `(n_objectives,)`.
    """
    # Any model in range per objective.
    player_any = np.any(player_norms_offset <= obj_radii, axis=0)
    opponent_any = np.any(opponent_norms_offset <= obj_radii, axis=0)

    # Contested objectives count as controlled by neither.
    player_controls = player_any & ~opponent_any
    opponent_controls = opponent_any & ~player_any
    return player_controls, opponent_controls
