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
    model_obj_norms_offset: np.ndarray  # (n_models, n_objectives)
    obj_radii: np.ndarray  # (n_objectives,)
    model_model_norms: np.ndarray | None  # (n_models, n_models) or None

    def all_models_at_objectives(self) -> bool:
        """True if every model is within the radius of at least one objective."""
        at_objective = self.model_obj_norms_offset <= self.obj_radii
        return bool(at_objective.any(axis=1).all())

    def min_distances_to_same_group(self, group_ids: np.ndarray) -> np.ndarray:
        """Per-model min distance to nearest same-group model.

        For models with no same-group member, returns 0 (treated as within range).
        Requires model_model_norms to be set.
        """
        if self.model_model_norms is None:
            raise ValueError("model_model_norms is required")
        n = len(group_ids)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            same = (np.arange(n) != i) & (group_ids == group_ids[i])
            if not same.any():
                out[i] = 0.0
            else:
                out[i] = float(self.model_model_norms[i, same].min())
        return out

    def all_models_within_group_distance(
        self, group_ids: np.ndarray, max_distance: float
    ) -> bool:
        """True if every model is within max_distance of at least one same-group model (or alone in its group)."""
        min_dists = self.min_distances_to_same_group(group_ids)
        return bool((min_dists <= max_distance).all())


def compute_distances(
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    compute_model_model: bool = False,
) -> DistanceCache:
    model_locs = np.array([m.location for m in wargame_models])  # (n_models, 2)
    obj_locs = np.array([o.location for o in objectives])  # (n_objectives, 2)
    obj_radii = np.array([o.radius_size for o in objectives], dtype=float)  # (n_obj,)

    # (n_models, n_objectives, 2)
    deltas = model_locs[:, np.newaxis, :] - obj_locs[np.newaxis, :, :]

    # (n_models, n_objectives)
    norms = np.linalg.norm(deltas, axis=2, ord=2)

    # offset: model.location - objective.location + radius_size / 2
    # radius_size/2 is a scalar per objective, broadcast to both x/y components
    offsets = deltas + (obj_radii[np.newaxis, :, np.newaxis] / 2)
    norms_offset = np.linalg.norm(offsets, axis=2, ord=2)

    model_model = None
    if compute_model_model:
        mm_deltas = model_locs[:, np.newaxis, :] - model_locs[np.newaxis, :, :]
        model_model = np.linalg.norm(mm_deltas, axis=2, ord=2)

    return DistanceCache(
        model_obj_deltas=deltas,
        model_obj_norms=norms,
        model_obj_norms_offset=norms_offset,
        obj_radii=obj_radii,
        model_model_norms=model_model,
    )


def compute_levels_of_control(
    player_models: list["WargameModel"],
    opponent_models: list["WargameModel"],
    objectives: list["WargameObjective"],
    control_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Level of Control (LoC) per objective for player and opponent.

    A model is in range of an objective if Euclidean distance from model to
    objective centre is <= control_range. LoC = sum of OC of models in range.

    Returns:
        player_loc: shape (n_objectives,) float
        opponent_loc: shape (n_objectives,) float
    """
    obj_locs = np.array([o.location for o in objectives], dtype=float)  # (n_obj, 2)
    n_objectives = len(objectives)

    def loc_for_models(models: list["WargameModel"]) -> np.ndarray:
        if not models:
            return np.zeros(n_objectives, dtype=np.float64)
        locs = np.array([m.location for m in models], dtype=float)  # (n_models, 2)
        oc_vals = np.array([getattr(m, "oc", 1) for m in models], dtype=np.float64)
        # (n_models, n_objectives, 2)
        deltas = locs[:, np.newaxis, :] - obj_locs[np.newaxis, :, :]
        norms = np.linalg.norm(deltas, axis=2, ord=2)  # (n_models, n_objectives)
        in_range = norms <= control_range  # (n_models, n_objectives)
        # Sum OC per objective: (n_objectives,)
        result: np.ndarray = np.where(in_range, oc_vals[:, np.newaxis], 0.0).sum(axis=0)
        return result

    player_loc = loc_for_models(player_models)
    opponent_loc = loc_for_models(opponent_models)
    return player_loc, opponent_loc
