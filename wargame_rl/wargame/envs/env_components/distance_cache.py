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
