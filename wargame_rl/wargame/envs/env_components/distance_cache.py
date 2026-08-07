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

    def fraction_at_objectives(self, alive_mask: np.ndarray | None = None) -> float:
        """Fraction of alive models within the radius of at least one objective.

        Unlike :meth:`all_models_at_objectives`, dead models are excluded from
        both numerator and denominator rather than counted as satisfied.
        Returns 0.0 when no model is alive.
        """
        at_objective = self.model_obj_norms_offset <= self.obj_radii
        per_model: np.ndarray = np.atleast_1d(at_objective.any(axis=1))
        if alive_mask is not None:
            per_model = per_model & alive_mask
            n_alive = int(alive_mask.sum())
        else:
            n_alive = int(per_model.size)
        if n_alive == 0:
            return 0.0
        return float(per_model.sum()) / n_alive

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

    # A model is within range of an objective when the closest part of its base is,
    # so the offset is the distance from the base *edge* to the objective centre.
    # Models overlapping the centre give a negative distance, which still compares
    # correctly against the radius.
    base_radii = np.array([m.base_radius for m in wargame_models], dtype=float).reshape(
        -1, 1
    )
    norms_offset = norms - base_radii
    if alive_mask is not None:
        norms_offset[~alive_mask] = np.inf

    return DistanceCache(
        model_obj_deltas=deltas,
        model_obj_norms=norms,
        model_obj_norms_offset=norms_offset,
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
    # OC/count rule (OC=1 per model): strictly greater in-range count controls.
    player_counts = np.sum(player_norms_offset <= obj_radii, axis=0)
    opponent_counts = np.sum(opponent_norms_offset <= obj_radii, axis=0)
    player_controls = player_counts > opponent_counts
    opponent_controls = opponent_counts > player_counts
    return player_controls, opponent_controls


def objective_states_from_norms_offset(
    player_norms_offset: np.ndarray,
    opponent_norms_offset: np.ndarray,
    obj_radii: np.ndarray,
) -> list[str]:
    """Per-objective control state under the same OC/count rule as VP scoring.

    "player": player count > opponent count; "opponent": opponent count > player
    count; "contested": equal and >=1 present; "neutral": none present.
    """
    player_counts = np.sum(player_norms_offset <= obj_radii, axis=0)
    opponent_counts = np.sum(opponent_norms_offset <= obj_radii, axis=0)
    states: list[str] = []
    for player_count, opponent_count in zip(player_counts, opponent_counts):
        p, o = int(player_count), int(opponent_count)
        if p <= 0 and o <= 0:
            states.append("neutral")
        elif p > o:
            states.append("player")
        elif o > p:
            states.append("opponent")
        else:
            states.append("contested")
    return states
