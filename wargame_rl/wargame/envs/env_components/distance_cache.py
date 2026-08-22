from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.types.geometry import polygons_distance_to_points

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameObjective
    from wargame_rl.wargame.envs.wargame_model import WargameModel


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

        Vectorised, and bit-identical to the per-row loop it replaces: filling
        non-candidates with ``inf`` and taking a row-wise ``min`` *selects* the
        same element as masking the row and taking ``min`` of the subset. No
        arithmetic happens, so there is no float reassociation. The loop version
        ran once per model from ``group_cohesion``, making the whole term O(n^3)
        and 55% of ``env.step()`` on the 25v25 configs.
        """
        if self.model_model_norms is None:
            raise ValueError("model_model_norms is required")
        same = group_ids[:, np.newaxis] == group_ids[np.newaxis, :]
        np.fill_diagonal(same, False)
        if alive_mask is not None:
            same = same & alive_mask[np.newaxis, :]
        out: np.ndarray = np.where(same, self.model_model_norms, np.inf).min(axis=1)
        # A model alone in its group has no candidates at all; `min` over an
        # all-inf row would give inf and read as "infinitely scattered".
        out = np.where(same.any(axis=1), out, 0.0)
        return np.atleast_1d(out).astype(np.float64, copy=False)

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
    """Model-to-objective (and optionally model-to-model) distances for one step.

    Each model's own `base_radius` shortens its objective distance, so "within
    range of the marker" is measured from the base *edge* — a model whose base
    touches the disc is in range, which is the rules' reading and what the
    renderer draws. It enters as an offset rather than as a branch in every
    consumer, so `norms_offset <= obj_radii` keeps working unchanged across the
    reward, VP and criteria layers.

    The radius is read off the models rather than passed in **because there are
    seventeen call sites**, and a parameter one of them forgot would silently
    fall back to dimensionless models in that one place — no exception, no
    failing test, just a reward or a VP score measured under different rules
    than the rest of the step.
    """
    model_locs = np.array([m.location for m in wargame_models])  # (n_models, 2)
    obj_locs = np.array([o.location for o in objectives])  # (n_objectives, 2)
    obj_radii = np.array([o.radius_size for o in objectives], dtype=float)  # (n_obj,)

    # (n_models, n_objectives, 2)
    deltas = model_locs[:, np.newaxis, :] - obj_locs[np.newaxis, :, :]

    # (n_models, n_objectives)
    norms = np.linalg.norm(deltas, axis=2, ord=2)
    base_radii = np.array(
        [m.base_radius for m in wargame_models], dtype=float
    )  # (n_models,)

    # An area objective measures to its own edge, not to its centroid. The
    # deltas keep pointing at the centroid — that is the steering target the
    # observation carries — while the *range* test uses the area's boundary.
    to_objective = _distances_to_objectives(model_locs, objectives, norms)
    norms_offset = (
        np.maximum(to_objective - base_radii[:, np.newaxis], 0.0)
        if base_radii.any()
        else to_objective
    )

    model_model = None
    if compute_model_model:
        mm_deltas = model_locs[:, np.newaxis, :] - model_locs[np.newaxis, :, :]
        model_model = np.linalg.norm(mm_deltas, axis=2, ord=2)

    if alive_mask is not None:
        dead = ~alive_mask
        norms = norms.copy()
        norms[dead] = np.inf
        norms_offset = norms_offset.copy()
        norms_offset[dead] = np.inf
        if model_model is not None:
            model_model = model_model.copy()
            model_model[dead, :] = np.inf
            model_model[:, dead] = np.inf

    return DistanceCache(
        model_obj_deltas=deltas,
        model_obj_norms=norms,
        # Distance to the objective centre from the model's base edge. With no
        # base it is the plain centre-to-centre distance, which is what the
        # field held before models had one.
        model_obj_norms_offset=norms_offset,
        obj_radii=obj_radii,
        model_model_norms=model_model,
    )


def _distances_to_objectives(
    model_locs: np.ndarray,
    objectives: list[WargameObjective],
    centre_norms: np.ndarray,
) -> np.ndarray:
    """``(n_models, n_objectives)`` distance to each objective's *range surface*.

    A marker's is the distance to its centre; an area's is the distance to its
    outline, zero inside. Returns `centre_norms` untouched when no objective is
    an area, so the common case allocates nothing and stays bit-identical.
    """
    areas = [obj.area for obj in objectives]
    if not any(area is not None for area in areas):
        return centre_norms

    distances = centre_norms.copy()
    for index, area in enumerate(areas):
        if area is None:
            continue
        distances[:, index] = polygons_distance_to_points(
            model_locs,
            area.vertices[np.newaxis, :, :],
            np.array([area.n_vertices]),
        )[:, 0]
    return distances


def objective_counts_from_norms_offset(
    norms_offset: np.ndarray, obj_radii: np.ndarray
) -> np.ndarray:
    """Models of one side in range of each objective, under the scoring rule.

    THE definition of "on an objective" for this project, and deliberately the
    only one. `norms_offset` already measures from the model's **base edge**
    (`compute_distances` subtracts `base_radius`) and already carries `inf` for
    dead models, so the comparison is the whole rule.

    Extracted because it was written out three times: twice here, and a third
    time in `observation_builder` as a point-in-polygon test on the model
    *centre*. That third copy disagreed with this one on **7.6% of
    (objective, step) slots** on the held-out nine -- so the count the agent
    observed was not the count the mission scored, in the one feature every
    objective-keyed reward and mission term reads.
    """
    counts: np.ndarray = np.sum(norms_offset <= obj_radii, axis=0)
    return counts


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
    player_counts = objective_counts_from_norms_offset(player_norms_offset, obj_radii)
    opponent_counts = objective_counts_from_norms_offset(
        opponent_norms_offset, obj_radii
    )
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
    player_counts = objective_counts_from_norms_offset(player_norms_offset, obj_radii)
    opponent_counts = objective_counts_from_norms_offset(
        opponent_norms_offset, obj_radii
    )
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
