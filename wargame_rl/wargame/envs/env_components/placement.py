"""Initial placement of models and objectives at reset.

Extracted so deployment strategies (random, fixed, zones) can be swapped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from wargame_rl.wargame.envs.types.config import ModelConfig, ObjectiveConfig
    from wargame_rl.wargame.envs.wargame_model import WargameModel
    from wargame_rl.wargame.envs.wargame_objective import WargameObjective

_MAX_PLACEMENT_RETRIES = 1000


def _sample_unoccupied(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    occupied: set[tuple[int, int]],
    rng: Generator,
) -> tuple[int, int]:
    """Return a random unoccupied cell within the given bounds."""
    for _ in range(_MAX_PLACEMENT_RETRIES):
        x = int(rng.integers(x_min, x_max))
        y = int(rng.integers(y_min, y_max))
        if (x, y) not in occupied:
            return (x, y)
    raise RuntimeError(
        "Could not find an unoccupied cell in deployment zone "
        f"[{x_min},{y_min})x[{x_max},{y_max})"
    )


def _sample_near_anchor(
    anchor: np.ndarray,
    max_dist: float,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    occupied: set[tuple[int, int]],
    rng: Generator,
) -> tuple[int, int]:
    """Return a random unoccupied cell within *max_dist* (L2) of *anchor*,
    clamped to the deployment zone."""
    lo_x = max(x_min, int(anchor[0] - max_dist))
    hi_x = min(x_max, int(anchor[0] + max_dist) + 1)
    lo_y = max(y_min, int(anchor[1] - max_dist))
    hi_y = min(y_max, int(anchor[1] + max_dist) + 1)

    if lo_x >= hi_x or lo_y >= hi_y:
        raise RuntimeError(
            f"No valid placement range near anchor {anchor} with "
            f"max_dist={max_dist} inside zone [{x_min},{y_min})x[{x_max},{y_max})"
        )

    max_dist_sq = max_dist * max_dist
    for _ in range(_MAX_PLACEMENT_RETRIES):
        x = int(rng.integers(lo_x, hi_x))
        y = int(rng.integers(lo_y, hi_y))
        dx = x - anchor[0]
        dy = y - anchor[1]
        if (x, y) not in occupied and (dx * dx + dy * dy) <= max_dist_sq:
            return (x, y)
    raise RuntimeError(
        f"Could not place model near anchor {anchor} within distance {max_dist}"
    )


def wargame_model_placement(
    wargame_models: list[WargameModel],
    deployment_zone: np.ndarray,
    group_max_distance: float,
    rng: Generator,
) -> None:
    """Place models randomly inside the deployment zone, group-aware.

    Models within the same group are placed within *group_max_distance* (L2)
    of at least one other model in the group. No two models share a cell.
    """
    occupied: set[tuple[int, int]] = set()
    x_min, y_min, x_max, y_max = (
        int(deployment_zone[0]),
        int(deployment_zone[1]),
        int(deployment_zone[2]),
        int(deployment_zone[3]),
    )

    groups: dict[int, list[WargameModel]] = {}
    for model in wargame_models:
        groups.setdefault(model.group_id, []).append(model)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)  # type: ignore[arg-type]

    for gid in group_ids:
        group = groups[gid]
        rng.shuffle(group)  # type: ignore[arg-type]
        placed: list[WargameModel] = []

        for model in group:
            if not placed:
                loc = _sample_unoccupied(x_min, y_min, x_max, y_max, occupied, rng)
            else:
                anchor = placed[int(rng.integers(len(placed)))]
                loc = _sample_near_anchor(
                    anchor.location,
                    group_max_distance,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    occupied,
                    rng,
                )

            model.location = np.array(loc, dtype=np.int32)
            model.previous_location = None
            model.stats["current_wounds"] = model.stats["max_wounds"]
            model.model_rewards_history.clear()
            occupied.add(loc)
            placed.append(model)


def objective_placement(
    objectives: list[WargameObjective],
    deployment_zone: np.ndarray,
    board_width: int,
    board_height: int,
    rng: Generator,
) -> None:
    """Place each objective at a random cell outside the deployment zone. Mutates objectives."""
    for objective in objectives:
        objective_x = rng.integers(deployment_zone[2], board_width, dtype=np.int32)
        objective_y = rng.integers(deployment_zone[1], board_height, dtype=np.int32)
        objective.location = np.array([objective_x, objective_y], dtype=np.int32)


def fixed_wargame_model_placement(
    wargame_models: list[WargameModel],
    model_configs: list[ModelConfig],
) -> None:
    """Place models at the exact positions specified in *model_configs*.

    Every entry must have x/y set (validated by WargameEnvConfig).
    """
    for model, cfg in zip(wargame_models, model_configs):
        assert cfg.x is not None and cfg.y is not None
        model.location = np.array([cfg.x, cfg.y], dtype=np.int32)
        model.previous_location = None
        model.stats["current_wounds"] = model.stats["max_wounds"]
        model.model_rewards_history.clear()


def fixed_objective_placement(
    objectives: list[WargameObjective],
    objective_configs: list[ObjectiveConfig],
) -> None:
    """Place objectives at the exact positions specified in *objective_configs*.

    Every entry must have x/y set (validated by WargameEnvConfig).
    """
    for objective, cfg in zip(objectives, objective_configs):
        assert cfg.x is not None and cfg.y is not None
        objective.location = np.array([cfg.x, cfg.y], dtype=np.int32)
