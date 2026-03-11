"""Placement domain service: place models and objectives for a new episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    ObjectiveConfig,
    WargameEnvConfig,
)

if TYPE_CHECKING:
    from numpy.random import Generator

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
    """Place models randomly inside the deployment zone, group-aware."""
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
            model.reset_for_episode()
            occupied.add(loc)
            placed.append(model)


def objective_placement(
    objectives: list[WargameObjective],
    deployment_zone: np.ndarray,
    board_width: int,
    board_height: int,
    rng: Generator,
    opponent_deployment_zone: np.ndarray | None = None,
) -> None:
    """Place each objective at a random cell outside both deployment zones."""
    x_min = int(deployment_zone[2])
    x_max = (
        int(opponent_deployment_zone[0])
        if opponent_deployment_zone is not None
        else board_width
    )
    for objective in objectives:
        objective_x = rng.integers(x_min, x_max, dtype=np.int32)
        objective_y = rng.integers(0, board_height, dtype=np.int32)
        objective.location = np.array([objective_x, objective_y], dtype=np.int32)


def fixed_wargame_model_placement(
    wargame_models: list[WargameModel],
    model_configs: list[ModelConfig],
) -> None:
    """Place models at the exact positions specified in *model_configs*."""
    for model, cfg in zip(wargame_models, model_configs):
        assert cfg.x is not None and cfg.y is not None
        model.location = np.array([cfg.x, cfg.y], dtype=np.int32)
        model.reset_for_episode()


def fixed_objective_placement(
    objectives: list[WargameObjective],
    objective_configs: list[ObjectiveConfig],
) -> None:
    """Place objectives at the exact positions specified in *objective_configs*."""
    for objective, cfg in zip(objectives, objective_configs):
        assert cfg.x is not None and cfg.y is not None
        objective.location = np.array([cfg.x, cfg.y], dtype=np.int32)


def place_for_episode(
    battle: Battle,
    config: WargameEnvConfig,
    rng: Generator,
) -> None:
    """Place all player models, objectives, and opponent models for a new episode.

    Uses fixed positions from config when available, otherwise random placement
    within deployment zones.
    """
    # Place player models
    if config.has_fixed_model_positions and config.models is not None:
        fixed_wargame_model_placement(battle.player_models, config.models)
    else:
        wargame_model_placement(
            battle.player_models,
            battle.deployment_zone,
            config.group_max_distance,
            rng,
        )

    # Place objectives
    if config.has_fixed_objective_positions and config.objectives is not None:
        fixed_objective_placement(battle.objectives, config.objectives)
    else:
        objective_placement(
            battle.objectives,
            battle.deployment_zone,
            battle.board_width,
            battle.board_height,
            rng,
            battle.opponent_deployment_zone,
        )

    # Place opponent models
    if battle.opponent_models:
        if config.has_fixed_opponent_positions and config.opponent_models is not None:
            fixed_wargame_model_placement(
                battle.opponent_models, config.opponent_models
            )
        else:
            wargame_model_placement(
                battle.opponent_models,
                battle.opponent_deployment_zone,
                config.group_max_distance,
                rng,
            )
