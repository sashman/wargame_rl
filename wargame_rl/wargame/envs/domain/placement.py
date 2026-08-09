"""Placement domain service: place models and objectives for a new episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.terrain import Terrain
from wargame_rl.wargame.envs.domain.terrain_placement import generate_terrain
from wargame_rl.wargame.envs.domain.value_objects import (
    BoardDimensions,
    Position,
    position,
    zero_position,
)
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

            model.location = position(*loc)
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
    min_separation: int | None = None,
    terrain: Terrain | None = None,
    terrain_clearance: int | None = None,
) -> None:
    """Place each objective at a random cell outside both deployment zones.

    `min_separation` keeps objective centres apart; without it each is drawn
    independently and the discs overlap in about a quarter of episodes, which
    quietly turns a three-objective mission into a two-objective one.
    `terrain_clearance` keeps them out of ruins.

    Both constraints are satisfied by rejection sampling and are best-effort:
    if a draw cannot be placed within the retry budget the last candidate is
    used, because a slightly crowded layout is better than a failed episode.
    """
    x_min = int(deployment_zone[2])
    x_max = (
        int(opponent_deployment_zone[0])
        if opponent_deployment_zone is not None
        else board_width
    )
    placed: list[Position] = []
    for objective in objectives:
        location = _sample_objective_location(
            x_min,
            x_max,
            board_height,
            rng,
            placed,
            min_separation,
            terrain,
            terrain_clearance,
        )
        objective.location = location
        placed.append(location)


def _sample_objective_location(
    x_min: int,
    x_max: int,
    board_height: int,
    rng: Generator,
    placed: list[Position],
    min_separation: int | None,
    terrain: Terrain | None,
    terrain_clearance: int | None,
) -> Position:
    """Draw one objective location satisfying the separation constraints."""
    candidate = zero_position()
    for _ in range(_MAX_PLACEMENT_RETRIES):
        # Each draw's dtype is part of the random stream rather than the storage
        # format, so changing it changes which layouts a given seed produces.
        candidate = position(
            rng.integers(x_min, x_max, dtype=np.int32),
            rng.integers(0, board_height, dtype=np.int32),
        )
        if min_separation is not None and any(
            float(np.linalg.norm(candidate - other)) < min_separation
            for other in placed
        ):
            continue
        if (
            terrain_clearance is not None
            and terrain is not None
            and _distance_to_terrain(candidate, terrain) < terrain_clearance
        ):
            continue
        return candidate
    return candidate


def _distance_to_terrain(location: np.ndarray, terrain: Terrain) -> float:
    """Euclidean distance from a cell to the nearest footprint, 0 inside one."""
    x, y = float(location[0]), float(location[1])
    if not terrain.footprints:
        return float("inf")
    return min(
        float(
            np.hypot(
                max(f.x0 - x, x - f.x1, 0.0),
                max(f.y0 - y, y - f.y1, 0.0),
            )
        )
        for f in terrain.footprints
    )


def fixed_wargame_model_placement(
    wargame_models: list[WargameModel],
    model_configs: list[ModelConfig],
) -> None:
    """Place models at the exact positions specified in *model_configs*."""
    for model, cfg in zip(wargame_models, model_configs):
        assert cfg.x is not None and cfg.y is not None
        model.location = position(cfg.x, cfg.y)
        model.reset_for_episode()


def fixed_objective_placement(
    objectives: list[WargameObjective],
    objective_configs: list[ObjectiveConfig],
) -> None:
    """Place objectives at the exact positions specified in *objective_configs*."""
    for objective, cfg in zip(objectives, objective_configs):
        assert cfg.x is not None and cfg.y is not None
        objective.location = position(cfg.x, cfg.y)


def place_for_episode(
    battle: Battle,
    config: WargameEnvConfig,
    rng: Generator,
) -> None:
    """Place terrain, player models, objectives, and opponent models for an episode.

    Uses fixed positions from config when available, otherwise random placement
    within deployment zones.
    """
    # Terrain first: it is the board the rest is placed onto. Models and
    # objectives may sit inside a footprint, exactly as they may with a fixed
    # layout — a model in a ruin can still see out and be seen.
    if config.random_terrain is not None:
        battle.set_terrain(
            generate_terrain(
                config.random_terrain,
                BoardDimensions(width=battle.board_width, height=battle.board_height),
                rng,
            )
        )

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
            min_separation=config.objective_min_separation,
            terrain=battle.terrain,
            terrain_clearance=config.objective_terrain_clearance,
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
