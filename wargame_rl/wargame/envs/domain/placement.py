"""Placement domain service: place models and objectives for a new episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
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


def _is_clear(
    candidate: tuple[float, float],
    occupied: list[tuple[float, float]],
    min_separation: float,
) -> bool:
    """True when *candidate* keeps its distance from every model already placed.

    `min_separation` is the base *diameter*: two bases of radius r overlap
    exactly when their centres are closer than 2r apart. At radius 0 this is
    vacuously true, which is what keeps dimensionless models placeable anywhere.
    """
    if min_separation <= 0.0:
        return True
    threshold_sq = min_separation * min_separation
    cx, cy = candidate
    for ox, oy in occupied:
        dx, dy = cx - ox, cy - oy
        if dx * dx + dy * dy < threshold_sq:
            return False
    return True


def _sample_unoccupied(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    occupied: list[tuple[float, float]],
    min_separation: float,
    rng: Generator,
) -> tuple[float, float]:
    """Return a random point in the zone whose base clears every placed base."""
    for _ in range(_MAX_PLACEMENT_RETRIES):
        candidate = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
        if _is_clear(candidate, occupied, min_separation):
            return candidate
    raise RuntimeError(
        f"Could not fit a base of diameter {min_separation} in deployment zone "
        f"[{x_min},{y_min})x[{x_max},{y_max}) alongside {len(occupied)} others. "
        "The zone is too small for the army at this base size."
    )


def _sample_near_anchor(
    anchor: np.ndarray,
    max_dist: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    occupied: list[tuple[float, float]],
    min_separation: float,
    rng: Generator,
) -> tuple[float, float]:
    """Return a random point within *max_dist* (L2) of *anchor*, inside the zone."""
    lo_x = max(x_min, float(anchor[0]) - max_dist)
    hi_x = min(x_max, float(anchor[0]) + max_dist)
    lo_y = max(y_min, float(anchor[1]) - max_dist)
    hi_y = min(y_max, float(anchor[1]) + max_dist)

    if lo_x >= hi_x or lo_y >= hi_y:
        raise RuntimeError(
            f"No valid placement range near anchor {anchor} with "
            f"max_dist={max_dist} inside zone [{x_min},{y_min})x[{x_max},{y_max})"
        )

    max_dist_sq = max_dist * max_dist
    for _ in range(_MAX_PLACEMENT_RETRIES):
        x = float(rng.uniform(lo_x, hi_x))
        y = float(rng.uniform(lo_y, hi_y))
        dx = x - float(anchor[0])
        dy = y - float(anchor[1])
        if (dx * dx + dy * dy) <= max_dist_sq and _is_clear(
            (x, y), occupied, min_separation
        ):
            return (x, y)
    raise RuntimeError(
        f"Could not place model near anchor {anchor} within distance {max_dist}"
    )


def wargame_model_placement(
    wargame_models: list[WargameModel],
    deployment_zone: np.ndarray,
    group_max_distance: float,
    rng: Generator,
    base_radius: float = 0.0,
) -> None:
    """Place models randomly inside the deployment zone, group-aware.

    Bases may not overlap, so the zone has to be big enough to hold the army at
    the configured base size. It fails loudly with the numbers rather than
    quietly stacking models: a 5x5 board's zone is 1 unit wide and a 32mm base
    is 1.26 across, so small demo configs have to grow.
    """
    occupied: list[tuple[float, float]] = []
    min_separation = 2.0 * base_radius
    # A base has to fit *within* the zone too, not just avoid its neighbours.
    x_min = float(deployment_zone[0]) + base_radius
    y_min = float(deployment_zone[1]) + base_radius
    x_max = float(deployment_zone[2]) - base_radius
    y_max = float(deployment_zone[3]) - base_radius
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"Deployment zone {tuple(deployment_zone)} is smaller than one base of "
            f"radius {base_radius}: it leaves no room to stand in."
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
                loc = _sample_unoccupied(
                    x_min, y_min, x_max, y_max, occupied, min_separation, rng
                )
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
                    min_separation,
                    rng,
                )

            model.location = position(*loc)
            model.reset_for_episode()
            occupied.append(loc)
            placed.append(model)


def objective_placement(
    objectives: list[WargameObjective],
    deployment_zone: np.ndarray,
    board_width: int,
    board_height: int,
    rng: Generator,
    opponent_deployment_zone: np.ndarray | None = None,
    min_separation: float | None = None,
    terrain: Terrain | None = None,
    terrain_clearance: float | None = None,
) -> None:
    """Place each objective at a random point outside both deployment zones.

    `min_separation` keeps objective centres apart; without it each is drawn
    independently and the discs overlap in about a quarter of episodes, which
    quietly turns a three-objective mission into a two-objective one.
    `terrain_clearance` keeps them out of ruins.

    Both constraints are satisfied by rejection sampling and are best-effort:
    if a draw cannot be placed within the retry budget the last candidate is
    used, because a slightly crowded layout is better than a failed episode.
    """
    x_min = float(deployment_zone[2])
    x_max = (
        float(opponent_deployment_zone[0])
        if opponent_deployment_zone is not None
        else float(board_width)
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
    x_min: float,
    x_max: float,
    board_height: int,
    rng: Generator,
    placed: list[Position],
    min_separation: float | None,
    terrain: Terrain | None,
    terrain_clearance: float | None,
) -> Position:
    """Draw one objective location satisfying the separation constraints."""
    candidate = zero_position()
    for _ in range(_MAX_PLACEMENT_RETRIES):
        candidate = position(
            rng.uniform(x_min, x_max),
            rng.uniform(0.0, board_height),
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
    base_radius = resolve_rules_quantities(config).base_radius
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
            base_radius=base_radius,
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
                base_radius=base_radius,
            )
