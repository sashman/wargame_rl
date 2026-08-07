"""Placement domain service: place models and objectives for a new episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.terrain import Terrain
from wargame_rl.wargame.envs.domain.terrain_placement import generate_terrain
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    ObjectiveConfig,
    WargameEnvConfig,
)

if TYPE_CHECKING:
    from numpy.random import Generator

_MAX_PLACEMENT_RETRIES = 1000


def _overlaps_any(
    candidate: np.ndarray,
    radius: float,
    placed_positions: list[np.ndarray],
    placed_radii: list[float],
) -> bool:
    """True if a base at *candidate* would overlap one already placed."""
    for position, other_radius in zip(placed_positions, placed_radii):
        gap = radius + other_radius
        if float(np.sum((candidate - position) ** 2)) < gap * gap:
            return True
    return False


def _sample_free(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    radius: float,
    placed_positions: list[np.ndarray],
    placed_radii: list[float],
    rng: Generator,
) -> np.ndarray:
    """Return a random position inside the zone whose base overlaps nothing placed.

    The base has to fit inside the zone, so the sampling window is inset by the
    radius on every side.
    """
    lo_x, hi_x = x_min + radius, x_max - radius
    lo_y, hi_y = y_min + radius, y_max - radius
    if lo_x >= hi_x or lo_y >= hi_y:
        raise RuntimeError(
            f"Deployment zone [{x_min},{y_min})x[{x_max},{y_max}) is too small for a "
            f"model of radius {radius}"
        )

    for _ in range(_MAX_PLACEMENT_RETRIES):
        candidate = np.array(
            [rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)], dtype=float
        )
        if not _overlaps_any(candidate, radius, placed_positions, placed_radii):
            return candidate
    raise RuntimeError(
        f"Could not place a model of radius {radius} in deployment zone "
        f"[{x_min},{y_min})x[{x_max},{y_max}) after {_MAX_PLACEMENT_RETRIES} tries"
    )


def _sample_free_near_anchor(
    anchor: np.ndarray,
    max_dist: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    radius: float,
    placed_positions: list[np.ndarray],
    placed_radii: list[float],
    rng: Generator,
) -> np.ndarray:
    """Return a free position within *max_dist* of *anchor*, inside the zone.

    Sampled in polar coordinates around the anchor, with the radius drawn as
    ``max_dist * sqrt(u)`` so points spread uniformly over the disc rather than
    bunching at the centre.
    """
    lo_x, hi_x = x_min + radius, x_max - radius
    lo_y, hi_y = y_min + radius, y_max - radius
    if lo_x >= hi_x or lo_y >= hi_y:
        raise RuntimeError(
            f"Deployment zone is too small for a model of radius {radius}"
        )

    for _ in range(_MAX_PLACEMENT_RETRIES):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        distance = max_dist * np.sqrt(rng.uniform(0.0, 1.0))
        candidate = anchor + distance * np.array([np.cos(angle), np.sin(angle)])
        candidate = np.clip(candidate, [lo_x, lo_y], [hi_x, hi_y])
        if float(np.linalg.norm(candidate - anchor)) > max_dist:
            continue  # clipping pushed it back outside the group radius
        if not _overlaps_any(candidate, radius, placed_positions, placed_radii):
            resolved: np.ndarray = candidate
            return resolved
    raise RuntimeError(
        f"Could not place a model near anchor {anchor} within distance {max_dist}"
    )


def wargame_model_placement(
    wargame_models: list[WargameModel],
    deployment_zone: np.ndarray,
    group_max_distance: float,
    rng: Generator,
) -> None:
    """Place models randomly inside the deployment zone, group-aware and non-overlapping.

    Bases may not overlap, so placement is rejection sampling against everything
    already down rather than a lookup in a set of occupied cells.
    """
    positions: list[np.ndarray] = []
    radii: list[float] = []
    x_min, y_min, x_max, y_max = (
        float(deployment_zone[0]),
        float(deployment_zone[1]),
        float(deployment_zone[2]),
        float(deployment_zone[3]),
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
                location = _sample_free(
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    model.base_radius,
                    positions,
                    radii,
                    rng,
                )
            else:
                anchor = placed[int(rng.integers(len(placed)))]
                location = _sample_free_near_anchor(
                    anchor.location,
                    group_max_distance,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    model.base_radius,
                    positions,
                    radii,
                    rng,
                )

            model.location = location
            model.reset_for_episode()
            positions.append(location)
            radii.append(model.base_radius)
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
    placed: list[np.ndarray] = []
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
    placed: list[np.ndarray],
    min_separation: float | None,
    terrain: Terrain | None,
    terrain_clearance: float | None,
) -> np.ndarray:
    """Draw one objective location satisfying the separation constraints."""
    candidate = np.zeros(2, dtype=float)
    for _ in range(_MAX_PLACEMENT_RETRIES):
        candidate = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(0.0, board_height)],
            dtype=float,
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
        model.location = np.array([cfg.x, cfg.y], dtype=float)
        model.reset_for_episode()


def fixed_objective_placement(
    objectives: list[WargameObjective],
    objective_configs: list[ObjectiveConfig],
) -> None:
    """Place objectives at the exact positions specified in *objective_configs*.

    An area objective is not placed: the outline *is* its position, and its location
    is already the area's centroid.
    """
    for objective, cfg in zip(objectives, objective_configs):
        if objective.is_area:
            continue
        assert cfg.x is not None and cfg.y is not None
        objective.location = np.array([cfg.x, cfg.y], dtype=float)


def place_for_episode(
    battle: Battle,
    config: WargameEnvConfig,
    rng: Generator,
) -> None:
    """Place terrain, player models, objectives, and opponent models for an episode.

    Uses fixed positions from config when available, otherwise random placement
    within deployment zones.
    """
    quantities = resolve_rules_quantities(config)
    # Terrain first: it is the board the rest is placed onto. Models and
    # objectives may sit inside a footprint, exactly as they may with a fixed
    # layout — a model in a ruin can still see out and be seen.
    if config.random_terrain is not None:
        battle.set_terrain(
            generate_terrain(
                config.random_terrain,
                BoardDimensions(width=battle.board_width, height=battle.board_height),
                rng,
                blocking_mask=config.blocking_mask,
            )
        )

    # Place player models
    if config.has_fixed_model_positions and config.models is not None:
        fixed_wargame_model_placement(battle.player_models, config.models)
    else:
        wargame_model_placement(
            battle.player_models,
            battle.deployment_zone,
            quantities.group_max_distance,
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
                quantities.group_max_distance,
                rng,
            )
