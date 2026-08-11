"""Placement domain service: place models and objectives for a new episode."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
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
        # An area objective is not *placed*: its outline is its position, and
        # drawing a random centre for it would move the marker off its own
        # ground while leaving the area where it was.
        if objective.is_area:
            continue
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


_MAX_TERRAIN_DRAWS = 40


def _generate_usable_terrain(
    config: WargameEnvConfig,
    board: BoardDimensions,
    deployment_zone: np.ndarray,
    opponent_deployment_zone: np.ndarray,
    rng: Generator,
) -> Terrain:
    """Draw a layout, redrawing until it can host the objectives it has to.

    Only `objectives_on_terrain` constrains the layout, and it is a genuine
    rejection-sampling constraint like every other placement rule here: a draw
    can leave too few pieces clear of both deployment zones, and the honest
    response is another draw rather than a failed episode.

    **This was a hard failure and it cost a 619-epoch run.** A profile measured
    over 200 layouts had a minimum of five eligible pieces, which read as ample
    -- but a training run resets tens of thousands of times, so a tail event with
    a per-episode probability of well under 1% is a certainty rather than a
    risk. Anything sampled per episode needs to survive the tail, not the mean.
    """
    assert config.random_terrain is not None
    spec = config.random_terrain
    if not config.objectives_on_terrain:
        return generate_terrain(spec, board, rng)

    terrain = generate_terrain(spec, board, rng)
    for _ in range(_MAX_TERRAIN_DRAWS):
        eligible = eligible_objective_pieces(
            terrain, deployment_zone, opponent_deployment_zone
        )
        if len(eligible) >= config.number_of_objectives:
            return terrain
        terrain = generate_terrain(spec, board, rng)
    # Fall through to the placement guard, which names the numbers.
    return terrain


def eligible_objective_pieces(
    terrain: Terrain,
    deployment_zone: np.ndarray,
    opponent_deployment_zone: np.ndarray,
) -> list[Footprint]:
    """Terrain pieces that could host an objective: clear of both deployment zones.

    Strict containment, not overlap. A piece straddling a zone edge would put
    contested ground inside somebody's deployment area, which hands that side
    the objective before the first move.
    """
    player_edge = float(deployment_zone[2])
    opponent_edge = float(opponent_deployment_zone[0])
    return [
        footprint
        for footprint in terrain.footprints
        if footprint.x0 >= player_edge and footprint.x1 <= opponent_edge
    ]


def objectives_from_terrain(
    objectives: list[WargameObjective],
    terrain: Terrain,
    deployment_zone: np.ndarray,
    opponent_deployment_zone: np.ndarray,
    board_width: int,
    board_height: int,
    spread: bool = False,
) -> None:
    """Make each objective *be* a terrain piece — the rules' terrain objective.

    The eligible pieces are the ones clear of both deployment zones, and the
    ones nearest the board centre are chosen. Choosing by distance to the centre
    is what keeps a mirrored layout fair: a mirrored pair sits at equal distance,
    so they are taken together, and an odd count takes the symmetric centre piece
    first. Picking randomly would hand one side the closer prize on some seeds.

    Raises:
        ValueError: when the layout has fewer eligible pieces than there are
            objectives. Silently reusing one piece for two objectives would make
            a three-objective mission a two-objective one without saying so.
    """
    centre = np.array([board_width / 2.0, board_height / 2.0])

    eligible = eligible_objective_pieces(
        terrain, deployment_zone, opponent_deployment_zone
    )
    if len(eligible) < len(objectives):
        raise ValueError(
            f"objectives_on_terrain needs {len(objectives)} terrain pieces clear "
            f"of both deployment zones, but the layout has {len(eligible)}. "
            "Raise the piece count, or widen the gap between the zones."
        )

    chosen = (
        _choose_spread_pieces(eligible, len(objectives))
        if spread
        else _choose_symmetric_pieces(eligible, len(objectives), centre, board_width)
    )
    for objective, footprint in zip(objectives, chosen):
        objective.set_area(footprint.polygon)


def _choose_spread_pieces(
    eligible: list[Footprint],
    n_wanted: int,
) -> list[Footprint]:
    """Pick the *n_wanted* pieces whose minimum pairwise separation is largest.

    Selecting the pieces nearest the board centre packs every objective into the
    middle: measured on 200 layouts, all three sat inside a ~16" circle on a
    60x44 board and 47% of objective pairs were within one weapon range, so one
    squad could shoot at two of them and there was no travel trade-off to make.

    Maximising the minimum separation is fair without needing distance rings.
    The rings existed because ordering by distance to the centre can take a
    mirrored pair plus *half* of the next one; a spread score is a property of
    the chosen *set*, and a mirrored layout's mirrored sets score identically,
    so neither side is handed the closer prize.

    Exhaustive over combinations. `n_wanted` is 3 and eligible counts are single
    digits, so this is a few dozen distance computations per episode.
    """
    if len(eligible) <= n_wanted:
        return list(eligible)

    centroids = [f.polygon.centroid for f in eligible]

    def min_separation(indices: tuple[int, ...]) -> float:
        return min(
            float(np.linalg.norm(centroids[a] - centroids[b]))
            for a, b in itertools.combinations(indices, 2)
        )

    best = max(
        itertools.combinations(range(len(eligible)), n_wanted), key=min_separation
    )
    # Left-to-right so objective index is a stable function of the layout rather
    # than of combination order.
    return sorted(
        (eligible[i] for i in best), key=lambda f: float(f.polygon.centroid[0])
    )


def _choose_symmetric_pieces(
    eligible: list[Footprint],
    n_wanted: int,
    centre: np.ndarray,
    board_width: int,
) -> list[Footprint]:
    """Pick *n_wanted* pieces nearest the board centre, in mirror-symmetric groups.

    Ordering by distance alone is not enough. A mirrored layout's pieces come in
    pairs at equal distance, so taking the nearest three can take a whole pair
    plus *half* of the next one — handing one side a closer prize on that seed,
    with nothing in any aggregate to show for it.

    So pieces are grouped by their distance ring first, and a group is taken
    whole or not at all. A ring that would overflow the budget is skipped in
    favour of the next one that fits; when nothing fits exactly (the count and
    the rings genuinely disagree) it falls back to nearest-first, because a
    slightly unfair layout beats a failed episode.
    """
    rings: dict[int, list[Footprint]] = {}
    for piece in eligible:
        distance = float(np.linalg.norm(piece.polygon.centroid - centre))
        rings.setdefault(round(distance * 1e6), []).append(piece)

    chosen: list[Footprint] = []
    for key in sorted(rings):
        ring = rings[key]
        if len(chosen) + len(ring) <= n_wanted:
            chosen.extend(sorted(ring, key=lambda f: float(f.polygon.centroid[0])))
        if len(chosen) == n_wanted:
            return chosen

    ordered = sorted(
        eligible, key=lambda f: float(np.linalg.norm(f.polygon.centroid - centre))
    )
    return ordered[:n_wanted]


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
            _generate_usable_terrain(
                config,
                BoardDimensions(width=battle.board_width, height=battle.board_height),
                battle.deployment_zone,
                battle.opponent_deployment_zone,
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
    if config.objectives_on_terrain:
        objectives_from_terrain(
            battle.objectives,
            battle.terrain,
            battle.deployment_zone,
            battle.opponent_deployment_zone,
            battle.board_width,
            battle.board_height,
            spread=config.objectives_spread_on_terrain,
        )
    elif config.has_fixed_objective_positions and config.objectives is not None:
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
