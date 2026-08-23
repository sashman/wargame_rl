"""Placement domain service: place models and objectives for a new episode."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.battle_factory import build_objectives
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.map_layout import MapLayout
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
    CoherencyConfig,
    ModelConfig,
    ObjectiveConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.geometry import Polygon

if TYPE_CHECKING:
    from numpy.random import Generator

_MAX_PLACEMENT_RETRIES = 1000
# How many times a coherent unit may be re-laid before the zone is declared
# too tight. Each attempt is independent, so a handful covers the rare corner.
_MAX_UNIT_LAYOUT_ATTEMPTS = 20


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


def _fits_in_zone(
    candidate: tuple[float, float], zone: Polygon | None, base_radius: float
) -> bool:
    """Whether a model at *candidate* stands *wholly* inside a deployment zone.

    The whole base has to fit, not just the centre -- the rules' *wholly within*.

    ⚠ This used to test four cardinal points of the base, justified by "the
    zones are convex or nearly so". Both halves of that were wrong. The zones
    are not convex -- only 11 of the 45 real tables deploy on axis-aligned
    bands, the other 34 being triangles, staircases and arcs -- and convexity
    was never the property that mattered. **A cardinal probe misses whenever the
    nearest boundary point does not lie in a cardinal direction**, which a
    single edge at any angle to the axes is enough to cause: no corner is
    needed, and it happens at convex and reflex vertices alike. On a 45 degree
    edge a cardinal step closes only `r / sqrt(2)` of the gap, so every centre
    between `r / sqrt(2)` and `r` from that edge was wrongly accepted.

    Measured over the real zones, the four-point test accepted **269 of 195,601
    positions (0.14%) whose base crossed the boundary, on 30 of the 45 tables**.

    Distance to the outline is exact for any simple polygon, and measured 2-3x
    FASTER than the four probes it replaces.
    """
    if zone is None:
        return True
    x, y = candidate
    if not zone.contains(x, y):
        return False
    if base_radius <= 0.0:
        return True
    return zone.distance_to_boundary(x, y) >= base_radius


def _sample_unoccupied(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    occupied: list[tuple[float, float]],
    min_separation: float,
    rng: Generator,
    zone: Polygon | None = None,
    base_radius: float = 0.0,
) -> tuple[float, float]:
    """Return a random point in the zone whose base clears every placed base."""
    for _ in range(_MAX_PLACEMENT_RETRIES):
        candidate = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
        if _is_clear(candidate, occupied, min_separation) and _fits_in_zone(
            candidate, zone, base_radius
        ):
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
    zone: Polygon | None = None,
    base_radius: float = 0.0,
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
        if (
            (dx * dx + dy * dy) <= max_dist_sq
            and _fits_in_zone((x, y), zone, base_radius)
            and _is_clear((x, y), occupied, min_separation)
        ):
            return (x, y)
    raise RuntimeError(
        f"Could not place model near anchor {anchor} within distance {max_dist}"
    )


def _sample_coherent_member(
    placed: list[tuple[float, float]],
    chain_span: float,
    spread_span: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    occupied: list[tuple[float, float]],
    min_separation: float,
    rng: Generator,
    zone: Polygon | None = None,
    base_radius: float = 0.0,
) -> tuple[float, float] | None:
    """A point that keeps its unit in coherency, given where the unit already is.

    Both spans are **centre to centre**, already widened from the rules' base to
    base figures by the caller. The candidate must land within ``chain_span`` of
    at least one placed member -- which is what makes the unit's chain graph
    connected by construction, since it links to a component that is already one
    piece -- and within ``spread_span`` of every one of them.

    Sampled from the box around an anchor rather than the whole zone: the legal
    annulus is a small part of a deployment zone, and uniform sampling over the
    zone would spend the retry budget missing it.

    **Every placed member is tried as an anchor, in a random order**, because
    one anchor's feasible region being empty says nothing about the others'. The
    first version drew a single anchor and raised if it failed, which killed two
    training runs ~20 minutes in: a model anchored against the board edge, with
    the spread cap measured to members already committed on the far side, has
    nowhere legal to stand while a member one row in has plenty. Returns None
    when *no* anchor works, which is the caller's signal to re-lay the unit
    rather than an error -- the unit's own earlier choices are what boxed it in.
    """
    chain_sq = chain_span * chain_span
    spread_sq = spread_span * spread_span
    order = rng.permutation(len(placed))
    budget = max(1, _MAX_PLACEMENT_RETRIES // max(1, len(placed)))

    for anchor_index in order:
        anchor = placed[int(anchor_index)]
        lo_x = max(x_min, anchor[0] - chain_span)
        hi_x = min(x_max, anchor[0] + chain_span)
        lo_y = max(y_min, anchor[1] - chain_span)
        hi_y = min(y_max, anchor[1] + chain_span)
        if lo_x >= hi_x or lo_y >= hi_y:
            continue
        for _ in range(budget):
            candidate = (
                float(rng.uniform(lo_x, hi_x)),
                float(rng.uniform(lo_y, hi_y)),
            )
            dx, dy = candidate[0] - anchor[0], candidate[1] - anchor[1]
            if dx * dx + dy * dy > chain_sq:
                continue
            if not _fits_in_zone(candidate, zone, base_radius):
                continue
            if not _is_clear(candidate, occupied, min_separation):
                continue
            if any(
                (candidate[0] - px) ** 2 + (candidate[1] - py) ** 2 > spread_sq
                for px, py in placed
            ):
                continue
            return candidate
    return None


def wargame_model_placement(
    wargame_models: list[WargameModel],
    deployment_zone: np.ndarray,
    group_max_distance: float,
    rng: Generator,
    base_radius: float = 0.0,
    coherency: CoherencyConfig | None = None,
    zone: Polygon | None = None,
) -> None:
    """Place models randomly inside the deployment zone, group-aware.

    `zone` is the layout's own deployment outline where it has one. The real
    deployments are triangles, staircases and arcs rather than the axis-aligned
    band `deployment_zone` describes, so the rectangle stays as the sampling
    *box* -- it is the outline's bounding box then -- and the outline rejects
    anything outside it.

    Bases may not overlap, so the zone has to be big enough to hold the army at
    the configured base size. It fails loudly with the numbers rather than
    quietly stacking models: a 5x5 board's zone is 1 unit wide and a 32mm base
    is 1.26 across, so small demo configs have to grow.

    Pass *coherency* with ``enforce_at_deployment`` set to satisfy
    `03-moving.md` § Setting up -- every unit is then placed in coherency, which
    is what the rules require of any set-up and what the default placement does
    not achieve: measured with `just measure-coherency`, **0 of 20 episodes**
    deploy coherently on the golden shooting config, because each model is
    anchored within ``group_max_distance`` of one *random* squadmate, which
    bounds the nearest neighbour and leaves the unit's overall span unbounded.
    """
    coherent = coherency is not None and coherency.enforce_at_deployment
    # The rules measure base to base; placement works in centres. Widening the
    # spans here once is what lets the sampler stay in plain centre distance.
    chain_span = (
        coherency.nearest_distance + 2.0 * base_radius
        if coherency is not None
        else group_max_distance
    )
    spread_span = (
        coherency.furthest_distance + 2.0 * base_radius
        if coherency is not None
        else group_max_distance
    )
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
        # A coherent unit can paint itself into a corner: the members already
        # down constrain the next one through both the spread cap and the bases
        # they occupy, and no anchor is left with anywhere legal. That is a
        # property of *this* attempt, not of the zone, so the unit is re-laid
        # from scratch rather than the episode failing. Bounded, because a zone
        # genuinely too tight must still fail loudly instead of spinning.
        for attempt in range(_MAX_UNIT_LAYOUT_ATTEMPTS):
            placed: list[WargameModel] = []
            placed_locations: list[tuple[float, float]] = []
            trial_occupied = list(occupied)
            for model in group:
                if not placed:
                    loc = _sample_unoccupied(
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        trial_occupied,
                        min_separation,
                        rng,
                        zone,
                        base_radius,
                    )
                elif coherent:
                    candidate = _sample_coherent_member(
                        placed_locations,
                        chain_span,
                        spread_span,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        trial_occupied,
                        min_separation,
                        rng,
                        zone,
                        base_radius,
                    )
                    if candidate is None:
                        break
                    loc = candidate
                else:
                    # Read the anchor from `placed_locations`, not from the
                    # model: positions are committed only once the whole unit
                    # lays successfully, so `model.location` is still the
                    # placeholder until then.
                    anchor_loc = placed_locations[int(rng.integers(len(placed)))]
                    loc = _sample_near_anchor(
                        np.asarray(anchor_loc, dtype=float),
                        group_max_distance,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        trial_occupied,
                        min_separation,
                        rng,
                        zone,
                        base_radius,
                    )

                trial_occupied.append(loc)
                placed.append(model)
                placed_locations.append(loc)

            if len(placed) == len(group):
                break
        else:
            raise RuntimeError(
                f"Could not lay unit {gid} of {len(group)} models in coherency after "
                f"{_MAX_UNIT_LAYOUT_ATTEMPTS} attempts: chain span {chain_span}, "
                f"spread span {spread_span}, base separation {min_separation}, zone "
                f"[{x_min},{y_min})x[{x_max},{y_max}). The zone is too tight for a "
                "unit this size at these distances."
            )

        for model, loc in zip(placed, placed_locations):
            model.location = position(*loc)
            model.reset_for_episode()
            occupied.append(loc)


def _sample_in_objective(
    objective: WargameObjective,
    occupied: list[tuple[float, float]],
    min_separation: float,
    rng: Generator,
    hostile: list[tuple[float, float]] | None = None,
    hostile_separation: float = 0.0,
) -> tuple[float, float] | None:
    """A random point whose centre is on *objective* and whose base is clear.

    Centre-inside rather than base-inside: control is scored from the centre for
    an area and from the base edge for a disc, so a point placed this way counts
    as holding under either rule. Returns None when the retry budget is spent,
    which a small objective genuinely can exhaust.
    """
    if objective.area is not None:
        x_min, y_min, x_max, y_max = objective.area.bounds
        inside = objective.area.contains
    else:
        centre = np.asarray(objective.location, dtype=float)
        radius = float(objective.radius_size)
        x_min, y_min = float(centre[0]) - radius, float(centre[1]) - radius
        x_max, y_max = float(centre[0]) + radius, float(centre[1]) + radius

        def inside(x: float, y: float) -> bool:
            return bool(np.hypot(x - centre[0], y - centre[1]) <= radius)

    for _ in range(_MAX_PLACEMENT_RETRIES):
        candidate = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
        if not inside(*candidate):
            continue
        if not _is_clear(candidate, occupied, min_separation):
            continue
        if hostile and not _is_clear(candidate, hostile, hostile_separation):
            continue
        return candidate
    return None


def start_group_on_objective(
    player_models: list[WargameModel],
    opponent_models: list[WargameModel],
    objectives: list[WargameObjective],
    rng: Generator,
    base_radius: float = 0.0,
    engagement_range: float = 0.0,
) -> int | None:
    """Move one whole player group off its deployment position onto an objective.

    A *training-time* start-state augmentation, and deliberately not a rule: it
    teleports a squad, which no legal turn can do.

    It exists because the abandonment this project keeps failing to fix is an
    optimisation problem rather than a pricing one. Measured on the trained
    agent, putting one squad on the objective it otherwise abandons is worth
    **+3.26 episode reward** (paired, 21 of 29 seeds, sign test p ~ 0.013)
    against a travel cost of roughly 0.27 — the policy is sitting where a
    deviation it can execute pays about twelve times what it costs, and does not
    take it. Five reward weightings moved abandonment by 2.3 points between them,
    because they all changed what the far peak *pays* rather than the odds of
    ever standing on it. This puts the far peak in the training distribution
    instead, so the policy only has to learn to stay.

    Returns the objective index that was occupied, or None if nothing moved.

    Best-effort by design: a model that cannot be fitted keeps its deployment
    position rather than failing the episode, matching `objective_placement`.

    **Known limitation, measured, and the reason to keep the probability low.**
    An objective can sit within one weapon range of the opponent's deployment
    zone -- objective selection actively pushes them apart -- and
    `reset` resolves the opponent's whole turn before the agent's first
    observation when the opponent has first turn. At probability 1.0 on the
    spread scenario that puts enemies in range at the first observation in
    123 of 200 episodes and leaves player models already dead in 52 of 200,
    against zero for the un-augmented start. The squad therefore learns the
    objective is a kill box as readily as it learns to hold it. The honest fix
    is to start the squad *part way* along the approach rather than on top of
    the objective, which would also give the value function a path to propagate
    along; that is not built.
    """
    if not objectives or not player_models:
        return None

    groups: dict[int, list[WargameModel]] = {}
    for model in player_models:
        groups.setdefault(model.group_id, []).append(model)

    group_ids = sorted(groups)
    chosen_group = int(group_ids[int(rng.integers(len(group_ids)))])
    objective_index = int(rng.integers(len(objectives)))
    moving = groups[chosen_group]

    # The movers vacate their deployment spots, so only the models staying put
    # constrain where they can land.
    staying = [m for m in player_models if m.group_id != chosen_group]
    occupied = [(float(m.location[0]), float(m.location[1])) for m in staying]
    # Enemies need a wider berth than friends. Clearing only the base diameter
    # sets models up in base contact, and a model within engagement range is
    # barred from shooting at all (`shooting_masks`), so it would deploy as a
    # free kill that cannot fire back -- and the rules spec requires a unit that
    # is *set up* to be unengaged.
    #
    # This guarantees the state at *placement* only. `reset` then resolves the
    # opponent's whole turn before the agent's first observation, and they walk
    # back into contact: measured, min separation is 2.44 here and 1.26 by the
    # first observation. Deployment legality is the most this can buy; the
    # first-observation problem is the opponent's free turn, not the placement.
    hostile = [(float(m.location[0]), float(m.location[1])) for m in opponent_models]
    min_separation = 2.0 * base_radius
    hostile_separation = min_separation + engagement_range

    moved = 0
    for model in moving:
        spot = _sample_in_objective(
            objectives[objective_index],
            occupied,
            min_separation,
            rng,
            hostile=hostile,
            hostile_separation=hostile_separation,
        )
        if spot is None:
            # A model that could not be fitted keeps its deployment position --
            # which was left out of `occupied` on the assumption it would move,
            # so put it back or a later mover can be placed on top of it.
            occupied.append((float(model.location[0]), float(model.location[1])))
            continue
        model.location = position(*spot)
        model.reset_for_episode()
        occupied.append(spot)
        moved += 1

    return objective_index if moved else None


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
    x_min, x_max = _band_between_zones(
        deployment_zone, opponent_deployment_zone, board_width
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


def _band_between_zones(
    deployment_zone: np.ndarray,
    opponent_deployment_zone: np.ndarray | None,
    board_width: int,
) -> tuple[float, float]:
    """The strip of board left over between the two deployment zones.

    **Which zone is on the left is not assumed.** This used to read the band as
    `(deployment_zone.x_max, opponent_deployment_zone.x_min)`, which is right
    only while the player deploys on the left -- every shipped config does, so
    the assumption was invisible. Give the player the right-hand zone and those
    two numbers are the *outer* edges of the board rather than the inner edges
    of the gap, so `x_max < x_min` and objective placement raises
    `high - low < 0` from numpy at reset.

    It surfaced from a seat-parity check, which plays one policy from both
    zones -- the whole point of which is to find asymmetries that no shipped
    config exercises.
    """
    if opponent_deployment_zone is None:
        return float(deployment_zone[2]), float(board_width)

    leftmost, rightmost = sorted(
        (deployment_zone, opponent_deployment_zone), key=lambda zone: float(zone[0])
    )
    return float(leftmost[2]), float(rightmost[0])


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
    # Redraw for `None` as well as `True`: under the default the objectives
    # should be the ruins wherever the layout allows it, and a draw that cannot
    # host them would otherwise quietly fall back to discs and change the
    # mission for that episode alone.
    if config.objectives_on_terrain is False:
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
    """Terrain pieces that could host an objective: those in the middle section.

    A piece belongs to the middle when its **centre** lies between the two
    deployment edges. Overlap is fine — a ruin may reach across an edge, and on
    a real table they routinely do. What is excluded is a piece that *sits in* a
    zone, which would hand that side an objective before the first move.

    This is deliberately laxer than the strict containment it replaces. Requiring
    the whole footprint to clear both edges rejects exactly the large pieces the
    selection now wants: on the fitted profile a 14" ruin cannot fit between the
    edges without touching one, so the widest candidates were being filtered out
    before the chooser ever saw them.
    """
    player_edge = float(deployment_zone[2])
    opponent_edge = float(opponent_deployment_zone[0])
    return [
        footprint
        for footprint in terrain.footprints
        if player_edge <= float(footprint.polygon.centroid[0]) <= opponent_edge
    ]


def objectives_from_terrain(
    objectives: list[WargameObjective],
    terrain: Terrain,
    deployment_zone: np.ndarray,
    opponent_deployment_zone: np.ndarray,
    board_width: int,
    board_height: int,
    min_separation: float | None = None,
) -> None:
    """Make each objective *be* a terrain piece — the rules' terrain objective.

    The eligible pieces are the ones whose centre lies in the middle section,
    and the **largest** of those are chosen: on a real table the objectives sit
    on the substantial ruins, not on whichever scatter happens to be nearest the
    middle. Size is also what makes an objective worth fighting over, since an
    area objective holds as many models as it has room for.

    Two constraints survive from the previous rule and are not negotiable:

    * **The chosen set must mirror onto itself.** Ranking by size alone has the
      same defect ranking by separation had — a mirrored layout's pieces come in
      equal-area pairs, so "the three biggest" can take a pair plus *half* of the
      next, handing one side the closer prize. That was measured at 38 of 200
      layouts before the constraint existed.
    * **They must not cluster.** Choosing the biggest pieces says nothing about
      where they are, and three large ruins in one corner is the failure the
      spread work was written for: all three inside a ~16" circle, 47% of pairs
      within one weapon range, so one squad covered two objectives and there was
      no travel trade-off to make. Separation is therefore the ranking criterion
      *within* the pool the size filter hands it, not an optional extra --
      ranking by area with separation as a tiebreak leaves it almost never
      running, since exact area ties are rare.

    Both are applied as filters ahead of the size ranking, and both fall back
    rather than fail — a slightly unfair or slightly clustered layout beats a
    failed episode, which is the same trade the rest of this module makes.

    Raises:
        ValueError: when the layout has fewer eligible pieces than there are
            objectives. Silently reusing one piece for two objectives would make
            a three-objective mission a two-objective one without saying so.
    """
    eligible = eligible_objective_pieces(
        terrain, deployment_zone, opponent_deployment_zone
    )
    if len(eligible) < len(objectives):
        raise ValueError(
            f"objectives_on_terrain needs {len(objectives)} terrain pieces whose "
            f"centres lie between the deployment zones, but the layout has "
            f"{len(eligible)}. Raise the piece count, or widen the gap between "
            "the zones."
        )

    # A floor derived from the board when the config does not set one. Without
    # it "not clustered" would silently do nothing on every config that leaves
    # `objective_min_separation` unset -- which is most of them, and is exactly
    # the settable-but-inert failure this repo keeps hitting.
    section_width = float(opponent_deployment_zone[0]) - float(deployment_zone[2])
    if min_separation is None:
        min_separation = _DEFAULT_SEPARATION_FRACTION * float(
            np.hypot(section_width, board_height)
        )

    chosen = _choose_largest_pieces(
        eligible,
        len(objectives),
        board_width,
        min_separation=min_separation,
    )
    for objective, footprint in zip(objectives, chosen):
        objective.set_area(footprint.polygon)


# The non-clustering floor, when the config does not set one, as a fraction of
# the middle section's diagonal. Derived from the layout rather than tuned: the
# measured failure was three objectives inside a ~16" circle on a 60x44 board
# with 47% of pairs within one weapon range, and a quarter of that section's
# diagonal is ~12" there -- one weapon range, the distance at which a single
# squad stops being able to cover two objectives.
_DEFAULT_SEPARATION_FRACTION = 0.25


def _choose_largest_pieces(
    eligible: list[Footprint],
    n_wanted: int,
    board_width: int,
    min_separation: float,
    tolerance: float = 1e-6,
) -> list[Footprint]:
    """Pick the largest *n_wanted* pieces that mirror onto themselves and spread.

    Three criteria, and the order between them is the design, because they
    genuinely conflict:

    1. **Mirror symmetry**, as a hard constraint. The chosen set must map onto
       itself under reflection about the board's centre line. This is a fairness
       guarantee rather than a preference: a mirrored layout's pieces come in
       equal-area pairs, so "the three biggest" can take a pair plus *half* of
       the next and hand one side the closer prize. An earlier separation-ranked
       version had the identical defect and it was measured — 38 of 200 layouts
       came out 2-1, up to 3.67" off centre. Non-mirrored terrain has no such
       guarantee to preserve, so there the filter drops out.
    2. **Not clustered**, as a hard floor on the minimum pairwise separation.
       Size says nothing about position, so the largest pieces may all sit in one
       corner — which removes the travel trade-off between objectives and lets
       one squad cover two.
    3. **Largest**, as the ranking among whatever survives both. Objectives
       should sit on the substantial ruins: an area objective holds as many
       models as it has room for, which is what makes a piece worth contesting.

    Ranking by size *within* a pool pre-filtered for separation was tried and is
    wrong — once the pool is fixed, separation becomes the only live criterion
    and size stops binding at all. One real layout offered areas
    [40, 40, 28, 27, 27] and that version chose [28, 27, 27], the *smaller*
    symmetric set, because it was marginally more spread. Size has to be the
    ranking and separation the constraint, not the other way round.

    Each filter falls back rather than fails: when nothing clears the separation
    floor the most separated set is taken anyway, since the layout cannot be made
    unclustered but can still be made as unclustered as it gets.

    Exhaustive over combinations. `n_wanted` is 3 and eligible counts are single
    digits, so this is a few dozen area computations per episode.
    """
    if len(eligible) <= n_wanted:
        return list(eligible)

    centroids = [f.polygon.centroid for f in eligible]
    areas = [float(f.polygon.area) for f in eligible]

    def min_gap(indices: tuple[int, ...]) -> float:
        return min(
            float(np.linalg.norm(centroids[a] - centroids[b]))
            for a, b in itertools.combinations(indices, 2)
        )

    def total_area(indices: tuple[int, ...]) -> float:
        return sum(areas[i] for i in indices)

    def is_symmetric(indices: tuple[int, ...]) -> bool:
        """True when reflecting the chosen centroids maps the set onto itself."""
        chosen = [centroids[i] for i in indices]
        for point in chosen:
            reflected = np.array([board_width - point[0], point[1]])
            if not any(
                bool(np.linalg.norm(reflected - other) <= tolerance) for other in chosen
            ):
                return False
        return True

    combinations = list(itertools.combinations(range(len(eligible)), n_wanted))
    candidates = [c for c in combinations if is_symmetric(c)] or combinations

    spread_enough = [c for c in candidates if min_gap(c) >= min_separation]
    if not spread_enough:
        # No set clears the floor, so "largest" would be free to cluster. Rank on
        # separation instead and take the best available.
        best_gap = max(candidates, key=min_gap)
        return sorted(
            (eligible[i] for i in best_gap),
            key=lambda f: float(f.polygon.centroid[0]),
        )

    # Area first, separation as the tiebreak -- equal-area sets are common on
    # mirrored terrain, and preferring the spread one among them is free.
    best = max(spread_enough, key=lambda c: (total_area(c), min_gap(c)))
    # Left-to-right so objective index is a stable function of the layout rather
    # than of combination order.
    return sorted(
        (eligible[i] for i in best), key=lambda f: float(f.polygon.centroid[0])
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


def _can_host_objectives(battle: Battle, config: WargameEnvConfig) -> bool:
    """True when the layout has enough eligible pieces for every objective.

    Only consulted when `objectives_on_terrain` came from the default rather
    than from the config, to decide whether to use terrain objectives or fall
    back to discs.
    """
    if not battle.terrain.footprints:
        return False
    eligible = eligible_objective_pieces(
        battle.terrain, battle.deployment_zone, battle.opponent_deployment_zone
    )
    return len(eligible) >= len(battle.objectives)


def install_layout(battle: Battle, config: WargameEnvConfig, layout: MapLayout) -> None:
    """Put a drawn layout's terrain, and its objectives if it has any, onto the battle.

    Shared by `place_for_episode` and by env construction, which installs a
    layout before the first reset so network sizing reads an observation of the
    right shape. Both go through here so a pool episode and the very first
    observation cannot describe different boards.
    """
    battle.set_terrain(layout.terrain)
    if layout.objectives is not None:
        battle.set_objectives(
            build_objectives(
                layout.objectives,
                len(layout.objectives),
                float(config.objective_radius_size),
            )
        )


def place_for_episode(
    battle: Battle,
    config: WargameEnvConfig,
    rng: Generator,
    augment_start: bool = False,
    layout: MapLayout | None = None,
) -> None:
    """Place terrain, player models, objectives, and opponent models for an episode.

    Uses fixed positions from config when available, otherwise random placement
    within deployment zones.

    `layout` is one map drawn from a `map_pool`, already resolved by the caller —
    the domain never reads the pool or the filesystem. Its terrain replaces the
    board's, and its objectives, if it carries any, replace the scenario's
    outright including their count. A layout that carries terrain alone falls
    through to the normal objective placement, which then runs against *this*
    layout's ruins because the terrain went in first.

    `augment_start` opts in to the start-state augmentation described on
    `start_group_on_objective`. It is off by default and **draws nothing from
    `rng` when off**, so a config carrying
    `start_on_objective_probability` produces layouts bit-identical to one
    without it whenever the augmentation is not requested. That is what keeps an
    augmented run's evaluation comparable to a baseline measured on the base
    config — the opposite discipline to `combat_seed`, which draws either way
    precisely so the layout stream *cannot* shift.
    """
    base_radius = resolve_rules_quantities(config).base_radius
    # A layout may bring its own deployment zones. They replace the scenario's
    # rectangles the same way its objectives replace the scenario's, and for the
    # same reason: they are part of the table, not of the scenario. The
    # rectangle handed to the sampler becomes the outline's bounding box, so the
    # existing "is the zone big enough" check still fires on a zone that is.
    player_zone = opponent_zone = None
    player_box, opponent_box = battle.deployment_zone, battle.opponent_deployment_zone
    if layout is not None and layout.deployment is not None:
        player_zone = layout.deployment.player_polygon()
        opponent_zone = layout.deployment.opponent_polygon()
    elif config.deployment_outline is not None:
        # The evaluation path installs a map onto a scenario rather than drawing
        # it from a pool, so it has no `layout` to carry the zones -- it puts
        # them on the config instead. Both routes have to work or a map would
        # train under its own deployment and be scored under the rectangle.
        player_zone = Polygon.from_points(config.deployment_outline)
        if config.opponent_deployment_outline is not None:
            opponent_zone = Polygon.from_points(config.opponent_deployment_outline)
    if player_zone is not None:
        player_box = np.array(player_zone.bounds, dtype=float)
    if opponent_zone is not None:
        opponent_box = np.array(opponent_zone.bounds, dtype=float)
    # Hand the resolved outlines to the aggregate so anything reading the
    # battle sees the zone models were placed in, not the scenario's
    # rectangle. Set unconditionally: on a map without deployments this
    # clears the previous episode's zones rather than leaving them.
    battle.set_deployment_outlines(player_zone, opponent_zone)

    # Terrain first: it is the board the rest is placed onto. Models and
    # objectives may sit inside a footprint, exactly as they may with a fixed
    # layout — a model in a ruin can still see out and be seen.
    if layout is not None:
        install_layout(battle, config, layout)
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
            player_box,
            config.group_max_distance,
            rng,
            base_radius=base_radius,
            coherency=config.coherency,
            zone=player_zone,
        )

    # Place objectives.
    #
    # `objectives_on_terrain` is tri-state. `True` means the author asked, so a
    # layout that cannot host the objectives is an error. `None` is the default,
    # which most configs now get without asking: there it defers, both to a
    # config that places its own objectives -- doing otherwise moved every fixed
    # objective onto a ruin and silently changed the scenario -- and to a layout
    # with nothing to stand on.
    #
    # A layout that brought its own has already installed them above: they are
    # part of the map in the same way its ruins are, and re-deriving them from
    # the terrain would discard the placement the table actually uses.
    brought_objectives = layout is not None and layout.objectives is not None
    required = config.objectives_on_terrain is True
    auto = (
        config.objectives_on_terrain is None
        and not config.has_fixed_objective_positions
        and _can_host_objectives(battle, config)
    )
    if brought_objectives:
        pass
    elif required or auto:
        objectives_from_terrain(
            battle.objectives,
            battle.terrain,
            battle.deployment_zone,
            battle.opponent_deployment_zone,
            battle.board_width,
            battle.board_height,
            min_separation=config.objective_min_separation,
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
                opponent_box,
                config.group_max_distance,
                rng,
                base_radius=base_radius,
                coherency=config.coherency,
                zone=opponent_zone,
            )

    # Last, so its draws cannot shift anything else *placed* this episode. They
    # do still shift what `reset` draws afterwards -- `run_until_player_phase`
    # can auto-execute the opponent's first turn, and a scripted policy picks
    # targets off the same `np_random`. So a firing augmentation changes that
    # episode's opponent rolls too; harmless, but it is why the non-firing half
    # of a partial-probability run is not a matched control.
    # A fixed-placement config exists to pin an exact layout, so the
    # augmentation must not silently discard it.
    if (
        augment_start
        and config.start_on_objective_probability > 0.0
        and not config.has_fixed_model_positions
    ):
        if float(rng.random()) < config.start_on_objective_probability:
            start_group_on_objective(
                battle.player_models,
                battle.opponent_models,
                battle.objectives,
                rng,
                base_radius=base_radius,
                engagement_range=resolve_rules_quantities(config).engagement_range,
            )
