"""Generate the evaluation tables from the public layout Data API.

The 45 tables in `configs/evaluation/maps/` were traced by hand from this same
source, and the tracing lost things. Outlines became quads; the objectives were
picked by eye for board symmetry rather than read off the layout -- on
`table_01` only two of the layout's five markers land inside an objective it
declares. This regenerates both from the data, so a table is what the layout
says it is and a new table costs a re-run rather than an afternoon of tracing.

Nothing from the API's vocabulary crosses this boundary. Layout slugs,
deployment names and mission-pack ids are the commercial product's names, and
`tests/test_no_ip_references.py` keeps them out of the repo; only geometry is
read, and the raw responses are never written to disk.

Usage: just fetch-maps [owner] [maps_dir]
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from wargame_rl.wargame.envs.domain.rules_constants import OBJECTIVE_MARKER_RANGE_IN
from wargame_rl.wargame.envs.types.geometry import Polygon
from wargame_rl.wargame.envs.types.terrain_observation import TERRAIN_VERTEX_BUDGET

API_BASE = "https://battlemaster.online/v1/public/data"
DEFAULT_OWNER = "superwutz"
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps")

# The board this project plays on. Layouts authored for any other size are
# skipped rather than scaled: a smaller table is a different scenario, not the
# same one at a different zoom.
BOARD_WIDTH_IN = 60.0
BOARD_HEIGHT_IN = 44.0

# Coordinates arrive with the origin at the centre of the board and y up. Ours
# has the origin at the corner, so every point shifts by half the board.
ORIGIN_SHIFT = (BOARD_WIDTH_IN / 2, BOARD_HEIGHT_IN / 2)

# Two pieces sharing at least this much boundary are one ruin. The layouts
# build a structure out of several kit pieces -- a rectangle split along a
# diagonal seam, or two bars meeting in an L -- and the source renders each
# such group as a single connected blob. Measured over all 45 tables, the
# shared-boundary lengths are strikingly discrete: 110 pairs at ~0.32in (a
# corner touch, incidental), then 84 at ~2.33 (a bar's end against another's
# side), 6 at ~6.33 and 18 at ~13.6 (one rectangle split in two). 1.0 sits in
# the empty gap between 0.33 and 1.45, 3x above the first mode and 2.3x below
# the second.
RUIN_CONTACT_IN = 1.0

# Boundary sampling for that measurement. `eps` has to exceed the two-decimal
# rounding the map files carry, or a shared edge reads as a near miss.
_CONTACT_EPS_IN = 0.15
_CONTACT_STEP_IN = 0.08

Point = tuple[float, float]


def fetch_bundle(owner: str) -> dict[str, Any]:
    """Every layout and deployment for one owner, in a single request.

    The bundle endpoint exists for exactly this: the catalogs *with* their
    details, so a full regeneration costs one round trip instead of ninety.
    """
    url = f"{API_BASE}/bundle?owner={owner}"
    request = urllib.request.Request(url, headers={"User-Agent": "wargame-rl"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload: dict[str, Any] = json.load(response)
    if payload.get("format") != "battlemaster.data.bundle":
        raise ValueError(f"unexpected payload format {payload.get('format')!r}")
    return payload


def piece_outline(piece: dict[str, Any]) -> list[Point]:
    """One terrain piece's silhouette, in board coordinates.

    Outline points are piece-local and **centred** on the footprint origin, so
    the piece frame is rotated about its own middle and then translated. That
    convention is not documented; it was established by measurement, and the
    test is decisive -- centred puts all 720 pieces on the board with zero
    overhang, while treating the points as corner-relative throws 70 of them off
    the edge by up to 3.75 inches.
    """
    footprint = piece["footprint"]
    origin_x, origin_y = footprint["origin"]["x"], footprint["origin"]["y"]
    angle = math.radians(footprint.get("rotationDeg") or 0.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    shift_x, shift_y = ORIGIN_SHIFT
    points = (piece.get("outline") or {}).get("points") or []
    return [
        (
            origin_x + p["x"] * cos_a - p["y"] * sin_a + shift_x,
            origin_y + p["x"] * sin_a + p["y"] * cos_a + shift_y,
        )
        for p in points
    ]


def _furthest_from_chord(
    points: list[Point], first: int, last: int
) -> tuple[int, float]:
    """The vertex furthest from the line through two others, and how far."""
    ax, ay = points[first]
    bx, by = points[last]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    best_index, best_distance = first, -1.0
    for index in range(first + 1, last):
        px, py = points[index]
        distance = (
            abs(dx * (ay - py) - (ax - px) * dy) / length
            if length
            else math.hypot(px - ax, py - ay)
        )
        if distance > best_distance:
            best_index, best_distance = index, distance
    return best_index, best_distance


def _douglas_peucker(points: list[Point], tolerance: float) -> list[Point]:
    """Drop vertices that sit within `tolerance` of the line they lie on."""

    def keep_between(first: int, last: int) -> list[int]:
        if last <= first + 1:
            return []
        index, distance = _furthest_from_chord(points, first, last)
        if distance <= tolerance:
            return []
        return keep_between(first, index) + [index] + keep_between(index, last)

    kept = [0, *keep_between(0, len(points) - 1), len(points) - 1]
    return [points[index] for index in kept]


def simplify_outline(points: list[Point], max_vertices: int) -> list[Point]:
    """Reduce a dense silhouette to at most `max_vertices`, keeping its shape.

    Mandatory, not an optimisation: every source outline carries 167 to 348
    vertices against an observation budget of 8, which is why the hand-traced
    tables were quads.

    Douglas-Peucker rather than the alternatives, chosen by measuring all 720
    pieces against their true area. It holds a median 1.027 and a worst 1.078;
    a convex hull reaches 1.160 because it fills in every concave bay, and the
    footprint rectangle has a 1.592 tail -- an angled or L-shaped ruin blocks
    half again as much board as it should. The tolerance is found by bisection
    because the vertex count, not the tolerance, is what the budget constrains.
    """
    ring = [*points, points[0]]
    low, high = 0.0, max(BOARD_WIDTH_IN, BOARD_HEIGHT_IN)
    for _ in range(40):
        middle = (low + high) / 2
        if len(_douglas_peucker(ring, middle)) - 1 <= max_vertices:
            high = middle
        else:
            low = middle
    simplified = _douglas_peucker(ring, high)[:-1]
    if len(simplified) < 3:
        raise ValueError(f"outline simplified to {len(simplified)} vertices")
    return simplified


def shared_boundary(first: Polygon, second: Polygon) -> float:
    """How much of the two outlines runs together, in inches.

    Measured by walking one boundary and asking the engine's own
    `distance_to_point` how close the other is, then taking the larger of the
    two directions -- a short piece lying along a long one shares all of its own
    edge and only part of the other's.
    """

    def along(source: Polygon, other: Polygon) -> float:
        total = 0.0
        vertices = source.vertices
        for index in range(len(vertices)):
            start, end = vertices[index], vertices[(index + 1) % len(vertices)]
            length = float(np.hypot(*(end - start)))
            if length <= 0.0:
                continue
            steps = max(2, int(length / _CONTACT_STEP_IN))
            fractions = np.linspace(0.0, 1.0, steps)
            points = start + fractions[:, np.newaxis] * (end - start)
            near = [
                other.distance_to_point(float(x), float(y)) < _CONTACT_EPS_IN
                for x, y in points
            ]
            total += float(np.mean(near)) * length
        return total

    return max(along(first, second), along(second, first))


def ruin_components(pieces: list[Polygon]) -> list[list[int]]:
    """Group pieces into ruins -- connected runs of substantial shared edge.

    A "terrain piece" in the source is a kit component, not a building: the
    layouts routinely split one rectangle along a diagonal or butt two bars into
    an L, and the source's own board render draws each such group as a single
    blob. Treating the components separately is what made an objective cover
    half a ruin and leave the other half neutral ground.
    """
    parent = list(range(len(pieces)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, first in enumerate(pieces):
        left = first.bounds
        for j in range(i + 1, len(pieces)):
            right = pieces[j].bounds
            # Cheap reject: outlines that cannot touch cannot share an edge.
            if (
                left[2] + _CONTACT_EPS_IN < right[0]
                or right[2] + _CONTACT_EPS_IN < left[0]
                or left[3] + _CONTACT_EPS_IN < right[1]
                or right[3] + _CONTACT_EPS_IN < left[1]
            ):
                continue
            if shared_boundary(first, pieces[j]) >= RUIN_CONTACT_IN:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for index in range(len(pieces)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def merge_ruin(pieces: list[Polygon], members: list[int]) -> list[Point]:
    """One ruin's outline: the union of the pieces it is built from.

    The pieces *abut* rather than overlap -- a diagonal seam or a butted L --
    and two polygons a rounded hundredth apart do not fuse, so a plain union
    returns a MultiPolygon and the ruin stays in halves. Grown by the same
    epsilon the contact measurement uses and then shrunk back, the seam closes
    while every real edge returns to where it was. Mitred joins, because a round
    join would bevel every corner of the outline.

    Exterior ring only. A union can enclose a courtyard, and an objective is the
    ground you stand on to hold it -- a model in the courtyard is on the
    objective, so the hole is not a hole for this purpose.

    No vertex budget applies here, unlike terrain: an objective reaches the
    network as its centroid, never as an outline.
    """
    if len(members) == 1:
        return [(float(x), float(y)) for x, y in pieces[members[0]].vertices]
    grown = [
        ShapelyPolygon([(float(x), float(y)) for x, y in pieces[m].vertices])
        .buffer(0)
        .buffer(_CONTACT_EPS_IN, join_style=2)
        for m in members
    ]
    union = unary_union(grown).buffer(-_CONTACT_EPS_IN, join_style=2)
    if union.geom_type != "Polygon":
        # Never silently take the biggest part: that is how half a ruin went
        # missing while every count still read five.
        raise ValueError(
            f"ruin of pieces {members} did not merge into one outline "
            f"(got {union.geom_type}); the contact threshold and the bridge "
            f"epsilon disagree"
        )
    return [(float(x), float(y)) for x, y in union.exterior.coords[:-1]]


def objectives_for(markers: list[Point], pieces: list[Polygon]) -> list[dict[str, Any]]:
    """Resolve the layout's objective markers against its terrain.

    **An objective is a ruin.** Free-standing markers you stand near are a
    previous edition's rule; here the marker only says *which ground* is being
    fought over, so it resolves to that ruin's outline. Measured across the
    pool, 146 of 225 markers sit inside a piece and 71 more within control
    range.

    **A ruin is not a terrain piece.** The layouts build one structure out of
    several kit pieces -- a rectangle split along a diagonal seam, two bars
    butted into an L -- and the source's own board render draws each group as a
    single blob. Resolving a marker to the nearest *piece* therefore made an
    objective cover half a ruin and leave the other half neutral: on `table_02`
    the centre rectangle came out green on one side of its diagonal and brown on
    the other. `ruin_components` groups them first.

    Two markers on one ruin collapse into a single objective, because that
    ground is held once.

    Eight markers sit further than control range from any piece, at most 5.01
    inches. Those are deliberate open ground rather than bad data: four of them
    are the exact board centre, and the distance 3.00 recurs to the inch, which
    is a marker placed exactly one control range off a ruin. They stay discs, so
    the table keeps the shape the layout gave it.
    """
    components = ruin_components(pieces)
    owner = {
        index: number for number, group in enumerate(components) for index in group
    }
    objectives: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for marker_x, marker_y in markers:
        distances = sorted(
            (piece.distance_to_point(marker_x, marker_y), index)
            for index, piece in enumerate(pieces)
        )
        distance, index = distances[0]
        if distance > OBJECTIVE_MARKER_RANGE_IN:
            objectives.append(
                {
                    "x": marker_x,
                    "y": marker_y,
                    "radius_size": OBJECTIVE_MARKER_RANGE_IN,
                }
            )
        elif owner[index] not in claimed:
            claimed.add(owner[index])
            # Unrounded on purpose. `render_map_yaml` formats every coordinate
            # to two places exactly once; rounding here as well rounds twice by
            # two different rules, and `round(33.455, 2)` and `f"{33.455:.2f}"`
            # disagree -- which silently made an objective's outline differ by a
            # hundredth from the very piece it *is*.
            objectives.append(
                {
                    "area": [
                        [x, y] for x, y in merge_ruin(pieces, components[owner[index]])
                    ]
                }
            )
    return objectives


def _bounds(pieces: list[Polygon]) -> list[tuple[float, float, float, float]]:
    """Each piece's axis-aligned extent, rounded, for comparing two layouts."""
    return sorted(tuple(round(value, 1) for value in p.bounds) for p in pieces)  # type: ignore[misc]


def _shared_pieces(left: list[Polygon], right: list[Polygon], tolerance: float) -> int:
    """How many pieces two layouts have in common, matched greedily by extent."""
    remaining = _bounds(right)
    shared = 0
    for candidate in _bounds(left):
        for index, other in enumerate(remaining):
            if max(abs(a - b) for a, b in zip(candidate, other)) <= tolerance:
                remaining.pop(index)
                shared += 1
                break
    return shared


def assign_names(
    layouts: list[list[Polygon]], existing: dict[str, list[Polygon]]
) -> list[str]:
    """Name each regenerated layout after the table it replaces.

    Identity is the *geometry*, deliberately. The API's own version keys are
    change-detection tokens that move whenever a layout is edited, and its slugs
    are product vocabulary that must not enter the repo -- while the terrain
    itself is the thing that makes `table_07` the table everyone means.

    Keeping the numbering is what makes this change readable: `maps_heldout` is
    nine specific tables and every current baseline was measured on them, so a
    renumbering would silently redefine the held-out set at the same moment the
    geometry changed, and nothing downstream would be comparable to anything.
    """
    scored = sorted(
        (
            (-_shared_pieces(pieces, other, tolerance=1.0), name, index)
            for index, pieces in enumerate(layouts)
            for name, other in existing.items()
        )
    )
    names: dict[int, str] = {}
    taken: set[str] = set()
    for _, name, index in scored:
        if index not in names and name not in taken:
            names[index] = name
            taken.add(name)
    spare = (f"table_{n:02d}" for n in range(1, 100) if f"table_{n:02d}" not in taken)
    return [names.get(index) or next(spare) for index in range(len(layouts))]


def render_map_yaml(
    name: str, pieces: list[Polygon], objectives: list[dict[str, Any]]
) -> str:
    """The map file, in the same shape the hand-written ones used."""
    lines = [
        "# Generated by `just fetch-maps` from the public layout API.",
        "# Edit the generator, not this file -- a re-run overwrites it.",
        "#",
        "# Objectives are terrain pieces: a layout's marker says which ruin is",
        "# fought over, so two markers on one ruin are one objective. A marker",
        "# further than control range from any ruin stays a disc on open ground.",
        f"name: {name}",
        "terrain:",
    ]
    for piece in pieces:
        outline = ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in piece.vertices)
        lines.append(f"  - outline: [{outline}]")
    lines.append("objectives:")
    for objective in objectives:
        if "area" in objective:
            area = ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in objective["area"])
            lines.append(f"  - area: [{area}]")
        else:
            lines.append(f"  - x: {objective['x']:.2f}")
            lines.append(f"    y: {objective['y']:.2f}")
            lines.append(f"    radius_size: {objective['radius_size']}")
    return "\n".join(lines) + "\n"


def _existing_layouts(maps_dir: Path) -> dict[str, list[Polygon]]:
    """The tables already on disk, as geometry, for the naming match."""
    from scripts.measure_maps import load_maps

    if not maps_dir.is_dir():
        return {}
    return {
        terrain_map.name: [piece.to_polygon() for piece in terrain_map.terrain]
        for terrain_map in load_maps(maps_dir)
    }


def convert(bundle: dict[str, Any]) -> list[tuple[list[Polygon], list[dict[str, Any]]]]:
    """Every usable layout in the bundle, as simplified terrain plus objectives."""
    converted = []
    for entry in bundle["layouts"]:
        layout = entry["layout"]
        board = layout["board"]
        deployment = entry.get("deployment")
        if (board["widthIn"], board["heightIn"]) != (BOARD_WIDTH_IN, BOARD_HEIGHT_IN):
            continue
        if not deployment:
            continue
        pieces = [
            Polygon.from_points(simplify_outline(outline, TERRAIN_VERTEX_BUDGET))
            for outline in (piece_outline(piece) for piece in entry["terrain"])
            if outline
        ]
        shift_x, shift_y = ORIGIN_SHIFT
        markers = [
            (o["center"]["x"] + shift_x, o["center"]["y"] + shift_y)
            for o in deployment.get("objectives") or []
        ]
        converted.append((pieces, objectives_for(markers, pieces)))
    return converted


def main() -> None:
    args = sys.argv[1:]
    owner = args[0] if args and args[0] else DEFAULT_OWNER
    maps_dir = Path(args[1]) if len(args) > 1 and args[1] else DEFAULT_MAPS_DIR

    bundle = fetch_bundle(owner)
    print(f"{bundle['layoutCount']} layouts, {bundle['deploymentCount']} deployments")

    converted = convert(bundle)
    skipped = bundle["layoutCount"] - len(converted)
    print(
        f"{len(converted)} usable on {BOARD_WIDTH_IN:.0f}x{BOARD_HEIGHT_IN:.0f}in, {skipped} skipped"
    )

    existing = _existing_layouts(maps_dir)
    names = assign_names([pieces for pieces, _ in converted], existing)

    maps_dir.mkdir(parents=True, exist_ok=True)
    # Keyed on the name alone. Without the key a tie would fall through to
    # comparing the payload, and a `Polygon` has no ordering -- so a duplicate
    # name would raise a TypeError from deep inside `sorted` rather than say so.
    for name, (pieces, objectives) in sorted(zip(names, converted), key=lambda p: p[0]):
        path = maps_dir / f"{name}.yaml"
        matched = _shared_pieces(pieces, existing[name], 1.0) if name in existing else 0
        path.write_text(render_map_yaml(name, pieces, objectives))
        discs = sum(1 for o in objectives if "area" not in o)
        print(
            f"  {name:10s} {len(pieces):2d} pieces  {len(objectives)} objectives"
            f"{f' ({discs} open ground)' if discs else '':17s}"
            f"  {matched}/{len(pieces)} pieces match the table it replaces"
        )

    heldout = maps_dir.parent / "maps_heldout"
    if heldout.is_dir():
        for path in sorted(heldout.glob("*.yaml")):
            source = maps_dir / path.name
            if source.is_file():
                path.write_text(source.read_text())
        print(f"synced {len(list(heldout.glob('*.yaml')))} held-out copies")


if __name__ == "__main__":
    main()
