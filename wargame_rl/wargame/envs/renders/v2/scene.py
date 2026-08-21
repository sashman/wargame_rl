"""The Scene: a backend- and source-independent description of one frame.

Primitives are plain dataclasses holding board-unit *positions* and pixel-space
*sizes* (stroke widths, radii-in-fallback, font sizes). A backend rasterises
them; a `Camera` maps positions to pixels. `build_scene` reads only the
read-only `BattleView` surface, so a snapshot-driven builder (Phase 3) can emit
the identical `Scene` type without an env.

Sizes are pixels because the legacy renderer's legibility floors (`max(3, ...)`,
the third-of-a-cell dead token) are pixel decisions; positions and base radii
stay in board units so the `Camera` owns the board→pixel scale. `build_scene`
therefore takes the current `scale` to resolve those pixel sizes.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, RGB, RGBA, Theme
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, PlayerSide

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.renders.v2.control import LosResult, ShadowRect


class Control(enum.Enum):
    """Which side holds an objective. Values match snapshot `objective_control`."""

    NEUTRAL = "none"
    PLAYER = "player"
    OPPONENT = "opponent"


# --- primitives -------------------------------------------------------------


@dataclass(frozen=True)
class Disc:
    """Filled/outlined circle. `radius` is in board units; the rest are pixels."""

    center: tuple[float, float]
    radius: float
    fill: RGBA | None
    outline: RGB | None
    outline_w: int


@dataclass(frozen=True)
class Poly:
    """Filled/outlined polygon in board-unit vertices."""

    points: tuple[tuple[float, float], ...]
    fill: RGBA | None
    outline: RGB | None
    outline_w: int


@dataclass(frozen=True)
class Seg:
    """A single line segment, board-unit endpoints, pixel width."""

    a: tuple[float, float]
    b: tuple[float, float]
    color: RGB
    width: int


@dataclass(frozen=True)
class Label:
    """Centred text at a board-unit anchor, pixel font size."""

    text: str
    center: tuple[float, float]
    size: int
    color: RGB


Primitive = Disc | Poly | Seg | Label


# Short labels for the phase chain — five slots that must fit side by side.
# Frames a volley stays on the board after it resolves. It fades rather than
# vanishing: at 4fps a single-frame flash is gone before it can be read, and a
# tracer that persists unchanged makes a movement frame look like a firefight.
SHOT_FADE_FRAMES = 4

_PHASE_LABELS: dict[BattlePhase, str] = {
    BattlePhase.command: "CMD",
    BattlePhase.movement: "MOVE",
    BattlePhase.shooting: "SHOOT",
    BattlePhase.charge: "CHRG",
    BattlePhase.fight: "FIGHT",
}


@dataclass(frozen=True)
class PhaseChip:
    """One phase of the round, and how the HUD should render it.

    A skipped phase is auto-advanced by the config, so it never becomes current;
    showing it dimmed makes `skip_phases` visible rather than implied.
    """

    label: str
    is_current: bool
    is_skipped: bool


@dataclass(frozen=True)
class HudData:
    """Scalar state the presenter draws into the panels."""

    round: int
    n_rounds: int
    phase: str
    phase_chips: tuple[PhaseChip, ...]
    step: int
    reward: float | None
    episode_reward: float | None
    # Last step's reward components, in the phase's calculator order.
    reward_breakdown: tuple[tuple[str, float], ...]
    player_vp: int
    player_vp_delta: int
    opponent_vp: int
    opponent_vp_delta: int
    # Objective control (one entry per objective) + the held tally.
    objective_controls: tuple[Control, ...]
    held_player: int
    held_opponent: int
    # Force strength per side (alive of total).
    player_alive: int
    player_total: int
    opponent_alive: int
    opponent_total: int
    # Whether the player takes the first turn of each round, or the opponent
    # does. None when the game is over and no side is active.
    #
    # Turn *order*, not "whose turn is it now" — that question has one answer in
    # every frame ever drawn. The opponent's whole turn executes inside the
    # player's `step()`, so the clock is always parked back on a player phase by
    # the time a frame is composed. Which side the player occupies is the part
    # that actually varies: under `turn_order: random` it is re-rolled every
    # reset, and it decides whether the positions on screen already include the
    # opponent's response.
    player_acts_first: bool | None = None
    # Steps currently available to rewind. None for presenters that cannot
    # rewind at all (recording, replay), which then draw nothing.
    #
    # Shown as *state* rather than reported as an event: "nothing to step back
    # to" is worth knowing before pressing the key, not after — and a log line
    # about it lands in a terminal nobody is looking at while the window has
    # their attention.
    undo_depth: int | None = None


@dataclass(frozen=True)
class Scene:
    """One frame: board size, background, ordered primitives, HUD data."""

    board_width: int
    board_height: int
    board_bg: RGB
    primitives: tuple[Primitive, ...]
    hud: HudData


# --- builders ---------------------------------------------------------------


def _faded(color: RGB) -> RGB:
    """Legacy movement-arrow fade: halve the distance from each channel to 255."""
    return (
        color[0] + (255 - color[0]) // 2,
        color[1] + (255 - color[1]) // 2,
        color[2] + (255 - color[2]) // 2,
    )


def _base_radius(model: object, fallback: float = 1.0 / 3.0) -> float:
    """Board-unit radius: the real base when present, else a third-cell token."""
    radius = float(getattr(model, "base_radius", 0.0))
    return radius if radius > 0.0 else fallback


def _dead_marker(cx: float, cy: float, color: RGB) -> list[Seg]:
    """The two-stroke X drawn over a dead model (half-length 0.25 board units)."""
    xr = 0.25
    return [
        Seg((cx - xr, cy - xr), (cx + xr, cy + xr), color, 2),
        Seg((cx + xr, cy - xr), (cx - xr, cy + xr), color, 2),
    ]


def _arrow(
    prims: list[Primitive],
    models: Sequence[object],
    color_of: Callable[[int], RGB],
    scale: float,
) -> None:
    """Emit a faded shaft + filled head for each model that moved."""
    for model in models:
        if not getattr(model, "is_alive", True):
            continue
        prev = getattr(model, "previous_location", None)
        if prev is None:
            continue
        curr = model.location  # type: ignore[attr-defined]
        px, py = float(prev[0]), float(prev[1])
        cx, cy = float(curr[0]), float(curr[1])
        if px == cx and py == cy:
            continue
        faded = _faded(color_of(model.group_id))  # type: ignore[attr-defined]
        prims.append(Seg((px, py), (cx, cy), faded, max(3, int(scale / 4))))

        dx, dy = cx - px, cy - py
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        ux, uy = dx / length, dy / length
        head_len = min(0.45, length * 0.4)
        head_w = head_len * 0.5
        left = (cx - ux * head_len - uy * head_w, cy - uy * head_len + ux * head_w)
        right = (cx - ux * head_len + uy * head_w, cy - uy * head_len - ux * head_w)
        prims.append(Poly(((cx, cy), left, right), (*faded, 255), None, 0))


def _draw_models(
    prims: list[Primitive],
    models: Sequence[object],
    color_of: Callable[[int], RGB],
    theme: Theme,
) -> None:
    """Draw each model as a circular base (real bases are round); the side is the
    colour, so both players and opponents are discs."""
    pal = theme.palette
    for model in models:
        cx, cy = float(model.location[0]), float(model.location[1])  # type: ignore[attr-defined]
        radius = _base_radius(model)
        if not getattr(model, "is_alive", True):
            prims.append(Disc((cx, cy), radius, (*pal.dead_fill, 255), None, 0))
            prims.extend(_dead_marker(cx, cy, pal.dead_mark))
            continue
        color = color_of(model.group_id)  # type: ignore[attr-defined]
        prims.append(Disc((cx, cy), radius, (*color, 255), pal.model_rim, 1))


def _shot_endpoints(
    attacker: object, target: object
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Base-edge to base-edge, so a line starts and ends at the models, not in
    them — with 25 shooters the overlap is what makes a volley unreadable."""
    ax, ay = float(attacker.location[0]), float(attacker.location[1])  # type: ignore[attr-defined]
    tx, ty = float(target.location[0]), float(target.location[1])  # type: ignore[attr-defined]
    dx, dy = tx - ax, ty - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (ax, ay), (tx, ty)
    ux, uy = dx / length, dy / length
    start = _base_radius(attacker), _base_radius(target)
    return (ax + ux * start[0], ay + uy * start[0]), (
        tx - ux * start[1],
        ty - uy * start[1],
    )


def _toward(color: RGB, ground: RGB, strength: float) -> RGB:
    """`color` blended `strength` of the way from the background toward itself."""
    return (
        int(ground[0] + (color[0] - ground[0]) * strength),
        int(ground[1] + (color[1] - ground[1]) * strength),
        int(ground[2] + (color[2] - ground[2]) * strength),
    )


def _draw_shots(
    prims: list[Primitive],
    results: Sequence[object],
    attackers: Sequence[object],
    targets: Sequence[object],
    color: RGB,
    theme: Theme,
    scale: float,
    fade: float,
) -> None:
    """A line per shot that did damage, with an impact ring on the target.

    Only damaging shots are drawn. A volley here is 21 shots of which 4 land,
    and drawing the misses — full length or as stubs — buries the four that
    matter. What the board answers is "who hit whom, and how hard"; the HUD's
    reward ledger is where the cost of firing and missing shows up.

    Drawn over the models, so a volley reads as traffic between two bodies
    rather than as marks on the board.
    """
    pal = theme.palette
    for entry in results:
        attacker_idx = int(entry.attacker_idx)  # type: ignore[attr-defined]
        target_idx = int(entry.target_idx)  # type: ignore[attr-defined]
        if attacker_idx >= len(attackers) or target_idx >= len(targets):
            continue
        outcome = entry.result  # type: ignore[attr-defined]
        damage = int(getattr(outcome, "damage_dealt", 0))
        if damage <= 0:
            continue
        killed = bool(getattr(entry, "killed", False))
        start, end = _shot_endpoints(attackers[attacker_idx], targets[target_idx])
        width = max(2, int(scale * 0.10)) if killed else max(1, int(scale * 0.06))
        ring = pal.shot_kill if killed else color
        prims.append(Seg(start, end, _toward(color, pal.board_bg, fade), width))
        # An impact ring scaled by the damage that landed, so focus fire on one
        # model reads without counting lines.
        radius = 0.22 + 0.10 * min(damage, 4)
        prims.append(Disc(end, radius, None, _toward(ring, pal.board_bg, fade), width))


def _reward_components(view: "BattleView") -> tuple[tuple[str, float], ...]:
    """The step's reward components: one entry per calculator, nothing nested.

    Calculators may also report their internals under `parent/child` keys, which
    sum into the parent — charting both would make a term compete with its own
    breakdown and double its weight in the composition bar.

    Order is the phase's declaration order, not magnitude: the dict is rebuilt
    from the same calculator list every step, so keeping it holds each term in
    the same slot of the bar frame after frame instead of letting segments swap
    places whenever two components cross.
    """
    return tuple(
        (name, value)
        for name, value in view.last_reward_breakdown.items()
        if "/" not in name
    )


def _phase_chips(view: "BattleView") -> tuple[PhaseChip, ...]:
    """The round's five phases, marking the current one and the skipped ones."""
    skipped = set(view.config.skip_phases)
    current = view.game_clock_state.phase
    return tuple(
        PhaseChip(
            label=label,
            is_current=phase == current,
            is_skipped=phase in skipped,
        )
        for phase, label in _PHASE_LABELS.items()
    )


def player_acts_first(active: PlayerSide | None) -> bool | None:
    """Whether the player takes the first turn of each round.

    Reads the *active* side as the player's own, which is sound only because of
    when a frame is drawn: `run_after_player_action` runs the opponent's entire
    turn inside the player's `step()`, so the clock has always come back round to
    a player phase before anything is composed. `player_1` is the side that acts
    first, so the player acting first is exactly the player holding `player_1`.

    None once the game is over, when no side is active.
    """
    if active is None:
        return None
    return active is PlayerSide.player_1


def shot_fade_for_age(age: int) -> float:
    """How strongly a volley `age` frames old is drawn, 1.0 fresh to 0.0 gone."""
    if age < 0 or age >= SHOT_FADE_FRAMES:
        return 0.0
    return 1.0 - age / SHOT_FADE_FRAMES


def build_scene(
    view: "BattleView",
    control: Sequence[Control],
    *,
    scale: float,
    theme: Theme = DEFAULT_THEME,
    debug_los: "LosResult | None" = None,
    los_shadow: Sequence["ShadowRect"] = (),
    show_grid: bool = True,
    shot_fade: float = 1.0,
) -> Scene:
    """Assemble the ordered primitives for one live frame from a `BattleView`."""
    pal = theme.palette
    prims: list[Primitive] = []
    board_w = view.config.board_width
    board_h = view.config.board_height

    # Deployment zones (drawn first, under everything). A map brings its own
    # outline where it has one -- only two of the six real deployments are an
    # axis-aligned band, so drawing the rectangle on the others puts the tint
    # somewhere the army is not.
    for zone, outline, color, label in (
        (
            view.deployment_zone,
            view.deployment_outline,
            pal.deployment_zone,
            "Deployment Zone",
        ),
        (
            view.opponent_deployment_zone,
            view.opponent_deployment_outline,
            pal.opponent_zone,
            "Opponent Zone",
        ),
    ):
        if outline is not None:
            points = tuple((float(x), float(y)) for x, y in outline.vertices)
            prims.append(Poly(points, (*color, 255), None, 0))
            x0, y0, x1, y1 = outline.bounds
        else:
            x0, y0, x1, y1 = (
                float(zone[0]),
                float(zone[1]),
                float(zone[2]),
                float(zone[3]),
            )
            prims.append(
                Poly(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), (*color, 255), None, 0)
            )
        cx = max(x0 + (x1 - x0) / 2, x0)
        cy = max(y0 + (y1 - y0) / 2, y0)
        prims.append(Label(label, (cx, cy), 48, pal.zone_label))

    # Grid is a substrate: above the (opaque) zone fills so it shows in the
    # deployment areas, but below terrain, objectives and models.
    if show_grid:
        for y in range(board_h + 1):
            prims.append(Seg((0.0, float(y)), (float(board_w), float(y)), pal.grid, 1))
        for x in range(board_w + 1):
            prims.append(Seg((float(x), 0.0), (float(x), float(board_h)), pal.grid, 1))

    # Terrain: real outline + translucent fill, "OBJECTIVE" when it is one.
    objective_outlines = {
        obj.area.vertices.tobytes() for obj in view.objectives if obj.area is not None
    }
    terrain_font = max(16, int(scale * 0.8))
    for fp in view.terrain.footprints:
        points = tuple((float(x), float(y)) for x, y in fp.polygon.vertices)
        prims.append(Poly(points, pal.terrain_fill, pal.terrain_outline, 2))
        is_objective = fp.polygon.vertices.tobytes() in objective_outlines
        centroid = fp.polygon.centroid
        prims.append(
            Label(
                "OBJECTIVE" if is_objective else "Ruin",
                (float(centroid[0]), float(centroid[1])),
                terrain_font,
                pal.terrain_label,
            )
        )

    # Objectives with ownership fill / wash.
    base_width = max(2, int(round(scale / 8)))
    for i, objective in enumerate(view.objectives):
        ctrl = control[i] if i < len(control) else Control.NEUTRAL
        fill_rgb = (
            pal.player_control
            if ctrl is Control.PLAYER
            else pal.opponent_control
            if ctrl is Control.OPPONENT
            else None
        )
        if objective.area is not None:
            wash = fill_rgb if fill_rgb is not None else pal.objective_rim
            points = tuple((float(x), float(y)) for x, y in objective.area.vertices)
            prims.append(
                Poly(
                    points,
                    (*wash, pal.area_wash_alpha),
                    pal.objective_rim,
                    max(3, int(scale / 6)),
                )
            )
            continue
        cx, cy = float(objective.location[0]), float(objective.location[1])
        radius_board = float(objective.radius_size)
        radius_px = max(1, int(round(radius_board * scale)))
        rim_width = min(base_width, max(1, radius_px // 2))
        fill_rgba = (*fill_rgb, 255) if fill_rgb is not None else None
        prims.append(
            Disc((cx, cy), radius_board, fill_rgba, pal.objective_rim, rim_width)
        )

    # Sight shadow: over the board and everything static on it, under the models.
    # It shades terrain and objectives because they *are* hidden — an objective
    # nobody can see is the interesting case — but leaving the models on top
    # keeps the pieces readable inside their own shadow.
    for x0, y0, x1, y1 in los_shadow:
        prims.append(
            Poly(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), pal.los_shadow, None, 0)
        )

    # Movement arrows, then models (opponents under players, matching legacy).
    _arrow(prims, view.player_models, theme.player_color, scale)
    if view.opponent_models:
        _arrow(prims, view.opponent_models, theme.opponent_color, scale)
        _draw_models(prims, view.opponent_models, theme.opponent_color, theme)
    _draw_models(prims, view.player_models, theme.player_color, theme)

    # Shots last, over the models: a volley is traffic between two bodies, and
    # under them it would read as marks on the board instead.
    if shot_fade > 0.0:
        _draw_shots(
            prims,
            view.last_player_shooting_results,
            view.player_models,
            view.opponent_models,
            pal.shot_player,
            theme,
            scale,
            shot_fade,
        )
        _draw_shots(
            prims,
            view.last_opponent_shooting_results,
            view.opponent_models,
            view.player_models,
            pal.shot_opponent,
            theme,
            scale,
            shot_fade,
        )

    if debug_los is not None:
        prims.append(
            Seg(
                debug_los.a,
                debug_los.b,
                pal.los_clear if debug_los.clear else pal.los_blocked,
                max(1, int(scale * 0.08)),
            )
        )

    controls = tuple(
        control[i] if i < len(control) else Control.NEUTRAL
        for i in range(len(view.objectives))
    )
    player_models = view.player_models
    opponent_models = view.opponent_models
    clock = view.game_clock_state
    hud = HudData(
        round=clock.battle_round or 0,
        n_rounds=view.n_rounds,
        phase=clock.phase.value.title() if clock.phase else "—",
        phase_chips=_phase_chips(view),
        step=view.current_turn,
        reward=view.last_reward,
        episode_reward=view.episode_reward,
        reward_breakdown=_reward_components(view),
        player_vp=view.player_vp,
        player_vp_delta=view.player_vp_delta,
        opponent_vp=view.opponent_vp,
        opponent_vp_delta=view.opponent_vp_delta,
        objective_controls=controls,
        held_player=sum(1 for c in controls if c is Control.PLAYER),
        held_opponent=sum(1 for c in controls if c is Control.OPPONENT),
        player_alive=sum(1 for m in player_models if getattr(m, "is_alive", True)),
        player_total=len(player_models),
        opponent_alive=sum(1 for m in opponent_models if getattr(m, "is_alive", True)),
        opponent_total=len(opponent_models),
        player_acts_first=player_acts_first(clock.active_player),
    )
    return Scene(board_w, board_h, pal.board_bg, tuple(prims), hud)
