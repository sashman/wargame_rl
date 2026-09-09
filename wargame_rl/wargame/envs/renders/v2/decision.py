"""The decision overlay: what the per-model facade is about to ask, drawn.

A frame of the phase facade shows a board between two whole-army steps. A
frame of the per-model facade sits between two DECISIONS, and a reader needs
to see which one: which models may be named, which unit is open and locked,
which model is forced to act next, who has already acted. `DecisionHighlight`
is that, as plain data the presenter grafts onto a scene; `build_scene` itself
is untouched, so every frame the phase facade draws is byte-identical.

Reads the decision point structurally (`DecisionLike`), so the renderer does
not import the per-model facade -- the facade is still a leaf of the
application, and `tests/test_per_model_layering.py` keeps it so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from wargame_rl.wargame.envs.renders.v2.scene import Disc, Label, Primitive
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView


class _Named(Protocol):
    @property
    def value(self) -> str: ...


class DecisionLike(Protocol):
    """The shape of a `DecisionPoint`, as this module reads it."""

    @property
    def kind(self) -> _Named: ...
    @property
    def phase(self) -> _Named | None: ...
    @property
    def seat_is_player(self) -> bool: ...
    @property
    def selector_mask(self) -> np.ndarray: ...
    @property
    def acted(self) -> np.ndarray: ...
    @property
    def open_unit(self) -> int | None: ...
    @property
    def forced_model(self) -> int | None: ...


@dataclass(frozen=True)
class DecisionHighlight:
    """One pending decision, ready to draw."""

    kind: str
    phase: str
    seat_is_player: bool
    selectable: tuple[int, ...]
    acted: tuple[int, ...]
    open_unit: int | None
    forced: int | None
    caption: str
    # The decision just taken, in a few characters ("m2 decl 2 r4", "m0 -> u1",
    # "m4 act 17"); folded into the caption.
    detail: str = ""


def highlight_from(
    point: DecisionLike, *, sub_step: int, detail: str = ""
) -> DecisionHighlight:
    """Read a decision point into a highlight, with its HUD caption.

    The caption is kept short enough for the context slot: the phase is
    already the bold chip in the south panel, so it names only the kind, the
    model or the count on offer, the open unit, the sub-step, and -- in a few
    characters -- the decision just taken.
    """
    kind = point.kind.value
    phase = point.phase.value if point.phase is not None else "-"
    selectable = tuple(int(i) for i in np.flatnonzero(point.selector_mask))
    acted = tuple(int(i) for i in np.flatnonzero(point.acted))
    if kind == "close_turn":
        parts = ["close turn"]
    else:
        parts = [kind]
        if point.forced_model is not None:
            parts.append(f"m{point.forced_model}")
        elif selectable:
            parts.append(f"{len(selectable)} sel")
        if point.open_unit is not None:
            parts.append(f"u{point.open_unit} open")
        parts.append(f"sub {sub_step}")
    if detail:
        parts.append(f"last {detail}")
    return DecisionHighlight(
        kind=kind,
        phase=phase,
        seat_is_player=bool(point.seat_is_player),
        selectable=selectable,
        acted=acted,
        open_unit=point.open_unit,
        forced=point.forced_model,
        caption=" · ".join(parts),
        detail=detail,
    )


def _radius(model: object) -> float:
    return float(getattr(model, "base_radius", 1.0 / 3.0)) or 1.0 / 3.0


def decision_primitives(
    view: BattleView, highlight: DecisionHighlight, theme: Theme = DEFAULT_THEME
) -> tuple[Primitive, ...]:
    """The overlay's primitives, drawn above the scene's.

    The open unit's members get a wash; every selectable model a ring; the
    forced model a heavier ring and its index; models that have acted a small
    grey dot. A closing point draws nothing on the board -- the caption says
    it. The opponent's decisions are never pending, so the highlight only
    ever concerns the player's models.
    """
    if highlight.kind == "close_turn" or not highlight.seat_is_player:
        return ()
    pal = theme.palette
    models: Sequence[object] = view.player_models
    prims: list[Primitive] = []
    ring = pal.shot_kill
    wash = (*pal.hud_player, 70)
    if highlight.open_unit is not None:
        for model in models:
            if int(getattr(model, "group_id", -1)) != highlight.open_unit:
                continue
            if not getattr(model, "is_alive", True):
                continue
            x, y = (float(v) for v in getattr(model, "location"))
            prims.append(Disc((x, y), _radius(model) * 2.2, wash, None, 0))
    for index in highlight.acted:
        if index >= len(models):
            continue
        model = models[index]
        if not getattr(model, "is_alive", True):
            continue
        x, y = (float(v) for v in getattr(model, "location"))
        prims.append(
            Disc(
                (x + _radius(model) * 0.9, y - _radius(model) * 0.9),
                0.18,
                (*pal.dead_mark, 255),
                None,
                0,
            )
        )
    for index in highlight.selectable:
        if index >= len(models):
            continue
        model = models[index]
        x, y = (float(v) for v in getattr(model, "location"))
        forced = highlight.forced == index
        prims.append(Disc((x, y), _radius(model) * 1.6, None, ring, 4 if forced else 2))
        if forced:
            prims.append(Label(str(index), (x, y - _radius(model) * 2.6), 12, ring))
    return tuple(prims)
