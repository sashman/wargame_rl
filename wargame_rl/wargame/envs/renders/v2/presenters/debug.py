"""A presenter whose stepping the *caller* owns, with a model inspector beside it.

Two things distinguish this from `InteractiveRenderer`, which it subclasses.

**Who owns the pause.** `InteractiveRenderer` pauses by spinning inside its own
`render()`, so the loop driving the match never learns a pause happened and
cannot be asked to advance one step. Stepping is therefore not a keybinding but
a change of ownership: the driver owns the pause, the presenter only reports
what was pressed. `Renderer.render` stays `-> None` and its ABC gains nothing —
the comment on `FrameSource` warns against adding abstract methods — so the
report travels through a shared mutable `DebugControls`.

**The side panel.** Click any model on either side and a fixed column shows what
it did this step: where it *meant* to go against where it ended up, what it
earned, what it shot and what shot it, and the row the network actually received
for it. The panel is reserved out of the window width rather than drawn over the
board, because a debugger that hides the thing being debugged is worse than a
narrower board.

The layout was picked from a mockup built on one real recorded step, which is
where three of the panel's rules came from: a dead model's reward of 0.000 needs
its reason printed beside it, per-calculator numbers are army means and must say
so, and the intended-versus-actual gap is the one field nothing else in the
renderer shows.

**The sight shading.** `S` shades the board the selected model cannot see, by
sampling the engine's own predicate on a grid rather than projecting terrain
silhouettes — a shadow the renderer computed for itself could not disagree with
the engine, and a disagreement is the bug worth finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pygame

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.renders.v2.backend import Canvas, RenderBackend
from wargame_rl.wargame.envs.renders.v2.control import (
    ShadowRect,
    compute_los_shadow,
    sight_from,
)
from wargame_rl.wargame.envs.renders.v2.debug_view import DebugView
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
    InteractiveRenderer,
)
from wargame_rl.wargame.envs.renders.v2.scene import Scene
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, RGB, Theme
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

# Paused, the window polls far faster than the match plays so a key press lands
# within a frame or two instead of waiting out a 4 fps tick.
PAUSED_POLL_FPS = 30
MIN_FPS = 1
MAX_FPS = 60

PANEL_W = 320
_PAD = 14
_HEAD_SIZE = 11
_ROW_SIZE = 13
_ROW_H = 18
_SECTION_GAP = 13
_NOTE_SIZE = 11
_NOTE_H = 14
# Below this, a displacement is float noise rather than the collision resolver
# having moved the model somewhere it did not ask to go.
_DISPLACED_EPS = 0.05

PLAYER = "player"
OPPONENT = "opponent"


@dataclass(frozen=True)
class MoveRecord:
    """Where a model meant to land on the last movement step, against where it did.

    Built by the session rather than read off the view: the env keeps only the
    *most recent* action, so by the time a shooting step is on screen the
    movement that produced these positions is two phases gone. Decoding it also
    needs the action handler, which the renderer layer may not import.
    """

    text: str
    intended: tuple[float, float]
    actual: tuple[float, float]

    @property
    def gap(self) -> float:
        """How far the model ended from where its own action pointed."""
        return math.dist(self.intended, self.actual)


@dataclass(frozen=True)
class Order:
    """A move a human authored for one model, resolved against the action space.

    Built by the session for the same reason as `MoveRecord`: turning a clicked
    point into an action needs the action handler, and checking it needs the
    action mask — neither of which the renderer layer may reach for.

    `landing` is where the model will *actually* end up, not where the click
    was. The bins are coarse, and that discrepancy is the informative part.
    """

    action: int
    text: str
    landing: tuple[float, float]
    legal: bool = True
    reason: str | None = None


@dataclass
class DebugControls:
    """What the user has asked the match to do next, and what they are looking at.

    The one-shot flags are set by the presenter and cleared by the driver once
    acted on. Only the driver clears them, so a key pressed between two frames
    cannot be silently dropped.
    """

    # Starts paused: a session opens on the first frame and waits, rather than
    # running an episode away before the window has even been looked at.
    paused: bool = True
    step_once: bool = False
    step_back: bool = False
    # ("player" | "opponent", index), or None when nothing is selected.
    selected: tuple[str, int] | None = None
    # Player model index -> its last move. Written by the session.
    moves: dict[int, MoveRecord] = field(default_factory=dict)
    # Steps available to rewind. Written by the session, drawn in the clock zone
    # so an empty history is visible before `,` is pressed rather than after.
    undo_depth: int = 0
    # A click asking for a move: (side, index, board_x, board_y). The presenter
    # can say *where* was clicked but not what action reaches it, so the session
    # resolves this into `orders` and clears it.
    order_at: tuple[str, int, float, float] | None = None
    # (side, index) -> the move authored for it, applied by the next step.
    orders: dict[tuple[str, int], Order] = field(default_factory=dict)
    # Whether a redo rolls fresh dice. Off by default: the rewind restores the
    # combat RNG with everything else, so re-running a step reproduces it
    # exactly, and holding the dice is what isolates the decision from the luck.
    reroll_dice: bool = False


class DebugPresenter(InteractiveRenderer):
    """Drives a live window that never blocks, with an inspector panel."""

    def __init__(
        self,
        backend: RenderBackend,
        controls: DebugControls,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        super().__init__(backend, theme)
        self._controls = controls
        # `view.observation` rebuilds on access and a paused window renders tens
        # of frames a second, so it is read once per step rather than per frame.
        self._obs_turn = -1
        self._obs: object | None = None
        self._show_shadow = False
        # A shadow sweep is thousands of rays; the window redraws tens of times a
        # second while paused and nothing about it changes between those frames.
        self._shadow_key: tuple[object, ...] | None = None
        self._shadow: tuple[ShadowRect, ...] = ()

    # -- control flow --------------------------------------------------------

    def render(self, view: BattleView) -> None:
        """Advance exactly one frame and return."""
        self._pump_and_present(view)

    def _is_paused(self) -> bool:
        # Read through to the controls, so the south panel's PAUSED chip still
        # reports the real state now that the driver owns it.
        return self._controls.paused

    def _present_fps(self) -> int:
        return max(self._fps, PAUSED_POLL_FPS) if self._controls.paused else self._fps

    def _reserved_width(self) -> int:
        return PANEL_W

    def key_map(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Space", "play / pause"),
            (".", "step one forward"),
            (",", "step one back"),
            ("[", "slower"),
            ("]", "faster"),
            ("Click", "a model to inspect it, the board to order it there"),
            ("Enter", "commit the orders and step"),
            ("Backspace", "cancel the orders"),
            ("D", "redo with fresh dice / with the same dice"),
            ("S", "shade what the selected model cannot see"),
            ("L", "line-of-sight debug ray"),
            ("Esc", "deselect, then quit"),
        )

    def _handle_key(self, event: pygame.event.Event, view: BattleView) -> None:
        controls = self._controls
        if event.key == pygame.K_SPACE:
            controls.paused = not controls.paused
        elif event.key in (pygame.K_PERIOD, pygame.K_RIGHT, pygame.K_RETURN):
            # Stepping implies pausing: you asked for one step, not for the match
            # to resume from here. Enter is the same request under the name the
            # authoring flow reaches for — any forward step applies the pending
            # orders, so a click followed by `.` cannot silently discard them.
            controls.step_once = True
            controls.paused = True
        elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            controls.orders.clear()
        elif event.key == pygame.K_d:
            controls.reroll_dice = not controls.reroll_dice
        elif event.key in (pygame.K_COMMA, pygame.K_LEFT):
            controls.step_back = True
            controls.paused = True
        elif event.key == pygame.K_LEFTBRACKET:
            self._fps = max(MIN_FPS, self._fps - 1)
        elif event.key == pygame.K_RIGHTBRACKET:
            self._fps = min(MAX_FPS, self._fps + 1)
        elif event.key == pygame.K_s:
            self._show_shadow = not self._show_shadow
        elif event.key == pygame.K_ESCAPE and controls.selected is not None:
            # Esc backs out of what you are looking at before it quits, so a
            # click cannot be undone only by leaving the session.
            controls.selected = None
        else:
            super()._handle_key(event, view)

    def _handle_click(self, view: BattleView, mx: int, my: int) -> None:
        """A model under the cursor selects it; open ground orders it there.

        One button, no modifier, because the two meanings never overlap: you
        cannot select empty ground, and ordering a model onto another model's
        base is not a move the action space can express anyway. `Esc` is what
        deselects, so nothing is lost by open ground meaning something else.
        """
        hit = self._model_at(view, mx, my)
        if hit is not None:
            self._controls.selected = hit
            return
        point = self._to_board(mx, my)
        if point is None:
            # Off the board entirely — the panel. Deselects, as it did before
            # ordering existed; only *board* clicks changed meaning.
            self._controls.selected = None
            return
        chosen = self._selected(view)
        if chosen is None:
            return
        side, index, _model = chosen
        self._controls.order_at = (side, index, point[0], point[1])

    def _to_board(self, mx: int, my: int) -> tuple[float, float] | None:
        """Window pixels to board units — the inverse of `_to_px`, or None off it."""
        if not (
            self._offset_x <= mx < self._offset_x + self._canvas_w
            and self._offset_y <= my < self._offset_y + self._canvas_h
        ):
            return None
        return (
            (mx - self._offset_x) / self._scale,
            (my - self._offset_y) / self._scale,
        )

    # -- selection -----------------------------------------------------------

    def _model_at(self, view: BattleView, mx: int, my: int) -> tuple[str, int] | None:
        """The model nearest the cursor, on either side, or None.

        Both sides, unlike the tooltip's player-only hit test — half the reason
        to open a debugger is to ask what the *opponent* just did.
        """
        if not (
            self._offset_x <= mx < self._offset_x + self._canvas_w
            and self._offset_y <= my < self._offset_y + self._canvas_h
        ):
            return None
        canvas_x = float(mx - self._offset_x)
        canvas_y = float(my - self._offset_y)
        best: tuple[str, int] | None = None
        best_d2 = max(self._scale / 2, 12.0) ** 2
        sides = ((PLAYER, view.player_models), (OPPONENT, view.opponent_models))
        for side, models in sides:
            for i, model in enumerate(models):
                dx = canvas_x - model.location[0] * self._scale
                dy = canvas_y - model.location[1] * self._scale
                d2 = dx * dx + dy * dy
                if d2 <= best_d2:
                    best, best_d2 = (side, i), d2
        return best

    def _selected(self, view: BattleView) -> tuple[str, int, WargameModel] | None:
        chosen = self._controls.selected
        if chosen is None:
            return None
        side, index = chosen
        models = view.player_models if side == PLAYER else view.opponent_models
        if not 0 <= index < len(models):
            return None
        return side, index, models[index]

    # -- composition ---------------------------------------------------------

    def _compose_with_tooltip(self, view: BattleView) -> Canvas:
        """The frame, plus the inspector column and (if open) the key map.

        Overrides the tooltip pipeline outright: the panel says everything the
        hover tooltip did and says it without covering the board.
        """
        frame = self._compose(view)
        self._draw_orders(frame, view)
        self._draw_selection(frame, view)
        self._draw_inspector(frame, view)
        if self._show_keys:
            self._draw_key_map(frame)
        return frame

    def _draw_orders(self, frame: Canvas, view: BattleView) -> None:
        """Every pending order, as a line to the *true* landing point.

        All of them, not just the selected model's: an authored turn is read as
        a whole, and a model whose order you can no longer see is a model you
        will forget you gave one. An illegal order is drawn too, in the casualty
        colour — silently dropping it would leave the click looking ignored.
        """
        pal = self._theme.palette
        for (side, index), order in self._controls.orders.items():
            models = view.player_models if side == PLAYER else view.opponent_models
            if not 0 <= index < len(models):
                continue
            model = models[index]
            accent = pal.hud_player if side == PLAYER else pal.hud_opponent
            color = accent if order.legal else pal.shot_kill
            start = self._to_px(*model.location)
            end = self._to_px(*order.landing)
            radius = max(model.base_radius * self._scale, 4.0)
            self._backend.draw_line(frame, start, end, color, 2)
            self._backend.draw_disc(frame, end, radius, None, color, 2)
            # A cross-hair at the landing point, so a short order still reads as
            # a destination rather than as a stray ring among the bases.
            self._backend.draw_line(
                frame, (end[0] - radius, end[1]), (end[0] + radius, end[1]), color, 1
            )
            self._backend.draw_line(
                frame, (end[0], end[1] - radius), (end[0], end[1] + radius), color, 1
            )

    def _to_px(self, x: float, y: float) -> tuple[float, float]:
        """Board units to window pixels — the inverse of the click hit test."""
        return (self._offset_x + x * self._scale, self._offset_y + y * self._scale)

    def _draw_selection(self, frame: Canvas, view: BattleView) -> None:
        """Ring the selected model, and ghost where its last move was aimed.

        Drawn onto the frame rather than pushed into the `Scene`: selection is a
        property of who is looking, not of the battle, and `scene.py` is shared
        with the recorder and the replay adapter, neither of which has a cursor.
        """
        chosen = self._selected(view)
        if chosen is None:
            return
        side, index, model = chosen
        pal = self._theme.palette
        accent = pal.hud_player if side == PLAYER else pal.hud_opponent
        # A halo rather than a tight outline: the accent is the model's own side
        # colour, so a thin ring on a same-coloured base is invisible in a crowd.
        centre = self._to_px(*model.location)
        radius = max(model.base_radius * self._scale, 5.0) + 8.0
        self._backend.draw_disc(frame, centre, radius + 2, None, pal.model_rim, 1)
        self._backend.draw_disc(frame, centre, radius, None, accent, 3)
        move = self._controls.moves.get(index) if side == PLAYER else None
        if move is None or move.gap <= _DISPLACED_EPS:
            return
        # Where the action pointed, against where the model ended up. The line is
        # the whole point: it is the only place collision displacement is visible.
        ghost = self._to_px(*move.intended)
        self._backend.draw_line(frame, ghost, self._to_px(*move.actual), accent, 1)
        self._backend.draw_disc(
            frame, ghost, max(model.base_radius * self._scale, 4.0), None, accent, 1
        )

    def _scene_for(self, view: BattleView) -> Scene:
        """The shared scene, plus the one HUD field only the driver knows.

        `build_scene` reads a `BattleView`, and how far back the session can
        rewind is not battle state — it is a property of the tool holding the
        history. So it is grafted on here rather than threaded through a builder
        that the recorder and the replay adapter also use.
        """
        scene = super()._scene_for(view)
        return replace(
            scene, hud=replace(scene.hud, undo_depth=self._controls.undo_depth)
        )

    def _los_shadow(self, view: BattleView) -> tuple[ShadowRect, ...]:
        """What the selected model cannot see, recomputed only when it changes.

        Keyed on the observer's position and on the terrain rather than on the
        turn, because a shadow depends on exactly those two — models do not
        occlude. So a step that moved only *other* models reuses the sweep, and
        an episode that regenerates terrain without moving anyone still gets a
        fresh one.
        """
        chosen = self._selected(view)
        if not self._show_shadow or chosen is None:
            return ()
        _side, _index, model = chosen
        origin = (float(model.location[0]), float(model.location[1]))
        key = (origin, view.terrain.outlines.tobytes())
        if key != self._shadow_key:
            self._shadow_key = key
            self._shadow = compute_los_shadow(view, origin)
        return self._shadow

    def _observation(self, view: BattleView) -> object | None:
        if not isinstance(view, DebugView):
            return None
        if view.current_turn != self._obs_turn or self._obs is None:
            self._obs_turn = view.current_turn
            self._obs = view.observation
        return self._obs

    # -- the panel -----------------------------------------------------------

    def _draw_inspector(self, frame: Canvas, view: BattleView) -> None:
        """Draw the reserved column: one model's step, or a prompt to pick one."""
        pal = self._theme.palette
        x0 = self._window_w - PANEL_W
        self._backend.fill_rect(
            frame, (x0, self._top_h, PANEL_W, self._canvas_h), pal.panel_bg
        )
        self._backend.draw_line(
            frame,
            (x0, self._top_h),
            (x0, self._top_h + self._canvas_h),
            pal.panel_line,
            1,
        )
        chosen = self._selected(view)
        if chosen is None:
            self._text(
                frame,
                "click a model",
                (x0 + PANEL_W // 2, self._top_h + 26),
                _ROW_SIZE,
                self._dim(),
            )
            return

        side, index, model = chosen
        cursor = _Cursor(self, frame, x0 + _PAD, self._top_h + _PAD, PANEL_W - 2 * _PAD)
        self._identity(cursor, side, index, model)
        self._orders(cursor, view)
        self._action(cursor, side, index, model)
        self._reward(cursor, view, side, index, model)
        self._sight(cursor, view, side, model)
        self._shooting(cursor, view, side, index)
        self._observation_rows(cursor, view, side, index)

    def _identity(
        self, cursor: _Cursor, side: str, index: int, model: WargameModel
    ) -> None:
        pal = self._theme.palette
        accent = pal.hud_player if side == PLAYER else pal.hud_opponent
        cursor.head(f"MODEL {index} · {side.upper()} · GROUP {model.group_id}", accent)
        cursor.row("position", f"{model.location[0]:.2f}, {model.location[1]:.2f}")
        cursor.row(
            "wounds",
            f"{model.stats['current_wounds']}/{model.stats['max_wounds']}",
            None if model.is_alive else pal.shot_kill,
        )
        cursor.row("T / Sv", f"{model.stats['toughness']} / {model.stats['save']}+")
        cursor.row("base radius", f'{model.base_radius:.2f}"')
        if not model.is_alive:
            cursor.note("Killed — every field below is its final state.")

    @staticmethod
    def _orders_caption(view: BattleView) -> str:
        """The ORDERS head, which doubles as the only warning that gets read.

        Ordering is movement-phase-only, and a real session clicked into a
        shooting phase **thirty times in ninety seconds** — each refused
        correctly, each explained to a terminal nobody was watching. Same
        mistake as the rewind depth before it: the answer to "will this click do
        anything" belongs on screen *before* the click, not in a log after it.
        """
        if view.game_clock_state.phase is BattlePhase.movement:
            return "ORDERS · PENDING"
        return "ORDERS · MOVEMENT PHASE ONLY"

    def _orders(self, cursor: _Cursor, view: BattleView) -> None:
        """Pending orders and the dice mode, above everything the *last* step did.

        High in the panel on purpose: it is the only section describing what has
        not happened yet, and the one the authoring flow needs to check before
        pressing a key. Outside the movement phase it draws even when empty —
        that emptiness is exactly what needs explaining.
        """
        pal = self._theme.palette
        orders = self._controls.orders
        movement = view.game_clock_state.phase is BattlePhase.movement
        if not orders and not self._controls.reroll_dice and movement:
            return
        cursor.head(self._orders_caption(view), None if movement else self._dim())
        if not movement:
            cursor.row("click", "does not order now", self._dim())
        for (side, index), order in sorted(orders.items()):
            cursor.row(
                f"{'you' if side == PLAYER else 'opp'} {index}",
                order.text,
                None if order.legal else pal.shot_kill,
            )
            if not order.legal and order.reason:
                cursor.note(order.reason)
        if orders:
            cursor.row("commit", "[Enter]")
        # Only worth a row once it is on: held dice are the default and the
        # thing a rewind is *for*, so saying so every frame would be noise.
        if self._controls.reroll_dice:
            cursor.row("dice", "reroll [D]", pal.shot_kill)

    def _action(
        self, cursor: _Cursor, side: str, index: int, model: WargameModel
    ) -> None:
        pal = self._theme.palette
        cursor.head("ACTION")
        move = self._controls.moves.get(index) if side == PLAYER else None
        if move is None:
            # No decoded intent for this model (an opponent, or before the first
            # movement step) — report the realised move, which is always there.
            previous = model.previous_location
            if previous is None:
                cursor.row("moved", "not yet")
                return
            travelled = math.dist(
                (float(previous[0]), float(previous[1])),
                (float(model.location[0]), float(model.location[1])),
            )
            cursor.row("moved", f'{travelled:.2f}"')
            return
        cursor.row("last move", move.text)
        cursor.row("intended", f"{move.intended[0]:.2f}, {move.intended[1]:.2f}")
        cursor.row("actual", f"{move.actual[0]:.2f}, {move.actual[1]:.2f}")
        displaced = move.gap
        cursor.row(
            "displaced",
            f'{displaced:.2f}"',
            pal.shot_kill if displaced > _DISPLACED_EPS else None,
        )
        if displaced > _DISPLACED_EPS:
            cursor.note("Off its own vector — collision resolution moved it.")

    def _reward(
        self,
        cursor: _Cursor,
        view: BattleView,
        side: str,
        index: int,
        model: WargameModel,
    ) -> None:
        pal = self._theme.palette
        cursor.head("REWARD")
        if side == OPPONENT:
            cursor.note("Only the player is rewarded.")
            return
        if not isinstance(view, DebugView):
            return
        per_model = view.last_per_model_reward
        if index >= len(per_model):
            return
        earned = float(per_model[index])
        cursor.row(
            "this model",
            f"{earned:+.3f}",
            pal.shot_kill if earned == 0.0 and not model.is_alive else None,
        )
        if not model.is_alive:
            cursor.note("0.000 — the reward loop iterates alive models only.")
        # Per-calculator numbers are the army mean: `phase_manager` divides each
        # calculator's total by the alive count before it reaches the breakdown,
        # and most calculators expose no per-model split at all. Unlabelled they
        # would read as this model's own earnings.
        components = [
            (name, value)
            for name, value in view.last_reward_breakdown.items()
            if "/" not in name and value != 0.0
        ]
        if components:
            cursor.head("BY CALCULATOR · ARMY MEAN")
            for name, value in components:
                cursor.row(name, f"{value:+.3f}")

    def _sight(
        self, cursor: _Cursor, view: BattleView, side: str, model: WargameModel
    ) -> None:
        """How much this model can see, in numbers beside the shading.

        Counted under the same predicate the shading is drawn from, so the
        number and the picture always answer the same question. It is not the
        shooting mask, which also gates on weapon range — a model with enemies
        in sight and no shots fired is exactly the case worth landing on, and
        this is the row that says sight was not the reason.
        """
        enemies = view.opponent_models if side == PLAYER else view.player_models
        alive = [enemy for enemy in enemies if enemy.is_alive]
        if not alive:
            return
        cursor.head("SIGHT")
        origin = (float(model.location[0]), float(model.location[1]))
        targets = np.array([[e.location[0], e.location[1]] for e in alive], dtype=float)
        cursor.row(
            "enemies in sight",
            f"{int(sight_from(view, origin, targets).sum())}/{len(alive)}",
        )
        if not self._show_shadow:
            cursor.row("board hidden", "[S] to sweep")
            return
        board = float(view.config.board_width * view.config.board_height)
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in self._shadow)
        cursor.row("board hidden", f"{area / board:.0%}")

    def _shooting(
        self, cursor: _Cursor, view: BattleView, side: str, index: int
    ) -> None:
        pal = self._theme.palette
        ours, theirs = (
            (view.last_player_shooting_results, view.last_opponent_shooting_results)
            if side == PLAYER
            else (
                view.last_opponent_shooting_results,
                view.last_player_shooting_results,
            )
        )
        fired = [s for s in ours if s.attacker_idx == index]
        taken = [s for s in theirs if s.target_idx == index]
        if not fired and not taken:
            return
        cursor.head("SHOOTING")
        for shot in fired:
            cursor.row(
                f"-> enemy {shot.target_idx}",
                _shot_text(shot),
                pal.shot_kill if shot.killed else None,
            )
        for shot in taken:
            cursor.row(
                f"<- enemy {shot.attacker_idx}",
                _shot_text(shot),
                pal.shot_kill if shot.killed else None,
            )

    def _observation_rows(
        self, cursor: _Cursor, view: BattleView, side: str, index: int
    ) -> None:
        """What the network actually receives for this model.

        The desk check `CLAUDE.md` calls for — "if two states differ only in what
        this term keys on, do they differ in the observation?" — made a glance.
        """
        observation = self._observation(view)
        if observation is None:
            return
        rows = (
            observation.wargame_models  # type: ignore[attr-defined]
            if side == PLAYER
            else observation.opponent_models  # type: ignore[attr-defined]
        )
        if index >= len(rows):
            return
        row = rows[index]
        cursor.head("OBSERVATION")
        for i, delta in enumerate(np.asarray(row.distances_to_objectives)):
            cursor.row(
                f"obj {i} delta", f"{float(delta[0]):+.1f}, {float(delta[1]):+.1f}"
            )
        cursor.row("alive", f"{float(row.alive):.1f}")
        cursor.row("group", f"{row.group_id}/{row.max_groups}")
        cursor.row("A / BS", f"{row.weapon_attacks} / {row.weapon_ballistic_skill}+")
        cursor.row(
            "S / AP / D",
            f"{row.weapon_strength} / {row.weapon_ap} / {row.weapon_damage}",
        )


def _shot_text(shot: object) -> str:
    """One shot as rolled: hits, wounds, unsaved, damage, and whether it killed."""
    result = shot.result  # type: ignore[attr-defined]
    text = f"{result.hits}h {result.wounds}w {result.unsaved}u"
    if result.damage_dealt:
        text += f" {result.damage_dealt}d"
    if shot.killed:  # type: ignore[attr-defined]
        text += " KILL"
    return text


class _Cursor:
    """Lays panel rows out top to bottom, clipping at the panel's foot.

    Rows that would run past the board's height are dropped rather than drawn
    over the south HUD. The count is not knowable in advance: objectives and
    reward calculators are both config-dependent, and a phase may declare a
    dozen of the latter.
    """

    def __init__(
        self, presenter: DebugPresenter, frame: Canvas, x: int, y: int, width: int
    ) -> None:
        self._presenter = presenter
        self._frame = frame
        self._x = x
        self._y = y
        self._w = width
        self._limit = presenter._top_h + presenter._canvas_h - _PAD

    def _room(self, height: int) -> bool:
        return self._y + height <= self._limit

    def head(self, text: str, color: RGB | None = None) -> None:
        """A section caption over a rule."""
        if not self._room(_ROW_H + _SECTION_GAP):
            return
        pal = self._presenter._theme.palette
        self._y += _SECTION_GAP
        self._presenter._text(
            self._frame,
            text,
            (self._x, self._y),
            _HEAD_SIZE,
            color or pal.text,
            "midleft",
            True,
        )
        self._y += 7
        self._presenter._backend.draw_line(
            self._frame,
            (self._x, self._y),
            (self._x + self._w, self._y),
            pal.panel_line,
            1,
        )
        self._y += _ROW_H - 7

    def row(self, key: str, value: str, color: RGB | None = None) -> None:
        """A label on the left, its value right-aligned to the panel edge."""
        if not self._room(_ROW_H):
            return
        pal = self._presenter._theme.palette
        self._presenter._text(
            self._frame,
            key,
            (self._x, self._y),
            _ROW_SIZE,
            self._presenter._dim(),
            "midleft",
        )
        self._presenter._text(
            self._frame,
            value,
            (self._x + self._w, self._y),
            _ROW_SIZE,
            color or pal.text,
            "midright",
        )
        self._y += _ROW_H

    def note(self, text: str) -> None:
        """A short reason, wrapped to the panel and marked with a rule."""
        pal = self._presenter._theme.palette
        lines: list[str] = []
        line = ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if line and self._presenter._tsize(candidate, _NOTE_SIZE)[0] > self._w - 10:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        if not lines or not self._room(_NOTE_H * len(lines) + 4):
            return
        self._y += 2
        top = self._y - _NOTE_H + 4
        for text_line in lines:
            self._presenter._text(
                self._frame,
                text_line,
                (self._x + 9, self._y),
                _NOTE_SIZE,
                pal.shot_kill,
                "midleft",
            )
            self._y += _NOTE_H
        self._presenter._backend.draw_line(
            self._frame,
            (self._x, top),
            (self._x, self._y - _NOTE_H + 5),
            pal.shot_kill,
            2,
        )
