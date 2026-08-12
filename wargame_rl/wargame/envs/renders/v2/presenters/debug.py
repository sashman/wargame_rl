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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pygame

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.renders.v2.backend import Canvas, RenderBackend
from wargame_rl.wargame.envs.renders.v2.debug_view import DebugView
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
    InteractiveRenderer,
)
from wargame_rl.wargame.envs.renders.v2.scene import Scene
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, RGB, Theme

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
            ("Click", "inspect a model, either side"),
            ("L", "line-of-sight debug ray"),
            ("Esc", "deselect, then quit"),
        )

    def _handle_key(self, event: pygame.event.Event, view: BattleView) -> None:
        controls = self._controls
        if event.key == pygame.K_SPACE:
            controls.paused = not controls.paused
        elif event.key in (pygame.K_PERIOD, pygame.K_RIGHT):
            # Stepping implies pausing: you asked for one step, not for the match
            # to resume from here.
            controls.step_once = True
            controls.paused = True
        elif event.key in (pygame.K_COMMA, pygame.K_LEFT):
            controls.step_back = True
            controls.paused = True
        elif event.key == pygame.K_LEFTBRACKET:
            self._fps = max(MIN_FPS, self._fps - 1)
        elif event.key == pygame.K_RIGHTBRACKET:
            self._fps = min(MAX_FPS, self._fps + 1)
        elif event.key == pygame.K_ESCAPE and controls.selected is not None:
            # Esc backs out of what you are looking at before it quits, so a
            # click cannot be undone only by leaving the session.
            controls.selected = None
        else:
            super()._handle_key(event, view)

    def _handle_click(self, view: BattleView, mx: int, my: int) -> None:
        self._controls.selected = self._model_at(view, mx, my)

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
        self._draw_selection(frame, view)
        self._draw_inspector(frame, view)
        if self._show_keys:
            self._draw_key_map(frame)
        return frame

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
        self._action(cursor, side, index, model)
        self._reward(cursor, view, side, index, model)
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
