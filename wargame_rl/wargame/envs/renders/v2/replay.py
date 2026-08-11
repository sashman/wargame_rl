"""Replay a recorded match: turn `GameStateSnapshot`s back into v2 frames.

A recording is a JSONL event log; `ReplayController.iter_snapshots()` gives one
full snapshot per frame. `build_scene_from_snapshot` feeds the *same*
`build_scene` the live renderer uses, via a tiny adapter that mimics the
`BattleView` surface it reads — so a replayed frame is pixel-identical to the
live one, with no duplicated board geometry. Control is precomputed in the
snapshot (`objective_control`) and terrain rides along as `terrain_footprints`
(schema 2.1), so no env is re-hydrated.

`ReplayPresenter` wraps the shared `BasePresenter` pipeline with a scrubbable
timeline and play/pause, and exports MP4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from wargame_rl.wargame.envs.renders.v2.backend import Canvas, RenderBackend
from wargame_rl.wargame.envs.renders.v2.presenters.base import BasePresenter
from wargame_rl.wargame.envs.renders.v2.scene import Control, Scene, build_scene
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme
from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.types.geometry import Polygon

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.state.replay import ReplayController

Point = tuple[float, float]


# --- snapshot -> Scene ------------------------------------------------------
#
# Frozen adapters exposing only the attributes `build_scene` reads off a
# `BattleView`. Backing the polygons with the real `Polygon` value object keeps
# `centroid` and the vertex-bytes dedup (a footprint that is also an objective)
# bit-identical to a live render.


@dataclass(frozen=True)
class _ClockView:
    battle_round: int | None
    phase: BattlePhase | None


@dataclass(frozen=True)
class _ConfigView:
    board_width: int
    board_height: int
    skip_phases: tuple[BattlePhase, ...]


@dataclass(frozen=True)
class _ModelView:
    location: Point
    group_id: int
    is_alive: bool
    previous_location: Point | None
    base_radius: float


@dataclass(frozen=True)
class _ObjectiveView:
    location: Point
    radius_size: float
    area: Polygon | None


@dataclass(frozen=True)
class _Footprint:
    polygon: Polygon


@dataclass(frozen=True)
class _TerrainView:
    footprints: tuple[_Footprint, ...]


@dataclass(frozen=True)
class _SnapshotView:
    config: _ConfigView
    deployment_zone: tuple[int, int, int, int]
    opponent_deployment_zone: tuple[int, int, int, int]
    objectives: tuple[_ObjectiveView, ...]
    player_models: tuple[_ModelView, ...]
    opponent_models: tuple[_ModelView, ...]
    terrain: _TerrainView
    n_rounds: int
    current_turn: int
    last_reward: float | None
    last_reward_breakdown: dict[str, float]
    episode_reward: float | None
    game_clock_state: _ClockView
    player_vp: int
    player_vp_delta: int
    opponent_vp: int
    opponent_vp_delta: int


def _polygon(vertices: list[list[float]] | None) -> Polygon | None:
    if not vertices:
        return None
    return Polygon(vertices=np.asarray(vertices, dtype=np.float64))


def _model_view(snap_model: object) -> _ModelView:
    m = snap_model  # ModelSnapshot; typed loosely to keep the adapter local
    loc = m.location  # type: ignore[attr-defined]
    prev = m.previous_location  # type: ignore[attr-defined]
    return _ModelView(
        location=(float(loc[0]), float(loc[1])),
        group_id=int(m.group_id),  # type: ignore[attr-defined]
        is_alive=bool(m.alive),  # type: ignore[attr-defined]
        previous_location=(
            (float(prev[0]), float(prev[1])) if prev is not None else None
        ),
        base_radius=float(m.base_radius),  # type: ignore[attr-defined]
    )


def _snapshot_to_view(snapshot: GameStateSnapshot) -> _SnapshotView:
    """Adapt a snapshot to the read-only surface `build_scene` consumes."""
    dz = snapshot.deployment_zone
    odz = snapshot.opponent_deployment_zone
    footprints = tuple(
        _Footprint(polygon=Polygon(vertices=np.asarray(fp, dtype=np.float64)))
        for fp in (snapshot.terrain_footprints or [])
    )
    phase = snapshot.clock.battle_phase
    return _SnapshotView(
        config=_ConfigView(
            snapshot.board_width,
            snapshot.board_height,
            # Pre-2.2 recordings did not carry it; no phase is then dimmed.
            tuple(BattlePhase(p) for p in (snapshot.skip_phases or [])),
        ),
        deployment_zone=(int(dz[0]), int(dz[1]), int(dz[2]), int(dz[3])),
        opponent_deployment_zone=(
            int(odz[0]),
            int(odz[1]),
            int(odz[2]),
            int(odz[3]),
        ),
        objectives=tuple(
            _ObjectiveView(
                location=(float(o.location[0]), float(o.location[1])),
                radius_size=float(o.radius_size),
                area=_polygon(o.area),
            )
            for o in snapshot.objectives
        ),
        player_models=tuple(_model_view(m) for m in snapshot.player_models),
        opponent_models=tuple(_model_view(m) for m in snapshot.opponent_models),
        terrain=_TerrainView(footprints=footprints),
        n_rounds=snapshot.n_rounds,
        current_turn=snapshot.step,
        last_reward=snapshot.reward.total,
        last_reward_breakdown=dict(snapshot.reward.breakdown),
        episode_reward=snapshot.reward.episode_total,
        game_clock_state=_ClockView(
            battle_round=snapshot.clock.battle_round,
            phase=BattlePhase(phase) if phase else None,
        ),
        player_vp=snapshot.player_vp,
        player_vp_delta=snapshot.player_vp_delta,
        opponent_vp=snapshot.opponent_vp,
        opponent_vp_delta=snapshot.opponent_vp_delta,
    )


def build_scene_from_snapshot(
    snapshot: GameStateSnapshot,
    *,
    scale: float,
    theme: Theme = DEFAULT_THEME,
    show_grid: bool = True,
) -> Scene:
    """Build the same `Scene` a live `BattleView` would, from a recorded snapshot."""
    control = tuple(Control(c) for c in snapshot.objective_control)
    view = cast("BattleView", _snapshot_to_view(snapshot))
    return build_scene(view, control, scale=scale, theme=theme, show_grid=show_grid)


# --- random-access source over a recording ----------------------------------


@dataclass(frozen=True)
class ReplaySource:
    """Materialised snapshots plus which indices are anchor frames (timeline ticks)."""

    snapshots: list[GameStateSnapshot]
    anchor_indices: frozenset[int]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int) -> GameStateSnapshot:
        return self.snapshots[index]

    @property
    def steps(self) -> list[int]:
        return [s.step for s in self.snapshots]

    @classmethod
    def from_controller(cls, controller: "ReplayController") -> "ReplaySource":
        """Read every snapshot and flag the reset + anchor frames."""
        from wargame_rl.wargame.envs.state.events import StepEvent

        snapshots = controller.iter_snapshots()
        # Snapshot i corresponds to event i (reset is index 0, always full).
        anchors = {0}
        for i, event in enumerate(controller.event_log.events):
            if isinstance(event, StepEvent) and event.anchor is not None:
                anchors.add(i)
        return cls(snapshots=snapshots, anchor_indices=frozenset(anchors))


# --- interactive player + MP4 export ----------------------------------------


class ReplayPresenter(BasePresenter):
    """Plays a `ReplaySource` back in a window with a scrubbable timeline."""

    def __init__(
        self,
        backend: RenderBackend,
        source: ReplaySource,
        theme: Theme = DEFAULT_THEME,
        fps: int = 5,
    ) -> None:
        super().__init__(backend, theme)
        if len(source) == 0:
            raise ValueError("Cannot replay an empty recording")
        self._source = source
        self._fps = fps
        self._index = 0
        self._playing = False
        self._show_keys = False
        self._timeline_h = theme.north_panel_h
        first = source[0]
        self._board_w = first.board_width
        self._board_h = first.board_height
        self._scale = min(
            self._grid_size / self._board_w, self._grid_size / self._board_h
        )
        self._recompute_layout()

    @property
    def _grid_size(self) -> int:
        from wargame_rl.wargame.envs.renders.v2.presenters.base import GRID_SIZE

        return GRID_SIZE

    def _is_paused(self) -> bool:
        return not self._playing

    def key_map(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Space", "play / pause (restarts at the end)"),
            ("<-  ->", "step one frame"),
            ("Home  End", "first / last frame"),
            ("Click", "scrub the timeline"),
            ("Esc  Q", "quit"),
        )

    # -- frame composition ---------------------------------------------------

    def frame_at(self, index: int) -> Canvas:
        """Compose the board+panels for a snapshot, then add the timeline strip."""
        snapshot = self._source[index]
        scene = build_scene_from_snapshot(
            snapshot,
            scale=self._scale,
            theme=self._theme,
            show_grid=self._theme.show_grid,
        )
        base = self._compose_scene(scene)
        frame = self._backend.new_canvas(
            self._window_w,
            self._window_h + self._timeline_h,
            self._theme.palette.window_bg,
        )
        self._backend.blit(frame, base, (0, 0))
        self._draw_timeline(frame, index)
        if self._show_keys:
            self._draw_key_map(frame)
        return frame

    def _draw_timeline(self, frame: Canvas, index: int) -> None:
        pal = self._theme.palette
        y0 = self._window_h
        h = self._timeline_h
        width = self._window_w
        self._backend.fill_rect(frame, (0, y0, width, h), pal.panel_bg)
        self._backend.draw_line(frame, (0, y0), (width, y0), pal.panel_line, 1)

        margin = 12
        track_y = y0 + h // 2
        track_x0, track_x1 = margin, width - margin
        self._backend.draw_line(
            frame, (track_x0, track_y), (track_x1, track_y), pal.panel_line, 2
        )
        n = max(1, len(self._source) - 1)
        for anchor in self._source.anchor_indices:
            ax = track_x0 + (track_x1 - track_x0) * anchor / n
            self._backend.draw_line(
                frame, (ax, track_y - 5), (ax, track_y + 5), pal.objective_rim, 1
            )
        cursor_x = track_x0 + (track_x1 - track_x0) * index / n
        self._backend.draw_disc(
            frame, (cursor_x, track_y), 6, (*pal.player_control, 255), pal.text, 1
        )
        # The keys moved behind Tab, so the strip carries the position and one hint.
        label = f"{index + 1}/{len(self._source)}  |  [Tab] keys"
        # Centred in the strip: anchored near its bottom edge, a 20px face
        # overflowed the strip and the frame clipped its descenders.
        self._backend.draw_text(
            frame, label, (width // 2, y0 + h - 11), 16, pal.text, "center"
        )

    def _index_from_click(self, x: int, y: int) -> int | None:
        """Map a click on the timeline strip to a frame index."""
        if y < self._window_h:
            return None
        margin = 12
        track_x0, track_x1 = margin, self._window_w - margin
        fraction = (x - track_x0) / max(1, track_x1 - track_x0)
        fraction = min(1.0, max(0.0, fraction))
        return round(fraction * (len(self._source) - 1))

    # -- interactive loop ----------------------------------------------------

    def run(self) -> None:
        """Open a window and play the recording until the user quits."""
        import pygame

        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode(
            (self._window_w, self._window_h + self._timeline_h)
        )
        pygame.display.set_caption("Wargame replay (v2)")
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                running = self._handle_event(event, pygame) and running
            if self._playing and self._index < len(self._source) - 1:
                self._index += 1
            elif self._playing:
                self._playing = False  # stop at the end
            self._blit_to_window(window, self.frame_at(self._index), pygame)
            clock.tick(self._fps)
        pygame.display.quit()
        pygame.quit()

    def _handle_event(self, event: object, pygame: object) -> bool:
        """Return False to quit."""
        etype = event.type  # type: ignore[attr-defined]
        if etype == pygame.QUIT:  # type: ignore[attr-defined]
            return False
        if etype == pygame.KEYDOWN:  # type: ignore[attr-defined]
            return self._handle_key(event.key, pygame)  # type: ignore[attr-defined]
        if etype == pygame.MOUSEBUTTONDOWN and event.button == 1:  # type: ignore[attr-defined]
            hit = self._index_from_click(event.pos[0], event.pos[1])  # type: ignore[attr-defined]
            if hit is not None:
                self._index = hit
                self._playing = False
        return True

    def _handle_key(self, key: int, pygame: object) -> bool:
        last = len(self._source) - 1
        if key in (pygame.K_ESCAPE, pygame.K_q):  # type: ignore[attr-defined]
            return False
        if key == pygame.K_TAB:  # type: ignore[attr-defined]
            self._show_keys = not self._show_keys
            return True
        if key == pygame.K_SPACE:  # type: ignore[attr-defined]
            # Replaying from the end restarts; otherwise toggle.
            if not self._playing and self._index >= last:
                self._index = 0
            self._playing = not self._playing
        elif key == pygame.K_RIGHT:  # type: ignore[attr-defined]
            self._index = min(last, self._index + 1)
            self._playing = False
        elif key == pygame.K_LEFT:  # type: ignore[attr-defined]
            self._index = max(0, self._index - 1)
            self._playing = False
        elif key == pygame.K_HOME:  # type: ignore[attr-defined]
            self._index = 0
            self._playing = False
        elif key == pygame.K_END:  # type: ignore[attr-defined]
            self._index = last
            self._playing = False
        return True

    def _blit_to_window(self, window: object, frame: Canvas, pygame: object) -> None:
        rgb = self._backend.to_rgb_array(frame)
        surface = pygame.image.frombuffer(  # type: ignore[attr-defined]
            rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"
        )
        window.blit(surface, (0, 0))  # type: ignore[attr-defined]
        pygame.event.pump()  # type: ignore[attr-defined]
        pygame.display.update()  # type: ignore[attr-defined]

    # -- MP4 export ----------------------------------------------------------

    def export_mp4(self, path: str, fps: int | None = None) -> None:
        """Write every frame to an MP4 (headless — set SDL_VIDEODRIVER=dummy first)."""
        import imageio  # type: ignore[import-untyped]

        writer = imageio.get_writer(
            path,
            format="FFMPEG",  # type: ignore[arg-type]
            mode="I",
            fps=fps or self._fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for index in range(len(self._source)):
                writer.append_data(self._backend.to_rgb_array(self.frame_at(index)))
        finally:
            writer.close()
