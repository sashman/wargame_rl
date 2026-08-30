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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from wargame_rl.wargame.envs.domain.rules_quantities import RulesQuantities
from wargame_rl.wargame.envs.domain.scale import Scale
from wargame_rl.wargame.envs.domain.sight import BlockingMask
from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
from wargame_rl.wargame.envs.renders.v2.backend import Canvas, RenderBackend
from wargame_rl.wargame.envs.renders.v2.control import (
    ThreatOptions,
    ThreatOverlay,
    sight_matrix_from_terrain,
)
from wargame_rl.wargame.envs.renders.v2.presenters.base import BasePresenter
from wargame_rl.wargame.envs.renders.v2.scene import (
    SHOT_FADE_FRAMES,
    Control,
    Scene,
    build_scene,
    shot_fade_for_age,
)
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme
from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, PlayerSide
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
    # The side that was active when the snapshot was taken — which is the
    # player's own, since a step always leaves the clock on a player phase. It
    # is what the HUD reads to say who takes the first turn of a round.
    active_player: PlayerSide | None


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
class _ShotOutcome:
    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int


@dataclass(frozen=True)
class _Shot:
    """A recorded shot, shaped like `PairedShootingResult` for `build_scene`."""

    attacker_idx: int
    target_idx: int
    result: _ShotOutcome
    killed: bool


@dataclass(frozen=True)
class _SnapshotView:
    config: _ConfigView
    deployment_zone: tuple[int, int, int, int]
    opponent_deployment_zone: tuple[int, int, int, int]
    deployment_outline: Polygon | None
    opponent_deployment_outline: Polygon | None
    objectives: tuple[_ObjectiveView, ...]
    player_models: tuple[_ModelView, ...]
    opponent_models: tuple[_ModelView, ...]
    terrain: Terrain
    # Weapon range per model, from `ModelSnapshot.weapons` — already recorded for
    # both sides, so the threat sweep needed no new field for it.
    player_max_ranges: np.ndarray
    opponent_max_ranges: np.ndarray
    # None on a pre-2.6 recording, which is what makes the overlays unavailable
    # there rather than wrong.
    rules_quantities: RulesQuantities | None
    blocking_mask: BlockingMask | None
    n_rounds: int
    current_turn: int
    last_reward: float | None
    last_player_shooting_results: tuple[_Shot, ...]
    last_opponent_shooting_results: tuple[_Shot, ...]
    # Empty on every pre-2.7 recording and on every melee-off config, which is
    # every config shipped today -- so a replay of one draws no clashes rather
    # than inventing them.
    last_player_fight_results: tuple[_Shot, ...]
    last_opponent_fight_results: tuple[_Shot, ...]
    last_reward_breakdown: dict[str, float]
    episode_reward: float | None
    game_clock_state: _ClockView
    player_vp: int
    player_vp_delta: int
    opponent_vp: int
    opponent_vp_delta: int

    def line_of_sight_matrix(
        self,
        origins: np.ndarray,
        targets: np.ndarray,
        candidates: np.ndarray | None = None,
    ) -> np.ndarray:
        """The engine's own sight test, run against the recorded terrain.

        This is what keeps "the renderer never invents its own answer to what is
        visible" true in replay as well as live — and it is why
        `compute_threat_overlay` is one function that serves both paths rather
        than two that can drift.

        A pre-2.6 recording has no `los_sample_step`, and guessing one would
        answer a *different* question convincingly, so it refuses instead. The
        presenter checks `rules_quantities` before it ever gets here.
        """
        if self.rules_quantities is None:
            raise ValueError(
                "this recording predates schema 2.6 and carries no sample step"
            )
        return sight_matrix_from_terrain(
            origins,
            targets,
            self.terrain,
            self.blocking_mask,
            sample_step=self.rules_quantities.los_sample_step,
            candidates=candidates,
        )


def _polygon(vertices: list[list[float]] | None) -> Polygon | None:
    if not vertices:
        return None
    return Polygon(vertices=np.asarray(vertices, dtype=np.float64))


def _max_ranges(models: "Sequence[object]") -> np.ndarray:
    """Longest weapon range per model, the rule `max_weapon_ranges` uses.

    Already on the recording — `ModelSnapshot.weapons` carries `weapon_range`
    for both sides — so the threat sweep needed no new snapshot field for it.
    A model with no weapons gets 0.0, which the sweep's `range > 0` guard then
    rules out rather than marking the cell it stands on.
    """
    return np.array(
        [
            max((float(w.weapon_range) for w in m.weapons), default=0.0)  # type: ignore[attr-defined]
            for m in models
        ],
        dtype=float,
    )


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


def _shots(recorded: Sequence[object]) -> tuple[_Shot, ...]:
    """Adapt recorded combat results to the shape `build_scene` draws from.

    `killed` rides on the snapshot from schema 2.3, and defaults to False on
    older recordings, where a killing shot replays as an ordinary hit.
    """
    return tuple(
        _Shot(
            attacker_idx=int(entry.attacker_idx),  # type: ignore[attr-defined]
            target_idx=int(entry.target_idx),  # type: ignore[attr-defined]
            result=_ShotOutcome(
                hits=int(entry.hits),  # type: ignore[attr-defined]
                wounds=int(entry.wounds),  # type: ignore[attr-defined]
                unsaved=int(entry.unsaved),  # type: ignore[attr-defined]
                damage_dealt=int(entry.damage_dealt),  # type: ignore[attr-defined]
            ),
            killed=bool(getattr(entry, "killed", False)),
        )
        for entry in recorded
    )


def _snapshot_to_view(snapshot: GameStateSnapshot) -> _SnapshotView:
    """Adapt a snapshot to the read-only surface `build_scene` consumes."""
    dz = snapshot.deployment_zone
    odz = snapshot.opponent_deployment_zone
    # The real `Terrain`, not a stub: it derives its own padded outlines and
    # vertex counts, which is exactly what `domain.sight` needs to trace against.
    # `build_scene` only reads `.footprints[].polygon`, so this is invisible to it.
    terrain = Terrain(
        [
            Footprint(Polygon(vertices=np.asarray(fp, dtype=np.float64)))
            for fp in (snapshot.terrain_footprints or [])
        ]
    )
    rules = snapshot.rules
    phase = snapshot.clock.battle_phase
    return _SnapshotView(
        config=_ConfigView(
            snapshot.board_width,
            snapshot.board_height,
            # Pre-2.2 recordings did not carry it; no phase is then dimmed.
            tuple(BattlePhase(p) for p in (snapshot.skip_phases or [])),
        ),
        deployment_outline=_polygon(snapshot.deployment_outline),
        opponent_deployment_outline=_polygon(snapshot.opponent_deployment_outline),
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
        terrain=terrain,
        player_max_ranges=_max_ranges(snapshot.player_models),
        opponent_max_ranges=_max_ranges(snapshot.opponent_models),
        rules_quantities=(
            RulesQuantities(
                scale=Scale(inches_per_unit=1.0),
                engagement_range=rules.engagement_range,
                max_move_speed=0.0,
                los_sample_step=rules.los_sample_step,
                base_radius=rules.base_radius,
                coherency_distance=0.0,
            )
            if rules is not None
            else None
        ),
        blocking_mask=rules.blocking_mask if rules is not None else None,
        n_rounds=snapshot.n_rounds,
        current_turn=snapshot.step,
        last_reward=snapshot.reward.total,
        last_player_shooting_results=_shots(snapshot.player_combat_results),
        last_opponent_shooting_results=_shots(snapshot.opponent_combat_results),
        last_player_fight_results=_shots(snapshot.player_melee_results),
        last_opponent_fight_results=_shots(snapshot.opponent_melee_results),
        last_reward_breakdown=dict(snapshot.reward.breakdown),
        episode_reward=snapshot.reward.episode_total,
        game_clock_state=_ClockView(
            battle_round=snapshot.clock.battle_round,
            phase=BattlePhase(phase) if phase else None,
            active_player=(
                PlayerSide(snapshot.clock.active_player)
                if snapshot.clock.active_player
                else None
            ),
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
    shot_fade: float = 1.0,
    threat: "ThreatOverlay | None" = None,
) -> Scene:
    """Build the same `Scene` a live `BattleView` would, from a recorded snapshot."""
    control = tuple(Control(c) for c in snapshot.objective_control)
    view = cast("BattleView", _snapshot_to_view(snapshot))
    return build_scene(
        view,
        control,
        scale=scale,
        theme=theme,
        threat=threat,
        show_grid=show_grid,
        shot_fade=shot_fade,
    )


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
        threat_options: ThreatOptions | None = None,
    ) -> None:
        super().__init__(backend, theme, threat_options)
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
            ("R", "shooting threat range, both sides"),
            ("E", "engagement range, both sides"),
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
            shot_fade=shot_fade_for_age(self._volley_age(index)),
            # The adapter is built twice per frame — once here for the overlay
            # and once inside `build_scene_from_snapshot`. Cheap (it is a
            # dataclass over data already in memory) and it keeps the snapshot
            # the single source for both, rather than threading a view out.
            threat=self._threat_overlay(
                cast("BattleView", _snapshot_to_view(snapshot))
            ),
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

    def _volley_age(self, index: int) -> int:
        """Frames since this frame's shooting results first appeared.

        Computed by walking back from `index` rather than counted as frames go
        by, because the timeline can be scrubbed: the fade has to be a function
        of where you are, not of how you got there.
        """

        def signature(i: int) -> tuple[object, ...]:
            snapshot = self._source[i]
            return tuple(
                (r.attacker_idx, r.target_idx, r.damage_dealt, r.killed)
                for r in (
                    *snapshot.player_combat_results,
                    *snapshot.opponent_combat_results,
                )
            )

        current = signature(index)
        age = 0
        while index - age > 0 and signature(index - age - 1) == current:
            age += 1
            if age >= SHOT_FADE_FRAMES:
                break
        return age

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
        if key == pygame.K_r:  # type: ignore[attr-defined]
            self.toggle_threat()
            return True
        if key == pygame.K_e:  # type: ignore[attr-defined]
            self.toggle_engagement()
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
