"""Renderer v2: Scene purity, control extraction, FrameSource, and A/B vs legacy.

The Scene tests need no pygame — they pin the board-coordinate conventions (no
`+0.5`, the base-radius rule) that only the renderer can get wrong, now that
those live in `build_scene` rather than in a draw method. The pixel tests render
v2 and the untouched legacy renderer headlessly and check they agree.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from wargame_rl.wargame.envs.renders.human import HumanRender  # noqa: E402
from wargame_rl.wargame.envs.renders.renderer import FrameSource  # noqa: E402
from wargame_rl.wargame.envs.renders.v2 import build_renderer  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.control import (  # noqa: E402
    compute_objective_control,
)
from wargame_rl.wargame.envs.renders.v2.factory import (  # noqa: E402
    BACKENDS,
    _build_backend,
)
from wargame_rl.wargame.envs.renders.v2.presenters.recording import (  # noqa: E402
    RecordingRenderer,
)
from wargame_rl.wargame.envs.renders.v2.replay import (  # noqa: E402
    ReplayPresenter,
    ReplaySource,
    build_scene_from_snapshot,
)
from wargame_rl.wargame.envs.renders.v2.scene import (  # noqa: E402
    Control,
    Disc,
    Poly,
    Seg,
    build_scene,
)
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME  # noqa: E402
from wargame_rl.wargame.envs.state import EventLog, ReplayController  # noqa: E402
from wargame_rl.wargame.envs.types import (  # noqa: E402
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import (  # noqa: E402
    ModelConfig,
    ObjectiveConfig,
    TerrainPieceConfig,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase  # noqa: E402
from wargame_rl.wargame.envs.wargame import WargameEnv  # noqa: E402


def _env(**overrides: Any) -> WargameEnv:
    base: dict[str, Any] = dict(
        board_width=20,
        board_height=20,
        number_of_wargame_models=1,
        number_of_objectives=1,
        number_of_battle_rounds=3,
        render_mode=None,
    )
    base.update(overrides)
    env = WargameEnv(config=WargameEnvConfig(**base))
    env.reset(seed=0)
    return env


# --- Scene purity (no pygame) ----------------------------------------------


def test_build_scene_draws_player_at_its_board_location() -> None:
    env = _env(models=[ModelConfig(x=7, y=5)])
    scene = build_scene(env, compute_objective_control(env), scale=10.0)

    model = env.player_models[0]
    discs = [
        p for p in scene.primitives if isinstance(p, Disc) and p.center == (7.0, 5.0)
    ]
    assert discs, "expected a player disc at the model's exact board coordinate"
    assert discs[0].fill is not None
    assert discs[0].fill[:3] == DEFAULT_THEME.player_color(model.group_id)


def test_build_scene_uses_the_real_base_radius() -> None:
    env = _env(models=[ModelConfig(x=7, y=5)], base_radius=2.0)
    scene = build_scene(env, compute_objective_control(env), scale=10.0)

    resolved = env.player_models[0].base_radius
    assert resolved > 0.0  # config base_radius resolves onto the model
    disc = next(
        p for p in scene.primitives if isinstance(p, Disc) and p.center == (7.0, 5.0)
    )
    assert disc.radius == pytest.approx(resolved)  # board units, not the 1/3 token


def test_build_scene_grid_toggles() -> None:
    env = _env()
    with_grid = build_scene(
        env, compute_objective_control(env), scale=10.0, show_grid=True
    )
    without = build_scene(
        env, compute_objective_control(env), scale=10.0, show_grid=False
    )
    assert len(with_grid.primitives) > len(without.primitives)


def test_grid_is_drawn_under_terrain_and_models() -> None:
    """Grid is a substrate — every grid line precedes the terrain and the player
    token, so it never rasterises on top of either."""
    pal = DEFAULT_THEME.palette
    env = _env(
        models=[ModelConfig(x=7, y=5)],
        terrain=[TerrainPieceConfig(footprint=(10, 10, 14, 14))],
    )
    scene = build_scene(env, compute_objective_control(env), scale=10.0, show_grid=True)
    grid_idx = [
        i
        for i, p in enumerate(scene.primitives)
        if isinstance(p, Seg) and p.color == pal.grid
    ]
    terrain_idx = next(
        i
        for i, p in enumerate(scene.primitives)
        if isinstance(p, Poly) and p.fill == pal.terrain_fill
    )
    player_idx = next(
        i
        for i, p in enumerate(scene.primitives)
        if isinstance(p, Disc) and p.center == (7.0, 5.0)
    )
    assert grid_idx
    assert max(grid_idx) < terrain_idx  # grid below terrain
    assert max(grid_idx) < player_idx  # grid below models


# --- Control extraction (behaviour) ----------------------------------------


def test_uncontested_objective_is_player_held() -> None:
    env = _env(
        number_of_objectives=1,
        objectives=[ObjectiveConfig(x=5, y=5)],
        models=[ModelConfig(x=5, y=5)],
    )
    assert compute_objective_control(env) == (Control.PLAYER,)


def test_control_length_matches_objectives() -> None:
    env = _env(number_of_objectives=3)
    control = compute_objective_control(env)
    assert len(control) == 3
    assert all(isinstance(c, Control) for c in control)


# --- FrameSource ------------------------------------------------------------


def test_recording_presenter_and_legacy_are_frame_sources() -> None:
    assert isinstance(build_renderer("v2", "recording"), FrameSource)
    assert isinstance(HumanRender(), FrameSource)


# --- Theme selection --------------------------------------------------------


def test_build_renderer_resolves_theme_by_name() -> None:
    from wargame_rl.wargame.envs.renders.v2.factory import resolve_theme
    from wargame_rl.wargame.envs.renders.v2.theme import TABLETOP_THEME

    renderer = build_renderer("v2", "recording", theme="tabletop")
    assert renderer._theme is TABLETOP_THEME  # type: ignore[attr-defined]
    assert resolve_theme("default") is DEFAULT_THEME
    with pytest.raises(ValueError):
        build_renderer("v2", "recording", theme="nope")


# --- A/B pixel parity vs the untouched legacy renderer ---------------------


def _frame(
    renderer_name: str,
    config: WargameEnvConfig,
    seed: int = 0,
    backend: str = "pygame",
) -> np.ndarray:
    renderer = build_renderer(renderer_name, "recording", backend=backend)
    env = WargameEnv(config=config, renderer=renderer)
    env.reset(seed=seed)
    env.render()
    assert isinstance(renderer, FrameSource)
    return renderer.get_frame_array()


def _ab_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=1,
        number_of_objectives=1,
        number_of_battle_rounds=3,
        render_mode=None,
        models=[ModelConfig(x=5, y=5)],
        terrain=[TerrainPieceConfig(footprint=(10, 10, 14, 14))],
    )


def test_v2_board_region_renders_terrain_and_player() -> None:
    """v2's HUD diverges from legacy (Phase 4b redesign), so whole-frame parity no
    longer holds; assert the board region itself renders — terrain + a player over
    a mostly board-background field, below the two-row top panel."""
    config = _ab_config()
    frame = _frame("v2", config)
    top = 2 * DEFAULT_THEME.north_panel_h
    canvas_h = int(round(1024 / 20 * 20))
    board = frame[top : top + canvas_h]
    coverage = (board != 255).any(axis=2).mean()
    # Deployment-zone tints, grid, terrain and a player all mark the board, so it
    # is substantially non-white but not fully saturated.
    assert 0.1 < coverage < 0.98


def test_v2_draws_the_player_in_its_group_colour() -> None:
    config = _ab_config()
    v2 = _frame("v2", config)

    # Board is 20x20 fit to 1024; it sits below the two-row top HUD panel.
    scale = 1024 / 20
    top = 2 * DEFAULT_THEME.north_panel_h
    px = int(5 * scale)
    py = int(5 * scale) + top
    patch = v2[py - 3 : py + 3, px - 3 : px + 3].reshape(-1, 3).mean(axis=0)
    # Group 0 is blue (0, 0, 255): blue channel dominates.
    assert patch[2] > patch[0] and patch[2] > patch[1]


# --- Per-backend parity (every wired backend draws the same Scene) ----------


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_renders_the_player_disc(backend: str) -> None:
    """Each backend must produce the same-shaped frame with the player's group
    colour at its board location. Antialiasing differs, so assert channel
    dominance rather than exact pixels."""
    config = _ab_config()
    frame = _frame("v2", config, backend=backend)

    # 20-unit board fit to 1024, below the two-row top HUD, above the south panel.
    north = DEFAULT_THEME.north_panel_h
    top = 2 * north
    south = DEFAULT_THEME.south_panel_rows * north
    assert frame.shape == (1024 + top + south, 1024, 3)
    scale = 1024 / 20
    px = int(5 * scale)
    py = int(5 * scale) + top
    patch = frame[py - 3 : py + 3, px - 3 : px + 3].reshape(-1, 3).mean(axis=0)
    assert patch[2] > patch[0] and patch[2] > patch[1]  # blue group 0 dominates


# --- Replay: snapshot -> Scene fidelity and the player -----------------------


def _replay_env() -> WargameEnv:
    env = _env(
        number_of_wargame_models=3,
        number_of_objectives=2,
        base_radius=1.0,
        models=[ModelConfig(x=5, y=5), ModelConfig(x=6, y=6), ModelConfig(x=7, y=5)],
        objectives=[ObjectiveConfig(x=10, y=10), ObjectiveConfig(x=14, y=6)],
        terrain=[TerrainPieceConfig(footprint=(9, 9, 12, 12))],
    )
    env.action_space.seed(0)
    env.step(WargameEnvAction(actions=[int(a) for a in env.action_space.sample()]))
    return env


def test_build_scene_from_snapshot_matches_live() -> None:
    """A snapshot-built Scene must equal the live one — replay fidelity. Terrain,
    objective control and the footprint/objective dedup all have to round-trip."""
    env = _replay_env()
    scale = 1024 / 20
    live = build_scene(env, compute_objective_control(env), scale=scale)
    replay = build_scene_from_snapshot(env.to_snapshot(), scale=scale)
    assert replay.primitives == live.primitives
    assert replay.hud == live.hud


def test_replay_pre_2_1_snapshot_drops_terrain_without_crashing() -> None:
    """A pre-2.1 recording has no terrain; the replay just omits the ruins."""
    env = _replay_env()
    scale = 1024 / 20
    with_terrain = build_scene_from_snapshot(env.to_snapshot(), scale=scale)
    legacy_snap = env.to_snapshot().model_copy(update={"terrain_footprints": None})
    without = build_scene_from_snapshot(legacy_snap, scale=scale)
    # Dropping terrain removes at least the ruin polygon + its label.
    assert len(without.primitives) < len(with_terrain.primitives)


def _recording(env: WargameEnv, n_steps: int) -> ReplayController:
    log = EventLog(anchor_interval=2)
    log.record_reset(env.to_snapshot())
    for _ in range(n_steps):
        env.step(WargameEnvAction(actions=[int(a) for a in env.action_space.sample()]))
        log.record_step(env.to_snapshot())
    return ReplayController(log)


def test_replay_source_from_controller_flags_reset_and_anchors() -> None:
    env = _replay_env()
    source = ReplaySource.from_controller(_recording(env, 5))
    assert len(source) == 6  # reset + 5 steps
    assert 0 in source.anchor_indices  # reset is always a full frame
    assert len(source.anchor_indices) > 1  # anchor_interval=2 crossed


def test_replay_presenter_renders_and_exports(tmp_path: Any) -> None:
    env = _replay_env()
    source = ReplaySource.from_controller(_recording(env, 4))
    presenter = ReplayPresenter(_build_backend("pillow"), source)

    frame = presenter.frame_at(0)
    rgb = presenter._backend.to_rgb_array(frame)
    # The replay frame is the board+panels plus the timeline strip, so it is
    # taller than the 1132px live frame at this fit.
    assert rgb.shape[1] == 1024
    assert rgb.shape[0] > 1132

    out = tmp_path / "replay.mp4"
    presenter.export_mp4(str(out))
    assert out.exists() and out.stat().st_size > 0


# --- South HUD: the reward ledger and a round track that does not grow --------


def test_hud_carries_the_reward_ledger_in_calculator_order() -> None:
    """The composition bar reads `reward_breakdown` positionally, so the order
    must be the phase's declaration order — sorting by size would make segments
    swap seats between frames."""
    env = _env(number_of_wargame_models=2)
    env.step(WargameEnvAction(actions=[0, 0]))
    hud = build_scene(env, compute_objective_control(env), scale=51.2).hud

    assert hud.reward_breakdown == tuple(
        (name, value)
        for name, value in env.last_reward_breakdown.items()
        if "/" not in name
    )
    assert hud.episode_reward == pytest.approx(env.episode_reward)


def test_hud_ledger_excludes_a_calculator_s_own_sub_components() -> None:
    """`closest_objective` also reports `closest_objective/base_penalty` and
    friends, which sum into it — charting both would double its weight in the
    composition bar and inflate the paid/charged counts."""
    env = _env(number_of_wargame_models=2)
    env.step(WargameEnvAction(actions=[0, 0]))
    hud = build_scene(env, compute_objective_control(env), scale=51.2).hud

    assert any("/" in name for name in env.last_reward_breakdown)  # the trap exists
    assert all("/" not in name for name, _ in hud.reward_breakdown)


def test_episode_reward_accumulates_and_resets() -> None:
    env = _env(number_of_wargame_models=2)
    for _ in range(3):
        env.step(WargameEnvAction(actions=[0, 0]))
    assert env.episode_reward != 0.0
    env.reset(seed=0)
    assert env.episode_reward == 0.0


@pytest.mark.parametrize(
    "skipped",
    [[], [BattlePhase.shooting, BattlePhase.charge], list(BattlePhase)],
)
def test_phase_chips_mark_skipped_phases(skipped: list[BattlePhase]) -> None:
    """Every phase gets a chip; the config's skipped ones are flagged so the HUD
    can dim them instead of leaving `skip_phases` invisible.

    The clock points at a phase whether or not it is skipped, so exactly one chip
    is current in every case — including when that chip is also a skipped one.
    """
    env = _env(skip_phases=skipped)
    hud = build_scene(env, compute_objective_control(env), scale=51.2).hud

    assert len(hud.phase_chips) == len(BattlePhase)
    assert [chip.is_skipped for chip in hud.phase_chips] == [
        phase in skipped for phase in BattlePhase
    ]
    assert sum(chip.is_current for chip in hud.phase_chips) == 1


@pytest.mark.parametrize("n_rounds", [1, 5, 30, 200])
def test_round_track_holds_its_width_at_any_round_count(n_rounds: int) -> None:
    """A 200-round config must not draw a 3000px track: the panel is fixed width
    and degrades to a continuous fill once segments get thinner than the gaps."""
    from wargame_rl.wargame.envs.renders.v2.presenters.base import _TRACK_W

    presenter = RecordingRenderer(_build_backend("pillow"))
    env = _env(number_of_battle_rounds=n_rounds)
    presenter.setup(env)
    frame = presenter._backend.new_canvas(400, 40, (0, 0, 0))
    presenter._draw_round_track(frame, 10, 20, min(2, n_rounds), n_rounds)

    rgb = presenter._backend.to_rgb_array(frame)
    painted = np.argwhere(rgb.any(axis=2))
    assert painted.size > 0
    assert painted[:, 1].max() <= 10 + _TRACK_W


def test_south_panel_renders_with_many_calculators() -> None:
    """Eleven components is eleven thinner segments, not an overflowing row."""
    presenter = RecordingRenderer(_build_backend("pillow"))
    env = _env(number_of_wargame_models=2)
    presenter.setup(env)
    env.step(WargameEnvAction(actions=[0, 0]))
    scene = presenter._scene_for(env)
    many = {f"calc_{i}": (0.1 if i % 2 else -0.05) for i in range(11)}
    hud = replace(scene.hud, reward_breakdown=tuple(many.items()))

    frame = presenter._compose_scene(replace(scene, hud=hud))
    rgb = presenter._backend.to_rgb_array(frame)
    assert rgb.shape[1] == 1024  # nothing widened the frame


# --- Key map behind Tab, and a readout that does not shift -------------------


def test_tab_toggles_the_key_map_and_the_panel_only_hints_at_it() -> None:
    """The panel names exactly one key; the rest live behind it. A recording has
    no keys at all, so it gets neither the hint nor the overlay."""
    import pygame

    from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
        InteractiveRenderer,
    )

    env = _env(number_of_wargame_models=2)
    presenter = InteractiveRenderer(_build_backend("pillow"))
    presenter.setup(env)

    assert presenter._hotkey_hint() == "[Tab] keys"
    assert presenter.key_map()  # the overlay has something to show
    assert RecordingRenderer(_build_backend("pillow"))._hotkey_hint() is None

    plain = presenter._backend.to_rgb_array(presenter._compose_with_tooltip(env))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
    presenter._process_events(env)
    assert presenter._show_keys
    with_keys = presenter._backend.to_rgb_array(presenter._compose_with_tooltip(env))
    assert not np.array_equal(plain, with_keys)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
    presenter._process_events(env)
    assert not presenter._show_keys


def test_round_readout_holds_its_width_when_the_round_gains_a_digit() -> None:
    """Round 6 → 10 of 20 must not shove the track along: the round sits in a
    field as wide as the round count, so the readout is the same width all game."""
    presenter = RecordingRenderer(_build_backend("pillow"))
    env = _env(number_of_battle_rounds=20)
    presenter.setup(env)
    scene = presenter._scene_for(env)

    def track_right_edge(round_number: int) -> int:
        hud = replace(scene.hud, round=round_number, n_rounds=20)
        rgb = presenter._backend.to_rgb_array(
            presenter._compose_scene(replace(scene, hud=hud))
        )
        # The clock row of the south panel, left half only (the reward is centred).
        row = presenter._window_h - 2 * DEFAULT_THEME.north_panel_h + 18
        band = rgb[row - 3 : row + 4, : presenter._window_w // 3]
        painted = np.argwhere((band != DEFAULT_THEME.palette.panel_bg).any(axis=2))
        return int(painted[:, 1].max())

    assert track_right_edge(6) == track_right_edge(16)


# --- Shooting: what landed, on whom, and for how long ------------------------


def _volley_env() -> tuple[WargameEnv, Any]:
    """A 25v25 shooting scenario stepped to a volley that did damage."""
    from pydantic_yaml import parse_yaml_file_as

    from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy

    config = parse_yaml_file_as(
        WargameEnvConfig, "configs/golden/25v25_shooting_opponent.yaml"
    )
    env = WargameEnv(config=config)
    observation, _ = env.reset(seed=4)
    policy = build_baseline_policy("squad_march_shoot")
    for _ in range(6):
        action = policy.select_action(
            env.player_models, env, action_mask=observation.action_mask
        )
        observation, _r, _t, _tr, _i = env.step(action)
    return env, observation


def test_only_damaging_shots_are_drawn() -> None:
    """A volley is mostly misses — 21 shots, 4 of them damaging on this seed —
    and drawing the misses buries the ones that landed."""
    env, _ = _volley_env()
    shots = env.last_player_shooting_results + env.last_opponent_shooting_results
    damaging = [s for s in shots if s.result.damage_dealt > 0]
    assert len(damaging) < len(shots)  # the seed still has misses to omit

    scene = build_scene(env, compute_objective_control(env), scale=51.2)
    pal = DEFAULT_THEME.palette
    tracers = [
        p
        for p in scene.primitives
        if isinstance(p, Seg) and p.color in (pal.shot_player, pal.shot_opponent)
    ]
    assert len(tracers) == len(damaging)


def test_a_tracer_runs_between_the_two_models_involved() -> None:
    """The line has to start at the shooter and end at its target, or "who shot
    whom" is decoration rather than information."""
    env, _ = _volley_env()
    scene = build_scene(env, compute_objective_control(env), scale=51.2)
    pal = DEFAULT_THEME.palette

    hits = [s for s in env.last_opponent_shooting_results if s.result.damage_dealt > 0]
    assert hits, "seed expected to produce opponent hits"
    shot = hits[0]
    attacker = env.opponent_models[shot.attacker_idx].location
    target = env.player_models[shot.target_idx].location

    tracer = next(
        p
        for p in scene.primitives
        if isinstance(p, Seg)
        and p.color == pal.shot_opponent
        and math.isclose(p.a[0], attacker[0], abs_tol=1.0)
        and math.isclose(p.b[0], target[0], abs_tol=1.0)
    )
    assert math.isclose(tracer.a[1], attacker[1], abs_tol=1.0)
    assert math.isclose(tracer.b[1], target[1], abs_tol=1.0)


def test_a_volley_fades_out_over_a_few_frames() -> None:
    """Shooting results sit on the env until the next volley, so without a fade
    a movement frame keeps drawing the last firefight at full strength."""
    from wargame_rl.wargame.envs.renders.v2.scene import (
        SHOT_FADE_FRAMES,
        shot_fade_for_age,
    )

    env, _ = _volley_env()
    presenter = RecordingRenderer(_build_backend("pillow"))
    presenter.setup(env)
    pal = DEFAULT_THEME.palette

    def tracer_colours() -> list[tuple[int, int, int]]:
        scene = presenter._scene_for(env)
        return [
            p.color
            for p in scene.primitives
            if isinstance(p, Seg) and p.width >= 3 and p.color != pal.grid
        ]

    first = presenter._scene_for(env)
    fresh = [p for p in first.primitives if isinstance(p, Seg)]
    # Re-rendering the same unchanged results ages the volley out.
    for _ in range(SHOT_FADE_FRAMES):
        presenter._scene_for(env)
    faded = presenter._scene_for(env)

    assert shot_fade_for_age(0) == 1.0
    assert shot_fade_for_age(SHOT_FADE_FRAMES) == 0.0
    assert len([p for p in faded.primitives if isinstance(p, Seg)]) < len(fresh)
    assert tracer_colours() is not None  # colours resolve at every age


def test_a_kill_round_trips_through_a_recording() -> None:
    """`killed` cannot be recovered from a snapshot after the fact — several
    shooters may hit one model and only one made the kill — so schema 2.3
    records it, and a replayed killing shot draws as a kill."""
    env, _ = _volley_env()
    kills = [
        s
        for s in env.last_player_shooting_results + env.last_opponent_shooting_results
        if s.killed
    ]
    assert kills, "seed expected to produce a kill"

    snapshot = env.to_snapshot()
    recorded = snapshot.player_combat_results + snapshot.opponent_combat_results
    assert any(r.killed for r in recorded)

    live = build_scene(env, compute_objective_control(env), scale=51.2)
    replayed = build_scene_from_snapshot(snapshot, scale=51.2)
    assert replayed.primitives == live.primitives
