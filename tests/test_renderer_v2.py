"""Renderer v2: Scene purity, control extraction, FrameSource, and A/B vs legacy.

The Scene tests need no pygame — they pin the board-coordinate conventions (no
`+0.5`, the base-radius rule) that only the renderer can get wrong, now that
those live in `build_scene` rather than in a draw method. The pixel tests render
v2 and the untouched legacy renderer headlessly and check they agree.
"""

from __future__ import annotations

import os
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
from wargame_rl.wargame.envs.renders.v2.replay import (  # noqa: E402
    ReplayPresenter,
    ReplaySource,
    build_scene_from_snapshot,
)
from wargame_rl.wargame.envs.renders.v2.scene import (  # noqa: E402
    Control,
    Disc,
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


def test_v2_frame_matches_legacy_shape_and_coverage() -> None:
    config = _ab_config()
    legacy = _frame("legacy", config)
    v2 = _frame("v2", config)

    assert v2.shape == legacy.shape
    legacy_nonwhite = int((legacy != 255).any(axis=2).sum())
    v2_nonwhite = int((v2 != 255).any(axis=2).sum())
    # Same board, same primitives: coverage should be within a couple of percent
    # (fonts and alpha compositing differ by a hair, so not exact).
    assert abs(v2_nonwhite - legacy_nonwhite) < 0.02 * legacy_nonwhite


def test_v2_draws_the_player_in_its_group_colour() -> None:
    config = _ab_config()
    v2 = _frame("v2", config)

    # Board is 20x20 fit to 1024; the board sits below the 36px north panel.
    scale = 1024 / 20
    north = 36
    px = int(5 * scale)
    py = int(5 * scale) + north
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

    # Same fit as the legacy A/B path: 20-unit board to 1024px under the panel.
    assert frame.shape == (1132, 1024, 3)
    scale = 1024 / 20
    north = 36
    px = int(5 * scale)
    py = int(5 * scale) + north
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
