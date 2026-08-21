"""The threat-range overlays: do they draw the rule the engine actually applies?

The load-bearing test is `test_the_threat_region_is_the_engines_own_answer`.
The overlay exists to be *believed* when it disagrees with intuition about the
geometry — "why can that squad shoot me from there" — and it can only be
believed if it is a picture of the engine's own range-and-sight test rather than
of a shape the renderer worked out for itself.
"""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from wargame_rl.wargame.envs.renders.v2.backend import rasterize  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.camera import Camera  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.control import (  # noqa: E402
    THREAT_SPACING,
    Ring,
    ThreatOptions,
    compute_engagement_zone,
    compute_objective_control,
    compute_threat_overlay,
    compute_threat_region,
    engagement_radius,
)
from wargame_rl.wargame.envs.renders.v2.factory import build_backend  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.scene import (  # noqa: E402
    Disc,
    DiscUnion,
    Poly,
    Scene,
    build_scene,
)
from wargame_rl.wargame.envs.types.config import (  # noqa: E402
    ModelConfig,
    OpponentPolicyConfig,
    TerrainPieceConfig,
    WargameEnvConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase  # noqa: E402
from wargame_rl.wargame.envs.wargame import WargameEnv  # noqa: E402

_ON = ThreatOptions(show_threat=True, show_engagement=True)


def _env(**overrides: Any) -> WargameEnv:
    """Two small squads that can actually shoot, on a 20x20 board."""
    rifle = [WeaponProfile(range=8, attacks=1)]
    base: dict[str, Any] = dict(
        board_width=20,
        board_height=20,
        number_of_wargame_models=3,
        number_of_opponent_models=3,
        number_of_objectives=1,
        number_of_battle_rounds=4,
        models=[ModelConfig(x=4, y=5 + i, weapons=rifle) for i in range(3)],
        opponent_models=[ModelConfig(x=15, y=5 + i, weapons=rifle) for i in range(3)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        render_mode=None,
    )
    base.update(overrides)
    return WargameEnv(config=WargameEnvConfig(**base))


def _walled_env() -> WargameEnv:
    """A wall beside the player squad, so some in-range ground is hidden."""
    return _env(terrain=[TerrainPieceConfig(footprint=(7, 3, 8, 16))])


def _cell_centres(env: WargameEnv) -> list[tuple[float, float]]:
    return [
        (x + 0.5, y + 0.5)
        for y in range(env.config.board_height)
        for x in range(env.config.board_width)
    ]


def _inside(rings: tuple[Ring, ...], px: float, py: float) -> bool:
    """Even-odd containment. A hole ring flips the parity, which is the right
    answer for free — an unthreatened pocket comes back as its own ring."""
    crossings = 0
    for ring in rings:
        for i, (x0, y0) in enumerate(ring):
            x1, y1 = ring[(i + 1) % len(ring)]
            if (y0 > py) != (y1 > py):
                if px < x0 + (py - y0) * (x1 - x0) / (y1 - y0):
                    crossings += 1
    return crossings % 2 == 1


def _threatened_by_engine(env: WargameEnv, px: float, py: float) -> bool:
    """The engine's own answer: some alive shooter in range of, and seeing, here."""
    return any(
        model.is_alive
        and rng > 0
        and float(np.hypot(model.location[0] - px, model.location[1] - py)) <= rng
        and env.has_line_of_sight_between_points(
            float(model.location[0]), float(model.location[1]), px, py
        )
        for model, rng in zip(env.player_models, env.player_max_ranges, strict=True)
    )


def test_the_threat_region_is_the_engines_own_answer() -> None:
    """Every sampled cell is inside the rings exactly when the engine says it is
    both in range and visible.

    Asserted at `smooth=0`, because Chaikin deliberately moves the boundary — a
    smoothed ring answers a slightly different question at the edge, which is
    precisely why the smoothing is a parameter and not baked in.
    """
    env = _walled_env()
    env.reset(seed=5)

    rings = compute_threat_region(
        env, env.player_models, env.player_max_ranges, smooth=0
    )

    for px, py in _cell_centres(env):
        assert _inside(rings, px, py) is _threatened_by_engine(env, px, py), (px, py)


def test_the_threat_stops_at_the_wall() -> None:
    """Stated on its own, though (1) subsumes it: the region is range *and*
    sight, not a circle. In-range ground behind the wall is outside it."""
    env = _walled_env()
    env.reset(seed=5)

    rings = compute_threat_region(
        env, env.player_models, env.player_max_ranges, smooth=0
    )

    hidden = [
        (px, py)
        for px, py in _cell_centres(env)
        if not _threatened_by_engine(env, px, py)
        and min(
            float(np.hypot(m.location[0] - px, m.location[1] - py))
            for m in env.player_models
        )
        <= 8.0
    ]
    assert hidden, "the wall should hide some ground the squad is in range of"
    assert not any(_inside(rings, px, py) for px, py in hidden)


@pytest.mark.parametrize("smooth", [0, 1, 2, 3, 5])
def test_no_alive_shooter_falls_outside_its_own_threat_region(smooth: int) -> None:
    """A shooter always threatens the ground it is standing on, so this holds by
    construction — which is what makes it a cheap trap for the whole class of
    boundary bug.

    It caught a real one: collinear-collapsing a ring before smoothing let
    Chaikin move a vertex by a quarter of a twenty-inch board-edge run rather
    than of a cell, hauling the boundary five inches inward and swallowing the
    models standing on it. Only the edge models were affected, so nothing else
    noticed.
    """
    env = _walled_env()
    env.reset(seed=5)

    rings = compute_threat_region(
        env, env.player_models, env.player_max_ranges, smooth=smooth
    )

    for model in env.player_models:
        if not model.is_alive:
            continue
        x, y = float(model.location[0]), float(model.location[1])
        assert _inside(rings, x, y), (smooth, x, y)


def test_the_engagement_radius_is_the_one_the_shooting_gate_uses() -> None:
    """The gate is `nearest_live - 2 * base_radius > engagement_range`, so the
    radius drawn around a centre is `engagement_range + 2 * base_radius`.

    The trap is `model.base_radius`: the engine measures with the config's
    resolved global, so a per-model reading would agree today and diverge
    silently the day a config sets them differently.
    """
    env = _env()
    env.reset(seed=5)
    rules = env.rules_quantities

    assert engagement_radius(env) == pytest.approx(
        rules.engagement_range + 2.0 * rules.base_radius
    )


def test_casualties_cast_no_threat_and_no_engagement() -> None:
    """The engine masks dead models out of both the range gate and the
    engagement gate — a corpse that kept projecting either would be drawing the
    bug that pinned models for whole episodes until 2026-08-19."""
    env = _walled_env()
    env.reset(seed=5)

    before_threat = compute_threat_region(
        env, env.player_models, env.player_max_ranges, smooth=0
    )
    before_zone = compute_engagement_zone(env.player_models)

    env.player_models[0].stats["current_wounds"] = 0
    assert not env.player_models[0].is_alive

    after_threat = compute_threat_region(
        env, env.player_models, env.player_max_ranges, smooth=0
    )
    after_zone = compute_engagement_zone(env.player_models)

    assert len(after_zone) == len(before_zone) - 1
    dead = (
        float(env.player_models[0].location[0]),
        float(env.player_models[0].location[1]),
    )
    assert dead not in after_zone
    assert after_threat != before_threat


def test_an_unarmed_side_threatens_nothing() -> None:
    """`range > 0` guards the gate: an unarmed model has range 0.0, and `0 <= 0`
    would mark the cell it stands on as threatened by a model that cannot fire."""
    env = _env(models=[ModelConfig(x=4, y=5 + i, weapons=[]) for i in range(3)])
    env.reset(seed=5)

    assert (
        compute_threat_region(env, env.player_models, env.player_max_ranges, smooth=0)
        == ()
    )


def test_the_overlays_are_drawn_under_the_models() -> None:
    """A tool that hides the pieces it is describing is worse than one that
    draws less — the same rule the sight shadow follows."""
    env = _walled_env()
    env.reset(seed=5)
    overlay = compute_threat_overlay(env, _ON)

    scene = build_scene(env, compute_objective_control(env), scale=16.0, threat=overlay)

    prims = list(scene.primitives)
    last_overlay = max(
        i
        for i, p in enumerate(prims)
        if isinstance(p, DiscUnion)
        or (isinstance(p, Poly) and p.outline is not None and p.fill is not None)
    )
    first_model = min(
        i for i, p in enumerate(prims) if isinstance(p, Disc) and p.outline is not None
    )
    assert last_overlay < first_model


def test_the_overlay_is_absent_unless_asked_for() -> None:
    """Off by default everywhere: `build_scene` without a `threat` argument must
    emit byte-identical primitives to before the feature existed."""
    env = _walled_env()
    env.reset(seed=5)
    control = compute_objective_control(env)

    assert (
        build_scene(env, control, scale=16.0).primitives
        == build_scene(env, control, scale=16.0, threat=None).primitives
    )
    assert not any(
        isinstance(p, DiscUnion)
        for p in build_scene(env, control, scale=16.0).primitives
    )


@pytest.mark.parametrize("backend_name", ["pygame", "pygame_aa", "pillow"])
def test_the_engagement_fill_composites_once(backend_name: str) -> None:
    """The whole reason `DiscUnion` exists, and it is only testable in pixels.

    Two overlapping translucent discs drawn as two primitives blend twice, so
    their intersection comes out darker than either. Painted into one layer and
    composited once, the overlap must be the *same* colour as a lone disc.
    """
    backend = build_backend(backend_name)
    fill = (60, 130, 255, 96)
    canvas = backend.new_canvas(120, 60, (255, 255, 255))
    backend.draw_disc_union(canvas, [(40.0, 30.0), (70.0, 30.0)], 24.0, fill)
    pixels = backend.to_rgb_array(canvas)

    overlap = pixels[30, 55]  # between the two centres, inside both discs
    lone = pixels[30, 20]  # inside the left disc only
    np.testing.assert_allclose(overlap, lone, atol=2)
    assert not np.array_equal(lone, np.array([255, 255, 255], dtype=np.uint8))


def test_threat_rings_and_engagement_reach_the_scene() -> None:
    """The overlay must actually become primitives — a silently empty overlay
    would make every other test here vacuous."""
    env = _walled_env()
    env.reset(seed=5)
    overlay = compute_threat_overlay(env, _ON)
    assert overlay.player_threat and overlay.player_engagement
    assert not overlay.is_empty()

    scene: Scene = build_scene(
        env, compute_objective_control(env), scale=16.0, threat=overlay
    )
    canvas = build_backend("pillow").new_canvas(320, 320, scene.board_bg)
    rasterize(build_backend("pillow"), scene, Camera(16.0), canvas)

    assert sum(isinstance(p, DiscUnion) for p in scene.primitives) == 2
    assert sum(
        isinstance(p, Poly) and p.fill is not None and p.outline is not None
        for p in scene.primitives
    ) >= len(overlay.player_threat)


def test_the_sampled_grid_is_shared_with_the_sight_shadow() -> None:
    """Both sweeps must sample the same points, or the two overlays can disagree
    about the same piece of ground while both claiming to be the engine."""
    from wargame_rl.wargame.envs.renders.v2.control import SHADOW_SPACING

    assert THREAT_SPACING == SHADOW_SPACING


# --- configuration surface ---------------------------------------------------


def test_the_overlays_are_off_by_default_everywhere() -> None:
    """Without this, "off by default" is an unpinned intention — and a training
    run that quietly started drawing a 12" footprint over every video would be
    found by watching one, which is late."""
    from wargame_rl.wargame.envs.renders.v2.factory import build_renderer

    assert not ThreatOptions().enabled
    for mode in ("interactive", "recording"):
        renderer = build_renderer("v2", mode)
        assert renderer.threat_options == ThreatOptions()  # type: ignore[attr-defined]
        assert not renderer.threat_options.enabled  # type: ignore[attr-defined]


def test_a_nonsense_sweep_is_rejected_when_the_renderer_is_built() -> None:
    """Validation at construction, not at draw time: a bad grid should fail on
    the command line, not several thousand frames into a recording."""
    with pytest.raises(ValueError, match="spacing must be positive"):
        ThreatOptions(spacing=0.0)
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        ThreatOptions(smoothing=-1)


def test_the_toggles_flip_one_switch_and_leave_the_sweep_alone() -> None:
    """`[R]` and `[E]` must not quietly reset a tuned grid or smoothing."""
    from wargame_rl.wargame.envs.renders.v2.factory import build_renderer

    renderer = build_renderer(
        "v2", "recording", threat_options=ThreatOptions(spacing=2.0, smoothing=0)
    )
    renderer.toggle_threat()  # type: ignore[attr-defined]
    options = renderer.threat_options  # type: ignore[attr-defined]
    assert options.show_threat and not options.show_engagement
    assert (options.spacing, options.smoothing) == (2.0, 0)

    renderer.toggle_engagement()  # type: ignore[attr-defined]
    assert renderer.threat_options.show_engagement  # type: ignore[attr-defined]
    assert renderer.threat_options.show_threat  # type: ignore[attr-defined]


def test_the_cli_helper_and_the_options_object_agree() -> None:
    """One parser for five call sites, so `simulate`, `play`, `debug`, the
    training recorder and `replay-render` cannot drift."""
    from wargame_rl.wargame.envs.renders.v2 import threat_options

    assert threat_options() == ThreatOptions()
    assert threat_options(True, True, 2.0, 1) == ThreatOptions(
        show_threat=True, show_engagement=True, spacing=2.0, smoothing=1
    )


def test_every_presenter_takes_the_options() -> None:
    """The recorder has no keyboard, so if it did not accept them at
    construction there would be no way to get an overlay into a video at all."""
    from wargame_rl.wargame.envs.renders.v2.factory import build_backend
    from wargame_rl.wargame.envs.renders.v2.presenters.debug import (
        DebugControls,
        DebugPresenter,
    )

    on = ThreatOptions(show_threat=True)
    backend = build_backend("pillow")
    assert DebugPresenter(backend, DebugControls(), threat_options=on).threat_options
    from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (
        InteractiveRenderer,
    )
    from wargame_rl.wargame.envs.renders.v2.presenters.recording import (
        RecordingRenderer,
    )

    assert InteractiveRenderer(backend, threat_options=on).threat_options == on
    assert RecordingRenderer(backend, threat_options=on).threat_options == on


def test_the_sweep_is_cached_between_frames_of_one_board() -> None:
    """A paused debug window polls at 30 fps and the sweep is thousands of rays;
    without the cache it would recompute all of them for an unchanged board."""
    from wargame_rl.wargame.envs.renders.v2.factory import build_renderer

    env = _walled_env()
    env.reset(seed=5)
    renderer = build_renderer(
        "v2", "recording", threat_options=ThreatOptions(show_threat=True)
    )

    first = renderer._threat_overlay(env)  # type: ignore[attr-defined]
    assert first is not None
    assert renderer._threat_overlay(env) is first  # type: ignore[attr-defined]

    env.player_models[0].location = env.player_models[0].location + 3.0
    assert renderer._threat_overlay(env) is not first  # type: ignore[attr-defined]


def test_a_view_that_cannot_trace_sight_draws_nothing() -> None:
    """The replay adapter has terrain but not yet the ranges or its own sight
    predicate. Drawing the engagement half alone would read as "nothing is
    threatened" rather than "not known here", which is worse than drawing none.
    """
    from wargame_rl.wargame.envs.renders.v2.factory import build_renderer

    class _Partial:
        player_models: list[Any] = []
        opponent_models: list[Any] = []

    renderer = build_renderer(
        "v2", "recording", threat_options=ThreatOptions(show_threat=True)
    )
    assert renderer._threat_overlay(_Partial()) is None  # type: ignore[attr-defined]


# --- replay ------------------------------------------------------------------


def _recorded(env: WargameEnv) -> Any:
    """One snapshot of the env as it stands, through the env's own exporter."""
    return env.to_snapshot()


def test_a_replayed_threat_region_is_the_live_one() -> None:
    """The strongest test here: it is the only thing that catches a lost
    `los_sample_step`, a dropped `blocking_mask`, or a weapon-range derivation
    that disagrees with `max_weapon_ranges`.

    Any of those would draw a *plausible* region that quietly answered a
    different question than the engine did.
    """
    from wargame_rl.wargame.envs.renders.v2.replay import _snapshot_to_view

    env = _walled_env()
    env.reset(seed=5)

    live = compute_threat_overlay(env, _ON)
    replayed = compute_threat_overlay(cast(Any, _snapshot_to_view(_recorded(env))), _ON)

    assert live.player_threat == replayed.player_threat
    assert live.opponent_threat == replayed.opponent_threat
    assert live.player_engagement == replayed.player_engagement
    assert live.engagement_radius == replayed.engagement_radius


def test_a_replayed_view_traces_the_same_sight_as_the_engine() -> None:
    """Cell for cell, not just in aggregate — a region can match while the two
    sight answers differ in compensating places."""
    from wargame_rl.wargame.envs.renders.v2.replay import _snapshot_to_view

    env = _walled_env()
    env.reset(seed=5)
    view = _snapshot_to_view(_recorded(env))

    origin = np.array([[2.5, 10.0]], dtype=float)
    targets = np.array(_cell_centres(env), dtype=float)
    replayed = np.asarray(view.line_of_sight_matrix(origin, targets))[0]

    for (px, py), seen in zip(_cell_centres(env), replayed, strict=True):
        assert bool(seen) is env.has_line_of_sight_between_points(2.5, 10.0, px, py), (
            px,
            py,
        )


def test_a_pre_2_6_recording_replays_without_the_overlay() -> None:
    """Older recordings carry no sample step. Drawing a *plausible wrong* region
    would be worse than drawing none, so the presenter declines rather than
    defaulting the values."""
    from wargame_rl.wargame.envs.renders.v2.factory import build_renderer
    from wargame_rl.wargame.envs.renders.v2.replay import _snapshot_to_view

    env = _walled_env()
    env.reset(seed=5)
    old = _recorded(env).model_copy(update={"rules": None, "schema_version": "2.5"})
    view = _snapshot_to_view(old)

    renderer = build_renderer("v2", "recording", threat_options=_ON)
    assert renderer._threat_overlay(view) is None  # type: ignore[attr-defined]
    # ...and the frame still builds, which is what "degrades" has to mean.
    assert build_scene(cast(Any, view), (), scale=16.0, threat=None).primitives


def test_the_recorded_rules_are_the_resolved_ones() -> None:
    """In units, already scaled — a replay that divided by `inches_per_unit`
    itself would be re-deriving a quantity the env already owns."""
    env = _env(inches_per_unit=2.0)
    env.reset(seed=5)

    rules = _recorded(env).rules

    assert rules is not None
    assert rules.engagement_range == pytest.approx(
        env.rules_quantities.engagement_range
    )
    assert rules.base_radius == pytest.approx(env.rules_quantities.base_radius)
    assert rules.los_sample_step == pytest.approx(env.rules_quantities.los_sample_step)


def _replay_source(env: WargameEnv, steps: int = 2) -> Any:
    """A tiny recording of `env`, as the replay player consumes it."""
    from wargame_rl.wargame.envs.renders.v2.replay import ReplaySource
    from wargame_rl.wargame.envs.state.event_log import EventLog
    from wargame_rl.wargame.envs.types.env_action import WargameEnvAction

    log = EventLog()
    log.record_reset(env.to_snapshot())
    for _ in range(steps):
        mask = env.observation.action_mask
        assert mask is not None
        env.step(WargameEnvAction.random(mask))
        log.record_step(env.to_snapshot())
    return ReplaySource(
        snapshots=[log.snapshot_at(i) for i in range(len(log))],
        anchor_indices=frozenset({0}),
    )


def _replayed_frame(source: Any, options: ThreatOptions) -> np.ndarray:
    from wargame_rl.wargame.envs.renders.v2.factory import build_backend
    from wargame_rl.wargame.envs.renders.v2.replay import ReplayPresenter

    backend = build_backend("pillow")
    presenter = ReplayPresenter(backend, source, threat_options=options)
    return np.asarray(
        backend.to_rgb_array(presenter.frame_at(len(source) - 1)), dtype=int
    )


def test_the_overlay_actually_reaches_the_replayed_frame() -> None:
    """Computing the overlay is not drawing it.

    `ReplayPresenter` composes through `build_scene_from_snapshot`, not through
    `BasePresenter._scene_for`, so every unit test here passed while the replay
    frame came out untouched — the geometry was right and nothing carried it to
    the scene. Only a frame comparison catches that, which is why this asserts
    on pixels rather than on primitives.
    """
    env = _walled_env()
    env.reset(seed=5)
    source = _replay_source(env)

    off = _replayed_frame(source, ThreatOptions())
    on = _replayed_frame(source, _ON)

    assert (np.abs(on - off).sum(axis=2) > 6).mean() > 0.02


def test_a_pre_2_6_replay_frame_is_the_overlay_off_frame() -> None:
    """Degrading has to mean *identical*, not merely "did not crash"."""
    from wargame_rl.wargame.envs.renders.v2.replay import ReplaySource

    env = _walled_env()
    env.reset(seed=5)
    source = _replay_source(env)
    old = ReplaySource(
        snapshots=[
            s.model_copy(update={"rules": None, "schema_version": "2.5"})
            for s in source.snapshots
        ],
        anchor_indices=source.anchor_indices,
    )

    np.testing.assert_array_equal(
        _replayed_frame(old, _ON), _replayed_frame(source, ThreatOptions())
    )
