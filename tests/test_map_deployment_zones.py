"""A map brings its own deployment zones, and they are outlines not rectangles.

Of the six deployments the real tables use, only one pair is an axis-aligned
band. Two are triangles split by a board diagonal, two are stepped staircases
and one is bounded by arcs, so `deployment_zone`'s `(x0, y0, x1, y1)` cannot
describe them and a map carries its own.

The zones come from the layout API, unlike the objective positions: rasterised
against the published layout cards, the tinted region is at least 98% inside the
API's polygon on all 45 tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_maps import config_for_map, load_maps
from wargame_rl.wargame.envs.domain import battle_factory
from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.placement import place_for_episode
from wargame_rl.wargame.envs.map_pool import MapPool
from wargame_rl.wargame.envs.renders.v2.control import compute_objective_control
from wargame_rl.wargame.envs.renders.v2.replay import _snapshot_to_view
from wargame_rl.wargame.envs.renders.v2.scene import Poly, build_scene
from wargame_rl.wargame.envs.types import TerrainMapConfig, WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

CONFIG_PATH = Path("configs/golden/25v25_maps_two_mode.yaml")
SHIPPED_MAPS = Path("configs/evaluation/maps")


def _scenario() -> WargameEnvConfig:
    config = cast(
        WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, CONFIG_PATH.read_text())
    )
    config.render_mode = None
    return config


def _deployed(
    config: WargameEnvConfig, terrain_map: TerrainMapConfig, seed: int = 7
) -> Battle:
    """Positions straight out of placement, before anyone has moved.

    Deliberately not through `env.reset`: that runs the opponent's whole turn
    before it returns, so the opponent's positions there are post-move and a
    check against the zone reads as a failure when nothing is wrong.
    """
    battle = battle_factory.from_config(config_for_map(config, terrain_map))
    place_for_episode(
        battle, config_for_map(config, terrain_map), np.random.default_rng(seed)
    )
    return battle


def test_every_shipped_map_carries_a_deployment() -> None:
    maps = load_maps(SHIPPED_MAPS)

    assert len(maps) == 45
    for terrain_map in maps:
        assert terrain_map.deployment is not None, terrain_map.name
        # Both sides get the same ground: these are tournament tables.
        player = terrain_map.deployment.player_polygon().area
        opponent = terrain_map.deployment.opponent_polygon().area
        assert player == pytest.approx(opponent, rel=0.02), terrain_map.name


def test_the_pool_uses_all_six_shapes() -> None:
    """Six distinct deployments, and the mix is what makes the pool a mission set.

    One of them puts the armies along the *long* edges, twenty inches apart
    across the short axis -- at a twelve inch weapon range that is a materially
    different game from the others, and it is a quarter of the pool.
    """
    names = {m.deployment.name for m in load_maps(SHIPPED_MAPS) if m.deployment}

    assert names == {
        "diagonal_halves",
        "long_edges",
        "opposed_quadrants",
        "short_edges",
        "stepped_bands",
        "stepped_columns",
    }


def test_every_model_deploys_inside_its_own_zone() -> None:
    """The whole point: an outline zone actually constrains placement.

    Sampling happens in the outline's bounding box, so without the outline test
    an army spills outside its zone on every non-rectangular deployment -- which
    is five of the six.
    """
    config = _scenario()

    for terrain_map in load_maps(SHIPPED_MAPS):
        assert terrain_map.deployment is not None
        battle = _deployed(config, terrain_map)
        for models, polygon in (
            (battle.player_models, terrain_map.deployment.player_polygon()),
            (battle.opponent_models, terrain_map.deployment.opponent_polygon()),
        ):
            outside = [
                m
                for m in models
                if not polygon.contains(float(m.location[0]), float(m.location[1]))
            ]
            assert not outside, f"{terrain_map.name}: {len(outside)} models outside"


def test_a_map_without_a_deployment_still_uses_the_rectangle() -> None:
    """The no-op that keeps every generated-terrain config unchanged."""
    config = _scenario()
    terrain_map = load_maps(SHIPPED_MAPS)[0]
    without = terrain_map.model_copy(update={"deployment": None})

    battle = _deployed(config, without)

    x0, _, x1, _ = config.deployment_zone or (0, 0, 20, 44)
    assert all(x0 <= float(m.location[0]) <= x1 for m in battle.player_models)


def test_the_rendered_zone_is_the_zone_models_were_placed_in() -> None:
    """What is drawn must be what was placed into, on both routes.

    This is the check whose absence let the renderers draw the config rectangle
    for a whole training run while placement correctly used each map's polygon:
    every existing test asserted placement, and none compared placement against
    what a viewer sees. On `long_edges` and `stepped_bands` the two do not
    overlap at all.
    """
    config = _scenario()

    for terrain_map in load_maps(SHIPPED_MAPS):
        assert terrain_map.deployment is not None
        battle = _deployed(config, terrain_map)

        for outline, models, side in (
            (battle.deployment_outline, battle.player_models, "player"),
            (battle.opponent_deployment_outline, battle.opponent_models, "opponent"),
        ):
            assert outline is not None, f"{terrain_map.name} {side}: nothing to draw"
            outside = [
                m
                for m in models
                if not outline.contains(float(m.location[0]), float(m.location[1]))
            ]
            assert not outside, (
                f"{terrain_map.name} {side}: {len(outside)} models outside the "
                f"zone the renderer would draw"
            )


def test_the_pool_route_carries_the_zone_to_the_renderer_too() -> None:
    """Training draws from a pool rather than installing one map onto a scenario.

    Two separate wiring routes reach `place_for_episode`, and only the map-pool
    one is what a training run uses -- so a zone that reaches the renderer under
    `config_for_map` says nothing about what the videos show.
    """
    config = _scenario()
    pool = MapPool.from_config(config)
    assert pool is not None

    rect = config.deployment_zone
    assert rect is not None
    outside_the_old_rectangle = 0

    for index in range(12):
        layout = pool.draw(np.random.default_rng(1000 + index))
        battle = battle_factory.from_config(config)
        place_for_episode(battle, config, np.random.default_rng(index), layout=layout)

        outline = battle.deployment_outline
        assert outline is not None, layout.name
        for model in battle.player_models:
            x, y = float(model.location[0]), float(model.location[1])
            assert outline.contains(x, y), layout.name
            if not (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]):
                outside_the_old_rectangle += 1

    # Without this the test passes on a renderer that ignores the outline
    # entirely, since a zone inside the rectangle is drawn correctly by accident.
    assert outside_the_old_rectangle > 0


def test_a_map_without_a_deployment_reports_no_outline_to_draw() -> None:
    """None means "draw the rectangle", and it must not leak between episodes."""
    config = _scenario()
    terrain_map = load_maps(SHIPPED_MAPS)[0]

    battle = _deployed(config, terrain_map)
    assert battle.deployment_outline is not None

    # The same battle, replaced by a map that carries no zones.
    without = terrain_map.model_copy(update={"deployment": None})
    place_for_episode(battle, config_for_map(config, without), np.random.default_rng(3))

    assert battle.deployment_outline is None
    assert battle.opponent_deployment_outline is None


def test_the_v2_scene_draws_the_outline_not_the_rectangle() -> None:
    """The renderer-level check: the zone primitive IS the map's polygon.

    The tests above assert the outline reaches the aggregate. This one asserts
    the renderer uses it, which is the half that was missing -- for a whole
    training run the videos drew `deployment_zone`'s band while the army stood
    somewhere else entirely.
    """
    config = _scenario()
    terrain_map = next(
        m
        for m in load_maps(SHIPPED_MAPS)
        if m.deployment and m.deployment.name == "long_edges"
    )
    assert terrain_map.deployment is not None
    env = create_environment(
        env_config=config_for_map(config, terrain_map), renderer=None
    )
    env.reset()

    scene = build_scene(env, compute_objective_control(env), scale=10.0)
    expected = {
        (round(float(x), 3), round(float(y), 3))
        for x, y in terrain_map.deployment.player_polygon().vertices
    }
    drawn = [
        p
        for p in scene.primitives
        if isinstance(p, Poly)
        and {(round(x, 3), round(y, 3)) for x, y in p.points} == expected
    ]

    assert drawn, "the player deployment zone was not drawn as its own outline"
    # And the old rectangle must be gone, or both are drawn and the band remains.
    rect = config.deployment_zone
    assert rect is not None
    rect_points = {
        (float(rect[0]), float(rect[1])),
        (float(rect[2]), float(rect[1])),
        (float(rect[2]), float(rect[3])),
        (float(rect[0]), float(rect[3])),
    }
    stale = [
        p
        for p in scene.primitives
        if isinstance(p, Poly) and set(p.points) == rect_points
    ]
    assert not stale, "the config rectangle is still being drawn over the real zone"


def test_a_replayed_snapshot_still_knows_its_deployment_zone() -> None:
    """Replay is the second route to a video, and it must not lose the zone.

    A recording carries only what the snapshot schema holds, so an outline that
    lives on the aggregate alone reaches the live renderer and not `just
    replay-render` -- which would leave the same wrong band in half the videos
    this project produces.
    """
    config = _scenario()
    terrain_map = next(
        m
        for m in load_maps(SHIPPED_MAPS)
        if m.deployment and m.deployment.name == "long_edges"
    )
    assert terrain_map.deployment is not None
    env = create_environment(
        env_config=config_for_map(config, terrain_map), renderer=None
    )
    env.reset()

    snapshot = env.to_snapshot()
    assert snapshot.deployment_outline is not None
    assert snapshot.opponent_deployment_outline is not None

    restored = _snapshot_to_view(snapshot)
    assert restored.deployment_outline is not None
    expected = terrain_map.deployment.player_polygon()
    assert restored.deployment_outline.vertices.tolist() == expected.vertices.tolist()
