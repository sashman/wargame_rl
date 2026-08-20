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
from wargame_rl.wargame.envs.types import TerrainMapConfig, WargameEnvConfig

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
