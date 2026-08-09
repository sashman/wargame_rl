"""The real-map evaluation swaps terrain onto an unchanged scenario.

The failure this guards against is silent: if `random_terrain` survives the
override, `reset` regenerates a layout and discards the map, so the run scores
the training distribution while printing a map's name. Nothing raises, and the
table looks exactly as it should.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_maps import config_for_map, load_maps
from wargame_rl.wargame.envs.types import (
    RandomTerrainConfig,
    TerrainMapConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.model.common.factory import create_environment

MAP_YAML = """
name: table_01
terrain:
  - { footprint: [12, 8, 18, 14] }
  - { footprint: [27, 20, 33, 26] }
"""


@pytest.fixture
def maps_dir(tmp_path: Path) -> Path:
    (tmp_path / "table_01.yaml").write_text(MAP_YAML)
    return tmp_path


def _scenario_with_random_terrain() -> WargameEnvConfig:
    return WargameEnvConfig(
        board_width=60,
        board_height=44,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=3,
        number_of_battle_rounds=2,
        random_terrain=RandomTerrainConfig(count=5, min_size=3, max_size=5),
    )


def test_load_maps_reads_every_file(maps_dir: Path) -> None:
    maps = load_maps(maps_dir)

    assert [m.name for m in maps] == ["table_01"]
    assert len(maps[0].terrain) == 2


def test_missing_maps_directory_says_so(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="maps directory not found"):
        load_maps(tmp_path / "nope")


def test_empty_maps_directory_explains_the_format(tmp_path: Path) -> None:
    """An empty dir is the normal starting state, so it must not look like a bug."""
    with pytest.raises(SystemExit, match="no maps in"):
        load_maps(tmp_path)


def test_override_replaces_terrain_and_disables_the_generator() -> None:
    """The generator must be cleared, not just overwritten."""
    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_YAML)

    config = config_for_map(base, terrain_map)

    assert config.random_terrain is None
    assert config.terrain is not None
    assert [piece.footprint for piece in config.terrain] == [
        (12, 8, 18, 14),
        (27, 20, 33, 26),
    ]


def test_the_env_actually_runs_on_the_map_terrain() -> None:
    """End to end: the layout the env resets to is the map's, on every episode.

    Asserted across two resets with different seeds, since the defect being
    guarded against only shows up when the generator re-runs.
    """
    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_YAML)
    expected = [(12, 8, 18, 14), (27, 20, 33, 26)]

    env = create_environment(env_config=config_for_map(base, terrain_map))
    try:
        for seed in (700_000, 700_001):
            env.reset(seed=seed)
            actual = [(f.x0, f.y0, f.x1, f.y1) for f in env.terrain.footprints]
            assert actual == expected
    finally:
        env.close()


def test_the_scenario_is_otherwise_untouched() -> None:
    """Only terrain may differ — that is what keeps evaluation comparable."""
    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_YAML)

    config = config_for_map(base, terrain_map)

    ignored = {"terrain", "random_terrain", "render_mode"}
    before = base.model_dump(exclude=ignored)
    after = config.model_dump(exclude=ignored)
    assert before == after
