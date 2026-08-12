"""The real-map evaluation swaps terrain onto an unchanged scenario.

The failure this guards against is silent: if `random_terrain` survives the
override, `reset` regenerates a layout and discards the map, so the run scores
the training distribution while printing a map's name. Nothing raises, and the
table looks exactly as it should.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
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

MAP_WITH_OBJECTIVES_YAML = """
name: table_02
terrain:
  - { footprint: [12, 8, 18, 14] }
objectives:
  - { x: 11.86, y: 17.45 }
  - { x: 24.79, y: 24.08 }
  - { x: 48.16, y: 26.55 }
"""

MAP_WITH_AREA_OBJECTIVES_YAML = """
name: table_03
terrain:
  - { footprint: [12, 8, 18, 14] }
  - { footprint: [27, 20, 33, 26] }
objectives:
  - area: [[12, 8], [19, 8], [19, 15], [12, 15]]
  - area: [[27, 20], [34, 20], [34, 27], [27, 27]]
"""

SHIPPED_MAPS = Path(__file__).resolve().parents[1] / "configs" / "evaluation" / "maps"


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
    # The map authors inclusive cell rectangles; the env holds continuous ones,
    # so the far corner is one further out. `Footprint.from_cell_rect` is the
    # single boundary where that conversion happens.
    expected = [(12.0, 8.0, 19.0, 15.0), (27.0, 20.0, 34.0, 27.0)]

    env = create_environment(env_config=config_for_map(base, terrain_map))
    try:
        for seed in (700_000, 700_001):
            env.reset(seed=seed)
            actual = [(f.x0, f.y0, f.x1, f.y1) for f in env.terrain.footprints]
            assert actual == expected
    finally:
        env.close()


def test_the_scenario_is_otherwise_untouched() -> None:
    """Only terrain may differ — that is what keeps evaluation comparable.

    A terrain-only map must leave the objective settings alone too, which is
    what keeps every map written before objectives existed scoring as it did.
    """
    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_YAML)

    config = config_for_map(base, terrain_map)

    ignored = {"terrain", "random_terrain", "render_mode"}
    before = base.model_dump(exclude=ignored)
    after = config.model_dump(exclude=ignored)
    assert before == after


def test_a_maps_objectives_replace_the_scenarios() -> None:
    """A map's objectives are part of the layout, count included."""
    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_WITH_OBJECTIVES_YAML)

    config = config_for_map(base, terrain_map)

    assert config.number_of_objectives == 3  # the scenario asked for 1
    assert config.objectives is not None
    assert [(o.x, o.y) for o in config.objectives] == [
        (11.86, 17.45),
        (24.79, 24.08),
        (48.16, 26.55),
    ]
    assert config.has_fixed_objective_positions


def test_map_objectives_survive_to_the_board() -> None:
    """End to end: the env resets objectives onto the map's exact points.

    Sub-inch coordinates are the point of the assertion. The real layouts were
    measured off the printed boards, and the two centre markers can sit 4in
    apart — rounding them to whole inches would move an objective by up to a
    quarter of its own radius and break the layout's rotational symmetry.
    """
    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_WITH_OBJECTIVES_YAML)

    env = create_environment(env_config=config_for_map(base, terrain_map))
    try:
        env.reset(seed=700_000)
        placed = [(o.location[0], o.location[1]) for o in env.objectives]
        assert placed == pytest.approx([(11.86, 17.45), (24.79, 24.08), (48.16, 26.55)])
    finally:
        env.close()


def test_an_undetermined_objective_is_rejected() -> None:
    """One bare entry would silently randomise every objective on the map."""
    with pytest.raises(ValueError, match="must be determined"):
        parse_yaml_raw_as(
            TerrainMapConfig,
            "name: t\nterrain: []\nobjectives:\n  - { x: 5, y: 5 }\n  - {}\n",
        )


def test_area_objectives_are_the_ground_and_do_not_move() -> None:
    """The shipped maps make each objective a ruin, so it is placed by its outline.

    An area is not *placed* — `objective_placement` skips it and its location is
    the outline's centroid — so the guard is that a reseed leaves it alone. A
    regression here would draw a fresh centre and move the marker off its own
    ground while leaving the area behind.
    """
    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_WITH_AREA_OBJECTIVES_YAML)

    config = config_for_map(base, terrain_map)
    assert config.number_of_objectives == 2

    env = create_environment(env_config=config)
    try:
        env.reset(seed=700_000)
        assert all(objective.is_area for objective in env.objectives)
        before = [tuple(objective.location) for objective in env.objectives]
        env.reset(seed=700_001)
        assert [tuple(o.location) for o in env.objectives] == before
    finally:
        env.close()


def test_a_preview_renders_from_the_map_file(tmp_path: Path) -> None:
    """The preview generator must keep working, or the images go stale again.

    The 45 previews were originally produced ad hoc and committed without the
    code that made them, so the first objective change left every one of them
    wrong with nothing to regenerate from.
    """
    from scripts.render_maps import render_map

    base = _scenario_with_random_terrain()
    terrain_map = parse_yaml_raw_as(TerrainMapConfig, MAP_WITH_OBJECTIVES_YAML)
    out = tmp_path / "table_02.png"

    render_map(base, terrain_map, out)

    with Image.open(out) as image:
        assert image.width == 1024
        assert image.height > 0


def test_every_shipped_map_objective_is_one_of_its_ruins() -> None:
    """The 45 real layouts make each objective the ground a marker sits on.

    Six markers per layout, but the two centre ones share the board's largest
    ruin on 24 of the 45, and one piece of ground is held once — so a map
    carries five or six objectives and each outline must be a terrain piece it
    actually has. An outline that matched nothing would mean a marker was
    attributed to the wrong ruin.
    """
    maps = load_maps(SHIPPED_MAPS)

    assert len(maps) == 45
    sizes = set()
    for terrain_map in maps:
        assert terrain_map.objectives is not None, terrain_map.name
        sizes.add(len(terrain_map.objectives))
        pieces = {
            tuple(map(tuple, piece.to_polygon().vertices))
            for piece in terrain_map.terrain
        }
        outlines = []
        for objective in terrain_map.objectives:
            area = objective.to_polygon()
            assert area is not None, terrain_map.name
            outline = tuple(map(tuple, area.vertices))
            assert outline in pieces, terrain_map.name
            outlines.append(outline)
        assert len(set(outlines)) == len(outlines), f"{terrain_map.name}: duplicate"
    assert sizes == {5, 6}
