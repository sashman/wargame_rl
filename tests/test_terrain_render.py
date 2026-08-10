"""Tests for terrain rendering helpers and demo config loading."""

from __future__ import annotations

from wargame_rl.wargame.envs.renders.human import los_line_color
from wargame_rl.wargame.envs.types import TerrainPieceConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def test_los_line_color_blocked() -> None:
    """Returns red when LOS is blocked by a footprint."""
    cfg = WargameEnvConfig(
        board_width=20,
        board_height=20,
        terrain=[TerrainPieceConfig(footprint=(8, 4, 12, 6))],
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    env = WargameEnv(cfg)
    color = los_line_color(env, 0, 5, 19, 5)
    assert color == (255, 80, 80)


def test_los_line_color_clear() -> None:
    """Returns green when LOS is clear."""
    cfg = WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    env = WargameEnv(cfg)
    color = los_line_color(env, 0, 0, 5, 0)
    assert color == (80, 200, 80)


def test_terrain_los_demo_config_loads() -> None:
    """Demo YAML loads, constructs WargameEnv, and has 2 footprints."""
    from pydantic_yaml import parse_yaml_file_as

    cfg = parse_yaml_file_as(
        WargameEnvConfig,
        "configs/dev/terrain_los_demo.yaml",
    )
    env = WargameEnv(cfg)
    assert len(env.terrain.footprints) == 2
