"""Tests for terrain rendering helpers and demo config loading."""

from __future__ import annotations

import os

import numpy as np

from wargame_rl.wargame.envs.renders.human import los_line_color
from wargame_rl.wargame.envs.types import TerrainPieceConfig, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import ModelConfig, ObjectiveConfig
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


def _render_frame(config: WargameEnvConfig, seed: int = 0) -> np.ndarray:
    """Draw one frame headlessly and return it as pixels.

    Asserted on the *pixels* rather than on the draw calls. The renderer is the
    only place a coordinate convention can be wrong without any other test
    noticing — an off-by-half-a-unit or a bounding box drawn in place of an
    outline breaks nothing and produces a picture of a board the rules are not
    playing on.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from wargame_rl.wargame.envs.renders.human import HumanRender

    renderer = HumanRender()
    env = WargameEnv(config=config, renderer=renderer)
    env.reset(seed=seed)
    renderer.setup(env)
    renderer._render_frame(env)
    assert renderer.canvas is not None
    return pygame.surfarray.array3d(renderer.canvas)


def _patch(pixels: np.ndarray, x: float, y: float, board: int = 40) -> np.ndarray:
    """Mean colour of a small square of board around (x, y).

    A mean rather than a single pixel: the canvas carries a faint grid, and a
    point sample lands on a gridline often enough to make an otherwise sound
    assertion look like a rendering bug.
    """
    scale = pixels.shape[0] / board
    half = max(1, int(0.3 * scale))
    cx, cy = int(x * scale), int(y * scale)
    patch = pixels[cx - half : cx + half, cy - half : cy + half]
    return np.asarray(patch.reshape(-1, 3).mean(axis=0))


def test_a_model_is_drawn_at_its_base_radius_not_a_fixed_token() -> None:
    """Two conventions checked at once, both otherwise silent.

    The `+ 0.5` that moved a cell index to the cell's centre is gone — a
    continuous coordinate already *is* the point — and the base is drawn at the
    radius the rules use rather than at a third of a cell. Either being wrong
    produces a picture of a board the rules are not playing on, and nothing else
    in the suite would notice.
    """
    config = WargameEnvConfig(
        board_width=40,
        board_height=40,
        number_of_wargame_models=1,
        number_of_objectives=1,
        base_radius=2.0,
        models=[ModelConfig(x=20, y=20)],
        objectives=[ObjectiveConfig(x=35, y=35)],
        render_mode="human",
        number_of_battle_rounds=1,
    )
    pixels = _render_frame(config)

    # Group 0 is blue, so the base reads as a strong blue channel.
    on_model = _patch(pixels, 20, 20)
    assert on_model[2] > on_model[0] + 50, f"no model drawn at (20, 20): {on_model}"

    inside_edge = _patch(pixels, 20, 20 - 1.5)
    outside_edge = _patch(pixels, 20, 20 - 2.6)
    assert inside_edge[2] > inside_edge[0] + 50, "base smaller than its radius"
    assert outside_edge[2] < outside_edge[0] + 50, "base larger than its radius"


def test_an_area_objective_is_drawn_on_its_terrain_piece() -> None:
    """The whole point of `objectives_on_terrain`, checked where it is visible.

    The objective outline and the ruin outline are the same shape, so the ruin's
    interior carries the objective's ownership wash rather than the plain
    terrain fill.
    """
    outline = [(20.0, 20.0), (28.0, 20.0), (28.0, 28.0), (20.0, 28.0)]
    config = WargameEnvConfig(
        board_width=40,
        board_height=40,
        number_of_wargame_models=1,
        number_of_objectives=1,
        terrain=[TerrainPieceConfig(outline=outline)],
        objectives=[ObjectiveConfig(area=outline)],
        models=[ModelConfig(x=24, y=24)],
        render_mode="human",
        number_of_battle_rounds=1,
    )
    pixels = _render_frame(config)

    inside_area = _patch(pixels, 22, 22)
    open_ground = _patch(pixels, 17, 35)

    assert open_ground.min() > 240, f"expected bare board, got {open_ground}"
    # The single model stands inside, so the player controls it: a green wash,
    # not the terrain's brown and not the neutral grey.
    assert inside_area[1] > inside_area[0] + 20
    assert inside_area[1] > inside_area[2] + 20
