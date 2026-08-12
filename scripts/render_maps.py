"""Render a preview PNG beside every map in `configs/evaluation/maps/`.

The previews exist so a map can be read without running anything, and they are
the only place its terrain and its objectives are visible together. They were
originally produced ad hoc and committed without the code that made them, so
the first objective change left 45 stale images and no way to refresh them.
This is that code.

Drawn from the map file rather than from a reset episode, so a preview shows
what the map *is* and cannot vary with a seed. Rendering one through the env
was the obvious alternative and is worse: fifty deployed models cover the
layout the picture exists to show. Deployment zones come from the scenario and
are drawn because most of these maps put an objective inside one, which is the
thing you want to notice while looking at a map.

Usage: just render-maps [env_config] [maps_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_maps import DEFAULT_MAPS_DIR, load_maps
from wargame_rl.wargame.envs.renders.v2.theme import TABLETOP_THEME
from wargame_rl.wargame.envs.types import TerrainMapConfig, WargameEnvConfig

DEFAULT_CONFIG = Path("configs/golden/25v25_shooting_opponent.yaml")
WIDTH_PX = 1024
MARGIN_PX = 12
LABEL_H = 26


def _zone(config: WargameEnvConfig, opponent: bool) -> tuple[int, int, int, int]:
    """A deployment zone, falling back to the same thirds the env defaults to."""
    explicit = config.opponent_deployment_zone if opponent else config.deployment_zone
    if explicit is not None:
        return explicit
    if opponent:
        return (config.board_width * 2 // 3, 0, config.board_width, config.board_height)
    return (0, 0, config.board_width // 3, config.board_height)


def render_map(
    base_config: WargameEnvConfig, terrain_map: TerrainMapConfig, out: Path
) -> None:
    """Draw one map's terrain, objectives and deployment zones to *out*."""
    palette = TABLETOP_THEME.palette
    board_w, board_h = base_config.board_width, base_config.board_height
    scale = (WIDTH_PX - 2 * MARGIN_PX) / board_w
    height = round(board_h * scale) + 2 * MARGIN_PX + LABEL_H
    image = Image.new("RGB", (WIDTH_PX, height), palette.window_bg)
    draw = ImageDraw.Draw(image, "RGBA")

    def px(x: float, y: float) -> tuple[float, float]:
        return (MARGIN_PX + x * scale, MARGIN_PX + y * scale)

    draw.rectangle([px(0, 0), px(board_w, board_h)], fill=palette.board_bg)
    for opponent, colour in (
        (False, palette.deployment_zone),
        (True, palette.opponent_zone),
    ):
        x0, y0, x1, y1 = _zone(base_config, opponent)
        draw.rectangle([px(x0, y0), px(x1, y1)], fill=colour)
    for inch in range(1, board_w):
        draw.line([px(inch, 0), px(inch, board_h)], fill=palette.grid, width=1)
    for inch in range(1, board_h):
        draw.line([px(0, inch), px(board_w, inch)], fill=palette.grid, width=1)

    for piece in terrain_map.terrain:
        points = [px(x, y) for x, y in piece.to_polygon().vertices]
        draw.polygon(points, fill=palette.terrain_fill, outline=palette.terrain_outline)

    radius = base_config.objective_radius_size
    wash = (*palette.player_control, palette.area_wash_alpha)
    for objective in terrain_map.objectives or []:
        area = objective.to_polygon()
        if area is not None:
            draw.polygon(
                [px(x, y) for x, y in area.vertices],
                fill=wash,
                outline=palette.objective_rim,
                width=2,
            )
            centre = px(*area.centroid)
        else:
            assert objective.x is not None and objective.y is not None
            size = (objective.radius_size or radius) * scale
            centre = px(objective.x, objective.y)
            draw.ellipse(
                [
                    centre[0] - size,
                    centre[1] - size,
                    centre[0] + size,
                    centre[1] + size,
                ],
                fill=wash,
                outline=palette.objective_rim,
                width=2,
            )
        draw.ellipse(
            [centre[0] - 3, centre[1] - 3, centre[0] + 3, centre[1] + 3],
            fill=palette.objective_rim,
        )

    draw.rectangle([px(0, 0), px(board_w, board_h)], outline=palette.terrain_outline)
    n_objectives = len(terrain_map.objectives or [])
    draw.text(
        (MARGIN_PX, height - LABEL_H + 4),
        f"{terrain_map.name}   {board_w}x{board_h}in   "
        f"{len(terrain_map.terrain)} pieces   {n_objectives} objectives",
        fill=palette.text,
    )
    image.save(out, optimize=True)


def main() -> None:
    args = sys.argv[1:]
    config_path = Path(args[0]) if args and args[0] else DEFAULT_CONFIG
    maps_dir = Path(args[1]) if len(args) > 1 and args[1] else DEFAULT_MAPS_DIR
    base_config = parse_yaml_raw_as(WargameEnvConfig, config_path.read_text())
    maps = load_maps(maps_dir)
    print(f"{len(maps)} maps from {maps_dir} on {config_path.name}")
    for terrain_map in maps:
        out = maps_dir / f"{terrain_map.name}.png"
        render_map(base_config, terrain_map, out)
        print(
            f"  {terrain_map.name:12s} -> {out.name}  {out.stat().st_size / 1024:5.1f} KB"
        )


if __name__ == "__main__":
    main()
