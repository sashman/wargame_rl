"""Colours and layout metrics for the v2 renderer.

Everything the legacy `HumanRender` scattered as inline literals and the two
`_GROUP_COLORS` / `_OPPONENT_COLORS` class lists lives here as one `Theme`, so a
future look is one object to edit rather than a hunt through draw methods.
`DEFAULT_THEME` reproduces the legacy colours exactly, so v2 starts visually
identical and any divergence is a deliberate theme change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

# Cool palette for player groups, warm for opponents (legacy order preserved).
_PLAYER_GROUPS: tuple[RGB, ...] = (
    (0, 0, 255),
    (60, 180, 80),
    (255, 180, 0),
    (180, 80, 220),
    (0, 200, 200),
    (200, 100, 0),
    (220, 100, 180),
    (220, 80, 60),
)
_OPPONENT_GROUPS: tuple[RGB, ...] = (
    (200, 40, 40),
    (220, 100, 30),
    (180, 30, 80),
    (160, 60, 140),
    (200, 80, 80),
    (180, 100, 20),
    (140, 30, 50),
    (210, 60, 100),
)


@dataclass(frozen=True)
class Palette:
    """Every colour the board and HUD use."""

    window_bg: RGB = (45, 45, 48)
    board_bg: RGB = (255, 255, 255)
    grid: RGB = (210, 210, 210)
    player_groups: tuple[RGB, ...] = _PLAYER_GROUPS
    opponent_groups: tuple[RGB, ...] = _OPPONENT_GROUPS
    dead_fill: RGB = (180, 180, 180)
    dead_mark: RGB = (120, 120, 120)
    model_rim: RGB = (30, 30, 30)
    deployment_zone: RGB = (200, 200, 200)
    opponent_zone: RGB = (220, 200, 200)
    zone_label: RGB = (240, 240, 240)
    terrain_fill: RGBA = (140, 120, 90, 90)
    terrain_outline: RGB = (100, 80, 60)
    terrain_label: RGB = (60, 50, 40)
    objective_rim: RGB = (90, 90, 90)
    player_control: RGB = (120, 220, 140)
    opponent_control: RGB = (255, 105, 180)
    area_wash_alpha: int = 120
    los_clear: RGB = (80, 200, 80)
    los_blocked: RGB = (255, 80, 80)
    panel_bg: RGB = (45, 45, 48)
    panel_line: RGB = (80, 80, 84)
    text: RGB = (220, 220, 220)


@dataclass(frozen=True)
class Theme:
    """A palette plus the panel layout metrics."""

    palette: Palette = field(default_factory=Palette)
    north_panel_h: int = 36
    south_panel_rows: int = 2
    show_grid: bool = True

    def player_color(self, group_id: int) -> RGB:
        """Distinct colour for a player group, cycling the palette."""
        groups = self.palette.player_groups
        return groups[group_id % len(groups)]

    def opponent_color(self, group_id: int) -> RGB:
        """Distinct colour for an opponent group, cycling the palette."""
        groups = self.palette.opponent_groups
        return groups[group_id % len(groups)]


DEFAULT_THEME = Theme()
