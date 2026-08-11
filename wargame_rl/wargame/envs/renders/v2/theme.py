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


# --- Tabletop theme ---------------------------------------------------------
#
# A physical-wargame-table look: a warm parchment board, muted stone/wood
# terrain, and unit tokens desaturated to sit on the parchment (cool muted
# families for the player, warm brick for the opponent). Objective control uses
# sage / terracotta rather than the default green / hot-pink. `DEFAULT_THEME`
# stays the legacy mirror (the A/B baseline); this is opt-in via `build_renderer`.

# Player is blue, opponent is red — the sides read at a glance; groups within a
# side are distinguished by shade (a muted ramp that stays on the parchment).
_TABLETOP_PLAYER_GROUPS: tuple[RGB, ...] = (
    (52, 80, 110),  # navy
    (74, 110, 158),  # medium blue
    (30, 64, 100),  # deep blue
    (100, 140, 185),  # steel blue
    (58, 96, 138),  # blue
    (18, 44, 78),  # dark navy
    (86, 124, 170),  # light steel
    (44, 82, 128),  # slate blue
)
_TABLETOP_OPPONENT_GROUPS: tuple[RGB, ...] = (
    (168, 69, 47),  # brick
    (196, 96, 74),  # light brick
    (138, 42, 42),  # deep red
    (216, 120, 96),  # terracotta
    (110, 32, 32),  # maroon
    (180, 72, 60),  # rust red
    (150, 52, 48),  # dark brick
    (92, 26, 26),  # oxblood
)

TABLETOP_PALETTE = Palette(
    window_bg=(48, 41, 31),
    board_bg=(239, 231, 212),
    grid=(221, 210, 184),
    player_groups=_TABLETOP_PLAYER_GROUPS,
    opponent_groups=_TABLETOP_OPPONENT_GROUPS,
    dead_fill=(195, 183, 156),
    dead_mark=(122, 111, 87),
    model_rim=(43, 34, 22),
    deployment_zone=(223, 228, 207),
    opponent_zone=(238, 220, 207),
    zone_label=(166, 146, 110),
    terrain_fill=(120, 95, 60, 92),
    terrain_outline=(107, 79, 52),
    terrain_label=(74, 56, 38),
    objective_rim=(138, 122, 92),
    player_control=(150, 180, 120),
    opponent_control=(200, 140, 105),
    area_wash_alpha=120,
    los_clear=(95, 140, 70),
    los_blocked=(190, 70, 50),
    panel_bg=(58, 50, 38),
    panel_line=(87, 73, 58),
    text=(236, 226, 207),
)

TABLETOP_THEME = Theme(palette=TABLETOP_PALETTE)

# Named themes selectable by string (CLI flags, factory).
THEMES: dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "tabletop": TABLETOP_THEME,
}
