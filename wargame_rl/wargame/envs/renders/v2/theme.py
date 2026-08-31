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
    # The sight shadow: what the selected model cannot see. Neutral and dark
    # rather than tinted, so it reads as *absence of light* over whatever it
    # covers — a coloured wash would compete with objective control, which is
    # the thing most worth seeing through it.
    los_shadow: RGBA = (18, 20, 30, 96)
    panel_bg: RGB = (45, 45, 48)
    panel_line: RGB = (80, 80, 84)
    text: RGB = (220, 220, 220)
    # HUD side accents — bright enough to read on the dark panel (the board
    # group colours are tuned for the board, not the panel).
    hud_player: RGB = (120, 170, 235)
    hud_opponent: RGB = (235, 130, 115)
    # Tracers. Deliberately not the group colours: a group ramp answers "which
    # squad", and a shot has to answer "which side" at a glance across a board
    # with eight of them.
    shot_player: RGB = (60, 130, 255)
    shot_opponent: RGB = (230, 60, 50)
    # A killing shot's impact ring. One colour for both sides: a casualty is an
    # event on the board, and reading it should not depend on knowing whose
    # tracer it was — the line already carries the shooter's side.
    shot_kill: RGB = (255, 240, 120)
    # The blade of a melee clash. Its own colour rather than the attacker's,
    # because a clash sits ON the two models making it: a player-blue blade
    # drawn over a player-blue base is invisible, which a tracer spanning open
    # board never is. Painted as a light core over a `model_rim` stroke so it
    # reads on the board, on either side's models and on a casualty alike; the
    # attacker's side is carried by the crossguard instead.
    melee_blade: RGB = (238, 241, 247)
    # Threat overlays. By side, like the tracers and for the same reason: the
    # shape is one army's reach and merging is per side, so a group ramp has
    # nothing to say about it. The shooting threat is a line on a busy board, so
    # it takes the stronger colour; the engagement wash covers ground under the
    # pieces, so its alpha sits below the terrain fill's (90) and the shadow's
    # (96) -- at 2.26" a model it can blanket a third of the table.
    threat_player: RGB = (40, 90, 200)
    threat_opponent: RGB = (200, 45, 40)
    # The threat outline gets a wash of its own colour behind it. An outline
    # alone does not say which SIDE of the frontier is threatened -- the region
    # covers 36-41% of the board, so its boundary meanders down the middle --
    # and without the wash the band where both armies' reach overlaps, which is
    # the most useful thing on the board, cannot be seen at all. Kept very low:
    # this sits under the models and must not compete with objective control.
    threat_fill_alpha: int = 22
    # The NEXT-turn field, as a low/medium/high ramp. It answers a different
    # question from the threat outlines above -- where can they shoot me after
    # they move, rather than where can they shoot me now -- so it gets its own
    # colours rather than a shade of `threat_opponent`, and a viewer with both
    # on can tell which is which. Amber to red because the field is *about* the
    # opponent's fire; the alphas stay under the terrain fill's 90 for the same
    # reason the engagement wash does, since the bands blanket most of the board.
    threat_field_bands: tuple[RGBA, ...] = (
        (210, 170, 60, 26),
        (215, 120, 45, 40),
        (205, 55, 40, 56),
    )
    engagement_player: RGBA = (60, 130, 255, 48)
    engagement_opponent: RGBA = (230, 60, 50, 48)


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
    los_shadow=(46, 34, 20, 104),
    panel_bg=(58, 50, 38),
    panel_line=(87, 73, 58),
    text=(236, 226, 207),
    hud_player=(139, 178, 232),
    hud_opponent=(231, 148, 112),
    shot_player=(46, 92, 148),
    shot_opponent=(176, 48, 38),
    shot_kill=(206, 138, 47),
    # Warmer than the default's cool white: on parchment a blue-white blade
    # reads as a hole in the board rather than as steel.
    melee_blade=(250, 246, 236),
    # A little more alpha than the default theme: the parchment ground is warm
    # and swallows a cool wash.
    threat_player=(36, 74, 122),
    threat_opponent=(150, 40, 32),
    threat_fill_alpha=26,
    engagement_player=(46, 92, 148, 52),
    engagement_opponent=(176, 48, 38, 52),
)

TABLETOP_THEME = Theme(palette=TABLETOP_PALETTE)

# Named themes selectable by string (CLI flags, factory).
THEMES: dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "tabletop": TABLETOP_THEME,
}
