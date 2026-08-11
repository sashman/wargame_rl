"""Pygame drawing backend — the primitive calls the legacy renderer used.

`to_rgb_array` matches `HumanRender.get_frame_array` (transpose pygame's
`(w, h, 3)` to `(h, w, 3)`) so recorded frames are identical in shape and
orientation to the legacy path.
"""

from __future__ import annotations

import numpy as np
import pygame

from wargame_rl.wargame.envs.renders.v2.fonts import mono_font_path
from wargame_rl.wargame.envs.renders.v2.theme import RGB, RGBA

Point = tuple[float, float]

_ALIGNS = {"center", "midleft", "midright", "topleft"}


class PygameBackend:
    """Renders primitives onto `pygame.Surface` canvases."""

    def __init__(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

    def new_canvas(self, width_px: int, height_px: int, bg: RGB) -> pygame.Surface:
        surface = pygame.Surface((max(1, width_px), max(1, height_px)))
        surface.fill(bg)
        return surface

    def fill_rect(
        self, canvas: pygame.Surface, rect: tuple[int, int, int, int], color: RGB
    ) -> None:
        pygame.draw.rect(canvas, color, pygame.Rect(*rect))

    def draw_line(
        self, canvas: pygame.Surface, a: Point, b: Point, color: RGB, width: int
    ) -> None:
        pygame.draw.line(canvas, color, a, b, max(1, int(width)))

    def draw_polygon(
        self,
        canvas: pygame.Surface,
        points: list[Point],
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        if len(points) >= 3:
            if fill is not None:
                if len(fill) == 4 and fill[3] < 255:
                    layer = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
                    pygame.draw.polygon(layer, fill, points)
                    canvas.blit(layer, (0, 0))
                else:
                    pygame.draw.polygon(canvas, fill[:3], points)
            if outline is not None:
                pygame.draw.polygon(canvas, outline, points, max(1, int(width)))

    def draw_disc(
        self,
        canvas: pygame.Surface,
        center: Point,
        radius_px: float,
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        radius = max(1, int(round(radius_px)))
        if fill is not None:
            if len(fill) == 4 and fill[3] < 255:
                layer = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
                pygame.draw.circle(layer, fill, center, radius)
                canvas.blit(layer, (0, 0))
            else:
                pygame.draw.circle(canvas, fill[:3], center, radius)
        if outline is not None:
            pygame.draw.circle(canvas, outline, center, radius, max(1, int(width)))

    def draw_text(
        self,
        canvas: pygame.Surface,
        text: str,
        anchor: Point,
        size_px: int,
        color: RGB,
        align: str = "center",
        mono: bool = False,
        bold: bool = False,
    ) -> None:
        surface = self._font(size_px, mono, bold).render(text, True, color)
        anchor_kw = align if align in _ALIGNS else "center"
        rect = surface.get_rect(**{anchor_kw: (int(anchor[0]), int(anchor[1]))})
        canvas.blit(surface, rect)

    def text_size(
        self, text: str, size_px: int, mono: bool = False, bold: bool = False
    ) -> tuple[int, int]:
        width, height = self._font(size_px, mono, bold).size(text)
        return (int(width), int(height))

    def _font(self, size_px: int, mono: bool, bold: bool = False) -> pygame.font.Font:
        path = mono_font_path(bold) if mono else None
        return pygame.font.Font(path, max(1, int(size_px)))

    def blit(self, canvas: pygame.Surface, src: pygame.Surface, pos: Point) -> None:
        canvas.blit(src, (int(pos[0]), int(pos[1])))

    def to_rgb_array(self, canvas: pygame.Surface) -> np.ndarray:
        return np.asarray(
            np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2)), dtype=np.uint8
        )
