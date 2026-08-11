"""Antialiased pygame backend via supersampling.

Pinned pygame (2.6.1) has no antialiased *filled* primitives with stroke width,
so instead of special-casing `gfxdraw` per shape (which offers AA outlines but
no thick AA lines), this backend draws every primitive at ``SUPERSAMPLE``× the
requested resolution with the ordinary `pygame.draw` calls and downsamples once
in `to_rgb_array` with `smoothscale`. One code path antialiases discs, thick
arrows, grid lines and text uniformly.

Canvases are created at the scaled size and carry no logical-size metadata, so
`to_rgb_array` recovers the final size by dividing by ``SUPERSAMPLE``. Every
method therefore scales its incoming pixel coordinates and sizes by the same
factor; `text_size` is the exception — the presenter lays panels out in final
pixels, so it reports the unscaled size.
"""

from __future__ import annotations

import numpy as np
import pygame

from wargame_rl.wargame.envs.renders.v2.theme import RGB, RGBA

Point = tuple[float, float]

SUPERSAMPLE = 3  # 3× keeps text legible without the 16× cost of 4×.
_ALIGNS = {"center", "midleft", "midright", "topleft"}


class PygameAABackend:
    """Renders primitives supersampled, then downscales for smooth edges."""

    def __init__(self, supersample: int = SUPERSAMPLE) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        self._ss = max(1, int(supersample))

    def new_canvas(self, width_px: int, height_px: int, bg: RGB) -> pygame.Surface:
        surface = pygame.Surface(
            (max(1, width_px) * self._ss, max(1, height_px) * self._ss)
        )
        surface.fill(bg)
        return surface

    def fill_rect(
        self, canvas: pygame.Surface, rect: tuple[int, int, int, int], color: RGB
    ) -> None:
        x, y, w, h = (int(v * self._ss) for v in rect)
        pygame.draw.rect(canvas, color, pygame.Rect(x, y, w, h))

    def draw_line(
        self, canvas: pygame.Surface, a: Point, b: Point, color: RGB, width: int
    ) -> None:
        pygame.draw.line(canvas, color, self._scale(a), self._scale(b), self._px(width))

    def draw_polygon(
        self,
        canvas: pygame.Surface,
        points: list[Point],
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        pts = [self._scale(p) for p in points]
        if len(pts) < 3:
            return
        if fill is not None:
            if len(fill) == 4 and fill[3] < 255:
                layer = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
                pygame.draw.polygon(layer, fill, pts)
                canvas.blit(layer, (0, 0))
            else:
                pygame.draw.polygon(canvas, fill[:3], pts)
        if outline is not None:
            pygame.draw.polygon(canvas, outline, pts, self._px(width))

    def draw_disc(
        self,
        canvas: pygame.Surface,
        center: Point,
        radius_px: float,
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        c = self._scale(center)
        radius = max(1, int(round(radius_px * self._ss)))
        if fill is not None:
            if len(fill) == 4 and fill[3] < 255:
                layer = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
                pygame.draw.circle(layer, fill, c, radius)
                canvas.blit(layer, (0, 0))
            else:
                pygame.draw.circle(canvas, fill[:3], c, radius)
        if outline is not None:
            pygame.draw.circle(canvas, outline, c, radius, self._px(width))

    def draw_text(
        self,
        canvas: pygame.Surface,
        text: str,
        anchor: Point,
        size_px: int,
        color: RGB,
        align: str = "center",
    ) -> None:
        font = pygame.font.Font(None, max(1, int(size_px) * self._ss))
        surface = font.render(text, True, color)
        anchor_kw = align if align in _ALIGNS else "center"
        rect = surface.get_rect(**{anchor_kw: self._scale(anchor)})
        canvas.blit(surface, rect)

    def text_size(self, text: str, size_px: int) -> tuple[int, int]:
        width, height = pygame.font.Font(None, max(1, int(size_px))).size(text)
        return (int(width), int(height))

    def blit(self, canvas: pygame.Surface, src: pygame.Surface, pos: Point) -> None:
        canvas.blit(src, self._scale(pos))

    def to_rgb_array(self, canvas: pygame.Surface) -> np.ndarray:
        width, height = canvas.get_size()
        final = pygame.transform.smoothscale(
            canvas, (max(1, width // self._ss), max(1, height // self._ss))
        )
        return np.asarray(
            np.transpose(pygame.surfarray.array3d(final), (1, 0, 2)), dtype=np.uint8
        )

    def _scale(self, point: Point) -> tuple[int, int]:
        return (int(point[0] * self._ss), int(point[1] * self._ss))

    def _px(self, width: int) -> int:
        return max(1, int(width) * self._ss)
