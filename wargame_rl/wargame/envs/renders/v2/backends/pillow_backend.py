"""Pillow drawing backend — no new dependency (Pillow is already installed).

Pillow rasterises polygons and ellipses with antialiased *edges* on filled
shapes, which pygame's `draw` cannot, so this is the second candidate in the
Phase 2 bake-off. Canvases are kept in ``RGBA`` so translucent terrain and
control washes composite correctly: opaque primitives are drawn straight onto
the canvas, translucent ones onto a throwaway overlay that is blended in with
`Image.alpha_composite` (the in-place instance method), matching the pygame
backend's per-piece alpha layers. `to_rgb_array` drops the alpha at the end.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wargame_rl.wargame.envs.renders.v2.theme import RGB, RGBA

Point = tuple[float, float]

# Pillow's text anchor codes (horizontal + vertical), keyed by our align names.
_ANCHORS = {
    "center": "mm",
    "midleft": "lm",
    "midright": "rm",
    "topleft": "la",
}


class PillowBackend:
    """Renders primitives onto `PIL.Image` RGBA canvases."""

    def __init__(self) -> None:
        # Fonts are scalable and reused across draws; cache by pixel size.
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def new_canvas(self, width_px: int, height_px: int, bg: RGB) -> Image.Image:
        return Image.new("RGBA", (max(1, width_px), max(1, height_px)), (*bg, 255))

    def fill_rect(
        self, canvas: Image.Image, rect: tuple[int, int, int, int], color: RGB
    ) -> None:
        x, y, w, h = rect
        ImageDraw.Draw(canvas).rectangle((x, y, x + w, y + h), fill=(*color, 255))

    def draw_line(
        self, canvas: Image.Image, a: Point, b: Point, color: RGB, width: int
    ) -> None:
        ImageDraw.Draw(canvas).line(
            [(a[0], a[1]), (b[0], b[1])], fill=(*color, 255), width=max(1, int(width))
        )

    def draw_polygon(
        self,
        canvas: Image.Image,
        points: list[Point],
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        if len(points) < 3:
            return
        xy = [(p[0], p[1]) for p in points]
        if fill is not None:
            if len(fill) == 4 and fill[3] < 255:
                self._composite_translucent(canvas, "polygon", xy, fill)
            else:
                ImageDraw.Draw(canvas).polygon(xy, fill=(*fill[:3], 255))
        if outline is not None:
            ImageDraw.Draw(canvas).polygon(
                xy, outline=(*outline, 255), width=max(1, int(width))
            )

    def draw_disc(
        self,
        canvas: Image.Image,
        center: Point,
        radius_px: float,
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        radius = max(1.0, float(radius_px))
        box = [
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
        ]
        if fill is not None:
            if len(fill) == 4 and fill[3] < 255:
                self._composite_translucent(canvas, "ellipse", box, fill)
            else:
                ImageDraw.Draw(canvas).ellipse(box, fill=(*fill[:3], 255))
        if outline is not None:
            ImageDraw.Draw(canvas).ellipse(
                box, outline=(*outline, 255), width=max(1, int(width))
            )

    def draw_text(
        self,
        canvas: Image.Image,
        text: str,
        anchor: Point,
        size_px: int,
        color: RGB,
        align: str = "center",
    ) -> None:
        ImageDraw.Draw(canvas).text(
            (anchor[0], anchor[1]),
            text,
            font=self._font(size_px),
            fill=(*color, 255),
            anchor=_ANCHORS.get(align, "mm"),
        )

    def text_size(self, text: str, size_px: int) -> tuple[int, int]:
        left, top, right, bottom = self._font(size_px).getbbox(text)
        return (int(right - left), int(bottom - top))

    def blit(self, canvas: Image.Image, src: Image.Image, pos: Point) -> None:
        canvas.alpha_composite(src, dest=(int(pos[0]), int(pos[1])))

    def to_rgb_array(self, canvas: Image.Image) -> np.ndarray:
        return np.asarray(canvas.convert("RGB"), dtype=np.uint8)

    # -- helpers -------------------------------------------------------------

    def _composite_translucent(
        self,
        canvas: Image.Image,
        shape: str,
        geometry: list[tuple[float, float]],
        fill: RGBA,
    ) -> None:
        """Draw a translucent shape on an overlay and blend it in place."""
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if shape == "polygon":
            draw.polygon(geometry, fill=fill)
        else:
            draw.ellipse(geometry, fill=fill)
        canvas.alpha_composite(overlay)

    def _font(self, size_px: int) -> ImageFont.FreeTypeFont:
        size = max(1, int(size_px))
        font = self._fonts.get(size)
        if font is None:
            # Pillow 10.1+ returns a scalable FreeTypeFont when given a size.
            font = cast(ImageFont.FreeTypeFont, ImageFont.load_default(size=size))
            self._fonts[size] = font
        return font
