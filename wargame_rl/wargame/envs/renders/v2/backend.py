"""The swappable drawing seam.

A `RenderBackend` implements a handful of primitive draws against its own canvas
type (pygame Surface, PIL Image, ...). `rasterize` walks a `Scene`'s primitives,
applies the `Camera`, and dispatches to those draws, so every backend renders the
identical primitives — which is what makes the Phase 2 bake-off apples-to-apples.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from wargame_rl.wargame.envs.renders.v2.camera import Camera
from wargame_rl.wargame.envs.renders.v2.scene import Disc, Label, Poly, Scene, Seg
from wargame_rl.wargame.envs.renders.v2.theme import RGB, RGBA

# The canvas is the backend's own surface type; callers pass it back opaquely.
Canvas = Any
Point = tuple[float, float]


class RenderBackend(Protocol):
    """Minimal primitive set every backend implements."""

    def new_canvas(self, width_px: int, height_px: int, bg: RGB) -> Canvas:
        """A fresh canvas filled with ``bg``."""
        ...

    def fill_rect(
        self, canvas: Canvas, rect: tuple[int, int, int, int], color: RGB
    ) -> None:
        """Fill an ``(x, y, w, h)`` rectangle."""
        ...

    def draw_line(
        self, canvas: Canvas, a: Point, b: Point, color: RGB, width: int
    ) -> None:
        """Stroke a line segment."""
        ...

    def draw_polygon(
        self,
        canvas: Canvas,
        points: list[Point],
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        """Fill (honouring alpha) and/or outline a polygon."""
        ...

    def draw_disc(
        self,
        canvas: Canvas,
        center: Point,
        radius_px: float,
        fill: RGBA | None,
        outline: RGB | None,
        width: int,
    ) -> None:
        """Fill (honouring alpha) and/or outline a circle."""
        ...

    def draw_text(
        self,
        canvas: Canvas,
        text: str,
        anchor: Point,
        size_px: int,
        color: RGB,
        align: str = "center",
    ) -> None:
        """Draw text anchored ``center`` / ``midleft`` / ``midright``."""
        ...

    def text_size(self, text: str, size_px: int) -> tuple[int, int]:
        """Rendered ``(width, height)`` of text at a size — for panel layout."""
        ...

    def blit(self, canvas: Canvas, src: Canvas, pos: Point) -> None:
        """Composite one canvas onto another."""
        ...

    def to_rgb_array(self, canvas: Canvas) -> np.ndarray:
        """Canvas as an ``(H, W, 3)`` uint8 array."""
        ...


def rasterize(
    backend: RenderBackend, scene: Scene, camera: Camera, canvas: Canvas
) -> None:
    """Draw a scene's primitives onto a canvas via a backend and camera."""
    for primitive in scene.primitives:
        if isinstance(primitive, Poly):
            points = [camera.to_px(x, y) for x, y in primitive.points]
            backend.draw_polygon(
                canvas, points, primitive.fill, primitive.outline, primitive.outline_w
            )
        elif isinstance(primitive, Disc):
            backend.draw_disc(
                canvas,
                camera.to_px(*primitive.center),
                primitive.radius * camera.scale,
                primitive.fill,
                primitive.outline,
                primitive.outline_w,
            )
        elif isinstance(primitive, Seg):
            backend.draw_line(
                canvas,
                camera.to_px(*primitive.a),
                camera.to_px(*primitive.b),
                primitive.color,
                primitive.width,
            )
        elif isinstance(primitive, Label):
            backend.draw_text(
                canvas,
                primitive.text,
                camera.to_px(*primitive.center),
                primitive.size,
                primitive.color,
            )
