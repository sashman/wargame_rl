"""The board→pixel transform, isolated so pan/zoom has one place to live.

Phase 1 uses only `fit` (the legacy fit-to-`GRID_SIZE` letterbox) with a zero
offset; `pan`/`zoom_at` are stubbed for Phase 4. Keeping the transform here is
what lets the interactive presenter gain pan/zoom later without touching the
scene or any backend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Camera:
    """Maps board units to pixels: ``px = unit * scale + offset``."""

    scale: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        """Board coordinate to pixel coordinate."""
        return (x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    @classmethod
    def fit(cls, board_width: float, board_height: float, scale: float) -> "Camera":
        """A camera at a given board→pixel scale with no pan offset."""
        return cls(scale=scale)

    def pan(self, dx_px: float, dy_px: float) -> None:
        """Shift the view (Phase 4)."""
        self.offset_x += dx_px
        self.offset_y += dy_px

    def zoom_at(self, cursor_px: tuple[float, float], factor: float) -> None:
        """Zoom keeping the cursor point fixed (Phase 4)."""
        cx, cy = cursor_px
        self.offset_x = cx - (cx - self.offset_x) * factor
        self.offset_y = cy - (cy - self.offset_y) * factor
        self.scale *= factor
