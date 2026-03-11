"""Value objects for the wargame domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BoardDimensions:
    """Board size (width and height)."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(
                f"Board dimensions must be positive, got {self.width}x{self.height}"
            )


@dataclass(frozen=True, slots=True)
class DeploymentZone:
    """Axis-aligned rectangle (x_min, y_min, x_max, y_max) for model placement."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError(
                f"Deployment zone min must be non-negative, got ({self.x_min}, {self.y_min})"
            )
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                f"Deployment zone max must be > min, got "
                f"({self.x_min},{self.y_min})-({self.x_max},{self.y_max})"
            )

    def as_array(self) -> np.ndarray:
        """Return (4,) int array [x_min, y_min, x_max, y_max] for placement helpers."""
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max], dtype=int)
