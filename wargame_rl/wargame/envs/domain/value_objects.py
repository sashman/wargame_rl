"""Value objects for the wargame domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, SupportsFloat, TypeAlias

import numpy as np
import numpy.typing as npt

# The one place the coordinate type is declared. Board coordinates are whole
# cells, matching the dtype the Gym observation space advertises in
# `entities.to_space`. Every position is built through `position` or
# `zero_position` so that making the board continuous is a change to this
# constant rather than a hunt through the codebase -- assigning a float array
# into an int one truncates *silently*, with no exception and no failing test,
# so a missed site would be invisible.
#
# Construction is not the only way to get the dtype wrong -- *arithmetic* is.
# A position was int32 from placement and int64 from its first move onward,
# because three things in `env_components/actions.py` widened it: an int64
# displacement table, an int64 STAY displacement, and `np.clip` bounds passed
# as python lists. Anything combined with a position has to carry this dtype
# too, and `test_positions_keep_the_declared_dtype_through_a_whole_episode`
# pins the result across a whole episode.
POSITION_DTYPE: Final = np.int32

# A board coordinate pair, shape (2,).
Position: TypeAlias = npt.NDArray[np.int32]


def position(x: SupportsFloat, y: SupportsFloat) -> Position:
    """Build a board coordinate from an x and a y."""
    return np.array([x, y], dtype=POSITION_DTYPE)


def zero_position() -> Position:
    """Build the origin coordinate, for entities not yet placed."""
    return np.zeros(2, dtype=POSITION_DTYPE)


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
