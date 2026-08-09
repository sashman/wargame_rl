"""Terrain configuration: fixed pieces, named maps and the random generator."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TerrainPieceConfig(BaseModel):
    """Configuration for a single terrain piece (axis-aligned rectangle)."""

    footprint: tuple[int, int, int, int] = Field(
        description="Bounding rectangle (x0, y0, x1, y1) in grid cells."
    )


class TerrainMapConfig(BaseModel):
    """A named fixed terrain layout, stored on its own in `configs/evaluation/maps/`.

    Kept out of `WargameEnvConfig` deliberately. A map is meant to be swapped
    onto an *existing* scenario — `just measure-maps` overrides `terrain` on the
    golden config once per map — so that final evaluation runs the same reward,
    opponent and force composition the agent was trained under. A config per map
    would duplicate a 13 KB scenario N times and let evaluation drift from
    training the first time a reward term changed.
    """

    name: str = Field(description="Map identifier, used as the row label.")
    terrain: list[TerrainPieceConfig] = Field(
        description="The layout's pieces. Replaces the scenario's own terrain."
    )


class RandomTerrainConfig(BaseModel):
    """Regenerate terrain footprints randomly at the start of every episode.

    The piece *count* is fixed while size and position vary. This is a hard
    constraint, not a simplification: `observations_to_tensor_batch` stacks the
    terrain arrays of a whole batch with `np.stack`, so a batch containing
    episodes with different piece counts cannot be collated.

    Randomising terrain is what makes a cover result falsifiable. With a fixed
    layout a policy can memorise a handful of rectangles; with a fresh layout
    every episode it has to read the terrain tokens in the observation.
    """

    count: int = Field(
        gt=0,
        default=7,
        description="Number of terrain pieces. Constant across episodes.",
    )
    min_size: int = Field(
        gt=0, default=5, description="Minimum footprint side length in cells."
    )
    max_size: int = Field(
        gt=0, default=7, description="Maximum footprint side length in cells."
    )
    mirror: bool = Field(
        default=True,
        description="Mirror the layout across the vertical centre line. Deployment "
        "zones are fixed to the left and right of the board, so an asymmetric "
        "random layout would systematically favour one side.",
    )
    edge_margin: int = Field(
        ge=0, default=2, description="Keep footprints this far from the board edge."
    )
    min_gap: int = Field(
        ge=0,
        default=1,
        description="Minimum clear cells between two footprints. 0 lets them touch.",
    )

    @model_validator(mode="after")
    def sizes_ordered(self) -> "RandomTerrainConfig":
        """Reject an inverted size range."""
        if self.min_size > self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) must not exceed max_size ({self.max_size})"
            )
        return self
