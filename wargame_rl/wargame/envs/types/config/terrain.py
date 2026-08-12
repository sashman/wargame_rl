"""Terrain configuration: fixed pieces, named maps and the random generator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wargame_rl.wargame.envs.types.config.entities import ObjectiveConfig
from wargame_rl.wargame.envs.types.geometry import Polygon


class TerrainPieceConfig(BaseModel):
    """One terrain piece: either an axis-aligned rectangle or an explicit outline.

    Exactly one of the two. A rectangle is the historical form and is authored in
    *inclusive cell* coordinates, so `(5, 5, 5, 5)` is one cell; an outline is
    authored in continuous board units and is taken literally. Mixing the two
    conventions in one field is how a layout silently comes out a unit small, so
    they are separate fields with a validator rather than one overloaded one.
    """

    model_config = ConfigDict(extra="forbid")

    footprint: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Bounding rectangle (x0, y0, x1, y1) in inclusive grid cells.",
    )
    outline: list[tuple[float, float]] | None = Field(
        default=None,
        min_length=3,
        description="Closed outline as (x, y) vertices in board units. May be "
        "concave. Mutually exclusive with `footprint`.",
    )

    @model_validator(mode="after")
    def exactly_one_shape(self) -> "TerrainPieceConfig":
        """A piece is a rectangle or an outline, never both and never neither."""
        if (self.footprint is None) == (self.outline is None):
            raise ValueError(
                "a terrain piece needs exactly one of `footprint` or `outline`"
            )
        return self

    def to_polygon(self) -> Polygon:
        """Resolve whichever form was authored into the one shape type."""
        if self.outline is not None:
            return Polygon.from_points(self.outline)
        assert self.footprint is not None
        return Polygon.from_cell_rect(*self.footprint)


class TerrainMapConfig(BaseModel):
    """A named fixed terrain layout, stored on its own in `configs/evaluation/maps/`.

    Kept out of `WargameEnvConfig` deliberately. A map is meant to be swapped
    onto an *existing* scenario — `just measure-maps` overrides `terrain` on the
    golden config once per map — so that final evaluation runs the same reward,
    opponent and force composition the agent was trained under. A config per map
    would duplicate a 13 KB scenario N times and let evaluation drift from
    training the first time a reward term changed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Map identifier, used as the row label.")
    terrain: list[TerrainPieceConfig] = Field(
        description="The layout's pieces. Replaces the scenario's own terrain."
    )
    objectives: list[ObjectiveConfig] | None = Field(
        default=None,
        description="The layout's objectives, replacing the scenario's own and "
        "setting `number_of_objectives` to this length. Optional, so a map that "
        "carries terrain alone keeps scoring under the scenario's own objective "
        "placement. Every entry must be determined — an `area` outline or x/y — "
        "since a map's objectives are part of the layout and a randomly placed "
        "one would make the map's rows differ run to run.",
    )

    @model_validator(mode="after")
    def objectives_are_determined(self) -> "TerrainMapConfig":
        """Reject an objective with neither an area nor coordinates.

        A fixed map exists so a row means the same thing every time it is run.
        A random objective inside one would reintroduce exactly the variance the
        map was written to remove, and `place_for_episode` only honours fixed
        positions when *every* entry has them — so one bare entry would silently
        randomise all of them.
        """
        if self.objectives is None:
            return self
        undetermined = [
            i
            for i, objective in enumerate(self.objectives)
            if objective.x is None and objective.area is None
        ]
        if undetermined:
            raise ValueError(
                f"map '{self.name}': objectives {undetermined} have neither an "
                "area nor x/y; every objective in a fixed map must be determined"
            )
        return self


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

    model_config = ConfigDict(extra="forbid")

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
    n_vertices: int | None = Field(
        default=None,
        ge=3,
        description="Generate convex n-gons with this many vertices instead of "
        "axis-aligned rectangles. None (the default) keeps rectangles, which is "
        "what every terrain profile in the repo was tuned against. Outlines hide "
        "*less* board than the rectangles they replace at equal size, and pack "
        "*tighter* — so re-derive a profile with `just measure-terrain` after "
        "turning this on rather than porting the old numbers.",
    )

    @model_validator(mode="after")
    def sizes_ordered(self) -> "RandomTerrainConfig":
        """Reject an inverted size range."""
        if self.min_size > self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) must not exceed max_size ({self.max_size})"
            )
        return self
