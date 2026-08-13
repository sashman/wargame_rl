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


class MapPoolConfig(BaseModel):
    """Draw a whole layout — terrain *and* objectives — from a set of fixed maps.

    The third terrain mode, beside a fixed `terrain` list and a generated
    `random_terrain`, and the only one that trains on real tables. A layout is
    drawn per episode from the pool, so a run sees the distribution the maps
    describe rather than one board or a generator's idea of one.

    `names` is what splits the pool. Training on every map in `directory` leaves
    no layout the agent has not seen, and a transfer number quoted against it
    means nothing; naming a subset here and its complement in the evaluation
    config keeps a held-out set. None means every map in the directory, which is
    the right default for *evaluation* and the wrong one for training.
    """

    model_config = ConfigDict(extra="forbid")

    directory: str = Field(
        description="Directory of `TerrainMapConfig` YAML files, relative to the "
        "working directory."
    )
    mirror: bool = Field(
        default=False,
        description=(
            "Draw each layout in one of four orientations — as authored, "
            "reflected in x, in y, or in both — chosen uniformly per episode. "
            "This targets a *measured* defect: the same checkpoint scored "
            "-1.4 vp on tables it trained on and -23.8 on tables it had not, so "
            "the gap was generalisation and 36 layouts is a small distribution. "
            "**It buys about 2x, not 4x.** The real tables are laid out with "
            "near-180-degree rotational symmetry, the way fair tournament "
            "tables are: measured over all 45, a table sits a median of 1.7 "
            "board units from its own 180-degree rotation and at worst 3.9, on "
            "a board 60 across. So the xy reflection is very nearly the "
            "authored table again, and the x and y reflections are very nearly "
            "each other. The one genuinely distinct orientation is the "
            "side-swap, which is legitimate only while the two deployment "
            "zones mirror each other — check that before enabling it on an "
            "asymmetric scenario. All four are still drawn, because they are "
            "near-duplicates rather than exact ones and the redundancy costs "
            "nothing. Off by default, so an existing pool draws exactly what "
            "it always did."
        ),
    )
    names: list[str] | None = Field(
        default=None,
        description="Map names to draw from, by `name` rather than filename. "
        "None = every map in the directory. Name the split explicitly for a "
        "training run, or it consumes the evaluation set.",
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
    mirror_mode: str = Field(
        default="reflect_x",
        pattern="^(reflect_x|rotate_180)$",
        description=(
            "How `mirror` pairs pieces. `reflect_x` puts a mirror line down the "
            "middle of the table. `rotate_180` makes the layout point-symmetric "
            "about the board centre, which is **what the real tables actually "
            "are**: measured over all 45 authored layouts, a table sits a median "
            "of 1.7 board units from its own 180-degree rotation and at worst "
            "3.9, because a fair tournament table is built so neither side gets "
            "the better ground. A generated table reflected in x alone has a "
            "mirror line the real ones do not, and that is a regularity a policy "
            "can learn and the eval maps will never reward. `reflect_x` remains "
            "the default so existing runs are unchanged."
        ),
    )
    angled_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Share of *small* pieces given a random facing rather than sitting "
            "square to the board. A layout of perfectly axis-aligned boxes is a "
            "tell, and the real tables are not laid out that way. Small only — "
            "the lower half of the spec's size range — because a large ruin "
            "turned off-axis reads as a mistake, and because an angled piece is "
            "shrunk to fit the footprint it was allotted, which costs a bigger "
            "piece more coverage. That shrink is not optional: the rectangles "
            "are chosen to be mutually clear, and a turned rectangle sweeps "
            "outside its box, so turning without shrinking would generate "
            "overlapping ruins. 0.0 by default, which is the exact no-op."
        ),
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
