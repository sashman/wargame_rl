"""Per-entity configuration: weapon profiles, models and objectives."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wargame_rl.wargame.envs.types.config._validation import (
    _validate_coords_both_or_neither,
)
from wargame_rl.wargame.envs.types.geometry import Polygon


class WeaponProfile(BaseModel):
    """Weapon stat block with range and resolution stats."""

    model_config = ConfigDict(extra="forbid")

    range: int = Field(gt=0, description="Maximum range in grid cells")
    attacks: int = Field(
        default=2, gt=0, description="Number of hit rolls per shooting action"
    )
    ballistic_skill: int = Field(
        default=3, ge=2, le=6, description="D6 roll needed to hit (e.g. 3 means 3+)"
    )
    strength: int = Field(
        default=4, gt=0, description="For wound roll comparison vs target toughness"
    )
    ap: int = Field(
        default=1,
        ge=0,
        description="Armour penetration (worsens target save by this amount)",
    )
    damage: int = Field(default=1, gt=0, description="Wounds inflicted per failed save")


class ModelConfig(BaseModel):
    """Per-model configuration (position, group, stats, etc.).

    When *x* and *y* are provided the model is placed at that exact cell;
    otherwise it is placed randomly in the deployment zone.
    """

    model_config = ConfigDict(extra="forbid")

    x: int | None = Field(
        default=None,
        ge=0,
        description="X coordinate on the board. If None, placed randomly.",
    )
    y: int | None = Field(
        default=None,
        ge=0,
        description="Y coordinate on the board. If None, placed randomly.",
    )
    group_id: int = Field(default=0, ge=0, description="Group this model belongs to")
    max_wounds: int = Field(default=1, gt=0)
    toughness: int = Field(default=3, gt=0, description="Wound roll comparison stat")
    save: int = Field(
        default=4,
        ge=2,
        le=7,
        description="Base armour save (e.g. 4 means 4+, 7 means no armour)",
    )
    weapons: list[WeaponProfile] = Field(
        default_factory=list,
        description="Weapon profiles. Empty = cannot shoot.",
    )

    @model_validator(mode="after")
    def coords_both_or_neither(self) -> "ModelConfig":
        _validate_coords_both_or_neither(self.x, self.y)
        return self


class ObjectiveConfig(BaseModel):
    """Per-objective configuration (position, radius, etc.).

    When *x* and *y* are provided the objective is placed at that exact cell;
    otherwise it is placed randomly outside the deployment zone.
    """

    model_config = ConfigDict(extra="forbid")

    x: int | None = Field(
        default=None,
        ge=0,
        description="X coordinate on the board. If None, placed randomly.",
    )
    y: int | None = Field(
        default=None,
        ge=0,
        description="Y coordinate on the board. If None, placed randomly.",
    )
    radius_size: float | None = Field(
        default=None,
        gt=0,
        description="Override the global objective_radius_size for this objective",
    )
    area: list[tuple[float, float]] | None = Field(
        default=None,
        min_length=3,
        description="Make this objective an *area* rather than a marker: the "
        "outline is the objective, and a model controls it by standing inside. "
        "Its `location` becomes the outline's centroid so anything steering "
        "toward an objective still has a point to aim at. Mutually exclusive "
        "with x/y and with radius_size.",
    )

    @model_validator(mode="after")
    def coords_both_or_neither(self) -> "ObjectiveConfig":
        _validate_coords_both_or_neither(self.x, self.y)
        return self

    @model_validator(mode="after")
    def an_area_is_not_also_a_disc(self) -> "ObjectiveConfig":
        """An area objective has no centre and no radius to override.

        Accepting both would leave two definitions of "in range" and no way to
        tell which one a result was measured under.
        """
        if self.area is None:
            return self
        if self.x is not None or self.y is not None:
            raise ValueError(
                "an area objective is positioned by its outline; drop x and y"
            )
        if self.radius_size is not None:
            raise ValueError(
                "an area objective has no radius; control is standing inside it"
            )
        return self

    def to_polygon(self) -> Polygon | None:
        """The objective's area, or None when it is a marker with a radius."""
        return None if self.area is None else Polygon.from_points(self.area)
