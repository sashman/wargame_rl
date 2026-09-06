"""Hyperparameters for the size-independent set network (issue #285)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SetNetworkConfig(BaseModel):
    """Trunk shape for `SetNetwork`.

    Nothing here may depend on a scenario count — that is Principle 2, and it
    is what lets one set of weights serve any army size. Action-resolution
    knobs (`n_move_actions`, `n_advance_actions`) come from the scenario's
    action handler at construction, exactly as `n_speed_bins` sizes the old
    movement slice: a *resolution* is a property of the action encoding, not
    of how many entities exist.
    """

    model_config = ConfigDict(extra="forbid")

    embedding_size: int = Field(default=128, gt=0)
    n_layers: int = Field(default=4, gt=0)
    n_heads: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bias: bool = True

    @model_validator(mode="after")
    def _heads_divide_width(self) -> SetNetworkConfig:
        if self.embedding_size % self.n_heads != 0:
            raise ValueError(
                f"embedding_size {self.embedding_size} is not divisible by "
                f"n_heads {self.n_heads}"
            )
        return self
