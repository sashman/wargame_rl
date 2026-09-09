"""The set network's trunk size.

`None` at every constructor means this default, exactly as `TransformerConfig`
does for the shipped transformer. The default is deliberately smaller than the
shipped 8 x 256 trunk: a per-model step runs a forward per decision, roughly
25x as often per round as the whole-army step, and the race (#288) compares
learning per round. A run that wants a bigger trunk passes one explicitly, and
`tests/test_set_network.py` pins this default so a fixture cannot shrink it.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class SetNetworkConfig(BaseModel):
    """Trunk width, depth and head count; the relation bias reads `n_heads`."""

    embedding_size: int = 128
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.0
    bias: bool = True

    @model_validator(mode="after")
    def _heads_divide_width(self) -> SetNetworkConfig:
        if self.embedding_size % self.n_heads != 0:
            raise ValueError(
                f"embedding_size {self.embedding_size} is not divisible by "
                f"n_heads {self.n_heads}"
            )
        return self
