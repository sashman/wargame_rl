"""Collation: pad to the maximum within the batch, never to a config budget.

A fixed budget is exactly the size dependence Principle 2 removes, so a batch
is padded to its own largest scenario and every tensor carries a mask that
says which rows are real. Padding and death share the network's key mask, so
a corpse and a pad row are the same thing to the attention: never attended
to, never selectable, never a pointer target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from wargame_rl.wargame.envs.per_model.tokens import (
    CONTEXT_DIM,
    MODEL_DIM,
    N_DECLARATIONS,
    RELATION_DIM,
    TokenObservation,
)
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER


@dataclass(slots=True)
class TokenBatch:
    """One batch of token observations, padded to the batch maximum."""

    players: torch.Tensor  # (B, P, MODEL_DIM) float32
    player_alive: torch.Tensor  # (B, P) bool
    player_pad: torch.Tensor  # (B, P) bool, True on real rows
    context: torch.Tensor  # (B, C, CONTEXT_DIM) float32
    context_kind: torch.Tensor  # (B, C) int64
    context_alive: torch.Tensor  # (B, C) bool
    context_pad: torch.Tensor  # (B, C) bool
    self_relations: torch.Tensor  # (B, P, P, RELATION_DIM)
    cross_relations: torch.Tensor  # (B, P, C, RELATION_DIM)
    opponent_unit_rows: torch.Tensor  # (B, U) int64
    unit_pad: torch.Tensor  # (B, U) bool
    selector_mask: torch.Tensor  # (B, P) bool
    declaration_mask: torch.Tensor  # (B, P, N_DECLARATIONS) bool
    displacement_mask: torch.Tensor  # (B, P, D) bool
    unit_mask: torch.Tensor  # (B, P, 1 + U) bool
    head: torch.Tensor  # (B,) int64 (`tokens.Head`)
    kind: torch.Tensor  # (B,) int64 (`StepKind` order)
    phase: torch.Tensor  # (B,) int64 (`BATTLE_PHASE_ORDER` index, -1 at close_turn)

    @property
    def batch_size(self) -> int:
        return int(self.players.shape[0])

    def to(self, device: torch.device | str) -> TokenBatch:
        """The same batch on `device`."""
        return TokenBatch(
            **{
                name: getattr(self, name).to(device)
                for name in self.__slots__  # type: ignore[attr-defined]
            }
        )


def collate(
    observations: Sequence[TokenObservation], device: torch.device | str | None = None
) -> TokenBatch:
    """Stack token observations into one padded batch."""
    if not observations:
        raise ValueError("collate needs at least one observation")
    n_p = max(o.n_players for o in observations)
    n_c = max(o.n_context for o in observations)
    n_u = max(o.n_units for o in observations)
    n_d = max(int(o.displacement_mask.shape[1]) for o in observations)
    batch = len(observations)

    players = np.zeros((batch, n_p, MODEL_DIM), dtype=np.float32)
    player_alive = np.zeros((batch, n_p), dtype=bool)
    player_pad = np.zeros((batch, n_p), dtype=bool)
    context = np.zeros((batch, n_c, CONTEXT_DIM), dtype=np.float32)
    context_kind = np.zeros((batch, n_c), dtype=np.int64)
    context_alive = np.zeros((batch, n_c), dtype=bool)
    context_pad = np.zeros((batch, n_c), dtype=bool)
    self_relations = np.zeros((batch, n_p, n_p, RELATION_DIM), dtype=np.float32)
    cross_relations = np.zeros((batch, n_p, n_c, RELATION_DIM), dtype=np.float32)
    unit_rows = np.zeros((batch, n_u), dtype=np.int64)
    unit_pad = np.zeros((batch, n_u), dtype=bool)
    selector = np.zeros((batch, n_p), dtype=bool)
    declaration = np.zeros((batch, n_p, N_DECLARATIONS), dtype=bool)
    displacement = np.zeros((batch, n_p, n_d), dtype=bool)
    unit_mask = np.zeros((batch, n_p, 1 + n_u), dtype=bool)
    head = np.zeros(batch, dtype=np.int64)
    kind = np.zeros(batch, dtype=np.int64)
    phase = np.full(batch, -1, dtype=np.int64)

    kinds = list(type(observations[0].kind))
    for b, o in enumerate(observations):
        p, c, u = o.n_players, o.n_context, o.n_units
        d = int(o.displacement_mask.shape[1])
        players[b, :p] = o.players
        player_alive[b, :p] = o.player_alive
        player_pad[b, :p] = True
        context[b, :c] = o.context
        context_kind[b, :c] = o.context_kind
        context_alive[b, :c] = o.context_alive
        context_pad[b, :c] = True
        self_relations[b, :p, :p] = o.self_relations
        cross_relations[b, :p, :c] = o.cross_relations
        unit_rows[b, :u] = o.opponent_unit_rows
        unit_pad[b, :u] = True
        selector[b, :p] = o.selector_mask
        declaration[b, :p] = o.declaration_mask
        displacement[b, :p, :d] = o.displacement_mask
        unit_mask[b, :p, : 1 + u] = o.unit_mask
        head[b] = int(o.head)
        kind[b] = kinds.index(o.kind)
        phase[b] = BATTLE_PHASE_ORDER.index(o.phase) if o.phase is not None else -1

    def tensor(array: np.ndarray) -> torch.Tensor:
        result = torch.from_numpy(array)
        return result.to(device) if device is not None else result

    return TokenBatch(
        players=tensor(players),
        player_alive=tensor(player_alive),
        player_pad=tensor(player_pad),
        context=tensor(context),
        context_kind=tensor(context_kind),
        context_alive=tensor(context_alive),
        context_pad=tensor(context_pad),
        self_relations=tensor(self_relations),
        cross_relations=tensor(cross_relations),
        opponent_unit_rows=tensor(unit_rows),
        unit_pad=tensor(unit_pad),
        selector_mask=tensor(selector),
        declaration_mask=tensor(declaration),
        displacement_mask=tensor(displacement),
        unit_mask=tensor(unit_mask),
        head=tensor(head),
        kind=tensor(kind),
        phase=tensor(phase),
    )
