"""Collation for token observations: pad to the batch maximum, mask the rest.

Principle 2's batching rule: tensors keep a fixed shape **within a batch** by
padding to the largest episode in it — never to a config budget, which is
exactly the size dependence the set network removes. Padding and death share
one mask; nothing attends to either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from wargame_rl.wargame.envs.per_model.observation import TokenObservation


@dataclass(slots=True)
class TokenBatch:
    """A batch of `TokenObservation`s as padded tensors.

    ``player_pad`` / ``context_pad`` / ``unit_pad`` mark REAL rows (True);
    padded rows are all-zero and masked out of attention, the selector and the
    pointers. ``*_mask`` fields keep their per-observation meaning.
    """

    phase_index: torch.Tensor  # (B,) long
    player_tokens: torch.Tensor  # (B, P, MODEL_DIM)
    player_alive: torch.Tensor  # (B, P) bool
    player_pad: torch.Tensor  # (B, P) bool — True = real
    selection_mask: torch.Tensor  # (B, P) bool
    context_tokens: torch.Tensor  # (B, C, CONTEXT_DIM)
    context_kinds: torch.Tensor  # (B, C) long
    context_mask: torch.Tensor  # (B, C) bool — real AND alive
    self_relations: torch.Tensor  # (B, P, P, R)
    cross_relations: torch.Tensor  # (B, P, C, R)
    opponent_unit_rows: torch.Tensor  # (B, U) long (0 where padded)
    unit_pad: torch.Tensor  # (B, U) bool — True = real target unit
    displacement_mask: torch.Tensor  # (B, P, 1 + n_move) bool
    advance_mask: torch.Tensor  # (B, P, A) bool (A may be 0)
    target_mask: torch.Tensor  # (B, P, 1 + U) bool
    declaration_mask: torch.Tensor  # (B, P, N_DECLARATION_OPTIONS) bool

    @property
    def batch_size(self) -> int:
        return int(self.player_tokens.shape[0])


def collate(
    observations: list[TokenObservation], device: torch.device | None = None
) -> TokenBatch:
    """Stack observations of possibly different sizes into one padded batch."""
    if not observations:
        raise ValueError("Cannot collate an empty batch.")
    n_p = max(o.player_tokens.shape[0] for o in observations)
    n_c = max(o.context_tokens.shape[0] for o in observations)
    n_u = max(o.opponent_unit_rows.shape[0] for o in observations)
    n_disp = max(o.displacement_mask.shape[1] for o in observations)
    n_adv = max(o.advance_mask.shape[1] for o in observations)
    n_decl = max(o.declaration_mask.shape[1] for o in observations)
    relation_dim = observations[0].self_relations.shape[-1]
    model_dim = observations[0].player_tokens.shape[-1]
    context_dim = observations[0].context_tokens.shape[-1]
    batch = len(observations)

    phase_index = np.zeros(batch, dtype=np.int64)
    player_tokens = np.zeros((batch, n_p, model_dim), dtype=np.float32)
    player_alive = np.zeros((batch, n_p), dtype=bool)
    player_pad = np.zeros((batch, n_p), dtype=bool)
    selection = np.zeros((batch, n_p), dtype=bool)
    context_tokens = np.zeros((batch, n_c, context_dim), dtype=np.float32)
    context_kinds = np.zeros((batch, n_c), dtype=np.int64)
    context_mask = np.zeros((batch, n_c), dtype=bool)
    self_relations = np.zeros((batch, n_p, n_p, relation_dim), dtype=np.float32)
    cross_relations = np.zeros((batch, n_p, n_c, relation_dim), dtype=np.float32)
    unit_rows = np.zeros((batch, n_u), dtype=np.int64)
    unit_pad = np.zeros((batch, n_u), dtype=bool)
    displacement = np.zeros((batch, n_p, n_disp), dtype=bool)
    advance = np.zeros((batch, n_p, n_adv), dtype=bool)
    target = np.zeros((batch, n_p, 1 + n_u), dtype=bool)
    declaration = np.zeros((batch, n_p, n_decl), dtype=bool)

    for row, obs in enumerate(observations):
        p = obs.player_tokens.shape[0]
        c = obs.context_tokens.shape[0]
        u = obs.opponent_unit_rows.shape[0]
        phase_index[row] = obs.phase_index
        player_tokens[row, :p] = obs.player_tokens
        player_alive[row, :p] = obs.player_alive
        player_pad[row, :p] = True
        selection[row, :p] = obs.selection_mask
        context_tokens[row, :c] = obs.context_tokens
        context_kinds[row, :c] = obs.context_kinds
        context_mask[row, :c] = obs.context_mask
        self_relations[row, :p, :p] = obs.self_relations
        cross_relations[row, :p, :c] = obs.cross_relations
        unit_rows[row, :u] = obs.opponent_unit_rows
        unit_pad[row, :u] = True
        displacement[row, :p, : obs.displacement_mask.shape[1]] = obs.displacement_mask
        if obs.advance_mask.shape[1]:
            advance[row, :p, : obs.advance_mask.shape[1]] = obs.advance_mask
        target[row, :p, 0] = obs.target_mask[:, 0]
        if u:
            target[row, :p, 1 : 1 + u] = obs.target_mask[:, 1 : 1 + u]
        declaration[row, :p, : obs.declaration_mask.shape[1]] = obs.declaration_mask

    def tensor(array: np.ndarray) -> torch.Tensor:
        result = torch.from_numpy(array)
        return result.to(device) if device is not None else result

    return TokenBatch(
        phase_index=tensor(phase_index),
        player_tokens=tensor(player_tokens),
        player_alive=tensor(player_alive),
        player_pad=tensor(player_pad),
        selection_mask=tensor(selection),
        context_tokens=tensor(context_tokens),
        context_kinds=tensor(context_kinds),
        context_mask=tensor(context_mask),
        self_relations=tensor(self_relations),
        cross_relations=tensor(cross_relations),
        opponent_unit_rows=tensor(unit_rows),
        unit_pad=tensor(unit_pad),
        displacement_mask=tensor(displacement),
        advance_mask=tensor(advance),
        target_mask=tensor(target),
        declaration_mask=tensor(declaration),
    )
