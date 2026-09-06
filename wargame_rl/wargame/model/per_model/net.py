"""The size-independent set network (issue #285, Principle 2).

Every entity is a token; every relation lives in the attention as an additive
per-head bias; every choice among entities is a pointer over the relevant
tokens. No weight shape depends on how many models, units, objectives,
opponents or terrain pieces exist, so one set of weights serves any scenario
with no retraining — which `tests/test_per_model_network.py` pins by running
the same instance on two differently sized scenarios.

The encoder is player-centric: player-model tokens are the queries; they
self-attend among themselves (the same-unit relation rides in the bias) and
cross-attend to the union of context tokens (game, both sides' unit tokens,
opponent models, objectives, terrain), each carrying a type embedding. One
forward pass yields the selector (who acts next), the value, and latents the
phase-conditioned action heads read — selection and action decided in the
same pass, one joint log-prob per env step.

There is **no autoregressive action decoder**: conditioning on earlier actions
comes from the env, because the board already reflects them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from wargame_rl.wargame.envs.per_model.observation import (
    CONTEXT_DIM,
    MODEL_DIM,
    N_CONTEXT_KINDS,
    N_DECLARATION_OPTIONS,
    RELATION_DIM,
)
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER
from wargame_rl.wargame.model.per_model.batch import TokenBatch
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.facade import PerModelEnv

NEG_INF = float("-inf")


@dataclass(slots=True)
class SetNetworkOutput:
    """One forward pass: latents, the selector, and the value."""

    player_latents: torch.Tensor  # (B, P, E)
    game_latent: torch.Tensor  # (B, E)
    selector_logits: torch.Tensor  # (B, P), -inf where not selectable
    value: torch.Tensor  # (B,)


@dataclass(slots=True)
class ActionLogits:
    """The phase heads for one selected model per batch row.

    Masked entries are -inf. ``displacement`` covers STAY + the movement bins;
    ``advance`` is the roll-gated advance rungs (empty width when the scenario
    registers none) — whether they are *offered* also depends on the unit's
    declaration, which is the agent's to decide (`SetAgent`). ``target`` is
    hold-fire + one pointer score per enemy unit token. ``declaration`` is the
    unit-level factor for opening steps.
    """

    displacement: torch.Tensor  # (B, 1 + n_move)
    advance: torch.Tensor  # (B, A); A may be 0
    target: torch.Tensor  # (B, 1 + U)
    declaration: torch.Tensor  # (B, N_DECLARATION_OPTIONS)


class RelationBiasedAttention(nn.Module):
    """Multi-head attention with an additive per-pair bias and a key mask.

    The bias is the relation embedder's output — one score per head per pair —
    which is how "relations live in the attention, not on the token" reaches
    the computation. Used for both the self block (y = x) and the cross block.
    """

    def __init__(self, config: SetNetworkConfig) -> None:
        super().__init__()
        size = config.embedding_size
        self.q_proj = nn.Linear(size, size, bias=config.bias)
        self.k_proj = nn.Linear(size, size, bias=config.bias)
        self.v_proj = nn.Linear(size, size, bias=config.bias)
        self.out_proj = nn.Linear(size, size, bias=config.bias)
        self.n_heads = config.n_heads
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bias: torch.Tensor | None,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_q, size = x.shape
        n_k = y.shape[1]
        head_size = size // self.n_heads
        q = self.q_proj(x).view(batch, n_q, self.n_heads, head_size).transpose(1, 2)
        k = self.k_proj(y).view(batch, n_k, self.n_heads, head_size).transpose(1, 2)
        v = self.v_proj(y).view(batch, n_k, self.n_heads, head_size).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(~key_mask[:, None, None, :], NEG_INF)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        merged = (attention @ v).transpose(1, 2).reshape(batch, n_q, size)
        result: torch.Tensor = self.out_proj(merged)
        return result


class _FeedForward(nn.Module):
    def __init__(self, config: SetNetworkConfig) -> None:
        super().__init__()
        size = config.embedding_size
        self.net = nn.Sequential(
            nn.Linear(size, 4 * size, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * size, size, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class SetBlock(nn.Module):
    """One encoder layer: self-attention, cross-attention, feed-forward."""

    def __init__(self, config: SetNetworkConfig) -> None:
        super().__init__()
        size = config.embedding_size
        self.ln_self = nn.LayerNorm(size)
        self.self_attention = RelationBiasedAttention(config)
        self.ln_cross = nn.LayerNorm(size)
        self.cross_attention = RelationBiasedAttention(config)
        self.ln_ff = nn.LayerNorm(size)
        self.feed_forward = _FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_bias: torch.Tensor | None,
        cross_bias: torch.Tensor | None,
        self_keys: torch.Tensor,
        context_keys: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attention(
            self.ln_self(x), self.ln_self(x), self_bias, self_keys
        )
        x = x + self.cross_attention(
            self.ln_cross(x), context, cross_bias, context_keys
        )
        x = x + self.feed_forward(self.ln_ff(x))
        return x


class SetNetwork(nn.Module):
    """Encoder + selector + value + phase-conditioned action heads.

    ``n_move_actions`` / ``n_advance_actions`` size the displacement heads —
    action-encoding *resolution* knobs (like ``n_speed_bins``), not entity
    counts, so Principle 2 is intact: the same weights load on any army.
    """

    def __init__(
        self,
        config: SetNetworkConfig | None = None,
        *,
        n_move_actions: int,
        n_advance_actions: int = 0,
    ) -> None:
        super().__init__()
        self.config = config or SetNetworkConfig()
        size = self.config.embedding_size
        self.n_move_actions = n_move_actions
        self.n_advance_actions = n_advance_actions

        # The three embedders are the only place raw features enter.
        self.player_embedder = nn.Linear(MODEL_DIM, size)
        self.context_embedder = nn.Linear(CONTEXT_DIM, size)
        self.type_embedding = nn.Embedding(N_CONTEXT_KINDS, size)
        self.phase_embedding = nn.Embedding(len(BATTLE_PHASE_ORDER), size)
        # One score per head per pair; shared across layers, as relative
        # position biases usually are.
        self.self_relation_bias = nn.Linear(RELATION_DIM, self.config.n_heads)
        self.cross_relation_bias = nn.Linear(RELATION_DIM, self.config.n_heads)
        # The pointer's own read of the same relation vector: one scalar.
        self.pointer_relation_bias = nn.Linear(RELATION_DIM, 1)

        self.blocks = nn.ModuleList(
            SetBlock(self.config) for _ in range(self.config.n_layers)
        )
        self.final_norm = nn.LayerNorm(size)

        self.selector_query = nn.Parameter(torch.randn(size) / math.sqrt(size))
        self.value_head = nn.Sequential(
            nn.Linear(2 * size, size), nn.GELU(), nn.Linear(size, 1)
        )
        head_input = 2 * size  # selected latent ‖ game token latent
        self.displacement_head = nn.Sequential(
            nn.Linear(head_input, size), nn.GELU(), nn.Linear(size, 1 + n_move_actions)
        )
        self.advance_head = (
            nn.Sequential(
                nn.Linear(head_input, size),
                nn.GELU(),
                nn.Linear(size, n_advance_actions),
            )
            if n_advance_actions > 0
            else None
        )
        self.declaration_head = nn.Sequential(
            nn.Linear(head_input, size),
            nn.GELU(),
            nn.Linear(size, N_DECLARATION_OPTIONS),
        )
        # One pointer mechanism for every "name an entity" choice: a query from
        # the selected latent against a key from each candidate token, plus the
        # pair's relation bias. The shooting target is its first user.
        self.target_query = nn.Linear(size, size)
        self.target_key = nn.Linear(size, size)
        self.hold_fire_head = nn.Sequential(
            nn.Linear(head_input, size), nn.GELU(), nn.Linear(size, 1)
        )

    @classmethod
    def from_env(
        cls, env: "PerModelEnv", config: SetNetworkConfig | None = None
    ) -> "SetNetwork":
        """Size the action heads from the env's own handler.

        Only action-encoding resolutions are read — never an entity count, so
        the returned network loads unchanged on any scenario sharing the same
        movement/advance bin counts.
        """
        handler = env.player_action_handler
        advance = handler.advance_slice
        return cls(
            config,
            n_move_actions=handler.n_move_actions,
            n_advance_actions=advance.size if advance is not None else 0,
        )

    # -- Forward --------------------------------------------------------------

    def forward(self, batch: TokenBatch) -> SetNetworkOutput:
        """Encode one batch; return latents, selector logits and values."""
        phase = self.phase_embedding(batch.phase_index)  # (B, E)
        x = self.player_embedder(batch.player_tokens) + phase[:, None, :]
        context = self.context_embedder(batch.context_tokens) + self.type_embedding(
            batch.context_kinds
        )

        self_bias = self.self_relation_bias(batch.self_relations).permute(0, 3, 1, 2)
        cross_bias = self.cross_relation_bias(batch.cross_relations).permute(0, 3, 1, 2)

        # Dead and padding share one mask as keys. A batch row whose every
        # player is dead (the game can outlive an army) falls back to its real
        # rows, so the softmax cannot collapse to NaN; the latents of such a
        # row feed nothing the agent uses — the selector is empty there.
        self_keys = batch.player_alive & batch.player_pad
        no_live_keys = ~self_keys.any(dim=1, keepdim=True)
        self_keys = torch.where(no_live_keys, batch.player_pad, self_keys)
        for block in self.blocks:
            x = block(x, context, self_bias, cross_bias, self_keys, batch.context_mask)
        x = self.final_norm(x)

        game_latent = context[:, 0, :]  # the game token is context row 0
        selector_logits = (x * self.selector_query).sum(-1) / math.sqrt(
            self.config.embedding_size
        )
        selector_logits = selector_logits.masked_fill(~batch.selection_mask, NEG_INF)

        pool_mask = (batch.player_alive & batch.player_pad).float()
        denominator = pool_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * pool_mask[:, :, None]).sum(dim=1) / denominator
        value = self.value_head(torch.cat([pooled, game_latent], dim=-1)).squeeze(-1)

        return SetNetworkOutput(
            player_latents=x,
            game_latent=game_latent,
            selector_logits=selector_logits,
            value=value,
        )

    def action_logits(
        self,
        output: SetNetworkOutput,
        batch: TokenBatch,
        model_index: torch.Tensor,
    ) -> ActionLogits:
        """The selected model's heads, masked by the batch's own legality.

        ``model_index`` is (B,) — one selected model per row. The advance
        logits carry only the roll gate; whether the rungs are offered at all
        depends on the unit's declaration, which the agent decides.
        """
        batch_rows = torch.arange(model_index.shape[0], device=model_index.device)
        latent = output.player_latents[batch_rows, model_index]  # (B, E)
        head_input = torch.cat([latent, output.game_latent], dim=-1)

        displacement = self.displacement_head(head_input)
        displacement = displacement.masked_fill(
            ~batch.displacement_mask[batch_rows, model_index], NEG_INF
        )
        if self.advance_head is not None and batch.advance_mask.shape[-1]:
            advance = self.advance_head(head_input)
            advance = advance.masked_fill(
                ~batch.advance_mask[batch_rows, model_index], NEG_INF
            )
        else:
            advance = torch.full(
                (model_index.shape[0], 0),
                NEG_INF,
                device=head_input.device,
                dtype=head_input.dtype,
            )

        # Pointer over the enemy-unit context tokens (embeddings, not updated
        # latents: the encoder is player-centric, and the pointer needs only
        # "what is this unit worth to me", which the relation bias carries).
        context = self.context_embedder(batch.context_tokens) + self.type_embedding(
            batch.context_kinds
        )
        unit_tokens = torch.gather(
            context,
            1,
            batch.opponent_unit_rows[:, :, None].expand(-1, -1, context.shape[-1]),
        )  # (B, U, E)
        query = self.target_query(latent)[:, None, :]  # (B, 1, E)
        keys = self.target_key(unit_tokens)  # (B, U, E)
        scores = (query * keys).sum(-1) / math.sqrt(self.config.embedding_size)
        unit_relations = torch.gather(
            batch.cross_relations[batch_rows, model_index],
            1,
            batch.opponent_unit_rows[:, :, None].expand(
                -1, -1, batch.cross_relations.shape[-1]
            ),
        )  # (B, U, R)
        scores = scores + self.pointer_relation_bias(unit_relations).squeeze(-1)
        hold = self.hold_fire_head(head_input)  # (B, 1)
        target = torch.cat([hold, scores], dim=1)
        target = target.masked_fill(
            ~batch.target_mask[batch_rows, model_index], NEG_INF
        )

        declaration = self.declaration_head(head_input)
        declaration = declaration.masked_fill(
            ~batch.declaration_mask[batch_rows, model_index], NEG_INF
        )
        return ActionLogits(
            displacement=displacement,
            advance=advance,
            target=target,
            declaration=declaration,
        )
