"""The size-independent set network over the per-model decision contract.

Every entity is a token, every relation is an additive per-head bias in the
attention, and every choice among entities is a pointer over the relevant
tokens. No weight shape depends on how many models, units, objectives,
opponents or terrain pieces exist -- `tests/test_set_network.py` pins that
one instance plays two differently sized scenarios and that its state dict
has the same shapes whichever it was built against.

The encoder is player-centric. The acting seat's model tokens are the
queries: they self-attend among themselves (the same-unit relation rides in
the bias) and cross-attend to the union of context tokens -- the game, both
sides' units, the enemy models, the objectives, the terrain -- each carrying
a type embedding. Context tokens are read and never updated, so an opponent
token does not know what *it* is near; the pointer heads need only "what is
this unit worth to me", which the relation bias supplies directly.

One forward pass yields the selector (which model acts next) and the value;
`heads` then scores the selected model's value factor -- a declaration, a
displacement, or a pointer over enemy units -- and which of the three a step
uses is a function of the decision's kind and phase, never learned. There is
no autoregressive decoder: conditioning on earlier decisions comes from the
env, because the board already reflects them.

Standalone `nn.Module`, deliberately not an `RL_Network`: that base fixes a
list-of-tensors forward, a policy/value split and a per-model value, all of
which are the whole-army step's contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from wargame_rl.wargame.envs.per_model.tokens import (
    CONTEXT_DIM,
    MODEL_DIM,
    N_CONTEXT_KINDS,
    N_DECLARATIONS,
    N_PHASES,
    N_STEP_KINDS,
    RELATION_DIM,
)
from wargame_rl.wargame.model.per_model.batch import TokenBatch
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.env_components.actions import ActionHandler
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv

NEG_INF = float("-inf")


@dataclass(slots=True)
class SetNetworkOutput:
    """One forward pass: the latents, the selector and the value."""

    player_latents: torch.Tensor  # (B, P, E)
    context_embedded: torch.Tensor  # (B, C, E), the embedded context (never updated)
    game_latent: torch.Tensor  # (B, E)
    selector_logits: torch.Tensor  # (B, P), -inf where not selectable
    value: torch.Tensor  # (B,)


@dataclass(slots=True)
class HeadLogits:
    """The three value-factor heads for one selected model per row, masked."""

    declaration: torch.Tensor  # (B, N_DECLARATIONS)
    displacement: torch.Tensor  # (B, 1 + n_move + n_advance)
    unit: torch.Tensor  # (B, 1 + U)


class RelationBiasedAttention(nn.Module):
    """Multi-head attention with an additive per-pair bias and a key mask.

    The bias is the relation embedder's output, one score per head per pair,
    which is how "relations live in the attention" reaches the computation.
    One manual path: scaled-dot-product attention cannot take a per-head
    additive bias and a boolean key mask together across torch versions.
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
        attention = self.dropout(F.softmax(scores, dim=-1))
        merged = (attention @ v).transpose(1, 2).reshape(batch, n_q, size)
        out: torch.Tensor = self.out_proj(merged)
        return out


class _FeedForward(nn.Module):
    def __init__(self, config: SetNetworkConfig) -> None:
        super().__init__()
        size = config.embedding_size
        self.fc = nn.Linear(size, 4 * size, bias=config.bias)
        self.proj = nn.Linear(4 * size, size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.dropout(self.proj(F.gelu(self.fc(x))))
        return out


class SetBlock(nn.Module):
    """One encoder layer: self-attention, cross-attention, feed-forward; pre-LN."""

    def __init__(self, config: SetNetworkConfig) -> None:
        super().__init__()
        size = config.embedding_size
        self.ln_self = nn.LayerNorm(size, bias=config.bias)
        self.self_attention = RelationBiasedAttention(config)
        self.ln_cross = nn.LayerNorm(size, bias=config.bias)
        self.ln_context = nn.LayerNorm(size, bias=config.bias)
        self.cross_attention = RelationBiasedAttention(config)
        self.ln_ff = nn.LayerNorm(size, bias=config.bias)
        self.feed_forward = _FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_bias: torch.Tensor,
        cross_bias: torch.Tensor,
        self_keys: torch.Tensor,
        context_keys: torch.Tensor,
    ) -> torch.Tensor:
        h = self.ln_self(x)
        x = x + self.self_attention(h, h, self_bias, self_keys)
        x = x + self.cross_attention(
            self.ln_cross(x), self.ln_context(context), cross_bias, context_keys
        )
        x = x + self.feed_forward(self.ln_ff(x))
        return x


class SetNetwork(nn.Module):
    """Encoder, selector, value and the three value-factor heads, in one module.

    `n_displacements` is the only number taken from the scenario, and it is a
    property of the action encoding (STAY, the movement bins, the advance
    rungs), not of how many entities exist.
    """

    def __init__(
        self, config: SetNetworkConfig | None = None, *, n_displacements: int
    ) -> None:
        super().__init__()
        self.config = config if config is not None else SetNetworkConfig()
        cfg = self.config
        size = cfg.embedding_size
        self.n_displacements = int(n_displacements)

        self.player_embed = nn.Linear(MODEL_DIM, size, bias=cfg.bias)
        self.context_embed = nn.Linear(CONTEXT_DIM, size, bias=cfg.bias)
        self.type_embedding = nn.Embedding(N_CONTEXT_KINDS, size)
        # Index phase + 1 so the `close_turn` point (phase -1) has its own row.
        self.phase_embedding = nn.Embedding(N_PHASES + 1, size)
        self.kind_embedding = nn.Embedding(N_STEP_KINDS, size)

        # One read of the relation vector per attention stream, shared by every
        # block, as relative-position biases usually are; and one scalar read
        # for the pointer.
        self.self_relation_bias = nn.Linear(RELATION_DIM, cfg.n_heads)
        self.cross_relation_bias = nn.Linear(RELATION_DIM, cfg.n_heads)
        self.pointer_relation_bias = nn.Linear(RELATION_DIM, 1)

        self.blocks = nn.ModuleList(SetBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(size, bias=cfg.bias)

        self.selector_query = nn.Parameter(torch.randn(size) / math.sqrt(size))
        self.value_head = nn.Sequential(
            nn.Linear(2 * size, size), nn.GELU(), nn.Linear(size, 1)
        )
        self.declaration_head = nn.Linear(2 * size, N_DECLARATIONS)
        self.displacement_head = nn.Linear(2 * size, self.n_displacements)
        self.no_target_head = nn.Linear(2 * size, 1)
        self.target_query = nn.Linear(size, size, bias=cfg.bias)
        self.target_key = nn.Linear(size, size, bias=cfg.bias)

    # ------------------------------------------------------------ factories

    @classmethod
    def from_handler(
        cls, handler: ActionHandler, config: SetNetworkConfig | None = None
    ) -> SetNetwork:
        """Size the displacement head from an action handler's slices."""
        advance = handler.advance_slice
        n_displacements = 1 + handler.movement_slice.size
        if advance is not None:
            n_displacements += advance.size
        return cls(config, n_displacements=n_displacements)

    @classmethod
    def from_env(
        cls, env: PerModelEnv, config: SetNetworkConfig | None = None
    ) -> SetNetwork:
        """Build against `env`'s action encoding; reads no entity count."""
        return cls.from_handler(env.player_action_handler, config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -------------------------------------------------------------- forward

    def forward(self, batch: TokenBatch) -> SetNetworkOutput:
        """Encode one batch; the selector and the value come with it."""
        size = self.config.embedding_size
        self_keys = batch.player_alive & batch.player_pad
        # A row whose every player is dead (the game outlives an army) keeps
        # its real rows as keys, so no softmax collapses to NaN.
        no_live = ~self_keys.any(dim=1, keepdim=True)
        self_keys = torch.where(no_live, batch.player_pad, self_keys)
        context_keys = batch.context_alive & batch.context_pad

        x = self.player_embed(batch.players)
        x = x + self.phase_embedding(batch.phase + 1)[:, None, :]
        x = x + self.kind_embedding(batch.kind)[:, None, :]
        context = self.context_embed(batch.context) + self.type_embedding(
            batch.context_kind
        )
        self_bias = self.self_relation_bias(batch.self_relations).permute(0, 3, 1, 2)
        cross_bias = self.cross_relation_bias(batch.cross_relations).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, context, self_bias, cross_bias, self_keys, context_keys)
        x = self.final_norm(x)

        game_latent = context[:, 0, :]
        selector = (x * self.selector_query).sum(dim=-1) / math.sqrt(size)
        selector = selector.masked_fill(~batch.selector_mask, NEG_INF)
        pool_mask = self_keys.to(x.dtype)
        pooled = (x * pool_mask[:, :, None]).sum(dim=1) / pool_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        value = self.value_head(torch.cat([pooled, game_latent], dim=-1)).squeeze(-1)
        return SetNetworkOutput(
            player_latents=x,
            context_embedded=context,
            game_latent=game_latent,
            selector_logits=selector,
            value=value,
        )

    def heads(
        self, output: SetNetworkOutput, batch: TokenBatch, model_index: torch.Tensor
    ) -> HeadLogits:
        """Score the selected model's value factor under each head, masked.

        Which head a step uses is `batch.head`; the others' rows are masked
        by their own (all-False) masks and mean nothing.
        """
        rows = torch.arange(batch.batch_size, device=model_index.device)
        latent = output.player_latents[rows, model_index]
        head_input = torch.cat([latent, output.game_latent], dim=-1)

        declaration = self.declaration_head(head_input).masked_fill(
            ~batch.declaration_mask[rows, model_index], NEG_INF
        )
        displacement = self.displacement_head(head_input).masked_fill(
            ~batch.displacement_mask[rows, model_index], NEG_INF
        )

        size = self.config.embedding_size
        unit_rows = batch.opponent_unit_rows  # (B, U)
        gather_rows = unit_rows[:, :, None].expand(-1, -1, size)
        unit_tokens = torch.gather(output.context_embedded, 1, gather_rows)
        query = self.target_query(latent)[:, None, :]
        keys = self.target_key(unit_tokens)
        scores = (query * keys).sum(dim=-1) / math.sqrt(size)  # (B, U)
        relations = batch.cross_relations[
            rows[:, None], model_index[:, None], unit_rows
        ]
        scores = scores + self.pointer_relation_bias(relations).squeeze(-1)
        unit = torch.cat([self.no_target_head(head_input), scores], dim=1)
        unit = unit.masked_fill(~batch.unit_mask[rows, model_index], NEG_INF)
        return HeadLogits(declaration=declaration, displacement=displacement, unit=unit)
