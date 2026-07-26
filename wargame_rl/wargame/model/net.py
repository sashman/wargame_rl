import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import nn

from wargame_rl.wargame.envs.types import WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common import Device, get_device
from wargame_rl.wargame.model.common.config import TransformerConfig
from wargame_rl.wargame.model.common.observation import (
    N_COMBAT_STATS,
    N_WOUND_FEATURES,
    observation_to_tensor,
)
from wargame_rl.wargame.model.dqn.layers import Block, LayerNorm


@dataclass(frozen=True, slots=True)
class EncodedState:
    """Transformer encoding plus the metadata needed to rebuild full-size logits.

    Returned by :meth:`TransformerNetwork.encode_state` and consumed by
    :meth:`TransformerNetwork.policy_from_encoded` /
    :meth:`TransformerNetwork.value_from_encoded`, so encoding metadata is
    passed explicitly between calls instead of being stored on the module.

    Token positions are fixed (game, objectives, all player rows, all opponent
    rows), so player ``p`` lives at ``n_prefix + p`` and opponent ``o`` at
    ``n_prefix + n_wargame_models + o``. ``player_alive`` (shape
    ``(batch, n_wargame_models)``) marks live player rows so dead ones can be
    forced stay-only without depending on the env action mask.
    """

    encoded: torch.Tensor
    n_prefix: int
    n_wargame_models: int
    n_opponents: int
    player_alive: torch.Tensor
    mask_tensor: torch.Tensor | None


class RL_Network(nn.Module, ABC):
    @property
    def device(self) -> torch.device:  # type: ignore[override]
        """Derive device from actual parameter location (stays correct after Lightning moves the model)."""
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device("cpu")

    def is_batched(self, xs: list[torch.Tensor]) -> bool:
        """Check if the input is batched."""
        game_state_tensor = xs[0]
        # Check if the game state tensor is batched
        return len(game_state_tensor.shape) > 1

    @abstractmethod
    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_env(cls, env: WargameEnv, is_policy: bool) -> Self:
        pass

    @classmethod
    def policy_from_env(cls, env: WargameEnv) -> Self:
        return cls.from_env(env, is_policy=True)

    @classmethod
    def value_from_env(cls, env: WargameEnv) -> Self:
        return cls.from_env(env, is_policy=False)

    @classmethod
    def from_checkpoint(cls, env: WargameEnv, checkpoint_path: str) -> Self:
        load_dict = torch.load(checkpoint_path, weights_only=False)
        if "state_dict" in load_dict:
            state_dict = convert_state_dict(load_dict["state_dict"])
        else:
            state_dict = load_dict
        return cls.from_state_dict(env, state_dict)

    @classmethod
    def from_state_dict(
        cls, env: WargameEnv, state_dict: dict, is_policy: bool = True
    ) -> Self:
        net = cls.from_env(env, is_policy=is_policy)
        net.load_state_dict(state_dict)
        return net


class MLPNetwork(RL_Network):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_wargame_models: int,
        device: Device | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        is_policy: bool = True,
    ) -> None:
        super().__init__()

        self.is_policy = is_policy
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))
        self.action_dim = action_dim
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        if self.is_policy:
            self.output_dim = n_wargame_models * action_dim
        else:
            self.output_dim = 1

        self.output = nn.Linear(hidden_dim, self.output_dim)
        self.activation = nn.GELU()
        self.to(get_device(device))
        self.n_wargame_models = n_wargame_models

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        # Exclude the action mask tensor (last element) — it's used
        # externally for action selection, not as network input.
        state_tensors = xs[:5]
        if self.is_batched(xs):
            x = torch.cat([x.flatten(start_dim=1) for x in state_tensors], dim=1)
        else:
            x = torch.cat(
                [x.flatten(start_dim=0) for x in state_tensors], dim=0
            ).unsqueeze(0)

        assert len(x.shape) == 2

        batch_size = x.shape[0]
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output(x)
        if self.is_policy:
            x = x.reshape(batch_size, self.n_wargame_models, self.action_dim)
        return cast(torch.Tensor, x)

    @classmethod
    def from_env(cls, env: WargameEnv, is_policy: bool) -> Self:
        observation: WargameEnvObservation
        observation, _ = env.reset()
        tensors = observation_to_tensor(observation)
        obs_size = sum(t.numel() for t in tensors[:5])
        n_wargame_models: int = observation.n_wargame_models
        n_actions: int = env._action_handler.n_actions
        print(
            f"Creating MLP network with obs_size: {obs_size}, n_wargame_models: {n_wargame_models}, n_actions: {n_actions}, is_policy: {is_policy}"
        )
        return cls(obs_size, n_actions, n_wargame_models, is_policy=is_policy)


class TransformerNetwork(RL_Network):
    # Transformer adapted from the NanoGPT implementation:
    # https://github.com/karpathy/nanoGPT
    def __init__(
        self,
        game_size: int,
        objective_size: int,
        wargame_model_size: int,
        n_actions: int,
        is_policy: bool,
        transformer_config: TransformerConfig,
        opponent_model_size: int = 0,
        terrain_size: int = 0,
        shooting_slice_start: int | None = None,
        shooting_slice_end: int | None = None,
        device: Device | None = None,
    ) -> None:
        self.game_size = game_size
        self.objective_size = objective_size
        self.wargame_model_size = wargame_model_size
        self.opponent_model_size = opponent_model_size
        self.terrain_size = terrain_size
        self.n_actions = n_actions
        self.is_policy = is_policy
        self.shooting_slice_start = shooting_slice_start
        self.shooting_slice_end = shooting_slice_end

        super().__init__()

        self.config = transformer_config
        self.embedding_size = transformer_config.embedding_size

        self.game_embedding = nn.Linear(
            self.game_size, self.config.embedding_size, bias=True
        )
        self.objective_embedding = nn.Linear(
            self.objective_size, self.config.embedding_size, bias=True
        )
        self.wargame_model_embedding = nn.Linear(
            self.wargame_model_size, self.config.embedding_size, bias=True
        )

        if self.opponent_model_size > 0:
            self.opponent_model_embedding: nn.Linear | None = nn.Linear(
                self.opponent_model_size, self.config.embedding_size, bias=True
            )
        else:
            self.opponent_model_embedding = None

        if self.terrain_size > 0:
            self.terrain_embedding: nn.Linear | None = nn.Linear(
                self.terrain_size, self.config.embedding_size, bias=True
            )
        else:
            self.terrain_embedding = None

        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(self.config.dropout),
                h=nn.ModuleList(
                    [Block(self.config) for _ in range(self.config.n_layers)]
                ),
                ln_f=LayerNorm(self.config.embedding_size, bias=self.config.bias),
            )
        )
        if self.is_policy:
            self.policy_head: nn.Linear | None = nn.Linear(
                self.config.embedding_size, self.n_actions, bias=False
            )
            self.value_head: nn.Linear | None = None
            self.shoot_query_proj: nn.Linear | None = None
            self.shoot_key_proj: nn.Linear | None = None
            if (
                self.opponent_model_embedding is not None
                and self.shooting_slice_start is not None
                and self.shooting_slice_end is not None
                and self.shooting_slice_end > self.shooting_slice_start
            ):
                self.shoot_query_proj = nn.Linear(
                    self.config.embedding_size, self.config.embedding_size, bias=True
                )
                self.shoot_key_proj = nn.Linear(
                    self.config.embedding_size, self.config.embedding_size, bias=True
                )
        else:
            self.policy_head = None
            self.value_head = nn.Linear(self.config.embedding_size, 1, bias=False)
            self.shoot_query_proj = None
            self.shoot_key_proj = None

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.to(get_device(device))

    def get_num_params(self) -> int:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return int(n_params)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the module.

        This is taken from the original GPT implementation, but I believe we should change it.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def embed_game_state(
        self, game_tensor: torch.Tensor, is_batched: bool = False
    ) -> torch.Tensor:
        """Embed the game state.

        Args:
            game_tensor: Tensor of shape (batch_size, game_size)
            is_batched: Whether the game tensor is batched

        Returns:
            Tensor of shape (batch_size, 1, embedding_size)
        """
        if not is_batched:
            game_tensor = game_tensor.unsqueeze(0)
        assert game_tensor.ndim == 2
        result: torch.Tensor = self.game_embedding(game_tensor).unsqueeze(
            1
        )  # shape (batch_size, 1, embedding_size)
        return result

    def embed_objective_state(
        self, objective_tensor: torch.Tensor, is_batched: bool = False
    ) -> torch.Tensor:
        """Embed the objective state.

        Args:
            objective_tensor: Tensor of shape (batch_size, num_objectives, objective_size)
            is_batched: Whether the objective tensor is batched

        Returns:
            Tensor of shape (batch_size, num_objectives, embedding_size)
        """
        if not is_batched:
            objective_tensor = objective_tensor.unsqueeze(0)
        assert objective_tensor.ndim == 3
        result: torch.Tensor = self.objective_embedding(
            objective_tensor
        )  # shape (batch_size, num_objectives, embedding_size)
        return result

    def embed_wargame_model_state(
        self, wargame_model_tensor: torch.Tensor, is_batched: bool = False
    ) -> torch.Tensor:
        """Embed the wargame model state.

        Args:
            wargame_model_tensor: Tensor of shape (batch_size, num_models, wargame_model_size)
            is_batched: Whether the wargame model tensor is batched

        Returns:
            Tensor of shape (batch_size, num_models, embedding_size)
        """
        if not is_batched:
            wargame_model_tensor = wargame_model_tensor.unsqueeze(0)
        assert wargame_model_tensor.ndim == 3
        result: torch.Tensor = self.wargame_model_embedding(
            wargame_model_tensor
        )  # shape (batch_size, num_models, embedding_size)
        return result

    def _embed_opponent_models(
        self, opp_tensor: torch.Tensor, is_batched: bool = False
    ) -> torch.Tensor | None:
        """Embed opponent model tokens.  Returns None when there are no opponents."""
        if self.opponent_model_embedding is None:
            return None
        if not is_batched:
            opp_tensor = opp_tensor.unsqueeze(0)
        if opp_tensor.shape[1] == 0:
            return None
        result: torch.Tensor = self.opponent_model_embedding(opp_tensor)
        return result

    def _alive_feature_index(self, feature_dim: int, n_opponents: int) -> int:
        """Infer the alive-feature column index from observation feature layout.

        Per-model rows end with ``N_WOUND_FEATURES + N_COMBAT_STATS`` columns
        (``alive`` is the first of these), optionally followed by ``n_opponents``
        expected-damage / padding columns. The ``alive`` flag therefore sits that
        many columns before the end. This is coupled to the layout produced by
        ``model/common/observation.py``; the constants keep the two in sync.
        """
        trailing = n_opponents if n_opponents > 0 else 0
        return feature_dim - trailing - (N_WOUND_FEATURES + N_COMBAT_STATS)

    def _alive_from_features(
        self, model_tensor: torch.Tensor, n_opponents: int
    ) -> torch.Tensor:
        """Per-row alive mask ``(batch, n_models)`` from the ``alive`` feature column.

        Falls back to all-alive if the inferred column index is out of range, so a
        layout change degrades to "everything alive" rather than crashing.
        """
        feature_dim = int(model_tensor.shape[-1])
        n_models = int(model_tensor.shape[1])
        if n_models == 0:
            return torch.ones(
                model_tensor.shape[0], 0, dtype=torch.bool, device=model_tensor.device
            )
        idx = self._alive_feature_index(feature_dim, n_opponents)
        if idx < 0 or idx >= feature_dim:
            return torch.ones(
                model_tensor.shape[0],
                n_models,
                dtype=torch.bool,
                device=model_tensor.device,
            )
        return model_tensor[:, :, idx] > 0.5

    def _embed_terrain(
        self, terrain_tensor: torch.Tensor, is_batched: bool = False
    ) -> torch.Tensor | None:
        """Embed terrain footprint tokens. Returns None when there is no terrain."""
        if self.terrain_embedding is None:
            return None
        if not is_batched:
            terrain_tensor = terrain_tensor.unsqueeze(0)
        if terrain_tensor.shape[1] == 0:
            return None
        result: torch.Tensor = self.terrain_embedding(terrain_tensor)
        return result

    def encode_state(self, xs: list[torch.Tensor]) -> EncodedState:
        """Encode observation tensors into contextual token representations.

        Runs a single batched transformer pass over the fixed-length token
        sequence ``[game, objectives, players, opponents, terrain]``; dead
        player/opponent rows are excluded from attention via a key-padding mask
        rather than being removed, so the batch stays a single forward.
        Terrain tokens are appended last (after opponents).

        Returns:
            An :class:`EncodedState` with the encoded sequence and the metadata
            needed to rebuild full-size policy logits.
        """
        game_tensor = xs[0]
        objective_tensor = xs[1]
        player_tensor = xs[2]
        opp_tensor = xs[3] if len(xs) > 3 else None
        terrain_tensor = xs[4] if len(xs) > 4 else None
        mask_tensor = xs[5] if len(xs) > 5 else None

        batched = self.is_batched(xs)
        if not batched:
            game_tensor = game_tensor.unsqueeze(0)
            objective_tensor = objective_tensor.unsqueeze(0)
            player_tensor = player_tensor.unsqueeze(0)
            if opp_tensor is not None:
                opp_tensor = opp_tensor.unsqueeze(0)
            if terrain_tensor is not None:
                terrain_tensor = terrain_tensor.unsqueeze(0)
            if mask_tensor is not None and mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

        game_embedding = self.embed_game_state(game_tensor, is_batched=True)
        objective_embedding = self.embed_objective_state(
            objective_tensor, is_batched=True
        )
        player_embedding = self.embed_wargame_model_state(
            player_tensor, is_batched=True
        )
        opp_embedding = (
            self._embed_opponent_models(opp_tensor, is_batched=True)
            if opp_tensor is not None
            else None
        )
        terrain_embedding = (
            self._embed_terrain(terrain_tensor, is_batched=True)
            if terrain_tensor is not None
            else None
        )

        batch_size = int(player_embedding.shape[0])
        n_wargame_models = int(player_embedding.shape[1])
        n_prefix = 1 + int(objective_embedding.shape[1])

        parts = [game_embedding, objective_embedding, player_embedding]
        if opp_embedding is not None:
            n_opponents = int(opp_embedding.shape[1])
            parts.append(opp_embedding)
        else:
            n_opponents = 0

        if terrain_embedding is not None:
            parts.append(terrain_embedding)

        tokens = torch.cat(parts, dim=1)
        seq_len = int(tokens.shape[1])

        # Key-padding mask: prefix + alive players + alive opponents + terrain
        # may be attended to; dead rows are dropped as keys (True = attend).
        player_alive = self._alive_from_features(player_tensor, n_opponents)
        key_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=tokens.device
        )
        key_mask[:, n_prefix : n_prefix + n_wargame_models] = player_alive
        if opp_embedding is not None and opp_tensor is not None and n_opponents > 0:
            opp_alive = self._alive_from_features(opp_tensor, n_opponents)
            opp_start = n_prefix + n_wargame_models
            key_mask[:, opp_start : opp_start + n_opponents] = opp_alive
        # Terrain tokens are always attendable (no alive/dead concept).
        attn_mask = key_mask[:, None, None, :]

        x = tokens
        for block in self.transformer.h:  # type: ignore
            x = block(x, attn_mask=attn_mask)
        encoded = cast(torch.Tensor, self.transformer.ln_f(x))  # type: ignore

        return EncodedState(
            encoded=encoded,
            n_prefix=n_prefix,
            n_wargame_models=n_wargame_models,
            n_opponents=n_opponents,
            player_alive=player_alive,
            mask_tensor=mask_tensor,
        )

    def _shooting_scores(
        self, player_latents: torch.Tensor, opponent_latents: torch.Tensor
    ) -> torch.Tensor:
        """Compute compact shooting logits from player/opponent latents."""
        if self.shoot_query_proj is None or self.shoot_key_proj is None:
            raise ValueError("Shooting head is not initialized.")
        q = self.shoot_query_proj(player_latents)
        k = self.shoot_key_proj(opponent_latents)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) * scale
        return scores

    def policy_from_encoded(self, state: EncodedState) -> torch.Tensor:
        """Rebuild full-size policy logits ``(batch, n_models, n_actions)``.

        Fully vectorized: player ``p`` is at token ``n_prefix + p`` and opponent
        ``o`` at ``n_prefix + n_models + o``, so no per-sample loop is needed.
        Dead player rows are forced stay-only; shooting columns carry bilinear
        player-vs-opponent scores. Remaining invalid actions are enforced by the
        env-provided action mask carried on ``state``.
        """
        if not self.is_policy:
            raise ValueError("Policy head requested from a value network.")
        if self.policy_head is None:
            raise ValueError("Policy head is not initialized.")

        encoded = state.encoded
        n_prefix = state.n_prefix
        n_wargame_models = state.n_wargame_models
        n_opponents = state.n_opponents

        player_latents = encoded[:, n_prefix : n_prefix + n_wargame_models, :]
        base_logits = self.policy_head(player_latents)

        can_score_shooting = (
            self.shooting_slice_start is not None
            and self.shooting_slice_end is not None
            and self.shooting_slice_end > self.shooting_slice_start
            and self.shoot_query_proj is not None
            and self.shoot_key_proj is not None
            and n_opponents > 0
        )
        if can_score_shooting:
            opp_start = n_prefix + n_wargame_models
            opponent_latents = encoded[:, opp_start : opp_start + n_opponents, :]
            shooting_scores = self._shooting_scores(player_latents, opponent_latents)
            start = cast(int, self.shooting_slice_start)
            base_logits = base_logits.clone()
            base_logits[:, :, start : start + n_opponents] = shooting_scores

        # Dead player rows: keep only ``stay`` (finite) so they sample as stay even
        # if the env mask is absent. Alive rows keep their head/shooting logits.
        dead_row = torch.full_like(base_logits, float("-inf"))
        dead_row[:, :, 0] = 0.0
        alive = state.player_alive.unsqueeze(-1)
        logits = torch.where(alive, base_logits, dead_row)

        mask_tensor = state.mask_tensor
        if (
            mask_tensor is not None
            and mask_tensor.shape[-1] == self.n_actions
            and mask_tensor.shape[-2] == n_wargame_models
        ):
            logits = logits.masked_fill(~mask_tensor.bool(), float("-inf"))

        all_invalid = torch.isneginf(logits).all(dim=-1)
        stay = torch.where(
            all_invalid, torch.zeros_like(logits[:, :, 0]), logits[:, :, 0]
        )
        logits = torch.cat([stay.unsqueeze(-1), logits[:, :, 1:]], dim=-1)
        return logits

    def value_from_encoded(self, state: EncodedState) -> torch.Tensor:
        """Apply value head to encoded tokens."""
        if self.is_policy:
            raise ValueError("Value head requested from a policy network.")
        if self.value_head is None:
            raise ValueError("Value head is not initialized.")
        # Use the global game token (first token) as the critic summary.
        game_token = state.encoded[:, 0, :]
        value: torch.Tensor = self.value_head(game_token)
        return value.squeeze(-1)

    def share_backbone_with(self, backbone_source: "TransformerNetwork") -> None:
        """Share embedding + transformer trunk with another TransformerNetwork."""
        self.game_embedding = backbone_source.game_embedding
        self.objective_embedding = backbone_source.objective_embedding
        self.wargame_model_embedding = backbone_source.wargame_model_embedding
        self.opponent_model_embedding = backbone_source.opponent_model_embedding
        self.terrain_embedding = backbone_source.terrain_embedding
        self.transformer = backbone_source.transformer

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        encoded_state = self.encode_state(xs)

        if self.is_policy:
            return self.policy_from_encoded(encoded_state)
        return self.value_from_encoded(encoded_state)

    # def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    #     # start with all of the candidate parameters
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     # filter out those that do not require grad
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {'params': decay_params, 'weight_decay': weight_decay},
    #         {'params': nodecay_params, 'weight_decay': 0.0}
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    #     # Create AdamW optimizer and use the fused version if it is available
    #     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == 'cuda'
    #     extra_args = dict(fused=True) if use_fused else dict()
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    #     print(f"using fused AdamW: {use_fused}")

    #     return optimizer

    @classmethod
    def from_env(cls, env: WargameEnv, is_policy: bool) -> Self:
        observation: WargameEnvObservation
        observation, _ = env.reset()
        tensors = observation_to_tensor(observation)
        game_size = int(tensors[0].shape[-1])
        objective_size = int(tensors[1].shape[-1])
        wargame_model_size = int(tensors[2].shape[-1]) if tensors[2].numel() > 0 else 0
        opponent_model_size = int(tensors[3].shape[-1]) if tensors[3].numel() > 0 else 0
        terrain_size = int(tensors[4].shape[-1]) if tensors[4].numel() > 0 else 0
        n_actions: int = env._action_handler.n_actions
        shooting_slice = env._action_handler.shooting_slice
        transformer_config = TransformerConfig()

        print(
            f"game_size: {game_size}, objective_size: {objective_size}, "
            f"wargame_model_size: {wargame_model_size}, "
            f"opponent_model_size: {opponent_model_size}, "
            f"terrain_size: {terrain_size}, "
            f"shooting_slice: {shooting_slice}, "
            f"transformer_config: {transformer_config}, n_actions: {n_actions}"
        )
        return cls(
            game_size=game_size,
            objective_size=objective_size,
            wargame_model_size=wargame_model_size,
            n_actions=n_actions,
            transformer_config=transformer_config,
            is_policy=is_policy,
            opponent_model_size=opponent_model_size,
            terrain_size=terrain_size,
            shooting_slice_start=shooting_slice.start if shooting_slice else None,
            shooting_slice_end=shooting_slice.end if shooting_slice else None,
        )


def convert_state_dict(state_dict: dict) -> dict:
    """Normalize state_dict keys (Lightning 'policy_net.', torch.compile '_orig_mod.')."""
    new_state_dict = {}
    prefix = "policy_net."
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[10:]
        new_state_dict[new_key] = value
    return new_state_dict
