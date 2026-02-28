import math
from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import nn

from wargame_rl.wargame.envs.types import WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common import Device, get_device
from wargame_rl.wargame.model.common.config import TransformerConfig
from wargame_rl.wargame.model.dqn.layers import Block, LayerNorm


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
        # 1 Concatenate all tensors in xs
        if self.is_batched(xs):
            x = torch.cat([x.flatten(start_dim=1) for x in xs], dim=1)
        else:
            x = torch.cat([x.flatten(start_dim=0) for x in xs], dim=0).unsqueeze(0)

        # 2 Forward through the network
        assert len(x.shape) == 2

        batch_size = x.shape[0]
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output(x)
        if self.is_policy:
            x = x.reshape(batch_size, self.n_wargame_models, self.action_dim)
        return x

    @classmethod
    def from_env(cls, env: WargameEnv, is_policy: bool) -> Self:
        observation: WargameEnvObservation
        observation, _ = env.reset()
        obs_size: int = observation.size
        n_wargame_models: int = observation.n_wargame_models
        n_actions: int = env._action_handler.n_actions
        print(
            f"Creating MLP network with obs_size: {obs_size}, n_wargame_models: {n_wargame_models}, n_actions: {n_actions}, is_policy: {is_policy}"
        )
        return MLPNetwork(obs_size, n_actions, n_wargame_models, is_policy=is_policy)


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
        device: Device | None = None,
    ) -> None:
        self.game_size = game_size
        self.objective_size = objective_size
        self.wargame_model_size = wargame_model_size
        self.opponent_model_size = opponent_model_size
        self.n_actions = n_actions
        self.is_policy = is_policy

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
            self.action_head = nn.Linear(
                self.config.embedding_size, self.n_actions, bias=False
            )
        else:
            self.action_head = nn.Linear(self.config.embedding_size, 1, bias=False)

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

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        game_tensor = xs[0]
        objective_tensor = xs[1]
        wargame_model_tensor = xs[2]
        opp_tensor = xs[3] if len(xs) > 3 else None

        batched = self.is_batched(xs)
        game_embedding = self.embed_game_state(game_tensor, batched)
        objective_embedding = self.embed_objective_state(objective_tensor, batched)
        wargame_model_embedding = self.embed_wargame_model_state(
            wargame_model_tensor, batched
        )
        n_wargame_models = wargame_model_embedding.shape[1]

        # Sequence: [game, objectives, player_models, (opponent_models)]
        parts = [game_embedding, objective_embedding, wargame_model_embedding]

        if opp_tensor is not None:
            opp_embedding = self._embed_opponent_models(opp_tensor, batched)
            if opp_embedding is not None:
                parts.append(opp_embedding)

        x = torch.cat(parts, dim=1)

        for block in self.transformer.h:  # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x)  # type: ignore

        if self.is_policy:
            # Player model tokens are right after game(1) + objectives(N_o).
            n_prefix = 1 + objective_embedding.shape[1]
            wargame_model_output = x[:, n_prefix : n_prefix + n_wargame_models, :]
            logits: torch.Tensor = self.action_head(wargame_model_output)
            return logits

        value: torch.Tensor = self.action_head(x)
        return value.mean(dim=[1, 2])

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
        objective_size: int = observation.size_objectives[0]
        wargame_model_size: int = observation.size_wargame_models[0]
        game_size: int = observation.size_game_observation
        n_actions: int = env._action_handler.n_actions
        transformer_config = TransformerConfig()

        opponent_model_size = 0
        if observation.size_opponent_models:
            opponent_model_size = observation.size_opponent_models[0]

        print(
            f"game_size: {game_size}, objective_size: {objective_size}, "
            f"wargame_model_size: {wargame_model_size}, "
            f"opponent_model_size: {opponent_model_size}, "
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
        )


def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("policy_net."):
            new_key = key[11:]
            new_state_dict[new_key] = value
    return new_state_dict
