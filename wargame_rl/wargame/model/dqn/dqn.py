import math
from abc import ABC, abstractmethod
from typing import Self

import gymnasium as gym
import torch
from torch import nn

from wargame_rl.wargame.envs.types import WargameEnvObservation
from wargame_rl.wargame.envs.wargame import MovementPhaseActions, WargameEnv
from wargame_rl.wargame.model.dqn.device import Device, get_device
from wargame_rl.wargame.model.dqn.layers import Block, LayerNorm, TransformerConfig


class RL_Network(nn.Module, ABC):
    device: torch.device

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
    def from_env(cls, env: WargameEnv) -> Self:
        pass

    @classmethod
    def from_checkpoint(cls, env: gym.Env, checkpoint_path: str) -> Self:
        load_dict = torch.load(checkpoint_path, weights_only=False)
        if "state_dict" in load_dict:
            state_dict = convert_state_dict(load_dict["state_dict"])
        else:
            state_dict = load_dict
        return cls.from_state_dict(env, state_dict)

    @classmethod
    def from_state_dict(cls, env: WargameEnv, state_dict: dict) -> Self:
        net = cls.from_env(env)
        net.load_state_dict(state_dict)
        return net


class DQN_MLP(RL_Network):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_wargame_models: int,
        device: Device | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, action_dim * n_wargame_models)
        self.activation = nn.GELU()
        self.device = get_device(device)
        self.to(self.device)
        self.n_wargame_models = n_wargame_models
        self.action_dim = action_dim

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
        return x.reshape(batch_size, self.n_wargame_models, self.action_dim)

    @classmethod
    def from_env(cls, env: WargameEnv) -> Self:
        observation: WargameEnvObservation
        observation, _ = env.reset()
        obs_size: int = observation.size
        n_wargame_models: int = observation.n_wargame_models
        n_actions: int = len(MovementPhaseActions)

        print(
            f"obs_size: {obs_size}, n_wargame_models: {n_wargame_models}, n_actions: {n_actions}"
        )
        return cls(obs_size, n_actions, n_wargame_models)


class DQN_Transformer(RL_Network):
    # Transformer adapted from the NanoGPT implementation:
    # https://github.com/karpathy/nanoGPT
    def __init__(
        self,
        game_size: int,
        objective_size: int,
        wargame_model_size: int,
        n_actions: int,
        transformer_config: TransformerConfig,
        device: Device | None = None,
    ) -> None:
        self.game_size = game_size
        self.objective_size = objective_size
        self.wargame_model_size = wargame_model_size
        self.n_actions = n_actions

        super().__init__()

        self.config = transformer_config
        self.embedding_size = transformer_config.embedding_size

        self.game_embedding = nn.Linear(
            self.game_size, self.config.embedding_size, bias=False
        )
        self.objective_embedding = nn.Linear(
            self.objective_size, self.config.embedding_size, bias=False
        )
        self.wargame_model_embedding = nn.Linear(
            self.wargame_model_size, self.config.embedding_size, bias=False
        )

        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(self.config.dropout),
                h=nn.ModuleList(
                    [Block(self.config) for _ in range(self.config.n_layers)]
                ),
                ln_f=LayerNorm(self.config.embedding_size, bias=self.config.bias),
            )
        )

        self.action_head = nn.Linear(
            self.config.embedding_size, self.n_actions, bias=False
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.device = get_device(device)
        self.to(self.device)

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
        return self.game_embedding(game_tensor).unsqueeze(
            1
        )  # shape (batch_size, 1, embedding_size)

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
        return self.objective_embedding(
            objective_tensor
        )  # shape (batch_size, num_objectives, embedding_size)

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
        return self.wargame_model_embedding(
            wargame_model_tensor
        )  # shape (batch_size, num_models, embedding_size)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        game_tensor, objective_tensor, wargame_model_tensor = xs

        # we compbute embeddings for each of the tensors in xs
        game_embedding = self.embed_game_state(game_tensor, self.is_batched(xs))
        objective_embedding = self.embed_objective_state(
            objective_tensor, self.is_batched(xs)
        )
        wargame_model_embedding = self.embed_wargame_model_state(
            wargame_model_tensor, self.is_batched(xs)
        )
        n_wargame_models = wargame_model_embedding.shape[1]
        # we concatenate the embeddings in a sequence for the transformer.
        # Note that the transformer does not know it is a sequence as there is no positional encoding.
        x = torch.cat(
            [game_embedding, objective_embedding, wargame_model_embedding], dim=1
        )
        # the shape of x is now (batch_size, ..., embedding_size)

        # we forward through the transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # we select the output for the wargame models
        wargame_model_output = x[:, -n_wargame_models:, :]
        logits = self.action_head(wargame_model_output)

        return logits

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
    def from_env(cls, env: WargameEnv) -> Self:
        observation: WargameEnvObservation
        observation, _ = env.reset()
        objective_size: int = observation.size_objectives[0]
        wargame_model_size: int = observation.size_wargame_models[0]
        game_size: int = observation.size_game_observation
        n_actions: int = len(MovementPhaseActions)
        transformer_config = TransformerConfig()

        print(
            f"game_size: {game_size}, objective_size: {objective_size}, wargame_model_size: {wargame_model_size}, transformer_config: {transformer_config}, n_actions: {n_actions}"
        )
        return cls(
            game_size=game_size,
            objective_size=objective_size,
            wargame_model_size=wargame_model_size,
            n_actions=n_actions,
            transformer_config=transformer_config,
        )


def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("policy_net."):
            new_key = key[11:]
            new_state_dict[new_key] = value
    return new_state_dict
