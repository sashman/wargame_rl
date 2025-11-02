from abc import ABC, abstractmethod
from typing import Self

import gymnasium as gym
import torch
from torch import nn

from wargame_rl.wargame.envs.wargame import MovementPhaseActions
from wargame_rl.wargame.model.dqn.device import Device, get_device


class RL_Network(nn.Module, ABC):
    device: torch.device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_env(cls, env: gym.Env) -> Self:
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
    def from_state_dict(cls, env: gym.Env, state_dict: dict) -> Self:
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

    def is_batched(self, xs: list[torch.Tensor]) -> bool:
        return len(xs[0].shape) > 1

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
    def from_env(cls, env: gym.Env) -> Self:
        observation, _ = env.reset()
        obs_size: int = observation.size
        n_wargame_models: int = observation.n_wargame_models
        n_actions: int = len(MovementPhaseActions)

        print(
            f"obs_size: {obs_size}, n_wargame_models: {n_wargame_models}, n_actions: {n_actions}"
        )
        return cls(obs_size, n_actions, n_wargame_models)


# class DQN_Transformer(RL_Network):

#     def __init__(self,
#         obs_size,
#         n_actions,
#         block_config: BlockConfig,
#         device: Device | None = None,
#     ) -> None:

#         self.obs_size = obs_size
#         self.n_actions = n_actions

#         super().__init__()
#         assert config.vocab_size is not None
#         assert config.block_size is not None
#         self.config = config

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.vocab_size, config.n_embd),
#             wpe = nn.Embedding(config.block_size, config.n_embd),
#             drop = nn.Dropout(config.dropout),
#             h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
#             ln_f = LayerNorm(config.n_embd, bias=config.bias),
#         ))
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
#         # with weight tying when using torch.compile() some warnings get generated:
#         # "UserWarning: functional_call was passed multiple values for tied weights.
#         # This behavior is deprecated and will be an error in future versions"
#         # not 100% sure what this is, so far seems to be harmless. TODO investigate
#         self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

#         # init all weights
#         self.apply(self._init_weights)
#         # apply special scaled init to the residual projections, per GPT-2 paper
#         for pn, p in self.named_parameters():
#             if pn.endswith('c_proj.weight'):
#                 torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

#         # report number of parameters
#         print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

#         self.device = get_device(device)
#         self.to(self.device)

#     def get_num_params(self, non_embedding=True):
#         """
#         Return the number of parameters in the model.
#         For non-embedding count (default), the position embeddings get subtracted.
#         The token embeddings would too, except due to the parameter sharing these
#         params are actually used as weights in the final layer, so we include them.
#         """
#         n_params = sum(p.numel() for p in self.parameters())
#         if non_embedding:
#             n_params -= self.transformer.wpe.weight.numel()
#         return n_params

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(self, idx, targets=None):
#         device = idx.device
#         b, t = idx.size()
#         assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
#         pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

#         # forward the GPT model itself
#         tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
#         pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
#         x = self.transformer.drop(tok_emb + pos_emb)
#         for block in self.transformer.h:
#             x = block(x)
#         x = self.transformer.ln_f(x)

#         if targets is not None:
#             # if we are given some desired targets also calculate the loss
#             logits = self.lm_head(x)
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
#         else:
#             # inference-time mini-optimization: only forward the lm_head on the very last position
#             logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
#             loss = None

#         return logits, loss


#     # def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
#     #     # start with all of the candidate parameters
#     #     param_dict = {pn: p for pn, p in self.named_parameters()}
#     #     # filter out those that do not require grad
#     #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#     #     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
#     #     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
#     #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#     #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#     #     optim_groups = [
#     #         {'params': decay_params, 'weight_decay': weight_decay},
#     #         {'params': nodecay_params, 'weight_decay': 0.0}
#     #     ]
#     #     num_decay_params = sum(p.numel() for p in decay_params)
#     #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
#     #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
#     #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
#     #     # Create AdamW optimizer and use the fused version if it is available
#     #     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#     #     use_fused = fused_available and device_type == 'cuda'
#     #     extra_args = dict(fused=True) if use_fused else dict()
#     #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
#     #     print(f"using fused AdamW: {use_fused}")

#     #     return optimizer


#     @classmethod
#     def from_env(cls, env: gym.Env) -> Self:
#         observation, _ = env.reset()
#         obs_size: int = observation.size
#         # n_wargame_models: int = observation.n_wargame_models
#         n_actions: int = len(MovementPhaseActions)
#         block_config = BlockConfig(block_size=obs_size,
#                     n_layer=6,
#                     n_head=8,
#                     n_embd=128,
#                     dropout=0.0,
#                     bias=True)

#         print(
#             f"obs_size: {obs_size}, block_config: {block_config}, n_actions: {n_actions}"
#         )
#         return cls(obs_size, n_actions, block_config)


def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("policy_net."):
            new_key = key[11:]
            new_state_dict[new_key] = value
    return new_state_dict
