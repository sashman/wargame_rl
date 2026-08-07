from __future__ import annotations

from typing import Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.model.common import Device, get_device
from wargame_rl.wargame.model.net import RL_Network, TransformerNetwork


def shooting_decode_applies(
    action_logits: Tensor, shooting_slice: tuple[int, int] | None
) -> bool:
    """True when target de-duplication is enabled and it is a shooting step.

    The phase is read off the logits rather than passed in: the env's action
    mask has already driven the shooting slice to -inf everywhere except during
    the shooting phase, so a finite entry there is exactly the condition under
    which de-duplicating targets means anything. This keeps the movement phase
    on the cheap single-shot path.
    """
    if shooting_slice is None:
        return False
    start, end = shooting_slice
    return bool(torch.isfinite(action_logits[..., start:end]).any())


def targets_taken_earlier(
    actions: Tensor, n_actions: int, shooting_slice: tuple[int, int]
) -> Tensor:
    """``(batch, n_models, n_actions)`` bool: targets claimed by a lower index.

    Exclusive by construction, so a model never forbids its own choice -- only
    the choices of models decoded before it. Vectorised, because here the
    choices are already known.
    """
    start, end = shooting_slice
    chosen = torch.zeros(
        (*actions.shape, n_actions), dtype=torch.bool, device=actions.device
    )
    chosen.scatter_(2, actions.unsqueeze(-1), True)
    taken_inclusive = chosen.cumsum(dim=1) > 0
    taken_before = torch.cat(
        [torch.zeros_like(taken_inclusive[:, :1]), taken_inclusive[:, :-1]], dim=1
    )
    forbidden = torch.zeros_like(taken_before)
    forbidden[..., start:end] = taken_before[..., start:end]
    return forbidden


def decode_distinct_targets(
    action_logits: Tensor, shooting_slice: tuple[int, int], deterministic: bool
) -> Tuple[Tensor, Tensor]:
    """Pick one action per model, no two models naming the same target.

    Sequential because the constraint is joint: model ``i``'s distribution
    depends on what models ``< i`` actually chose. Returns actions
    ``(batch, n_models)`` and the conditional log-probs under the same masking,
    whose sum is the joint log-prob of the autoregressive policy.

    "Stay" sits outside the shooting slice and is legal in every phase, so a
    model whose only target was claimed falls back to holding fire rather than
    being left with no legal action at all.
    """
    start, end = shooting_slice
    batch, n_models, _ = action_logits.shape
    actions = torch.zeros(
        (batch, n_models), dtype=torch.long, device=action_logits.device
    )
    log_probs = torch.zeros(
        (batch, n_models), dtype=action_logits.dtype, device=action_logits.device
    )
    taken = torch.zeros(
        (batch, end - start), dtype=torch.bool, device=action_logits.device
    )

    for model_idx in range(n_models):
        row = action_logits[:, model_idx, :].clone()
        row[:, start:end] = row[:, start:end].masked_fill(taken, float("-inf"))
        row_dist = Categorical(logits=row)
        picked = torch.argmax(row, dim=-1) if deterministic else row_dist.sample()
        actions[:, model_idx] = picked
        log_probs[:, model_idx] = row_dist.log_prob(picked)
        shot = (picked >= start) & (picked < end)
        if bool(shot.any()):
            taken[shot, picked[shot] - start] = True

    return actions, log_probs


def greedy_actions(
    action_logits: Tensor, shooting_slice: tuple[int, int] | None
) -> Tensor:
    """Greedy per-model actions, honouring autoregressive target decoding.

    Every greedy consumer must go through here rather than calling ``argmax``
    on the logits: a plain argmax lets all models name the highest-scoring
    target, which is exactly the duplicate-target behaviour the decode removes,
    and it would score a policy the agent never played.
    """
    if shooting_decode_applies(action_logits, shooting_slice):
        actions, _log_probs = decode_distinct_targets(
            action_logits, cast(Tuple[int, int], shooting_slice), deterministic=True
        )
        return actions
    greedy: Tensor = action_logits.argmax(dim=-1)
    return greedy


class PPOModel(nn.Module):
    """PPO Model combining policy and value networks."""

    def __init__(
        self,
        policy_network: RL_Network,
        value_network: RL_Network,
        device: Device = None,
        share_transformer: bool = False,
        shooting_slice: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if not policy_network.is_policy or value_network.is_policy:
            raise ValueError("Wrong network type.")
        if share_transformer and (
            not isinstance(policy_network, TransformerNetwork)
            or not isinstance(value_network, TransformerNetwork)
        ):
            raise ValueError(
                "`share_transformer=True` requires TransformerNetwork for both policy and value networks."
            )
        if share_transformer:
            assert isinstance(policy_network, TransformerNetwork)
            assert isinstance(value_network, TransformerNetwork)
            value_network.share_backbone_with(policy_network)
        self.policy_network = policy_network
        self.value_network = value_network
        self.share_transformer = share_transformer
        # None disables autoregressive shooting decode; the model then behaves
        # exactly as before, so every existing checkpoint keeps its semantics.
        self.shooting_slice = shooting_slice
        self.to(get_device(device))

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        """Derive device from actual parameter location (stays correct after Lightning moves the model)."""
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device("cpu")

    def forward(self, x: list[torch.Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward pass through both networks.

        Args:
            x: List of input tensors (game state, objectives, models).

        Returns:
            (action_logits, state_values) where action_logits has shape
            (batch, n_models, n_actions) and state_values has shape (batch,).
        """
        if self.share_transformer:
            policy_network = cast(TransformerNetwork, self.policy_network)
            value_network = cast(TransformerNetwork, self.value_network)
            encoded_state = policy_network.encode_state(x)
            action_logits = policy_network.policy_from_encoded(encoded_state)
            state_values = value_network.value_from_encoded(encoded_state)
            return action_logits, state_values

        action_logits = self.policy_network(x)
        state_values = self.value_network(x)
        return action_logits, state_values

    def get_action(
        self, state_tensors: list[torch.Tensor], deterministic: bool = False
    ) -> Tuple[WargameEnvAction, Tensor]:
        """Select one action per model, mirroring how DQN selects per-model actions.

        Args:
            state_tensors: Observation converted to tensors (single observation, not batched).
            deterministic: If True take argmax, otherwise sample.

        Returns:
            (env_action, per_model_log_probs) where env_action contains a
            per-model action list and per_model_log_probs has shape
            (n_models,) — kept unsummed so PPO can clip each model's
            importance ratio separately.
        """
        action_logits, _ = self.forward(state_tensors)

        if self._shooting_decode_applies(action_logits):
            actions, per_model_log_probs = self._decode_distinct_targets(
                action_logits, deterministic=deterministic
            )
            env_action = WargameEnvAction(actions=actions.flatten().tolist())
            return env_action, per_model_log_probs.squeeze(0)

        action_dist = Categorical(logits=action_logits)

        if deterministic:
            actions = torch.argmax(action_logits, dim=-1)
        else:
            actions = action_dist.sample()

        per_model_log_probs = action_dist.log_prob(actions).squeeze(0)

        env_action = WargameEnvAction(actions=actions.flatten().tolist())
        return env_action, per_model_log_probs

    def evaluate_actions(
        self, state_tensors: list[torch.Tensor], actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions under the current policy.

        Args:
            state_tensors: Batch of observations as tensors.
            actions: Per-model actions, shape (batch_size, n_models).

        Returns:
            (action_logits, log_probs, entropy) with the per-model quantities
            left unsummed, shape (batch_size, n_models).
        """
        action_logits, _ = self.forward(state_tensors)

        if self._shooting_decode_applies(action_logits):
            # Replay the rollout's masking rule. It depends only on the actions
            # of lower-indexed models, all of which are in `actions`, so the
            # conditionals here are exactly the ones the rollout sampled under
            # and the importance ratio stays correct.
            action_logits = action_logits.masked_fill(
                targets_taken_earlier(
                    actions,
                    action_logits.shape[-1],
                    cast(Tuple[int, int], self.shooting_slice),
                ),
                float("-inf"),
            )

        action_dist = Categorical(logits=action_logits)

        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return action_logits, log_probs, entropy

    def greedy_actions_from_logits(self, action_logits: Tensor) -> Tensor:
        """Greedy per-model actions, honouring autoregressive target decoding."""
        return greedy_actions(action_logits, self.shooting_slice)

    def _shooting_decode_applies(self, action_logits: Tensor) -> bool:
        return shooting_decode_applies(action_logits, self.shooting_slice)

    def _decode_distinct_targets(
        self, action_logits: Tensor, deterministic: bool
    ) -> Tuple[Tensor, Tensor]:
        return decode_distinct_targets(
            action_logits,
            cast(Tuple[int, int], self.shooting_slice),
            deterministic=deterministic,
        )
