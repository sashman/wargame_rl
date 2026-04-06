"""Helpers to load PPO policy weights from checkpoints/snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _strip_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        normalized[new_key] = value
    return normalized


def _ensure_tensor_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}


def load_ppo_policy_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load policy-network weights from a PPO checkpoint or policy snapshot."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and "policy_state_dict" in loaded:
        state_dict = _ensure_tensor_dict(loaded["policy_state_dict"])
        return _normalize_state_dict_keys(state_dict)

    if isinstance(loaded, dict) and "state_dict" in loaded:
        full_state = _ensure_tensor_dict(loaded["state_dict"])
        policy_sd = _strip_prefix(full_state, "ppo_model.policy_network.")
        if not policy_sd:
            policy_sd = _strip_prefix(full_state, "policy_network.")
        if not policy_sd:
            policy_sd = _strip_prefix(full_state, "policy_net.")
        if policy_sd:
            return _normalize_state_dict_keys(policy_sd)

    if isinstance(loaded, dict):
        raw = _ensure_tensor_dict(loaded)
        if raw:
            return _normalize_state_dict_keys(raw)

    raise ValueError(f"Unsupported checkpoint format for PPO policy: {checkpoint_path}")
