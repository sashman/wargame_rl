"""Shared utilities for training models (device, logging, callbacks)."""

from wargame_rl.wargame.model.common.checkpoint_callback import get_checkpoint_callback
from wargame_rl.wargame.model.common.device import Device, auto_device, get_device
from wargame_rl.wargame.model.common.elo import EloRatingSystem
from wargame_rl.wargame.model.common.env_config_callback import EnvConfigCallback
from wargame_rl.wargame.model.common.wandb import get_logger, init_wandb

__all__ = [
    "Device",
    "EloRatingSystem",
    "EnvConfigCallback",
    "get_checkpoint_callback",
    "get_device",
    "get_logger",
    "init_wandb",
    "auto_device",
]
