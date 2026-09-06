"""The size-independent set network for the per-model facade (issue #285)."""

from wargame_rl.wargame.model.per_model.agent import SetAgent, StepDecision
from wargame_rl.wargame.model.per_model.batch import TokenBatch, collate
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.net import (
    ActionLogits,
    SetNetwork,
    SetNetworkOutput,
)

__all__ = [
    "ActionLogits",
    "SetAgent",
    "SetNetwork",
    "SetNetworkConfig",
    "SetNetworkOutput",
    "StepDecision",
    "TokenBatch",
    "collate",
]
