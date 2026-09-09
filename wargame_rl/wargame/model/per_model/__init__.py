"""The size-independent set network over the per-model facade (issue #285).

The first client of `envs/per_model/`. Token observations come from
`envs/per_model/tokens.py` (numpy); this package collates them, encodes them
and drives a seat. ⚠ Stage 3 will pickle this package's path into every
checkpoint's hyper-parameters, so it is not to be moved without an alias.
"""

from wargame_rl.wargame.model.per_model.agent import SetAgent, StepDecision
from wargame_rl.wargame.model.per_model.batch import TokenBatch, collate
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.net import (
    HeadLogits,
    SetNetwork,
    SetNetworkOutput,
)

__all__ = [
    "HeadLogits",
    "SetAgent",
    "SetNetwork",
    "SetNetworkConfig",
    "SetNetworkOutput",
    "StepDecision",
    "TokenBatch",
    "collate",
]
