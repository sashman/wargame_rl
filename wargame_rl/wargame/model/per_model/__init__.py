"""The size-independent set network over the per-model facade (issue #285),
and the PPO loop that trains it over decision steps (issue #286).

The first client of `envs/per_model/`. Token observations come from
`envs/per_model/tokens.py` (numpy); this package collates them, encodes them,
drives a seat, and trains the weights. ⚠ Checkpoints do not pickle this
package (`checkpoint.py` stores plain tensors and config dicts), so the path
may move; the abandoned Lightning route would have pinned it.
"""

from wargame_rl.wargame.model.per_model.agent import NO_DRAW, SetAgent, StepDecision
from wargame_rl.wargame.model.per_model.batch import TokenBatch, collate
from wargame_rl.wargame.model.per_model.checkpoint import (
    LoadedCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.evaluate import EvalResult, evaluate_per_model
from wargame_rl.wargame.model.per_model.net import (
    HeadLogits,
    SetNetwork,
    SetNetworkOutput,
)
from wargame_rl.wargame.model.per_model.ppo import (
    PerModelPPOConfig,
    Rollout,
    Transition,
    collect_rollout,
    compute_gae,
    evaluate_transitions,
    ppo_update,
)

__all__ = [
    "NO_DRAW",
    "HeadLogits",
    "LoadedCheckpoint",
    "EvalResult",
    "PerModelPPOConfig",
    "Rollout",
    "SetAgent",
    "SetNetwork",
    "SetNetworkConfig",
    "SetNetworkOutput",
    "StepDecision",
    "TokenBatch",
    "Transition",
    "collate",
    "collect_rollout",
    "compute_gae",
    "evaluate_per_model",
    "evaluate_transitions",
    "load_checkpoint",
    "ppo_update",
    "save_checkpoint",
]
