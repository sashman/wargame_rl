"""The per-model env facade: one step is one model's action (issue #283/#284).

A second facade over the same ``domain/`` layer. The whole-phase ``WargameEnv``
is untouched and remains the default; nothing constructs this package unless
asked to.
"""

from wargame_rl.wargame.envs.per_model.adapter import ScriptedPolicyAdapter
from wargame_rl.wargame.envs.per_model.facade import PerModelEnv
from wargame_rl.wargame.envs.per_model.types import (
    ChargeDeclaration,
    MoveDeclaration,
    PerModelAction,
    PerModelObservation,
    ShootDeclaration,
    StepKind,
)

__all__ = [
    "ChargeDeclaration",
    "MoveDeclaration",
    "PerModelAction",
    "PerModelObservation",
    "PerModelEnv",
    "ScriptedPolicyAdapter",
    "ShootDeclaration",
    "StepKind",
]
