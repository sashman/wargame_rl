"""Baseline: never move, fire at the nearest valid target -- a blocker that shoots back."""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.hold_deployment import (
    ScriptedHoldDeploymentPolicy,
)
from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_shoot import (
    ScriptedSquadMarchShootPolicy,
)


class ScriptedHoldAndShootPolicy(ScriptedHoldDeploymentPolicy):
    """``STAY`` every movement phase; fire at the nearest valid unit every
    shooting phase.

    `hold_deployment` shoots at nothing on purpose, so that a null against it
    is about position alone. The curriculum's C2 rung (#340) needs the other
    half of that: an enemy that stands on a point and makes approaching it
    cost bodies, with no movement of its own to confound the question. The
    target rule is `squad_march_shoot`'s, unchanged -- nearest valid unit under
    the env's own mask -- so a shooting figure against this seat and against
    that bar are the same rule from two positions.
    """

    select_shooting = ScriptedSquadMarchShootPolicy.select_shooting


register_baseline("hold_and_shoot", ScriptedHoldAndShootPolicy)
