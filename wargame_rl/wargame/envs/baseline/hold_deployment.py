"""Baseline: never move — the floor for a mission that deals you ground at deployment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedHoldDeploymentPolicy(BaselinePolicy):
    """Issue ``STAY`` for every model, every step.

    A second floor beside ``random``, and the one that matters on any mission
    where objectives sit inside the deployment zones. The 45 real layouts do
    exactly that — a third of their objectives are in each player's own zone, so
    a player *starts* holding 1.98 of them — which raises the obvious worry that
    the mission can be won by standing still. This is the policy that answers it.

    On the nine held-out tables it scores **0.00 win rate at -70.2 vp_margin**,
    ending on 1.63 objectives with 99.8% of its force alive: doing nothing loses
    every episode, because the opponent takes the middle uncontested and then
    comes for the home points. Quote it whenever a scenario deals starting
    ground, so "the agent beat the opponent" cannot be confused with "the
    scenario handed it the win".

    It shoots at nothing, deliberately. The question this answers is whether
    *position* alone is enough, and adding fire would make a null result
    ambiguous between the two.
    """

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Return ``STAY`` for every model, alive or not."""
        return WargameEnvAction(actions=[STAY_ACTION] * len(models))


register_baseline("hold_deployment", ScriptedHoldDeploymentPolicy)
