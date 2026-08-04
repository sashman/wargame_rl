"""Baseline: spread models evenly across objectives, ignoring squad structure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.baseline.policy import ScriptedObjectiveAssignmentPolicy
from wargame_rl.wargame.envs.baseline.registry import register_baseline

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedSplitEvenlyPolicy(ScriptedObjectiveAssignmentPolicy):
    """Assign model *i* to objective *i mod n_objectives*.

    Beats `greedy_nearest` because it contests every objective, but it shreds
    every squad across the whole board — it is the one baseline that badly
    violates coherency, and it is also weaker than marching by squad. Kept as
    the middle rung of the scale precisely because that contrast isolates
    force concentration from mere spreading.
    """

    def assign_objective(
        self, model_index: int, model: WargameModel, env: WargameEnv
    ) -> int:
        """Return the objective index for a round-robin assignment."""
        return model_index % len(env.objectives)


register_baseline("split_evenly", ScriptedSplitEvenlyPolicy)
