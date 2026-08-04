"""Baseline: every model walks to whichever objective is closest to it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.policy import ScriptedObjectiveAssignmentPolicy
from wargame_rl.wargame.envs.baseline.registry import register_baseline

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedGreedyNearestPolicy(ScriptedObjectiveAssignmentPolicy):
    """Send every model to its nearest objective.

    The weakest of the scripted baselines, and instructively so: it piles the
    army onto whichever objective the deployment happens to favour and leaves
    the others uncontested. Control is a strict count comparison, so surplus
    models on a point already won are worth nothing.
    """

    def assign_objective(
        self, model_index: int, model: WargameModel, env: WargameEnv
    ) -> int:
        """Return the index of the objective closest to this model."""
        locations = np.array([o.location for o in env.objectives], dtype=float)
        distances = np.linalg.norm(
            locations - np.asarray(model.location, dtype=float), axis=1
        )
        return int(np.argmin(distances))


register_baseline("greedy_nearest", ScriptedGreedyNearestPolicy)
