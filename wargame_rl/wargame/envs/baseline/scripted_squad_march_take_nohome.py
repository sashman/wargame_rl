"""Baseline: `squad_march_take` with the HOME objective excised — §34's abandon pricer.

The pricer that replaced the degenerate pin-vs-stock form: stock `take` already
garrisons home by construction (cheapest-ground-first order; measured home
held-rate 0.912), so the informative contrast is stock MINUS this variant —
what abandoning home actually costs the strongest script, lever arm 0.912.
The greedy allocator redistributes the freed squad automatically.

Home is the §34-validated identity: the objective whose AREA CENTROID lies
inside this side's deployment outline (exactly one exists on all 45 tables).
On a scenario with no deployment outline (generated-terrain configs) nothing
is excised and this is `squad_march_take` exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedSquadMarchTakeNoHomePolicy(ScriptedSquadMarchTakePolicy):
    """`squad_march_take` that never assigns a squad to its own home objective."""

    def squad_objectives(
        self,
        models: list[WargameModel],
        env: WargameEnv,
        group_ids: list[int],
    ) -> list[WargameObjective]:
        """The stock assignment, computed over the non-home objectives only."""
        outline = getattr(env, "deployment_outline", None)
        objectives = env.objectives
        if outline is None:
            return super().squad_objectives(models, env, group_ids)

        def centre(objective: WargameObjective) -> np.ndarray:
            if objective.area is not None:
                return np.mean(np.asarray(objective.area.vertices, dtype=float), axis=0)
            return np.asarray(objective.location, dtype=float)

        away = [
            objective
            for objective in objectives
            if not outline.contains(*centre(objective))
        ]
        if len(away) == len(objectives) or not away:
            return super().squad_objectives(models, env, group_ids)
        # Reuse the stock allocator on the reduced board: swap the objective
        # list the parent reads, restore it after. The parent reads
        # `env.objectives` directly, so this is the narrowest seam.
        original = env.objectives
        try:
            env.objectives = away
            return super().squad_objectives(models, env, group_ids)
        finally:
            env.objectives = original


register_baseline("squad_march_take_nohome", ScriptedSquadMarchTakeNoHomePolicy)
