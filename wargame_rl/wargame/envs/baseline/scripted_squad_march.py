"""Baseline: each squad marches to one objective as a body, then holds it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.policy import (
    BaselinePolicy,
    objective_extent,
    step_toward_objective,
)
from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedSquadMarchPolicy(BaselinePolicy):
    """Send squad *k* to objective *k mod n_objectives*, moving as one body.

    The strongest scripted baseline and the reference bar for the learned
    policy. Two properties make it so, and both matter for what the agent is
    supposed to learn:

    - **Concentration.** Whole squads on whole objectives produce an uneven
      split (5 squads over 3 objectives gives 2/2/1, i.e. 10/10/5 models).
      Control is a strict count comparison, so concentrating beats spreading.
    - **Coherency.** Steering on the squad centroid rather than each model's
      own vector keeps squads together, which is legal under the tabletop
      rules that the per-model baselines violate.
    """

    def squad_objectives(
        self, models: list[WargameModel], env: WargameEnv, group_ids: list[int]
    ) -> list[WargameObjective]:
        """One objective per squad, in `group_ids` order.

        The seam subclasses change to play a different allocation while keeping
        this class's centroid-steered, coherency-preserving movement. The
        assignment here is fixed at squad *k* -> objective *k mod n* and never
        revised, which is what makes this baseline a stable reference bar.
        """
        objectives = env.objectives
        return [objectives[i % len(objectives)] for i in range(len(group_ids))]

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """March each squad toward its objective, settling onto the disc on arrival."""
        objectives = env.objectives
        actions = [STAY_ACTION] * len(models)
        if not objectives:
            return WargameEnvAction(actions=actions)

        max_step = float(env.config.max_move_speed)
        group_ids = sorted({model.group_id for model in models})
        targets = self.squad_objectives(models, env, group_ids)

        for squad_index, group_id in enumerate(group_ids):
            member_indices = [
                i
                for i, model in enumerate(models)
                if model.group_id == group_id and model.is_alive
            ]
            if not member_indices:
                continue

            objective = targets[squad_index]
            radius = objective_extent(objective)
            centroid = np.mean(
                [models[i].location for i in member_indices], axis=0, dtype=float
            )
            lead = np.asarray(objective.location, dtype=float) - centroid
            lead_distance = float(np.linalg.norm(lead))

            for i in member_indices:
                if lead_distance <= radius:
                    # The squad has arrived; each model settles onto the disc
                    # individually so the whole body ends up inside it.
                    actions[i] = step_toward_objective(models[i], objective, env)
                else:
                    # Every model follows the same squad vector, which keeps
                    # relative positions — and therefore coherency — intact.
                    actions[i] = env.player_action_handler.best_action_toward(
                        float(lead[0]),
                        float(lead[1]),
                        max_step_length=min(max_step, lead_distance),
                    )

        return WargameEnvAction(actions=actions)


register_baseline("squad_march", ScriptedSquadMarchPolicy)
