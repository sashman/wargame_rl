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

    # Advance when a normal full move cannot reach the objective: run while far,
    # walk once close. Advancing forbids this unit shooting for the turn, so a
    # marching squad trades fire it mostly could not deliver at range for
    # arriving sooner.
    #
    # A toggle rather than a hardcoded rule because the trade is a real one and
    # this project measures rather than assumes: `advance=0` reproduces the
    # pre-Advance bar exactly, so the two are directly comparable on one config.
    #
    # ⚠ It is a per-SQUAD decision, matching how the env resolves it: the roll is
    # one D6 per unit and `advanced_this_turn` is marked for every model of any
    # group that advanced, so a squad splitting its choice loses the shooting
    # anyway and gains only part of the distance.
    # ⚠ **REJECTED AS A DEFAULT, and the measurement that rejected it is the 2x2.**
    # "Run while far, walk once close" costs its USER about 78 vp. Measured on
    # `25v25_maps_advance_refereed`, held-out nine, n=10, `squad_march_take`
    # both sides, vp_margin to the player:
    #
    #                        opponent walks   opponent advances
    #     player walks            -4.1              +72.7
    #     player advances        -81.8               -3.6
    #
    # The both-advance cell (-3.6) is indistinguishable from both-walk (-4.1),
    # which is why a first measurement read this as "Advance is worth +15.5 to
    # the bar". It is worth nothing to the bar: the two sides were handicapping
    # themselves by the same amount and the effects cancelled. **Never measure a
    # symmetric change with both sides changed at once.**
    #
    # The mechanism (`ActionHandler.best_advance_toward`) is kept, because
    # Advance is a core rule and a scripted bar that cannot use it is not a bar.
    # What is rejected is this HEURISTIC. A better rule has to price the
    # forfeited shooting, which this one never does.
    advance_when_out_of_reach: bool = False

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """March each squad toward its objective, settling onto the disc on arrival."""
        objectives = env.objectives
        actions = [STAY_ACTION] * len(models)
        if not objectives:
            return WargameEnvAction(actions=actions)

        # Per SQUAD, and that distinction is load-bearing. A squad marches as a
        # body on one shared vector -- that is what keeps its formation rigid
        # and therefore legal -- so its step is capped by its own slowest
        # member, since a member that cannot cover the shared step is left
        # behind and breaks the property this policy relies on.
        #
        # Taking the minimum over the whole ARMY instead is identical while
        # every model is equally fast, and silently wrong the moment they are
        # not: one slow squad would cap a fast one, so a scripted bar could not
        # use a speed a learned policy can. That flatters the agent against a
        # hobbled bar, which is the most expensive class of error here.
        speeds = env.player_action_handler.move_speeds
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
            max_step = float(
                min(speeds[i] for i in member_indices) if speeds.size else 0.0
            )

            objective = targets[squad_index]
            radius = objective_extent(objective)
            centroid = np.mean(
                [models[i].location for i in member_indices], axis=0, dtype=float
            )
            lead = np.asarray(objective.location, dtype=float) - centroid
            lead_distance = float(np.linalg.norm(lead))

            # One decision for the whole squad, taken before any member moves:
            # a normal full move cannot close the gap, so run.
            squad_advances = (
                self.advance_when_out_of_reach
                and lead_distance > radius
                and lead_distance > max_step
            )

            for i in member_indices:
                if lead_distance <= radius:
                    # The squad has arrived; each model settles onto the disc
                    # individually so the whole body ends up inside it.
                    actions[i] = step_toward_objective(models[i], objective, env, i)
                else:
                    # Every model follows the same squad vector, which keeps
                    # relative positions — and therefore coherency — intact.
                    step = min(max_step, lead_distance)
                    advance = (
                        env.player_action_handler.best_advance_toward(
                            float(lead[0]),
                            float(lead[1]),
                            advance_roll=float(models[i].advance_roll),
                            max_step_length=min(
                                lead_distance, max_step + float(models[i].advance_roll)
                            ),
                            model_idx=i,
                        )
                        if squad_advances
                        else None
                    )
                    actions[i] = (
                        advance
                        if advance is not None
                        else env.player_action_handler.best_action_toward(
                            float(lead[0]),
                            float(lead[1]),
                            max_step_length=step,
                            model_idx=i,
                        )
                    )

        return WargameEnvAction(actions=actions)


register_baseline("squad_march", ScriptedSquadMarchPolicy)
