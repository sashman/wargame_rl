"""Baseline: `squad_march_take` that follows the ENVIRONMENT's assignment.

The bar for any rung on which the environment writes the ground commitment
(#393): where `squad_march_take` derives its own squad-to-objective plan by
the greedy rule, this one reads the seat's commitment state and marches each
squad to the objective it was assigned. On a config whose writer IS the
greedy rule (`assignment: greedy`) the two plans coincide and this is
`squad_march_take` exactly; on the legibility rung (`assignment: rotated`)
this is the only script that can satisfy `all_units_on_commitment`, and
plain `take` must fail it -- which is the check that the rung separates
reading the marking from walking to the nearest objective.

Where no commitment is written (the phase facade, or the layer off) it
falls back to the parent's plan, so it is a drop-in for `take` anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.kernel.entities import (
        WargameModel,
        WargameObjective,
    )
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedSquadMarchCommittedPolicy(ScriptedSquadMarchTakePolicy):
    """`squad_march_take` whose plan is the seat's written ground commitment."""

    def squad_objectives(
        self, models: list[WargameModel], env: WargameEnv, group_ids: list[int]
    ) -> list[WargameObjective]:
        """The committed objective per squad where one is written, the
        parent's greedy plan for the rest. Duck-typed on `player_commitments` so this
        module depends on nothing under `per_model/`."""
        state = getattr(env, "player_commitments", None)
        objectives = env.objectives
        if state is None or not group_ids:
            return super().squad_objectives(models, env, group_ids)
        targets = [int(state.ground_of(int(g))) for g in group_ids]
        committed = [0 <= t < len(objectives) for t in targets]
        if all(committed):
            return [objectives[t] for t in targets]
        if not any(committed):
            return super().squad_objectives(models, env, group_ids)
        # A partial plan (#384: the head commits one unit at its open, and the
        # units after it in the turn are not yet written): the committed
        # squads keep theirs, the rest take the parent's greedy plan over
        # themselves alone, so a unit's own commitment is never overridden by
        # another unit's absence.
        unset = [g for g, done in zip(group_ids, committed) if not done]
        fallback = dict(zip(unset, super().squad_objectives(models, env, unset)))
        return [
            objectives[t] if done else fallback[g]
            for g, t, done in zip(group_ids, targets, committed)
        ]


register_baseline("squad_march_committed", ScriptedSquadMarchCommittedPolicy)
