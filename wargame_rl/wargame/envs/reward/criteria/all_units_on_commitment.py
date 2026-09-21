from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class AllUnitsOnCommitmentCriteria(SuccessCriteria):
    """Succeeds when every living unit with a ground commitment has a member
    inside ITS committed objective, and at least one unit is committed.

    The legibility rung's criterion (#393): `all_objectives_occupied` is
    satisfied by any policy that covers the points, whichever unit takes
    which, so a policy that ignores the marked-target relation and walks to
    the nearest objective passes it. This one asks each unit to be on the
    objective it was ASSIGNED, so on a rung whose assignment is deliberately
    not the nearest (`commitments.assignment: rotated`) only a policy that
    reads the relation can satisfy it. With the layer off no unit is
    committed and it never succeeds, which is deliberate: a config naming it
    without a writer is a mistake, not a pass.
    """

    def is_successful(self, view: BattleView, ctx: StepContext) -> bool:
        committed = ctx.committed_objective
        if committed is None:
            return False
        cache = ctx.distance_cache
        alive = alive_mask_for(view.player_models)
        inside = cache.model_obj_norms_offset <= cache.obj_radii
        groups = np.array([int(m.group_id) for m in view.player_models])
        any_committed = False
        for group in np.unique(groups[alive]):
            members = np.flatnonzero(alive & (groups == group))
            target = int(committed[members[0]])
            if target < 0:
                continue
            any_committed = True
            if not bool(inside[members, target].any()):
                return False
        return any_committed
