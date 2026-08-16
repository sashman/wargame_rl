"""Baseline: bank the VP cap, then take the weakest ground rather than raid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_deny import (
    ScriptedSquadMarchDenyPolicy,
    occupants,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedSquadMarchTakePolicy(ScriptedSquadMarchDenyPolicy):
    """`squad_march_deny`, but surplus squads take the *weakest*-held objective.

    **The strongest scripted policy measured on the real tables.** It exists
    because `squad_march_deny` and `squad_march_shoot` reach the same score by
    opposite routes, and each is missing the other's half:

    | | vp_margin | player VP | opp VP | held |
    |---|---|---|---|---|
    | `squad_march_shoot` | 111.8 | 272.2 | **160.4** | 4.00 |
    | `squad_march_deny` | 112.3 | **277.3** | 165.1 | 3.00 |
    | **this policy** | **115.0** | 275.9 | **160.9** | **4.01** |

    `squad_march_deny` sends its surplus at the objectives the opponent
    *controls*, and its `held` sits at exactly 3.00 — **the raids never flip
    anything**, so they deny nothing while the squads making them die. The bar,
    which merely marches to `k mod n`, reaches 4.00 and denies more precisely
    because the ground it happens to walk onto is undefended.

    So the ordering is inverted here. Weak ground flips and then denies for the
    rest of the game; defended ground absorbs a squad and yields nothing. That
    recovers the bar's denial (160.9 against 160.4) while keeping most of the
    denier's scoring efficiency (275.9 against 277.3).

    **Mechanically this is simpler than its parent, not an extra step on top.**
    `squad_march_deny` builds a two-tier priority — bank
    ``cap // vp_per_objective`` objectives, then re-sort the remainder to prefer
    what the opponent controls. This drops the re-sort, and the two tiers
    collapse into one flat ascending-opponent-count list: **one squad per
    objective, cheapest ground first, nearest squad to each.** There is no cap
    arithmetic here and `mission.params` is not read; banking still happens
    because the head of that list *is* the cheapest ground. The other difference
    is the leftover branch, which reinforces down the whole order rather than the
    banked subset (it fires only when squads outnumber objectives).

    Note what this does *not* say: it is not that denial was the wrong target.
    Denial is still where the whole margin lives, since own VP is capped at
    three objectives. It is that **holding** an objective is a far more reliable
    way to deny it than contesting one — a raid that does not cross the count
    threshold changes nothing at all.
    """

    def squad_objectives(
        self, models: list[WargameModel], env: WargameEnv, group_ids: list[int]
    ) -> list[WargameObjective]:
        """Assign every squad by ascending opponent count — cheapest ground first."""
        objectives = env.objectives
        opponent_locations = np.array(
            [m.location for m in env.opponent_models if m.is_alive], dtype=float
        )
        if opponent_locations.size == 0:
            opponent_locations = np.empty((0, 2), dtype=float)
        opponent_counts = [occupants(opponent_locations, o) for o in objectives]

        centroids = []
        for group_id in group_ids:
            members = [
                m.location for m in models if m.group_id == group_id and m.is_alive
            ]
            centroids.append(
                np.mean(members, axis=0, dtype=float)
                if members
                else np.zeros(2, dtype=float)
            )

        # One order for everything: the cap is banked by the cheapest objectives
        # and the surplus continues down the same list. Ties break on index, so
        # the assignment is deterministic.
        order = sorted(range(len(objectives)), key=lambda i: (opponent_counts[i], i))
        unassigned = list(range(len(group_ids)))
        targets: list[int | None] = [None] * len(group_ids)
        for objective_index in order:
            if not unassigned:
                break
            location = np.asarray(objectives[objective_index].location, dtype=float)
            nearest = min(
                unassigned,
                key=lambda s: float(np.linalg.norm(centroids[s] - location)),
            )
            targets[nearest] = objective_index
            unassigned.remove(nearest)
        # More squads than objectives: the leftovers reinforce the cheapest.
        for squad in unassigned:
            targets[squad] = order[squad % len(order)]

        return [objectives[index] for index in targets if index is not None]


register_baseline("squad_march_take", ScriptedSquadMarchTakePolicy)
