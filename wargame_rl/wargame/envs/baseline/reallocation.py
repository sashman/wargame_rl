"""The surplus-reallocation rule, shared by the bar and the play-time decode.

One rule, stated once: **a squad that is genuinely surplus on the army's
biggest stack (the stack stays controlled without it) marches at the opponent's
weakest-held objective, else at the nearest empty one.**

It exists here — in the baseline package, importable by scripted policies —
because the play-time decode built on it measured **+14.54 ± 3.81 vp, positive
on 6 of 6 seeds and all four opponents** (`docs/melee-teaching-goal.md` §28),
and a rule that good creates a fairness obligation: an agent scored WITH it
against a bar WITHOUT it is not a same-game comparison. ⚠ The critic gate the
decode first shipped with is a measured TAX (−3.24 ± 1.29) — this is the bare
heuristic, deliberately.

Counting uses `objective_counts_from_norms_offset`, the scoring definition —
three implementations of "on an objective" once disagreed on 7.6% of slots
here, which is why this module computes nothing of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_counts_from_norms_offset,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


def choose_surplus_reallocation(
    models: list[WargameModel],
    env: WargameEnv,
    min_stack: int = 4,
) -> tuple[int, int] | None:
    """`(donor group_id, target objective index)`, or None on boards without one.

    Contest first — the opponent's weakest-held objective — falling back to the
    nearest empty one; nothing at all when no squad is surplus. The donor must
    leave its stack still controlled: the rule redistributes a SURPLUS, it
    never abandons a point, which is what separates it from the force-moves the
    2026-08-11 teleport audit priced at −29.4 income.
    """
    objectives = env.objectives
    if not objectives:
        return None
    alive = alive_mask_for(models)
    cache = compute_distances(models, objectives, alive_mask=alive)
    player_counts = objective_counts_from_norms_offset(
        cache.model_obj_norms_offset, cache.obj_radii
    )
    opponent_cache = compute_distances(
        env.opponent_models,
        objectives,
        alive_mask=alive_mask_for(env.opponent_models),
    )
    opponent_counts = objective_counts_from_norms_offset(
        opponent_cache.model_obj_norms_offset, opponent_cache.obj_radii
    )

    stacked = [
        index
        for index in range(len(objectives))
        if player_counts[index] >= min_stack
        and player_counts[index] > opponent_counts[index]
    ]
    if not stacked:
        return None
    source = max(stacked, key=lambda index: int(player_counts[index]))

    on_source = cache.model_obj_norms_offset[:, source] <= cache.obj_radii[source]
    by_group: dict[int, int] = {}
    for index, model in enumerate(models):
        if alive[index] and on_source[index]:
            by_group[int(model.group_id)] = by_group.get(int(model.group_id), 0) + 1
    if not by_group:
        return None
    donor = max(by_group, key=lambda group: by_group[group])
    # Genuinely surplus: the stack stays controlled after the donor leaves.
    if player_counts[source] - by_group[donor] <= opponent_counts[source]:
        return None

    theirs = [
        index
        for index in range(len(objectives))
        if opponent_counts[index] > player_counts[index]
    ]
    if theirs:
        return donor, min(theirs, key=lambda index: int(opponent_counts[index]))

    empty = [
        index
        for index in range(len(objectives))
        if player_counts[index] == 0 and opponent_counts[index] == 0
    ]
    if not empty:
        return None
    target = min(
        empty,
        key=lambda index: float(
            np.min(cache.model_obj_norms_offset[alive & on_source, index])
        ),
    )
    return donor, target
