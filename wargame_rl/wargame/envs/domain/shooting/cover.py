"""Cover (`13-terrain.md` § Cover): a unit-level, all-or-nothing property.

A unit has cover against an attack only when **every** model in it is not
fully visible to the attacker, so one model of a unit standing in the open
denies cover to the whole unit. "Not fully visible" spans partial and total
blockage, so only a `CLEAR` member denies it (#289).

One implementation for both facades: the phase facade traces every declared
(attacker, unit) pair of a phase at once, the per-model facade the pairs of one
attacking unit at its close. A pair traced alone samples the same segment the
batched trace does, so the answers cannot differ.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wargame_rl.wargame.envs.domain.battlefield.sight import CLEAR
from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel

# `(origins, targets, candidates, *, origin_models, target_models) -> visibility`:
# the env's corridor trace with both base radii applied.
VisibilityBetween = Callable[..., np.ndarray]


def unit_cover_mask(
    visibility_between: VisibilityBetween,
    shots: list[tuple[int, int]],
    attackers: list[WargameModel],
    targets: list[WargameModel],
    *,
    alive: np.ndarray | None = None,
) -> np.ndarray | None:
    """``(n_attackers, n_target_units)`` -- True where the *unit* has cover.

    Only the declared (attacker, unit) pairs are traced, expanded to the unit's
    members -- a handful out of the full product. `alive` is the membership the
    attacking unit faces; it defaults to the board's living models, and a
    caller judging against another moment's membership passes its own.
    Returns None when nothing was declared, so an empty phase costs nothing.
    """
    if not shots or not attackers or not targets:
        return None
    groups = np.array([m.group_id for m in targets], dtype=int)
    if alive is None:
        alive = np.array([m.is_alive for m in targets], dtype=bool)
    n_groups = int(groups.max()) + 1 if len(groups) else 0

    candidates = np.zeros((len(attackers), len(targets)), dtype=bool)
    declared = np.zeros((len(attackers), n_groups), dtype=bool)
    for attacker_idx, target_group in shots:
        if 0 <= target_group < n_groups and attacker_idx < len(attackers):
            declared[attacker_idx, target_group] = True
            candidates[attacker_idx, (groups == target_group) & alive] = True
    if not candidates.any():
        return None

    visibility = visibility_between(
        np.array([m.location for m in attackers], dtype=float),
        np.array([m.location for m in targets], dtype=float),
        candidates,
        origin_models=attackers,
        target_models=targets,
    )
    model_in_cover = visibility != CLEAR
    unit_in_cover = np.zeros((len(attackers), n_groups), dtype=bool)
    for group in range(n_groups):
        members = (groups == group) & alive
        if not members.any():
            continue
        # Every living model of the unit must be covered, and only for the
        # attackers that actually declared against it -- an undeclared pair
        # was never traced, so its cells are vacuously True under `all`.
        unit_in_cover[:, group] = (
            model_in_cover[:, members].all(axis=1) & declared[:, group]
        )
    return unit_in_cover
