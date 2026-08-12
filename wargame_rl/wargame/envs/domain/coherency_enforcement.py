"""Making a unit's move legal, once every model in the force has moved.

`docs/rules/03-moving.md` § Making a move is checked *after* the move, and its
consequence is a revert: "If any check fails, the move cannot be made: return
every model to where it started." That -- not the End-of-Turn attrition -- is
the rule's **primary** enforcement. On the table nobody loses models to
coherency in a normal game, because the illegal move is never made. So if
attrition ever fires often, this is what is wrong.

Two modes, because the spec's own rule and the shape this environment wants are
not obviously the same thing, and this project has already been burned reasoning
about movement geometry instead of measuring it (`domain/movement.py`: the
"obviously better" tangential slide measured ~20 vp *worse* than back-off).

- ``revert_unit`` is the spec: one model out of place cancels its whole unit's
  move. Faithful, and a cliff -- the unit's 5-model joint action is legal or it
  is nothing.
- ``revert_model`` returns only the models *in breach* -- those failing either
  condition, or sitting outside their unit's largest chain component. A
  divergence, and a gentler gradient: a straggler that breaks the chain is
  pulled back alone while its squadmates keep the ground they took.

  Note what that does **not** mean. The spread condition is collective: once one
  model is more than the cap from the rest, no model in the unit is within the
  cap of every other, so every one of them is in breach and the two modes
  coincide. They separate only while a break is local -- which is exactly why
  they tie on the bar and differ by 51 vp on ``split_evenly``, whose squads are
  shattered across the whole board every turn.

**A revert can leave two bases overlapping**, and that is inherent rather than a
bug here. Models resolve sequentially against live positions, so a model may
legally have moved onto ground a lower-indexed model vacated; sending the second
one back puts them on the same spot. The tabletop cannot hit this because moves
there are genuinely sequential and each is checked before the next. Rather than
invent a repair, the count is returned so a caller can measure how often it
happens before deciding whether it needs one.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel


class CoherencyEnforcement(str, Enum):
    """What happens to a unit that ends its move out of coherency."""

    off = "off"
    revert_unit = "revert_unit"
    revert_model = "revert_model"


def enforce_after_move(
    models: list[WargameModel],
    nearest_distance: float,
    furthest_distance: float,
    mode: CoherencyEnforcement,
) -> int:
    """Return out-of-coherency models to where they started. Mutates locations.

    Args:
        models: One force's models, moved but not yet committed to.
        nearest_distance: The chain distance, in board units.
        furthest_distance: The spread distance, in board units.
        mode: Which revert to apply. ``off`` returns 0 and touches nothing.

    Returns:
        How many models were sent back, which is the natural cost metric for
        this rule -- a policy paying it constantly is being told its moves do
        not happen, and that is a training failure worth catching early.
    """
    if mode is CoherencyEnforcement.off or not models:
        return 0

    report = evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=np.array([m.is_alive for m in models], dtype=bool),
        base_radii=np.array([m.base_radius for m in models], dtype=float),
        nearest_distance=nearest_distance,
        furthest_distance=furthest_distance,
    )

    reverted = 0
    for unit in report.units:
        if unit.coherent:
            continue
        if mode is CoherencyEnforcement.revert_unit:
            targets = unit.member_indices
        else:
            targets = unit.member_indices[~unit.member_coherency]
        for index in targets:
            reverted += _return_to_start(models[index])
    return reverted


def _return_to_start(model: WargameModel) -> int:
    """Put one model back where it began this move, if it moved at all.

    ``previous_location`` is written by the action handler on every model it
    displaces, and is None for a model that did not move this phase -- which is
    not a failure to revert but nothing to revert.
    """
    if model.previous_location is None:
        return 0
    model.location = model.previous_location.copy()
    return 1
