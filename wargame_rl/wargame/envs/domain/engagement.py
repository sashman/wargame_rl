"""Who is engaged with whom — one definition, several consumers.

`docs/rules/03-moving.md` § Engagement: a model's engagement range is everything
within 2" of it, and while a friendly model is inside an enemy's engagement
range, those models *and their units* are engaged with each other.

The predicate was written out three times before this module existed — twice
verbatim inside `env_components/shooting_masks.py` and once as a radius in
`renders/v2/control.py` — and charge legality, fight eligibility, fall-back
eligibility and the target-side shooting gate all need the same answer. This
project has already paid twice for a duplicated predicate that drifted: the
observation's objective-membership test disagreed with scoring's on **7.6%** of
slots, and the opponent's action mask is still a hand-written copy of the
player's. One definition, many readers, is the `domain/coherency.py` pattern.

⚠ **Measured base to base, not centre to centre.** Two models with `r`-radius
bases are `2r` closer than their centres suggest, so the caller passes
`base_diameter` and every comparison subtracts it. The movement path in
`env_components/actions.py` builds its own rings per *enemy radius* rather than
from a single global diameter — a finer formulation for a scenario with mixed
base sizes, and deliberately not folded in here, because collapsing it to one
global would agree on every shipped config and diverge silently on the first
that does not.

⚠ **Only LIVING models engage, on BOTH axes, and both masks are REQUIRED.** A
destroyed model keeps its position forever (`take_damage` writes only
`current_wounds`), so a corpse would otherwise pin a model for the rest of the
episode. That defect was real, fired on 8.74% of model-steps against the rule's
0.80%, and cost the agent 7.0 vp.

⚠ **It then happened a SECOND time, on the other axis, because only `other_alive`
was a parameter.** The target-side shooting gate asks *is this enemy engaged*,
whose subject is the ENEMY models -- and nothing masked them, so an enemy
casualty lying beside one of my models made its whole unit unshootable,
including a living squadmate thirty inches away. Every caller that got it right
was masking the subject axis by hand afterwards, which is a convention rather
than a rule. `subject_alive` is therefore **positional and required**: a caller
can pass the wrong array, but it can no longer forget there is one.
"""

from __future__ import annotations

import numpy as np


def engagement_matrix(
    positions: np.ndarray,
    other_positions: np.ndarray,
    other_alive: np.ndarray,
    subject_alive: np.ndarray,
    *,
    engagement_range: float,
    base_diameter: float = 0.0,
) -> np.ndarray:
    """``(n, n_other)`` bool — is each model within engagement range of each other.

    Rows AND columns for destroyed models are False: a casualty engages
    nobody, and nobody engages a casualty. Both masks are required -- see the
    module docstring for the two defects that bought that decision.
    """
    n, n_other = len(positions), len(other_positions)
    if n == 0 or n_other == 0 or engagement_range <= 0:
        return np.zeros((n, n_other), dtype=bool)
    deltas = positions[:, np.newaxis, :] - other_positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    within = (distances - base_diameter) <= engagement_range
    alive = np.asarray(other_alive, dtype=bool)[np.newaxis, :]
    mine = np.asarray(subject_alive, dtype=bool)[:, np.newaxis]
    return np.asarray(within & alive & mine, dtype=bool)


def engaged_with_any(
    positions: np.ndarray,
    other_positions: np.ndarray,
    other_alive: np.ndarray,
    subject_alive: np.ndarray,
    *,
    engagement_range: float,
    base_diameter: float = 0.0,
) -> np.ndarray:
    """``(n,)`` bool — is each LIVING model engaged with a living enemy."""
    matrix = engagement_matrix(
        positions,
        other_positions,
        other_alive,
        subject_alive,
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )
    return np.asarray(matrix.any(axis=1), dtype=bool)


def engaged_units(
    engaged_models: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """``(n_groups,)`` bool — a unit is engaged when ANY of its models is.

    Unit-level, because the rule is: *those models — and their units — are
    engaged with each other*. Reducing a per-model answer over the unit is what
    the shooting mask's own docstring warns against doing to visibility and
    range, but here it is the rule rather than an approximation of it.
    """
    out = np.zeros(n_groups, dtype=bool)
    if len(engaged_models) == 0:
        return out
    for group in range(n_groups):
        member = groups == group
        if member.any():
            out[group] = bool(np.asarray(engaged_models)[member].any())
    return out
