"""Movement with physical bases: a model cannot pass through an enemy, and no
two models may end a move overlapping.

The rules the geometry encodes, and why each is asymmetric:

- **Enemy bases block the path.** Walking through an enemy line is the one thing
  models physically cannot do, so contact stops the move where it happens.
- **Friendly bases may be passed through but not ended on.** A squad moving as a
  body would otherwise gridlock on its own front rank, which is a modelling
  artefact rather than a rule.
- **Resolution is sequential, in model index order.** That is a documented
  right-of-way bias — model 0 always gets the ground it wants — and it is the
  price of determinism. The seeded environment and every golden gate depend on
  the same actions producing the same board.

Everything here is pure geometry over arrays. It knows nothing about phases,
actions or whose turn it is.
"""

from __future__ import annotations

import numpy as np

# Fraction of a base radius left between two touching models. Stopping *exactly*
# at contact leaves the endpoint on the overlap boundary, where a float rounding
# either way decides whether the position is legal -- so a model would sometimes
# end up marginally inside another and sometimes not, from the same move.
_CONTACT_MARGIN = 1e-6

# How many times to back off before giving up and staying put. Backing off out
# of one base can put a model inside another; in practice one pass settles it,
# and the cap stops a pathological cluster from looping.
_MAX_BACKOFF_PASSES = 4


def _first_contact(
    start: np.ndarray,
    displacement: np.ndarray,
    radius: float,
    centres: np.ndarray,
    radii: np.ndarray,
) -> float:
    """Earliest ``t`` in [0, 1] at which the swept base touches an obstacle.

    Returns 1.0 when the whole move is clear. Solves the standard
    moving-circle/static-circle quadratic for every obstacle at once: contact is
    ``|start + t*d - c| = radius + r_c``.
    """
    if len(centres) == 0:
        return 1.0
    offsets = start - centres  # (N, 2)
    a = float(displacement @ displacement)
    if a == 0.0:
        return 1.0
    reach = radius + radii  # (N,)
    b = 2.0 * (offsets @ displacement)  # (N,)
    c = (offsets**2).sum(axis=1) - reach**2  # (N,)

    discriminant = b**2 - 4.0 * a * c
    # `reach > 0` excludes the degenerate case of two dimensionless models: with
    # both radii zero the discriminant is exactly 0 for a path passing through
    # the other's position, so a point would "collide" with a point.
    hits = (discriminant >= 0.0) & (reach > 0.0)
    if not hits.any():
        return 1.0
    root = np.sqrt(discriminant[hits])
    # The earlier root is the entry point; the later one is the exit.
    entry = (-b[hits] - root) / (2.0 * a)
    # Already-overlapping obstacles have a negative entry time and must not drag
    # the move to zero: a model that starts touching one is not blocked by it.
    entry = entry[(entry >= 0.0) & (entry <= 1.0)]
    if entry.size == 0:
        return 1.0
    return float(entry.min())


def _overlaps(
    point: np.ndarray, radius: float, centres: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """``(N,)`` — True where a base at *point* would overlap that obstacle."""
    if len(centres) == 0:
        return np.zeros(0, dtype=bool)
    reach = radius + radii
    result: np.ndarray = ((centres - point) ** 2).sum(axis=1) < reach**2
    return result


def _advance(
    start: np.ndarray,
    displacement: np.ndarray,
    radius: float,
    blocker_centres: np.ndarray,
    blocker_radii: np.ndarray,
    all_centres: np.ndarray,
    all_radii: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Slide along *displacement* as far as legal. Returns the point and fraction."""
    travel = _first_contact(start, displacement, radius, blocker_centres, blocker_radii)
    # Only back off the margin where something was actually hit. Applying it to
    # a clear move would shorten *every* move by a sliver, which compounds over
    # an episode and is invisible in any single step.
    if travel < 1.0:
        travel = max(0.0, travel - _CONTACT_MARGIN)

    for _ in range(_MAX_BACKOFF_PASSES):
        candidate = start + travel * displacement
        clashing = _overlaps(candidate, radius, all_centres, all_radii)
        if not clashing.any():
            return candidate, travel
        # Back off to just before entering the earliest offending base. Entry
        # times are recomputed against only the offenders, so a model is never
        # pushed further back than the one thing actually in its way.
        entry = _first_contact(
            start,
            displacement,
            radius,
            all_centres[clashing],
            all_radii[clashing],
        )
        new_travel = max(0.0, min(travel, entry) - _CONTACT_MARGIN)
        if new_travel >= travel:
            break
        travel = new_travel

    # Standing still is always legal: the model was not overlapping anything
    # before it moved, because the same rule applied last turn.
    return start.copy(), 0.0


def _stack(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Concatenate two obstacle arrays, either of which may be empty."""
    if not len(first):
        return second
    if not len(second):
        return first
    return np.concatenate([first, second])


def resolve_move(
    start: np.ndarray,
    displacement: np.ndarray,
    radius: float,
    blocker_centres: np.ndarray,
    blocker_radii: np.ndarray,
    passable_centres: np.ndarray,
    passable_radii: np.ndarray,
) -> np.ndarray:
    """Where a model actually ends up, given what is in the way.

    Args:
        start: ``(2,)`` current position.
        displacement: ``(2,)`` the move it is attempting.
        radius: the moving model's base radius.
        blocker_centres / blocker_radii: bases that stop the move on contact.
        passable_centres / passable_radii: bases the move may cross but not end
            inside.

    A model with no base (radius 0 and no obstacles with one) moves exactly as
    it asked, which is what keeps every pre-base result reproducible.

    **A tangential slide was tried here and measured worse. Do not re-add it
    without measuring.** The reasoning for one is sound — blocked models
    otherwise queue radially behind whoever reached the objective first — and the
    prototype's notes predicted it would recover most of the loss. On this
    scenario it did the opposite, at n=30 on identical layouts:

        policy               back-off only      + tangential slide
        squad_march_shoot    0.70 / +20.6       0.57 /  +1.0
        split_evenly         0.40 / -19.2       0.17 / -73.8
        alive (bar)          0.277              0.191

    The mechanism, once measured: a *fully* blocked model has its whole move
    left to spend, so the slide becomes a full-length sideways swing away from
    the objective. Models drift laterally, spend longer in the open, and are
    shot. Backing off and stopping is worse-looking and better.

    The plan this came from says the real fix is on the policy side — give each
    model a distinct target slot around the objective instead of aiming every
    model at the centre — and this measurement is consistent with that: collision
    response cannot substitute for allocation.
    """
    if radius <= 0.0 and not len(blocker_radii) and not len(passable_radii):
        unobstructed: np.ndarray = start + displacement
        return unobstructed

    all_centres = _stack(blocker_centres, passable_centres)
    all_radii = _stack(blocker_radii, passable_radii)

    point, _ = _advance(
        start,
        displacement,
        radius,
        blocker_centres,
        blocker_radii,
        all_centres,
        all_radii,
    )
    return point
