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
# Steps the endpoint just clear of an engagement ring. Small enough not to move
# a model visibly, large enough that the strict `<` test cannot re-trip on the
# boundary in floating point.
_ENDPOINT_EPSILON = 1e-6

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


def back_off_to_unengaged(
    start: np.ndarray,
    resolved: np.ndarray,
    enemy_centres: np.ndarray,
    enemy_reach: np.ndarray,
    occupied_centres: np.ndarray | None = None,
    occupied_reach: np.ndarray | None = None,
) -> np.ndarray:
    """Pull an endpoint back along its own heading until it is unengaged.

    `09-movement-phase.md` requires a unit to be unengaged *after* moving, and
    `03-moving.md` is explicit that only the endpoint counts:

        Passing through an enemy unit's engagement range during a move does not
        make the moving unit engaged. Only where it *ends* matters.

    ⚠ **This is why the first attempt was reverted.** Inflating enemy blockers
    by the engagement range turns an end-state rule into a 2"-thick impassable
    wall: review measured **87% of opponent-held objectives with no legal spot
    at all**, and it stopped a model at the ring's near edge even when its move
    would have carried it clean through and out the far side. Passing through is
    legal; only ending inside is not.

    The legal set along the ray is **not an interval** -- a ray can leave one
    ring and enter another -- so this walks back interval by interval rather
    than bisecting, which is the same mistake the movement solver made once
    already. Each enemy ring contributes the open span of `t` where the point
    lies inside it; the answer is the largest `t` in `[0, 1]` outside every span.

    ⚠ **`occupied_*` is not optional in practice, and omitting it was a real bug.**
    Backing off walks the endpoint into ground `resolve_move` had already cleared
    as passable-but-not-endable, so a model rescued from an engagement ring could
    come to rest inside a friendly base. Measured before the fix: **0.18% of
    friendly pairs ended a movement phase overlapping, worst penetration 0.68"**,
    against 0.0000% with the rule off. Both constraints have to be satisfied by
    the same point, so both contribute forbidden spans to one walk.

    Returns `start` when no legal point exists short of it, which is the rules'
    own remedy: the move is not made.
    """
    centres = enemy_centres
    reach = enemy_reach
    if (
        occupied_centres is not None
        and occupied_reach is not None
        and occupied_centres.shape[0] > 0
    ):
        centres = np.vstack([centres, occupied_centres])
        reach = np.concatenate([reach, occupied_reach])
    if centres.shape[0] == 0:
        return resolved
    direction = resolved - start
    length_squared = float(direction @ direction)
    if length_squared <= 0.0:
        return resolved

    # Inside ring i for t in (lo_i, hi_i), from |start + t*d - c|^2 < reach^2.
    offsets = start[None, :] - centres
    b = offsets @ direction
    c = np.einsum("ij,ij->i", offsets, offsets) - reach**2
    discriminant = b**2 - length_squared * c
    hit = discriminant > 0.0
    if not np.any(hit):
        return resolved
    root = np.sqrt(discriminant[hit])
    lo = (-b[hit] - root) / length_squared
    hi = (-b[hit] + root) / length_squared

    t = 1.0
    # Each step exits one ring; there are finitely many, so this terminates.
    for _ in range(len(lo) + 1):
        inside = (lo < t) & (t < hi)
        if not np.any(inside):
            return start + t * direction if t > 0.0 else start
        t = float(np.min(lo[inside])) - _ENDPOINT_EPSILON
        if t <= 0.0:
            return start
    return start


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
