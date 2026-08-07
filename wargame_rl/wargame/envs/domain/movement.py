"""Movement resolution with model bases that cannot overlap.

Models occupy space, so a move is a disc sweeping across the board rather than a point
jumping to a new coordinate. Two rules shape the result:

* a model may move **through** friendly bases but **not through** enemy ones;
* a model must **end** its move clear of every base, friendly and enemy alike.

Resolution is sequential in model order, and each model is tested against where
everyone else currently is. That makes it order-dependent -- a model earlier in the
list effectively has right of way -- which is the price of determinism. The
environment is seeded and its tests depend on reproducibility, so an order-independent
relaxation pass (which would also land models where no action asked them to go) is the
worse trade.

A packed formation does not deadlock under this: models simply stop short rather than
jittering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Leave a sliver of space at contact so that a model which stopped against another is
# not considered overlapping by the strict test on the next step.
_CONTACT_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class Blocker:
    """A base occupying space, as seen by a moving model."""

    position: np.ndarray
    radius: float
    blocks_path: bool
    """Enemy bases block the path; friendly ones only block the destination."""


def _first_contact_fraction(
    start: np.ndarray,
    displacement: np.ndarray,
    radius: float,
    blocker: Blocker,
) -> float | None:
    """Fraction of the move at which the sweeping disc first touches *blocker*.

    Solves |start + t*displacement - centre| = radius + blocker.radius for the smaller
    root in [0, 1]. Returns None when the sweep never touches it.
    """
    offset = start - blocker.position
    contact_distance = radius + blocker.radius

    a = float(displacement @ displacement)
    if a == 0.0:
        return None
    b = 2.0 * float(offset @ displacement)
    c = float(offset @ offset) - contact_distance * contact_distance

    if c <= 0.0:
        # Already touching at the start -- treat it as no new contact, or a model
        # nudged into an overlap could never move again.
        return None

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    root = (-b - np.sqrt(discriminant)) / (2.0 * a)
    if 0.0 <= root <= 1.0:
        return float(root)
    return None


def _overlaps(position: np.ndarray, radius: float, blockers: list[Blocker]) -> bool:
    """True if a base at *position* overlaps any blocker."""
    for blocker in blockers:
        gap = radius + blocker.radius
        delta = position - blocker.position
        if float(delta @ delta) < gap * gap - _CONTACT_EPSILON:
            return True
    return False


def resolve_move(
    start: np.ndarray,
    displacement: np.ndarray,
    radius: float,
    blockers: list[Blocker],
    board_width: float,
    board_height: float,
) -> np.ndarray:
    """Return where a model actually ends up after attempting *displacement*.

    The base is kept wholly on the board, stopped short of any enemy base in its path,
    and backed off until it rests clear of everything.
    """
    if not displacement.any():
        return start.copy()

    low = np.array([radius, radius], dtype=float)
    high = np.array([board_width - radius, board_height - radius], dtype=float)
    target = np.clip(start + displacement, low, high)
    travel = target - start
    if not travel.any():
        return start.copy()

    # Stop short of the first enemy base the path runs into.
    fraction = 1.0
    for blocker in blockers:
        if not blocker.blocks_path:
            continue
        contact = _first_contact_fraction(start, travel, radius, blocker)
        if contact is not None:
            fraction = min(fraction, contact)

    candidate = start + fraction * travel
    if not _overlaps(candidate, radius, blockers):
        clipped: np.ndarray = np.clip(candidate, low, high)
        return clipped

    # The destination is occupied -- typically by a friendly base the model was
    # allowed to pass through. Back off along the path to the last clear point.
    resting = _back_off(start, travel, radius, blockers, fraction)

    # Then slide along whatever is in the way, spending the movement that backing off
    # gave up. Without this, models converging on one point queue up radially behind
    # whoever arrives first: each stops dead the moment its destination is taken, so
    # a squad forms a line pointing at the objective instead of spreading around it.
    #
    # Only what is left of the move may be spent, or sliding would become a way to
    # travel further than the Move characteristic allows.
    remaining = float(np.linalg.norm(travel)) - float(np.linalg.norm(resting - start))
    if remaining <= 0.0:
        return np.clip(resting, low, high)
    slid = _slide(resting, travel, remaining, radius, blockers, low, high)
    return np.clip(slid, low, high)


def _back_off(
    start: np.ndarray,
    travel: np.ndarray,
    radius: float,
    blockers: list[Blocker],
    fraction: float,
) -> np.ndarray:
    """Return the furthest point along the path that is clear of every blocker."""
    lower, upper = 0.0, fraction
    for _ in range(24):  # ~1e-7 of the move length, far below any base radius
        middle = (lower + upper) / 2.0
        if _overlaps(start + middle * travel, radius, blockers):
            upper = middle
        else:
            lower = middle

    resolved = start + lower * travel
    return start.copy() if _overlaps(resolved, radius, blockers) else resolved


def _slide(
    position: np.ndarray,
    travel: np.ndarray,
    remaining: float,
    radius: float,
    blockers: list[Blocker],
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Spend *remaining* movement tangentially around the nearest obstruction.

    One pass, not a solver: it is enough to let a squad flow around a model already
    on an objective rather than stacking up behind it.
    """
    nearest = _nearest_blocker(position, blockers)
    if nearest is None:
        return position

    away = position - nearest.position
    norm = float(np.linalg.norm(away))
    if norm == 0.0:
        return position
    away = away / norm

    # Drop the component of the move that pushes into the obstacle; what is left runs
    # along its surface, capped at the movement still unspent.
    tangential = travel - float(travel @ away) * away
    length = float(np.linalg.norm(tangential))
    if length == 0.0:
        return position
    tangential = tangential / length * min(length, remaining)

    target = np.clip(position + tangential, low, high)
    step = target - position
    if not step.any():
        return position
    return _back_off(position, step, radius, blockers, 1.0)


def _nearest_blocker(position: np.ndarray, blockers: list[Blocker]) -> Blocker | None:
    """The blocker whose base edge is closest to *position*."""
    best: Blocker | None = None
    best_gap = float("inf")
    for blocker in blockers:
        gap = float(np.linalg.norm(position - blocker.position)) - blocker.radius
        if gap < best_gap:
            best_gap = gap
            best = blocker
    return best


def blockers_for(
    mover: object,
    friendly: list,
    enemy: list,
) -> list[Blocker]:
    """Build the blocker list a *mover* sees: friendly bases plus enemy bases.

    Dead models occupy no space. The mover is excluded from its own friendly list.
    """
    blockers: list[Blocker] = []
    for model in friendly:
        if model is mover or not model.is_alive:
            continue
        blockers.append(
            Blocker(
                position=model.location,
                radius=model.base_radius,
                blocks_path=False,
            )
        )
    for model in enemy:
        if not model.is_alive:
            continue
        blockers.append(
            Blocker(
                position=model.location,
                radius=model.base_radius,
                blocks_path=True,
            )
        )
    return blockers
