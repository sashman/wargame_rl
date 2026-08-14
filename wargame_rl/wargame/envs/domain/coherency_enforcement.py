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
  they tie on the bar and differ by 50 vp on ``split_evenly``, whose squads are
  shattered across the whole board every turn.

**Selection runs to a fixed point, because one pass does not make a unit
coherent.** Reverting the models in breach moves the goalposts: pull a straggler
back to where it started and it may now be too far from the squadmates who kept
the ground they took, so the unit is still broken and a single pass has enforced
nothing. Measured, before this was iterated: ``revert_model`` had the same
coherent-to-incoherent transition rate as *no enforcement at all* (0.024 against
0.025), where ``revert_unit`` had 0.000 in 1743 transitions.

So each pass re-evaluates against the tentatively reverted positions, and a unit
whose breaching models are already all reverted while it remains incoherent
escalates to reverting the whole unit. That terminates: every pass either adds a
model or escalates a unit, and both are bounded by the force size.

**It deliberately stops at the starting configuration rather than forcing
coherency.** A unit split by casualties is already incoherent when its move
begins, and no revert can repair that -- reverting is not a move. Enforcing here
would mean deleting models, which is `apply_attrition`'s job in the End of Turn
step. The guarantee is therefore *this move did not break the unit*, not *the
unit is coherent*.

**A naive revert leaves two bases overlapping, which is another illegal move,
so the revert cascades.** Models resolve sequentially against live positions, so
a model may legally have moved onto ground a lower-indexed model vacated;
sending the first one back would put two bases on the same spot. `03-moving.md`
checks that in the *same breath* as coherency -- "no model is left on top of
another model" sits under the same "if any check fails, the move cannot be
made" -- so the second model's move has failed too, and it goes back as well.

That converges rather than looping: each pass reverts at least one more model or
stops, and the worst case is the whole force at its start, which is legal by
construction because it is the configuration the previous step ended in. Enemies
cannot be drawn in, since they do not move during this force's move.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import (
    UnitCoherency,
    base_to_base_distances,
    evaluate_coherency,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel


class CoherencyEnforcement(str, Enum):
    """What happens to a unit that ends its move out of coherency."""

    off = "off"
    revert_unit = "revert_unit"
    revert_model = "revert_model"
    clamp = "clamp"


def enforce_after_move(
    models: list[WargameModel],
    nearest_distance: float,
    furthest_distance: float,
    mode: CoherencyEnforcement,
    probability: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Return out-of-coherency models to where they started. Mutates locations.

    Args:
        models: One force's models, moved but not yet committed to.
        nearest_distance: The chain distance, in board units.
        furthest_distance: The spread distance, in board units.
        mode: Which revert to apply. ``off`` returns 0 and touches nothing.
        probability: Fraction of illegal unit moves actually reverted, drawn per
            unit. 1.0 (the default) is full enforcement and takes the same code
            path as before this parameter existed -- no draw is made, so a
            seeded run is byte-identical. Below 1.0 the rule becomes a dial, so
            the price of compliance can be measured; see `CoherencyConfig`.
        rng: Episode RNG, required below 1.0 so a seeded run reproduces.

    Returns:
        How many models were sent back, which is the natural cost metric for
        this rule -- a policy paying it constantly is being told its moves do
        not happen, and that is a training failure worth catching early.
    """
    if mode is CoherencyEnforcement.off or not models:
        return 0
    if probability <= 0.0:
        return 0
    if probability < 1.0 and rng is None:
        raise ValueError(
            "enforce_after_move needs an rng when probability < 1.0, or a "
            "seeded run would not reproduce"
        )

    if mode is CoherencyEnforcement.clamp:
        return _clamp_after_move(
            models,
            nearest_distance,
            furthest_distance,
            probability=probability,
            rng=rng,
        )

    reverting: set[int] = set()
    units: tuple[UnitCoherency, ...] = ()
    # Selection and the overlap cascade each only ever *add*, and the cascade's
    # additions can break their own units, so the two alternate until neither
    # grows the set. Bounded by the force size for the same reason.
    for iteration in range(len(models) + 1):
        first_units = _select_reverting(
            models,
            reverting,
            nearest_distance,
            furthest_distance,
            mode,
            probability=probability,
            rng=rng,
        )
        if iteration == 0:
            units = first_units
        before = len(reverting)
        _cascade_displaced(models, reverting, report_units=units, mode=mode)
        if len(reverting) == before:
            break

    if not reverting:
        return 0

    reverted = 0
    for index in sorted(reverting):
        reverted += _return_to_start(models[index])
    return reverted


def _clamp_after_move(
    models: list[WargameModel],
    nearest_distance: float,
    furthest_distance: float,
    probability: float,
    rng: np.random.Generator | None,
) -> int:
    """Pull detached models back along their own line of advance. Mutates.

    Measured motivation. On the real-tables scenario a trained policy that had
    never seen the constraint pays it constantly: `revert_model` sends back
    **6.34 of 25 models every step**, cutting mean displacement 0.460 -> 0.370
    and `vp_margin` +104.2 -> +76.7 on identical layouts. Training under it for
    a hundred epochs does not recover that; the policy adapts by *moving less*.
    A rule that cancels a quarter of all movement is being learned as "movement
    does not work", which is a credit-assignment accident rather than the rule.

    So this mode keeps the model's decision and shortens it: a detached model
    slides back along the line toward the body it left until it is exactly
    within the chain distance. It ends somewhere it chose to go, just not as
    far -- which is also what a player does at the table. Nobody makes an
    illegal move and takes it back; they move as far as the rule allows.

    **It is a divergence, and a smaller one than it looks.** `03-moving.md` says
    an illegal move "cannot be made: return every model to where it started",
    and that is `revert_unit`. But the spec describes a player who chooses among
    legal moves, and reverting is only how a *referee* corrects one already
    made. Clamping is the continuous-action analogue of masking the illegal
    choice out, which is the closer analogue of choosing legally.

    **It never leaves an illegal position.** A clamp is accepted only if it
    makes the unit coherent and overlaps no live base; otherwise that unit falls
    back to the full revert, so the guarantee is exactly `revert_unit`'s. That
    fallback is also what handles a break the move did not cause -- a unit split
    by casualties cannot be repaired by shortening anyone's move.

    Returns:
        How many models enforcement moved off the position the policy chose,
        so the number stays comparable to the other modes'.
    """
    positions = np.array([m.location for m in models], dtype=float)
    base_radii = np.array([m.base_radius for m in models], dtype=float)
    alive_mask = np.array([m.is_alive for m in models], dtype=bool)
    report = evaluate_coherency(
        positions=positions,
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=alive_mask,
        base_radii=base_radii,
        nearest_distance=nearest_distance,
        furthest_distance=furthest_distance,
    )

    moved = 0
    falling_back: list[UnitCoherency] = []
    for unit in report.units:
        if unit.coherent:
            continue
        # Drawn once per unit, exactly as the revert modes draw, so the two are
        # comparable at the same probability.
        if probability < 1.0 and rng is not None:
            if float(rng.random()) >= probability:
                continue
        clamped = _pull_into_coherency(
            models,
            unit,
            positions,
            base_radii,
            alive_mask,
            nearest_distance,
            furthest_distance,
        )
        if clamped is None:
            falling_back.append(unit)
            continue
        for index, point in clamped.items():
            positions[index] = point
            models[index].location = np.array(point, dtype=models[index].location.dtype)
            moved += 1

    for unit in falling_back:
        for index in unit.member_indices:
            start = models[int(index)].previous_location
            if start is not None:
                positions[int(index)] = start
            moved += _return_to_start(models[int(index)])
    return moved


def _pull_into_coherency(
    models: list[WargameModel],
    unit: UnitCoherency,
    positions: np.ndarray,
    base_radii: np.ndarray,
    alive_mask: np.ndarray,
    nearest_distance: float,
    furthest_distance: float,
) -> dict[int, np.ndarray] | None:
    """New positions for one unit's detached models, or None if that fails.

    The unit's largest chain component keeps its move and is the body everyone
    else is pulled back toward. **A detached model is only ever moved backwards
    along its own move segment**, never off it and never past its start. That
    restriction is what makes this a shortened move rather than a free
    repositioning: anything else would let enforcement teleport a model further
    than its speed allows, and would repair breaks the move did not cause --
    which is `apply_attrition`'s job, not this one.

    Returns None when no legal point on the segment exists -- the body itself
    broken by the spread condition, a unit split by casualties rather than by
    its move, or a clamped model landing on an occupied base. The caller
    reverts those, so the guarantee is exactly `revert_unit`'s.
    """
    members = unit.member_indices
    body = members[unit.component == np.bincount(unit.component).argmax()]
    detached = [int(i) for i in members if int(i) not in {int(b) for b in body}]
    if not detached:
        return None

    proposed = {index: positions[index].copy() for index in detached}
    for index in detached:
        gaps = np.linalg.norm(positions[body] - positions[index], axis=1)
        anchor = int(body[int(np.argmin(gaps))])
        allowed = nearest_distance + float(base_radii[index] + base_radii[anchor])
        if float(np.linalg.norm(positions[index] - positions[anchor])) <= allowed:
            continue
        previous = models[index].previous_location
        if previous is None:
            return None
        start = np.array(previous, dtype=float)
        fraction = _furthest_legal_fraction(
            start, positions[index], positions[anchor], allowed
        )
        if fraction is None:
            return None
        proposed[index] = start + (positions[index] - start) * fraction

    candidate = positions.copy()
    for index, point in proposed.items():
        candidate[index] = point

    unit_mask = np.zeros(len(positions), dtype=bool)
    unit_mask[members] = True
    check = evaluate_coherency(
        positions=candidate,
        group_ids=np.where(unit_mask, unit.group_id, unit.group_id + 1),
        alive_mask=alive_mask & unit_mask,
        base_radii=base_radii,
        nearest_distance=nearest_distance,
        furthest_distance=furthest_distance,
    )
    if not all(reported.coherent for reported in check.units):
        return None
    if _overlaps_any_base(proposed, candidate, base_radii, alive_mask):
        return None
    return {index: point for index, point in proposed.items()}


def _furthest_legal_fraction(
    start: np.ndarray,
    destination: np.ndarray,
    anchor: np.ndarray,
    allowed: float,
) -> float | None:
    """How far along its move a model may go and still reach `anchor`.

    Returns the largest ``t`` in [0, 1] with
    ``|start + t*(destination - start) - anchor| <= allowed``, or None if no
    such ``t`` exists -- which means even standing still leaves the model too
    far from the body, so the move is not what broke the unit.

    Solved in closed form rather than searched: the constraint is a disc, the
    move is a segment, so ``t`` is a root of a quadratic. Exact and constant
    time, on a path that runs every step of every episode.
    """
    direction = destination - start
    offset = start - anchor
    a = float(direction @ direction)
    b = 2.0 * float(offset @ direction)
    c = float(offset @ offset) - allowed * allowed
    if a == 0.0:
        # The model did not move, so its distance to the body is fixed.
        return 0.0 if c <= 0.0 else None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = np.sqrt(discriminant)
    upper = (-b + root) / (2.0 * a)
    lower = (-b - root) / (2.0 * a)
    # The segment enters the disc at `lower` and leaves at `upper`; we want the
    # furthest point of [lower, upper] that is also on [0, 1].
    highest = min(1.0, float(upper))
    if highest < max(0.0, float(lower)):
        return None
    return highest


def _overlaps_any_base(
    proposed: dict[int, np.ndarray],
    candidate: np.ndarray,
    base_radii: np.ndarray,
    alive_mask: np.ndarray,
) -> bool:
    """True if any clamped model would sit on top of a live base.

    `03-moving.md` checks "no model is left on top of another model" in the same
    breath as coherency, so a clamp that lands on an occupied base has failed
    the same test the coherency breach did.
    """
    for index in proposed:
        separations = base_radii + float(base_radii[index])
        distances = np.linalg.norm(candidate - candidate[index], axis=1)
        others = alive_mask.copy()
        others[index] = False
        if np.any(others & (separations > 0.0) & (distances < separations)):
            return True
    return False


def _select_reverting(
    models: list[WargameModel],
    reverting: set[int],
    nearest_distance: float,
    furthest_distance: float,
    mode: CoherencyEnforcement,
    probability: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[UnitCoherency, ...]:
    """Grow *reverting* until no unit is left broken by its own move.

    Re-evaluates coherency against the *tentatively* reverted positions after
    every pass, because reverting a model changes whether its unit is coherent
    -- see the module docstring. Nothing is written to the models here; the
    caller applies the reverts once the set has settled. Models already in
    *reverting* on entry count as back at their start, so this composes with the
    overlap cascade.

    Returns:
        The units as first seen, which `_cascade_displaced` needs to keep a unit
        revert all-or-nothing.
    """
    positions = np.array([m.location for m in models], dtype=float)
    group_ids = np.array([m.group_id for m in models], dtype=np.intp)
    alive_mask = np.array([m.is_alive for m in models], dtype=bool)
    base_radii = np.array([m.base_radius for m in models], dtype=float)
    for index in reverting:
        start = models[index].previous_location
        if start is not None:
            positions[index] = start

    first_units: tuple[UnitCoherency, ...] = ()
    # Each pass adds a model or escalates a unit, so the force size bounds both.
    for pass_index in range(len(models) + 1):
        report = evaluate_coherency(
            positions=positions,
            group_ids=group_ids,
            alive_mask=alive_mask,
            base_radii=base_radii,
            nearest_distance=nearest_distance,
            furthest_distance=furthest_distance,
        )
        if pass_index == 0:
            first_units = report.units
        added = False
        for unit in report.units:
            if unit.coherent:
                continue
            # Drawn once per unit per pass. A unit spared here keeps its illegal
            # move for this step; the next step judges it afresh.
            if probability < 1.0 and rng is not None:
                if float(rng.random()) >= probability:
                    continue
            targets = _targets_for(unit, reverting, mode)
            for index in targets:
                if index in reverting:
                    continue
                reverting.add(index)
                start = models[index].previous_location
                if start is not None:
                    positions[index] = start
                added = True
        # Nothing left to send back: what remains is a break the move did not
        # cause, which only attrition can close.
        if not added:
            break
    return first_units


def _targets_for(
    unit: UnitCoherency,
    reverting: set[int],
    mode: CoherencyEnforcement,
) -> list[int]:
    """The models of one incoherent unit to send back on this pass.

    Under `revert_model` this is the models in breach, escalating to the whole
    unit once those are all back and the unit is still broken -- a local revert
    that has not worked has to widen or it enforces nothing.
    """
    if mode is CoherencyEnforcement.revert_unit:
        return [int(index) for index in unit.member_indices]
    breaching = [int(index) for index in unit.member_indices[~unit.member_coherency]]
    if any(index not in reverting for index in breaching):
        return breaching
    return [int(index) for index in unit.member_indices]


def _cascade_displaced(
    models: list[WargameModel],
    reverting: set[int],
    report_units: tuple[UnitCoherency, ...],
    mode: CoherencyEnforcement,
) -> None:
    """Grow *reverting* until no reverted model lands on ground still occupied.

    A model may legally have moved onto ground a lower-indexed model vacated --
    resolution runs in index order against live positions. Sending the first one
    back would then put two bases on the same spot, and `03-moving.md` checks
    that in the *same* breath as coherency: "no model is left on top of another
    model", under the same "if any check fails, the move cannot be made". So the
    second model's move has failed too, and it reverts as well.

    This terminates. Each pass adds at least one model or stops, and the worst
    case is the whole force back at its start -- which is legal by construction,
    because it is the configuration the previous step ended in. Enemies cannot
    be involved: they do not move during this force's move, so a start position
    that was clear of them still is.
    """
    unit_of: dict[int, np.ndarray] = {}
    if mode is CoherencyEnforcement.revert_unit:
        for unit in report_units:
            for index in unit.member_indices:
                unit_of[int(index)] = unit.member_indices

    for _ in range(len(models)):
        added = False
        for index in list(reverting):
            start = models[index].previous_location
            if start is None:
                continue
            radius = float(models[index].base_radius)
            for other_index, other in enumerate(models):
                if other_index in reverting or not other.is_alive:
                    continue
                separation = radius + float(other.base_radius)
                if separation <= 0.0:
                    continue
                if float(np.hypot(*(start - other.location))) >= separation:
                    continue
                # Reverting a unit is all-or-nothing, so a displaced model
                # brings its whole unit with it rather than splitting it.
                joining = unit_of.get(other_index)
                if joining is None:
                    reverting.add(other_index)
                else:
                    reverting.update(int(i) for i in joining)
                added = True
        if not added:
            return


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


def apply_attrition(
    models: list[WargameModel],
    nearest_distance: float,
    furthest_distance: float,
) -> list[int]:
    """Remove models until every unit is back in coherency. Mutates models.

    `03-moving.md` § Regaining coherency: in the End of Turn step, a unit on the
    board that is out of coherency loses models one at a time until coherency is
    restored. They are **destroyed, but trigger nothing that fires when a model
    is destroyed** -- so the caller must not credit these to the opponent, or
    `model_kills` pays for a death nobody caused and `models_lost` and the kill
    counter stop meaning the same thing.

    This is the **backstop**, not the enforcement. The move check is what keeps a
    unit legal; what reaches here is a break no move caused -- casualties
    splitting a unit, most of all. If this ever removes models in bulk, the
    end-of-move check is wrong, and that is the diagnostic to reach for first.

    Which model dies is the controlling player's choice on the table, and has to
    be deterministic here (seeded env, bit-identical golden gates). The rule:
    drop from the **smallest chain component** first, since detaching a lone
    straggler costs one model where cutting the body would cost several, and
    break ties by lowest index. Once the unit is connected, drop whichever model
    sits furthest from the rest, which is the only way a spread breach closes.

    Returns:
        Indices of the models destroyed, in removal order.
    """
    destroyed: list[int] = []
    if not models:
        return destroyed

    group_ids = np.array([m.group_id for m in models], dtype=np.intp)
    base_radii = np.array([m.base_radius for m in models], dtype=float)
    positions = np.array([m.location for m in models], dtype=float)

    # Bounded by the force size: every pass either removes a model or stops.
    for _ in range(len(models)):
        alive_mask = np.array([m.is_alive for m in models], dtype=bool)
        report = evaluate_coherency(
            positions=positions,
            group_ids=group_ids,
            alive_mask=alive_mask,
            base_radii=base_radii,
            nearest_distance=nearest_distance,
            furthest_distance=furthest_distance,
        )
        if report.all_coherent:
            return destroyed
        for unit in report.units:
            if unit.coherent:
                continue
            victim = _choose_casualty(unit, positions, base_radii)
            models[victim].take_damage(models[victim].stats["current_wounds"])
            destroyed.append(victim)
            # One at a time: removing a model re-shapes its unit, and the next
            # pass re-evaluates rather than assuming the rest still have to go.
            break
    return destroyed


def _choose_casualty(
    unit: UnitCoherency,
    positions: np.ndarray,
    base_radii: np.ndarray,
) -> int:
    """Pick the model this unit loses, deterministically."""
    members = unit.member_indices
    if not unit.connected:
        counts = np.bincount(unit.component, minlength=unit.n_components)
        smallest = int(np.argmin(counts))
        return int(members[unit.component == smallest][0])
    gaps = base_to_base_distances(positions[members], base_radii[members])
    return int(members[int(np.argmax(gaps.max(axis=1)))])
