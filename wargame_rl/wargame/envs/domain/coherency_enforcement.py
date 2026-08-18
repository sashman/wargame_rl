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

**This is a referee, not a teacher — do not train under it.** It guarantees a
legal board and teaches nothing: measured with it switched off, so the numbers
describe the policy rather than the wrapper, a policy trained under enforcement
intends 0.569 units coherent against 0.756-0.886 for the reward gate alone, and
loses on unseen ground too (70.3 vp_margin on nine held-out tables against
81.5). Every reverted action produces the identical outcome, so they share an
advantage and the policy gradient inside that whole set is exactly zero. Train
with ``objective_hold.require_coherent`` and no enforcement; switch this on for
play.

A third mode, ``clamp``, shortened the move instead of cancelling it. It was
removed once all three measured the same ~26 vp cost and the conclusion above
made the choice of mode a play-time detail rather than a training lever; it
could also never shorten a pure-spread breach, silently degrading to a full
revert. ``git log -- wargame_rl/wargame/envs/domain/coherency_enforcement.py``
restores it.

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
    repair = "repair"


# Passes of the repair loop. Each pass moves at most one model per unit, and a
# unit of five needs at most four pulls to gather every stray onto its body, so
# eight is slack rather than a tuning knob.
MAX_REPAIR_PASSES = 8


def _pull_toward(
    stray: np.ndarray,
    anchor: np.ndarray,
    stray_radius: float,
    anchor_radius: float,
    nearest_distance: float,
) -> np.ndarray:
    """The nearest point to *stray* that satisfies the chain against *anchor*.

    Kept at 90% of the chain distance rather than exactly on it: the boundary is
    where floating point and the next move's rounding both bite, and a model
    parked precisely on the limit re-breaks on any subsequent nudge.
    """
    delta = stray - anchor
    span = float(np.hypot(*delta))
    want = stray_radius + anchor_radius + nearest_distance * 0.9
    if span < 1e-9:
        # Coincident centres: any direction will do, and east is deterministic.
        placed: np.ndarray = anchor + np.array([want, 0.0])
        return placed
    pulled: np.ndarray = anchor + delta / span * want
    return pulled


def _repair_stragglers(
    models: list[WargameModel],
    nearest_distance: float,
    furthest_distance: float,
) -> int:
    """Gather each broken unit's strays back onto its body. Mutates locations.

    **This is a divergence from the rules, and a deliberate one.** `03-moving.md`
    says an illegal move cannot be made; this instead makes the *nearest legal*
    move, by pulling the models that broke the unit back toward the body they
    left rather than cancelling everyone's move.

    The reason is that the spec's own consequence is a catastrophic learning and
    play interface. A revert maps every illegal joint action to one outcome, so
    it is an absorbing state -- measured `P(frozen next | frozen now) = 0.62`,
    against 0.17 after a move -- and it destroys **48.9%** of all intended
    movement while cancelling **33.1%** of unit-moves. It also cannot *repair*:
    a unit split by casualties is incoherent before it moves, so reverting
    returns it to the split and it is frozen for the rest of the game.

    What is fixed here is the **chain** and **connectivity** clauses, which are
    ~95% of breaches; a pure spread breach is left to the caller's fallback.

    Returns:
        How many models the repair moved, which with the fallback's revert count
        is the honest cost of the rule.
    """
    moved = 0
    for _ in range(MAX_REPAIR_PASSES):
        positions = np.array([m.location for m in models], dtype=float)
        radii = np.array([m.base_radius for m in models], dtype=float)
        report = evaluate_coherency(
            positions=positions,
            group_ids=np.array([m.group_id for m in models], dtype=np.intp),
            alive_mask=np.array([m.is_alive for m in models], dtype=bool),
            base_radii=radii,
            nearest_distance=nearest_distance,
            furthest_distance=furthest_distance,
        )
        if report.all_coherent:
            break
        progressed = False
        for unit in report.units:
            if unit.coherent or unit.size < 2:
                continue
            members = unit.member_indices
            counts = np.bincount(unit.component, minlength=unit.n_components)
            body_id = int(np.argmax(counts))
            body = members[unit.component == body_id]
            strays = members[unit.component != body_id]
            if strays.size:
                # Disconnected: bring the stray nearest the body in first, so
                # each pass shortens the remaining distance rather than
                # thrashing between two equally distant models.
                gaps = base_to_base_distances(
                    positions[np.concatenate([strays, body])],
                    radii[np.concatenate([strays, body])],
                )[: strays.size, strays.size :]
                s, b = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
                stray_idx, anchor_idx = int(strays[s]), int(body[b])
            else:
                # Connected but a model is off the chain: pull the loosest one.
                gaps = base_to_base_distances(positions[members], radii[members])
                np.fill_diagonal(gaps, np.inf)
                nearest_gap = gaps.min(axis=1)
                k = int(np.argmax(nearest_gap))
                if nearest_gap[k] <= nearest_distance:
                    continue  # a pure spread breach; the caller falls back
                stray_idx = int(members[k])
                anchor_idx = int(members[int(np.argmin(gaps[k]))])
            target = _pull_toward(
                positions[stray_idx],
                positions[anchor_idx],
                float(radii[stray_idx]),
                float(radii[anchor_idx]),
                nearest_distance,
            )
            models[stray_idx].location = target.astype(models[stray_idx].location.dtype)
            moved += 1
            progressed = True
        if not progressed:
            break
    return moved


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

    if mode is CoherencyEnforcement.repair:
        # Gather what can be gathered, then hand whatever is still broken to the
        # spec's own consequence. Composing rather than duplicating keeps the
        # overlap cascade and the fixed-point iteration in one place.
        repaired = _repair_stragglers(models, nearest_distance, furthest_distance)
        return repaired + enforce_after_move(
            models,
            nearest_distance,
            furthest_distance,
            CoherencyEnforcement.revert_unit,
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


def _select_reverting(
    models: list[WargameModel],
    reverting: set[int],
    nearest_distance: float,
    furthest_distance: float,
    mode: CoherencyEnforcement,
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
