# 11 — Charge phase

The active player closes the distance into melee.

Steps:

1. **Start of Charge phase**
2. **Charge step**
3. **End of Charge phase**

## Charge step

The active player resolves charges **one unit at a time**, for as many units as they
choose:

1. **Declare charge.** Pick a friendly unit that is eligible to declare a charge and has
   not already declared one this phase. It declares a charge.
2. **Make charge roll.** Roll **2D6**. The result is the maximum distance for the charge
   move.
3. **Attempt charge.** If a legal [charge move](#charge-move) is possible and the
   controlling player still wants to make it, make it. Otherwise the unit does not move.
   Either way the charge is resolved.

Note the ordering: the roll comes **before** targets are chosen, so a charge is declared
without knowing how far the unit will reach.

### Eligibility

A unit on the board is eligible to declare a charge unless something prevents it. The
common blockers:

- it is **not within 12"** of any enemy unit;
- it is **engaged**;
- it made an **advance** or **fall-back** move this turn.

Charging does not require visibility.

### Failed charges

A charge that cannot legally complete simply fails and the unit does not move. With no
modifiers, a roll of 2 can never succeed: a unit cannot already be within engagement range
when it attempts a charge, so 2" is never enough to reach it.

Charge rolls are also capped — see [the modifier clamps](02-unit-profiles.md#clamps).

---

## Charge move

| | |
|---|---|
| **Maximum distance** | The charge roll. |
| **Eligible if** | The unit declared a charge this phase. |
| **Effect** | The unit moves as described in [Moving](03-moving.md). |

**Before moving.** Select one or more enemy units that are both within 12" of your unit
and within the charge roll's distance. Each is a **charge target** until the move ends.

**While moving.** Each model:

- must end its move **closer** to one or more charge targets;
- must end within 1" of one or more charge targets if it can;
- must end **engaged** with one or more charge targets if it can.

**After moving.**

- The unit must be engaged with **all** of its charge targets.
- The unit must **not** be engaged with any enemy unit that was not a charge target.
- Until the end of the turn, every model in the unit has
  [Strikes First](16-ability-reference.md#strikes-first).

Those two after-moving conditions are what make a charge fail even on a long roll: a
reachable enemy that cannot be engaged without also engaging a non-target is not a legal
charge. And because distance is measured around obstacles, a target 7" away as the crow
flies may be unreachable on a 7" roll.

---

## End of Charge phase

Resolve anything triggered at the end of the Charge phase.
