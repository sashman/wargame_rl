# 07 — The battle round

A battle is a fixed number of **battle rounds**. Five, unless the mission says otherwise.

Each battle round runs three steps:

1. **Start of battle round**
2. **Player turns**
3. **End of battle round**

## Start of battle round

Resolve anything triggered at the start of the battle round, then move on.

## Player turns

Both players take one turn each. The **same player always goes first in every battle
round** — the mission decides who. Once that player's turn ends, their opponent takes
theirs.

A turn is seven parts:

| # | Part | What happens |
|---|---|---|
| 1 | Start of Turn step | Anything triggered at the start of a turn resolves. |
| 2 | [Command phase](08-command-phase.md) | Check the army's nerve. |
| 3 | [Movement phase](09-movement-phase.md) | Units reposition. |
| 4 | [Shooting phase](10-shooting-phase.md) | Units make ranged attacks. |
| 5 | [Charge phase](11-charge-phase.md) | Units close into melee. |
| 6 | [Fight phase](12-fight-phase.md) | **Both** players' units make melee attacks. |
| 7 | End of Turn step | Anything triggered at the end of a turn resolves. |

The Fight phase is the one phase in which the non-active player's units act as a matter of
course.

### End of Turn step ordering

Resolve in this order:

1. Everything triggered at this point that is not a mission rule.
2. Then both players consult the mission; anything either player has achieved that
   triggers at this point resolves now.

Coherency is also restored in this step — see [Moving](03-moving.md#regaining-coherency).

## End of battle round

Same two-part ordering as the End of Turn step:

1. Non-mission rules triggered at this point.
2. Then mission rules triggered at this point, for both players.

The round then ends. Unless the battle has ended, the next battle round begins.

## Trigger vocabulary

- **Start / end of the battle round, turn or phase.** A rule triggered "at the start of the
  Movement phase" triggers at the start of *every* Movement phase, both players'.
- **"The turn" / "the phase"** without a possessive means both players' turns or phases.
  "At the end of the Movement phase, this model heals 1 Wound" fires twice per battle
  round.
- **"Your turn" / "your phase"** narrows it to the owner's.

## Out-of-phase rules

Some rules let a unit move, shoot, declare a charge or fight outside the normal sequence.
While using one, that unit cannot use any other **phase-locked** rule — any rule that
names the phase it applies in.

A rule that names when it triggers but not how long it lasts is active only for that
period: "in the Shooting phase" means until the end of that Shooting phase. A rule granted
to a unit with no stated duration lasts only for the phase in which it was granted.

## Persisting effects

An effect with a stated duration ("until the start of your next turn") is a **persisting
effect**. If a unit carrying one leaves the board, note the effect and its remaining
duration; if that unit is set up on the board again, the effect continues for the rest of
its duration.
