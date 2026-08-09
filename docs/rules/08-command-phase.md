# 08 — Command phase

The first phase of a turn. The active player checks whether their army's nerve is
holding.

Steps:

1. **Start of Command phase**
2. **Suppression step**
3. **End of Command phase**

> The wider game also spends this phase accruing and spending a command resource. That
> economy is [out of scope](README.md#deliberately-out-of-scope) here.

## Start of Command phase

Resolve anything triggered at the start of the Command phase.

## Suppression step

The **active player** makes one [suppression roll](01-core-concepts.md#suppression) for
each unit in their army that meets either condition:

- the unit is currently **suppressed**; or
- the unit is **at or below half-strength**.

A unit that was suppressed at the start of this step and whose roll now succeeds stops
being suppressed. A unit that fails becomes — or stays — suppressed.

Rolls are made for qualifying units whether or not they are on the board.

### One roll per unit

No unit makes more than one suppression roll in this step. If some other rule forces a
unit to roll here before it would have rolled for being suppressed or at or below
half-strength, that earlier roll counts and no second roll is made.

### Worked cases

| Unit | Situation | Rolls? |
|---|---|---|
| Starting strength 3, 3 models left, currently suppressed | Above half-strength, but suppressed | Yes — success clears the suppression |
| Starting strength 10, 5 models left | Exactly at half-strength | Yes |
| Starting strength 5, 2 models left | Below half-strength | Yes |
| Starting strength 1, W 11, 3 Wounds left | Below half-strength by Wounds | Yes |
| Starting strength 5, 4 models left, not suppressed | Below starting strength but above half | No |

## End of Command phase

Resolve in this order:

1. Everything triggered at this point that is not a mission rule.
2. Then both players consult the mission; anything achieved that triggers at this point
   resolves now.

This is one of the points at which objectives are scored — see
[Missions and scoring](15-missions-and-scoring.md).
