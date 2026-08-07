# 12 — Fight phase

**Both** players act in this phase. Units first close up, then trade melee attacks, then
reposition.

Steps:

1. **Start of Fight phase**
2. **Pile-in step**
3. **Fight step**
4. **Consolidate step**
5. **End of Fight phase**

## Pile-in step

Both players make [pile-in moves](#pile-in-move) with any eligible unit they choose. The
active player resolves all of theirs first, then the opposing player. No unit piles in
more than once in this step.

### Pile-in move

| | |
|---|---|
| **Maximum distance** | 3" |
| **Eligible if** | It is the Fight phase and at least one of: the unit is engaged; it made a charge move this turn; it was selected to make an [overrun fight](#overrun-fight) this phase. |
| **Effect** | The unit moves as described in [Moving](03-moving.md). |

**Before moving.** Select pile-in targets:

- if the unit is engaged, select **every** enemy unit it is engaged with;
- otherwise, select one or more enemy units within 5" of the unit.

**While moving.**

- Models in base contact with an enemy model cannot be moved.
- Every model that is moved must end **closer to the closest pile-in target**, and engaged
  with it if possible.

**After moving.**

- The unit must be engaged.
- Every model that started this move engaged with an enemy unit must still be engaged with
  that unit.

## Fight step

A unit is **eligible to fight** if it has not already been selected to fight this phase
and at least one of:

- it is engaged, **or was engaged at the start of this step**;
- it made a charge move this turn.

Resolve this sequence until every eligible unit has fought:

1. **Resolve Strikes First combats.** Starting with the active player, players alternate
   selecting one friendly [Strikes First](16-ability-reference.md#strikes-first) unit that
   is eligible to fight. If a player cannot:
   - if *no* Strikes First unit is eligible to fight, move to step 2, and that player
     selects first there;
   - otherwise the other player selects next.
2. **Resolve remaining combats.** Starting with whoever moved the sequence into this step,
   players alternate selecting one friendly eligible unit to fight. If a player cannot:
   - if no unit is eligible to fight at all, the Fight step ends;
   - otherwise the other player selects next.

After any fight resolved in step 2, if a Strikes First unit has *become* eligible to
fight, return to step 1.

Each time a unit is selected to fight, its controlling player picks one fight type it is
eligible for and resolves it.

### Passing

When the sequence returns to a player and every one of their eligible units is more than
5" from all enemy units, that player may **pass** and hand the sequence back. If both
players pass in succession — or one passes while the other has nothing eligible left — the
Fight step ends.

Passing exists because a unit's targets can all die before it swings, with nothing else in
reach; passing lets it wait and see whether an enemy pile-in brings something within 5".

---

## Normal fight

| | |
|---|---|
| **Eligible if** | The unit is engaged. |
| **Effect** | The unit fights as described in [Making attacks](04-making-attacks.md). |

## Overrun fight

| | |
|---|---|
| **Eligible if** | The unit is unengaged, or was unengaged at the start of the Fight step but became engaged during the Fight phase. |
| **Effect** | The unit makes **one additional pile-in move**, then fights as described in [Making attacks](04-making-attacks.md). |

This is how a unit that killed its charge target — or that was left behind when a target
was destroyed — can still reach a new one.

---

## Consolidate step

Both players make [consolidation moves](#consolidation-move) with any eligible unit they
choose. The active player resolves all of theirs first, then the opposing player. No unit
consolidates more than once.

### Consolidation move

| | |
|---|---|
| **Maximum distance** | 3" |
| **Eligible if** | It is the Fight phase and the unit **was eligible to fight** this phase. |
| **Effect** | The unit moves as described in [Moving](03-moving.md). |

**Before moving**, select a consolidation mode — assessed in order, and the first one whose
conditions are met is compulsory:

| Mode | Conditions | Selection |
|---|---|---|
| **Ongoing** | The unit is engaged. | Every enemy unit it is engaged with. |
| **Engaging** | Otherwise, the unit is within 3" of one or more enemy units. | One or more of those enemy units. |
| **Objective** | Otherwise, the unit is within 3" of one or more objectives. | One of those objectives. |

A unit meeting none of the three conditions cannot consolidate.

**While moving.**

- **Ongoing** — models in base contact with an enemy model cannot be moved. Every model
  that is moved must end closer to the closest selected enemy unit, and engaged with it if
  possible.
- **Engaging** — every model that is moved must end closer to the closest selected enemy
  unit, and engaged with it if possible.
- **Objective** — every model that is moved must end within range of the selected
  objective if possible, or closer to it if not.

**After moving.**

- **Ongoing** — every model that started this move engaged with an enemy unit must still be
  engaged with that unit.
- **Engaging** — the unit must be engaged with **all** of the selected enemy units. If any
  of the enemy units now engaged with it has not been selected to fight this phase, the
  opposing player must select each of those units in turn; each becomes eligible to fight
  and is selected to fight.
- **Objective** — the unit must be within range of the selected objective.

The engaging mode is therefore a way to drag fresh enemy units into the fight, at the cost
of giving them a swing.

---

## End of Fight phase

Resolve anything triggered at the end of the Fight phase.

## Fighting after death

Some rules let models attack after being destroyed. A model under such an effect is not
removed when destroyed. It stays on the board until its unit has been selected to attack
and has attacked, or until the end of the phase — whichever comes first. Then anything
triggered by its destruction resolves and it is removed.

If a rule instead tells a destroyed model to fight immediately after the attacking unit,
the model stays on the board until its unit has fought, or until the end of the phase.
This lets the whole unit swing together, and anything targeting that unit also reaches the
destroyed model.
