# 09 — Movement phase

The active player repositions their army. How a move is physically executed is covered in
[Moving](03-moving.md); this chapter covers which moves exist.

Steps:

1. **Start of Movement phase**
2. **Move units step**
3. **End of Movement phase**

## Move units step

The active player moves their units **one at a time**:

1. **Select unit.** Pick one friendly unit that has not yet been selected to move this
   phase. It is now selected to move.
2. **Select move type.** Pick one move type that unit is eligible for and resolve it.

By the end of this step, **every** unit in the active player's army must have been
selected to move. Any unit its controlling player never picks a type for defaults to
*remain stationary*.

### Modes

Some move types offer **modes**. Modes are mutually exclusive and must be assessed in the
order they are printed — take the first one whose conditions you meet, unless the mode is
explicitly optional. A unit meeting no mode's conditions cannot make that move at all.

Conditions in a move type labelled with a mode name apply only if that mode was selected;
unlabelled conditions always apply.

---

## Remain stationary

| | |
|---|---|
| **Maximum distance** | `-` |
| **Eligible if** | Any unit. |
| **Effect** | No model is moved or rotated. |

A unit that remains stationary triggers nothing that fires when a unit starts or ends a
move — it did not make one.

---

## Normal move

| | |
|---|---|
| **Maximum distance** | The unit's **M** characteristic. |
| **Eligible if** | The unit is on the board and unengaged. |
| **Effect** | The unit moves as described in [Moving](03-moving.md). |
| **After moving** | The unit must be unengaged. |

---

## Advance move

| | |
|---|---|
| **Maximum distance** | Advance roll **+** the unit's **M** characteristic. |
| **Eligible if** | The unit is on the board and unengaged. |
| **Before moving** | Make an **advance roll**: one D6. |
| **Effect** | The unit moves as described in [Moving](03-moving.md). |
| **After moving** | The unit must be unengaged. Until the end of the turn it cannot declare a charge. |

Advancing trades the turn's offensive options for reach. It does **not** by itself
prevent shooting — but only `[RUN AND GUN]` weapons may be fired after it, via
[run-and-gun shooting](10-shooting-phase.md#run-and-gun-shooting).

---

## Fall-back move

Disengaging from melee.

| | |
|---|---|
| **Maximum distance** | The unit's **M** characteristic. |
| **Eligible if** | The unit is engaged. |
| **Before moving** | Select a fall-back mode (below). |
| **Effect** | The unit moves as described in [Moving](03-moving.md). |
| **After moving** | The unit must be unengaged. Until the end of the turn it cannot shoot or declare a charge. |

**Modes**, assessed in order:

| Mode | Conditions | Consequences |
|---|---|---|
| **Ordered withdrawal** | The unit is not suppressed. Optional — a unit may choose reckless break instead. | None beyond the shared restrictions. |
| **Reckless break** | Otherwise, compulsory. | *Before moving:* one [backfire roll](06-visibility-and-damage.md#backfire-rolls) per model in the unit. *While moving:* models may move through enemy models. *After moving:* if the unit is not suppressed, it must make a suppression roll. |

Ordered withdrawal is not mandatory, so a unit that qualifies for it may still take the
reckless break — worth doing when moving through the enemy is the only way out.

---

## End of Movement phase

Resolve anything triggered at the end of the Movement phase.
