# 10 — Shooting phase

The active player makes ranged attacks. The mechanics of an attack are in
[Making attacks](04-making-attacks.md) and the [attack sequence](05-attack-sequence.md);
this chapter covers which units may shoot and under what constraints.

Steps:

1. **Start of Shooting phase**
2. **Shoot step**
3. **End of Shooting phase**

## Shoot step

The active player shoots with their units **one at a time**, for as many units as they
choose:

1. **Select unit.** Pick a friendly unit that is eligible to shoot — on the board, and not
   already selected to shoot this phase.
2. **Select shooting type.** Pick one shooting type that unit is eligible for and resolve
   it.

Unlike moving, shooting is optional: a player is never obliged to select a unit.

The four core shooting types are below. They are mutually exclusive; a unit resolves
exactly one.

---

## Normal shooting

| | |
|---|---|
| **Eligible if** | The unit is unengaged and did not make an advance move this turn. |
| **Effect** | The unit shoots as described in [Making attacks](04-making-attacks.md). |

The default. Targets must be visible, in range and unengaged.

---

## Run-and-gun shooting

| | |
|---|---|
| **Eligible if** | The unit is unengaged, **made an advance move this turn**, and has one or more `[RUN AND GUN]` weapons. |
| **Effect** | The unit shoots as described in [Making attacks](04-making-attacks.md). |
| **While shooting** | Only `[RUN AND GUN]` weapons may be selected. |

---

## Sidearm shooting

Firing while locked in melee.

| | |
|---|---|
| **Eligible if** | The unit is **engaged**, did not make an advance move this turn, and either has one or more `[SIDEARM]` weapons or is a `MONSTER`/`VEHICLE` unit. |
| **Effect** | The unit shoots as described in [Making attacks](04-making-attacks.md). |
| **While shooting** | Models may target enemy units their unit is engaged with. |

The *while shooting* restrictions then split by model type:

**Non-`MONSTER`/non-`VEHICLE` models** — may select only their `[SIDEARM]` weapons, and
may target only enemy units their unit is engaged with.

**`MONSTER`/`VEHICLE` models** — may fire anything, with two riders:

- subtract 1 from the hit roll, unless the attack is made with a `[SIDEARM]` weapon
  against a unit their unit is engaged with;
- `[SCATTER]` weapons still cannot target a unit their unit is engaged with.

---

## Indirect shooting

Firing at what you cannot see.

| | |
|---|---|
| **Eligible if** | The unit is unengaged, did not make an advance move this turn, and has one or more `[INDIRECT FIRE]` weapons. |
| **Effect** | The unit shoots as described in [Making attacks](04-making-attacks.md). |

**While shooting**, `[INDIRECT FIRE]` weapons may target units that are **not visible** to
the attacking model. Each attack made with one:

- gives the target [cover](13-terrain.md#cover);
- cannot re-roll its hit roll;
- **fails on an unmodified hit roll of 1–5** — unless the attacking unit remained
  stationary this turn *and* the target is visible to at least one friendly unit, in which
  case it fails on an unmodified 1–3 instead.

Spotting is therefore worth roughly a doubling of accuracy, and moving forfeits it.

---

## Shooting at engaged large models

Enemy `MONSTER`/`VEHICLE` units that are engaged **can** be selected as targets of ranged
attacks, which is the exception to the "targets must be unengaged" rule. `[SCATTER]`
weapons are excluded.

Each ranged attack against such a unit subtracts 1 from its hit roll — excluding attacks
made with `[SIDEARM]` weapons by models in a unit engaged with the target.

This does not cut both ways: a unit that is itself engaged with an enemy
`MONSTER`/`VEHICLE` unit is still not eligible to shoot at all, unless it qualifies for
sidearm shooting.

---

## End of Shooting phase

Resolve anything triggered at the end of the Shooting phase.
