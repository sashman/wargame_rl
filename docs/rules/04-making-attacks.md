# 04 — Making attacks

The wrapper around the [attack sequence](05-attack-sequence.md). Every time a unit shoots
or fights, the active player works through three steps in order:

1. **Select weapons**
2. **Select targets**
3. **Resolve attacks**

## 1. Select weapons

For each model in the attacking unit, choose which of its weapons will attack. Ranged
weapons make ranged attacks; melee weapons make melee attacks.

- **Shooting** — select any number of the model's ranged weapons.
- **Fighting** — select exactly one of the model's melee weapons.

A model with no ranged weapons cannot make ranged attacks; a model with no melee weapons
cannot make melee attacks. A unit may still be selected to shoot or fight when none of
its models can attack — it simply makes no attacks, and the shooting or fight type is
still considered resolved.

Two ability groups modify this step: `[SIDEARM]` weapons compete with a model's other
ranged weapons rather than adding to them, and `[EXTRA ATTACKS]` weapons attack in
addition to a model's chosen melee weapon. See the
[ability reference](16-ability-reference.md).

If a selected weapon has multiple profiles, pick one now.

### Attacks inherit from their weapon

An attack carries the characteristics and abilities of the weapon making it. Any modifier
or ability that applies to the attack applies to that weapon until the attacking unit's
whole attack sequence — and every consequence of those abilities — has finished
resolving.

## 2. Select targets

For each selected weapon, choose what it attacks.

**Shooting.** Select **one** enemy unit per weapon. Unless a rule says otherwise, that
target must be:

- **visible** to the model carrying the weapon (see [Visibility](06-visibility-and-damage.md));
- **within range** of that weapon;
- **unengaged**.

Visibility and range are checked independently and need not be satisfied by the same
enemy model — it is enough that some model in the target unit is visible and some model
in it is in range.

**Fighting.** Select one or more enemy units per weapon. Each must be **engaged** with
the model carrying the weapon, and a weapon may not name more targets than its Attacks
characteristic.

Different weapons on the same model may pick different targets. If a weapon has no legal
target — or if its controlling player simply declines to name one for a ranged weapon —
that weapon makes no attacks.

When every attack a model or unit makes goes into the same enemy unit, it is said to be
attacking a **single target**.

### Timing hooks

- Rules triggered *against* an attack fire once the Select Targets step is complete,
  provided their other conditions hold.
- Rules triggered when an attack is *allocated* fire later, in the Inflict Damage step.

### Targets that stop being legal

If a unit was a legal target when selected and then stops being one — because some
out-of-sequence rule moved it out of range, say — the attacking player may choose new
targets.

## 3. Resolve attacks

Work through this loop:

1. **Select enemy unit.** Pick one of the units targeted by one or more weapons.
2. **Gather attack dice.** Pick one weapon aimed at that unit that has not yet attacked
   it, and take a number of D6 equal to its Attacks characteristic. Each die is one
   attack.
   Any other weapons aimed at the same unit that make **identical attacks** and have not
   yet attacked it fire now as well, and their dice join the pool. Three identical
   weapons with A 2 gather six dice.
3. **Resolve attack dice.** Run the [attack sequence](05-attack-sequence.md) for the whole
   pool at once.
4. **Continue.** Take the first case that applies:
   - weapons still aimed at this unit have not attacked → return to step 2;
   - weapons aimed at a *different* unit have not attacked → return to step 1;
   - otherwise the unit has finished making its attacks.

### Identical attacks

Two attacks are identical when they share the same RS/MS, S, AP and D characteristics
**and** are affected by the same applicable abilities and rules. An ability that cannot
possibly trigger against the chosen target is not applicable, so it does not split the
pool.

### Splitting melee attacks

If a melee weapon names more than one target, its Attacks must be divided between them
during target selection, with at least one attack per target. When gathering dice for
that weapon against a given target, take only the number declared against that target.

## Completion vocabulary

- A ranged weapon has **shot** once all of its attacks are resolved; a model has shot once
  all its ranged weapons have; a unit has shot once every attacking model has. A unit
  selected to shoot that makes no attacks has *not* shot.
- The same construction applies to **fought** for melee weapons.
- A unit that has resolved every attack has **finished making its attacks**, and has
  **attacked**.
