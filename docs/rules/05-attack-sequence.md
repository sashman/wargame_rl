# 05 — Attack sequence

Every attack die gathered in [Making attacks](04-making-attacks.md) runs through four
steps:

1. **Hit rolls**
2. **Wound rolls**
3. **Save rolls**
4. **Inflict damage**

Within each step, every die that is due to be rolled is rolled at once. An attack that
fails at any step, or that reaches damage, is finished. When every attack has either
failed or inflicted damage, the sequence ends.

## 1. Hit rolls

Roll one D6 per attack die. Take the first row that matches:

| Result | Outcome |
|---|---|
| Unmodified 1 | Fails |
| Unmodified 6 | **Critical hit** |
| Equal to or greater than the attack's RS (shooting) or MS (melee) | Hit |
| Anything else | Fails |

A critical hit is still a hit; it just also satisfies anything that keys off critical
hits. An unmodified 1 fails whatever the modifiers say, and an unmodified 6 hits whatever
the skill characteristic says.

## 2. Wound rolls

Roll one D6 per hit. Take the first row that matches:

| Result | Outcome |
|---|---|
| Unmodified 1 | Fails |
| Unmodified 6 | **Critical wound** |
| Equal to or greater than the target below | Wound |
| Anything else | Fails |

The target comes from the attack's Strength against the target unit's Toughness:

| Strength vs Toughness | Needed |
|---|---|
| S is at least **double** T | `2+` |
| S is **greater than** T | `3+` |
| S **equals** T | `4+` |
| S is **less than** T | `5+` |
| S is at most **half** T | `6+` |

Critical wounds are still wounds.

### Mixed Toughness

If the target unit's models have different T characteristics, use the **highest** T among
its models still on the board.

## 3. Save rolls

The opposing player resolves this step.

1. **Create groups.** Divide the target unit into groups, repeating as needed:
   - one group per `CHARACTER` model;
   - one group per distinct combination of W, Sv and InSv among all other models.
2. **Declare allocation order.** State the order in which groups will take attacks,
   subject to:
   - a non-`CHARACTER` group containing a wounded model must come first;
   - no `CHARACTER` group may come before any non-`CHARACTER` group;
   - a `CHARACTER` group containing a wounded model must come before a `CHARACTER` group
     with none.

   Where several groups tie on those constraints, the defender picks the order among them.
3. **Roll saves.** Roll one D6 for each attack that wounded.

## 4. Inflict damage

The opposing player resolves each save roll, working from the **lowest** result upward,
until every attack is resolved or every model in the target unit is destroyed. Excess
attacks against a wiped-out unit are lost.

For each save roll:

1. **Select model.** Pick a model in the **current allocation group** — a model that has
   already lost Wounds if one is available.
2. **Check the save.** Take the first row that matches:

   | Condition | Outcome |
   |---|---|
   | Unmodified 1 | Inflicts damage |
   | The group has an InSv and the result is at least that InSv | Saved |
   | The result, modified by the attacking weapon's AP, is at least the group's Sv | Saved |
   | Anything else | Inflicts damage |

   The invulnerable save is checked first and ignores AP, so a defender always gets
   whichever save is better against that attack. An unmodified 1 never saves.
3. **Resolve damage.** The selected model loses Wounds equal to the attack's Damage
   characteristic. At 0 or fewer Wounds it is destroyed. Damage in excess of the model's
   remaining Wounds does **not** spill over to another model.

### Current allocation group

The first group in the declared order starts as the current group. When every model in
the current group is destroyed, the next group in the order becomes current.

### Save rolls that cannot succeed

Roll the dice even when a modifier makes a save mathematically unreachable for some models
in the unit — other groups may have better saves, and allocation can reach them.

## Suffering damage and being destroyed

A model has **suffered damage** as soon as an attack reaches the Resolve Damage step
against it, even if a later rule prevents the Wounds from actually being lost.

A model reduced to 0 or fewer Wounds is **destroyed**. A unit is destroyed when all its
models are.

When a model is destroyed:

1. Resolve anything triggered by its destruction.
2. Remove it from the board.

If the model was destroyed by an attack, both of those happen only **after the attacking
unit has resolved all of its attacks** — so a dying model keeps soaking allocation
decisions for the rest of that sequence. Destroyed models and units cannot use abilities
or be selected or targeted, unless a rule says otherwise.

Some rules fire only when a model or unit is destroyed **by you**: that means destroyed by
an attack from your army or by a rule of yours. Anything else — falling out of coherency,
a backfire roll — does not count.

### Measuring to something destroyed

If a rule needs a distance to a destroyed model, measure to any point that model occupied
before it was removed. For a destroyed unit, measure to the last model destroyed in it.
