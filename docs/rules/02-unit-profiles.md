# 02 — Unit profiles

Every unit has a **unit profile**: the characteristics of its models, the weapons they
carry, the abilities they have, and the keywords that let other rules find them.

## Model characteristics

| Symbol | Name | What it does |
|---|---|---|
| **M** | Move | Maximum distance in inches for a normal move. `-` means the model can be set up but never moved. |
| **T** | Toughness | Resilience. Compared against a weapon's Strength to find the wound target. |
| **Sv** | Save | Armour, written as a target (`3+`). Rolled against, modified by the attacking weapon's AP. |
| **InSv** | Invulnerable save | An alternative save that ignores AP. Optional — omitted from most profiles. |
| **W** | Wounds | Damage capacity. At 0 or fewer the model is destroyed. |
| **Rv** | Resolve | Nerve, written as a 2D6 target (`7+`). |
| **CV** | Control Value | Weight when contesting objectives. `-` means the model contests nothing. |

## Weapon characteristics

| Symbol | Name | What it does |
|---|---|---|
| **R** | Range | Reach in inches. A range of `Melee` marks a melee weapon; anything else is a ranged weapon. |
| **A** | Attacks | Attack dice contributed per use. |
| **RS** | Ranged Skill | Hit target for a ranged weapon. |
| **MS** | Melee Skill | Hit target for a melee weapon. |
| **S** | Strength | Compared against Toughness. |
| **AP** | Armour Penetration | Worsens the defender's save roll. `0` is no penalty; `-1` is a penalty of one. |
| **D** | Damage | Wounds removed per attack that gets through. |

A weapon with no Strength characteristic counts as **S 1** for any rule that reads its
Strength.

A weapon may list several **profiles** — alternative characteristic sets. When the weapon
is selected to attack, its controlling player also picks one profile, and that choice
holds for the rest of that attack sequence. Different models carrying the same weapon may
pick different profiles.

Some profiles are restricted: they may only be used against targets carrying stated
keywords.

## Modifiers

A rule that changes a value is a **modifier**. `+1` improves; `-1` worsens. What
"improve" means depends on the characteristic:

- **RS, MS, Sv, InSv, Rv** are targets, so improving *subtracts* from the number before
  the `+`: improving an `MS 3+` by 1 gives `MS 2+`. Worsening adds.
- **AP** is a penalty, so improving *subtracts*: improving `AP -1` by 1 gives `AP -2`;
  improving `AP 0` gives `AP -1`. Worsening adds, but never past `0`.
- Everything else is a plain number: improving S by 1 adds 1.

### Order of application

All modifiers are cumulative. Apply them in this order:

1. Replacements — any rule that *sets* a value to a specific new value. A characteristic
   set to `0`, `-` or `*` in this step is frozen and skips steps 2–5.
2. Multiplication.
3. Addition.
4. Division.
5. Subtraction.
6. Round any fraction up.

### Clamps

Characteristics of `-`, `*` and `N/A` can never be modified at all. After every modifier
has landed, the following bounds apply:

| Characteristic | Bound |
|---|---|
| M | not less than 1" |
| T | not less than 1 |
| Sv | never better than `2+` |
| InSv | never better than `2+` |
| Rv | never better than `4+`, never worse than `9+` |
| CV | not less than 0, and never `-` by modification |
| R | not less than 1" |
| A | not less than 1 |
| RS | never better than `2+`, never worse than `6+` |
| MS | never better than `2+`, never worse than `6+` |
| S | not less than 1 |
| AP | never worse than `0` |
| D | not less than 1 |

A rule that changes a model's RS or MS changes it for **every** weapon that model carries.

### Modifying dice rolls

- Modifiers apply after any re-roll.
- A rule that reads an **unmodified** roll reads the result after re-rolls but before
  modifiers.
- A result may be modified above 6.
- A result modified below 1 becomes 1.
- **Hit rolls and wound rolls can never be modified by more than ±1.** Sum every modifier
  first; a total of −2 or worse becomes −1, and +2 or better becomes +1.
- **Charge rolls cannot exceed 12.** A modified result of 13 or more becomes 12.

### Ignoring modifiers

Some rules let a player, model, unit or weapon ignore modifiers. Unless stated otherwise
this covers modifiers to that unit's rolls and to its profile and weapon characteristics.

Ignoring is selective: you may ignore all of the modifiers a rule covers, or only some.
That means you can keep the beneficial ones and drop the harmful ones — a unit under both
`+2" M` and `-2" M` may ignore only the penalty.

## Random characteristics

- **Random Move** — roll the stated dice once per unit, when it is selected to move; the
  result is that unit's maximum distance.
- **Random Attacks** — roll when gathering attack dice. If several weapons making
  identical attacks all have random Attacks, roll each separately and pool the results.
- **Random Damage** — roll each time an attack inflicts damage, after the defender has
  chosen which model the attack lands on. This roll is a **damage roll**. Any value after
  an operator (`D6+1`) is part of the characteristic, not a modifier.
- Anything else random is rolled per model or per weapon, each time the value is needed.

## Wounds

A model has its **full Wounds remaining** when its remaining Wounds equal its W
characteristic.

When a unit **heals** or regains N Wounds, resolve one Wound at a time:

1. If any model in the unit is missing Wounds, pick one; it regains one Wound.
2. Otherwise, if every surviving model is at full Wounds but the unit has destroyed
   models, revive one destroyed model with a single Wound.

No model may exceed the Wounds it started the battle with. If a rule heals a specific
*model* rather than the unit, excess healing is simply lost — it cannot revive anyone.

## Restoring models

Some rules return destroyed models to a unit. When they do:

- The models come back with the weapons they started the battle with and their full
  Wounds, unless stated otherwise.
- A unit can never exceed its starting strength this way.
- Returned models must be set up in coherency with the models that began the phase on the
  board.
- They may be set up engaged with an enemy unit, but only one their unit was already
  engaged with.

## Keywords

Keywords are tags written in `SMALL CAPS` that let other rules select models and units.

- A unit carries the keywords of every model in it. A model carries only its own.
- A rule that applies to a *unit* with keyword `K` applies to any unit containing at least
  one `K` model.
- A rule that applies to a *model* with keyword `K` applies only to `K` models — even
  inside a unit that has the keyword through someone else.
- A rule aimed at `non-K` applies to anything without `K`.
- Keywords separated by commas, slashes or "or" mean *any one of them*. Keywords written
  adjacent to each other mean *all of them*.
- Singular and plural forms of a keyword are the same keyword.

Attacks target units, not models — so a unit is exposed to any rule matching any keyword
any of its models carries.

## Abilities

Units may carry **abilities**; weapons may carry **weapon abilities**, written in
`[BRACKETS]`. Both are catalogued in the [ability reference](16-ability-reference.md).

A weapon ability followed by keywords only applies against targets carrying one of those
keywords — `[PIERCING STRIKES: VEHICLE]` does nothing against infantry.

When a rule asks you to pick something "with ability X", any number attached to X is
irrelevant to the selection.

### Duplicated abilities

Two instances of the same ability never stack, whatever numbers or keywords they carry.
The controlling player picks which instance applies, and for weapon abilities that choice
is made afresh each time the unit selects weapons to attack with.

Instances count as duplicates even when their numbers differ — a model with
`[EXTRA HITS 1]` and `[EXTRA HITS 2]` must choose one.

### Rules with two conditions

Some abilities layer a narrower condition on a broader one to grant a better effect. Both
conditions must hold for the better effect to apply — the second condition refines the
first, it does not replace it.
