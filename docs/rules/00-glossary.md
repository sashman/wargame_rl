# Glossary

Every term this specification relies on, defined once. Terms are defined on their own
footing — nothing here is a translation of anything.

## Entities

| Term | Meaning |
|---|---|
| **model** | One figure on the board. The smallest thing that occupies a position, moves, shoots and is destroyed. |
| **unit** | One or more models that are set up, move, shoot and fight together. A unit is the thing rules target; damage lands on models. |
| **army** | Every unit one player controls. |
| **friendly** / **enemy** | Relative to whoever is using the rule: friendly means "in your army", enemy means "in your opponent's army". A rule that says neither applies to both. |
| **controlling player** | The player whose army a model or unit belongs to. |
| **active player** / **opposing player** | Exactly one player is active at any moment; the other is opposing. See [Core concepts](01-core-concepts.md#sides). |
| **board** | The rectangular playing area. Nothing may be moved across its edge. |
| **terrain feature** | A physical obstacle on the board. |
| **terrain area** | A bounded region of the board holding one or more terrain features. Most terrain rules key off the *area*, not the feature. |
| **objective** | A location both sides are trying to hold. Either a terrain area, or a circular marker. |

## Characteristics

A model's characteristics come from its **unit profile**. Weapons carry their own.

| Symbol | Name | Meaning |
|---|---|---|
| **M** | Move | How far the model can travel in one normal move, in inches. `-` means it cannot be moved at all. |
| **T** | Toughness | How hard it is to wound. Compared against a weapon's Strength. |
| **Sv** | Save | The die result needed to shrug off a wound, written as a target (`3+`). Lower is better. |
| **InSv** | Invulnerable save | A second save that ignores Armour Penetration. Optional — not every model has one. |
| **W** | Wounds | Damage the model absorbs before it is destroyed. |
| **Rv** | Resolve | Nerve. Written as a target for a 2D6 roll (`7+`). Lower is better. |
| **CV** | Control Value | How much weight the model pulls when contesting an objective. `-` means it contests nothing. |

Weapon characteristics:

| Symbol | Name | Meaning |
|---|---|---|
| **R** | Range | How far the weapon reaches, in inches. `Melee` marks a melee weapon; everything else is a ranged weapon. |
| **A** | Attacks | How many attack dice the weapon contributes each time it is used. |
| **RS** | Ranged Skill | The hit-roll target for a ranged weapon. Lower is better. |
| **MS** | Melee Skill | The hit-roll target for a melee weapon. Lower is better. |
| **S** | Strength | Compared against the target's Toughness to find the wound-roll target. |
| **AP** | Armour Penetration | A penalty applied to the defender's save roll, written as `0`, `-1`, `-2`… |
| **D** | Damage | Wounds lost per attack that gets through. |

## States and conditions

| Term | Meaning |
|---|---|
| **engaged** | Two models are engaged while each is inside the other's engagement range. Their units are engaged too. A unit with no engaged models is **unengaged**. |
| **engagement range** | Within 2" horizontally and 5" vertically of a model. |
| **coherency** | The spacing requirement that keeps a unit together. See [Moving](03-moving.md#coherency). |
| **suppressed** | A unit whose nerve has broken. Its models' Control Value becomes `-`. |
| **suppression roll** | A resolve roll made to find out whether a unit becomes, or stops being, suppressed. |
| **resolve roll** | 2D6 against a unit's Resolve characteristic. |
| **backfire roll** | A D6 that punishes the roller on a low result, usually with piercing damage. |
| **hidden** | A model that enemy models can only see inside their detection range. |
| **detection range** | How far a model can see a hidden model. 15" unless stated otherwise. |
| **cover** | A defensive benefit that worsens the Ranged Skill of attacks against a unit by 1. |
| **starting strength** | How many models a unit had at the start of the first battle round. |
| **below starting strength** / **at half-strength** / **below half-strength** | Thresholds derived from starting strength, or from remaining Wounds for a one-model unit. See [Core concepts](01-core-concepts.md#unit-strength). |
| **destroyed** | A model reduced to 0 or fewer Wounds. A unit is destroyed when all its models are. |

## Actions and events

| Term | Meaning |
|---|---|
| **battle round** | One pass in which both players take a turn. |
| **player turn** | One player's Command → Movement → Shooting → Charge → Fight sequence. |
| **phase** | One of the five segments of a turn. |
| **move type** | A named kind of move with its own eligibility and limits: remain stationary, normal, advance, fall back, charge, pile in, consolidate. |
| **shooting type** | A named kind of shooting with its own eligibility: normal, run-and-gun, sidearm, indirect. |
| **attack** | One attack die and everything that happens to it: a hit roll, a wound roll, a save roll, and damage. |
| **hit roll** / **wound roll** / **save roll** | The three D6 rolls of the attack sequence. |
| **critical hit** / **critical wound** | An unmodified 6 on a hit or wound roll. Still a hit or a wound; may trigger further abilities. |
| **piercing damage** | Damage that skips hit, wound and save rolls entirely and removes Wounds directly. |
| **victory points (VP)** | The score. The player with the most at the end wins. |
| **level of control** | The summed Control Value a player has within range of an objective. |

## Measurement

| Term | Meaning |
|---|---|
| **within X"** | No more than X" away. A model is within X" if any part of it is; a unit is within X" if any of its models is. |
| **wholly within X"** | Every part of the model, and every model in the unit, is within X". |
| **visible** | At least one part of the observed model can be seen. |
| **fully visible** | Every part of the observed model that faces the observer can be seen. |
| **line of sight** | The imaginary 1 mm line drawn to test visibility. |
| **closest** / **nearest** | Measured from the model or unit using the rule. Ties are broken by that unit's controlling player. |
| **unmodified** | A die result after any re-roll but before any modifier. |
| **D6**, **2D6**, **D3** | One die; two dice summed; one die halved and rounded up. |
| **`2+`, `3+`…** | A target: that result or higher succeeds. |
