# 16 — Ability reference

Every ability, alphabetically. Weapon abilities are written in `[BRACKETS]` and attach to
a weapon profile; unit abilities are plain and attach to a unit profile.

Two general rules govern all of them, and are stated in
[Unit profiles](02-unit-profiles.md#abilities):

- an ability followed by keywords only applies against targets carrying one of them;
- duplicated abilities never stack — the controlling player picks one instance.

## Weapon abilities

### `[AUTO-HIT]`

Each attack made with this weapon automatically hits. No hit roll is made, so the hit is
never a critical hit.

### `[BANE-X Y+]`

Each time an attack is made with this weapon, if the target unit has keyword `X`, an
unmodified wound roll of `Y+` is a **critical wound**.

`[BANE-non-X Y+]` triggers against any unit that does *not* have keyword `X`.

### `[BRACED]`

In your Shooting phase, add 1 to the hit roll of each attack made with this weapon if all
of the following apply to the attacking unit:

- it is unengaged;
- it was not set up on the board this turn;
- no model in it has moved more than 3" this turn.

### `[BURST X]`

Each time you gather attack dice for this weapon, add **X** additional dice if the target
unit was within **half** the weapon's Range in the Select Targets step.

### `[EXTRA ATTACKS]`

Each time a unit containing a model with this weapon fights, that model attacks with it
**in addition** to its other attacks. In the Select Weapons step, for each such model you
must select:

- all of that model's `[EXTRA ATTACKS]` weapons, and
- one of that model's other melee weapons, if it has one.

A model with only `[EXTRA ATTACKS]` melee weapons still attacks with them.

### `[EXTRA HITS X]`

Each time an attack made with this weapon results in a **critical hit**, that attack scores
**X additional hits** on the target — so `[EXTRA HITS 2]` turns one critical hit into three
hits in total.

### `[IGNORES COVER]`

The target of an attack made with this weapon cannot have [cover](13-terrain.md#cover)
against that attack — including cover granted by an ability rather than by terrain.

### `[IMPACT]`

Each time an attack is made with this weapon, add 1 to the wound roll if the attacking
model's unit made a charge move this turn.

### `[INDIRECT FIRE]`

A unit containing a model with this weapon may use
[indirect shooting](10-shooting-phase.md#indirect-shooting).

### `[MARKSMAN]`

While resolving attacks made with one or more `[MARKSMAN]` weapons, at the start of the
Allocation Order step: if the target unit contains a `CHARACTER` model visible to one of
the attacking models, the active player may select an allocation group containing one of
those visible `CHARACTER` models. That group becomes the **current allocation group**
until those attacks are resolved or the group is destroyed.

This ability is only *applicable* — for the purposes of splitting identical attacks — when
the target actually contains a `CHARACTER`.

### `[ONE SHOT]`

This weapon may be selected to attack with only **once per battle**.

A model carrying two `[ONE SHOT]` weapons may fire each of them once. A revived model
cannot re-fire a `[ONE SHOT]` weapon it has already used, but a unit newly added to an
army brings fresh uses.

### `[OVERLOAD X]`

Each time a model attacks with this weapon, if the target unit was within **half** the
weapon's Range in the Select Targets step, add **X** to the weapon's Damage characteristic
until the attacking unit has resolved its attacks.

### `[PAIRED]`

Each time an attack is made with this weapon, you may re-roll the wound roll.

### `[PIERCING STRIKES]`

Each time an attack made with this weapon results in a **critical hit**, you may choose for
that attack to automatically wound the target.

It is a choice, not a compulsion: taking it skips the wound roll, which also forfeits any
chance of a critical wound and anything that keys off one — `[SHATTERING WOUNDS]`, most
obviously.

### `[PSIONIC]`

Each time an attack is made with this weapon, you may ignore any or all modifiers to that
attack's RS or MS characteristic, and any or all modifiers to the hit roll. Attacks made
with this weapon are **psionic attacks**.

### `[RUN AND GUN]`

A unit containing a model with this weapon may use
[run-and-gun shooting](10-shooting-phase.md#run-and-gun-shooting).

### `[SCATTER]`

Each time you gather attack dice for this weapon, add **one** additional die for every five
models that were in the target unit at the Select Targets step, rounding down.

`[SCATTER X]` adds **X** dice per five models instead.

`[SCATTER]` weapons can never target a unit that is engaged.

### `[SHATTERING WOUNDS]`

Each time an attack made with this weapon results in a **critical wound**, the attack
sequence ends for that attack and the target suffers
[piercing damage](06-visibility-and-damage.md#piercing-damage) equal to the weapon's Damage
characteristic. It is inflicted **after** any ordinary damage from the same pool of
attacks.

Piercing damage from this ability may damage at most **one model per critical wound** — any
surplus is lost, which is the one case where piercing damage does not spill.

### `[SIDEARM]`

A unit containing a model with this weapon may use
[sidearm shooting](10-shooting-phase.md#sidearm-shooting).

When using any other shooting type, for each model carrying one (excluding
`MONSTER`/`VEHICLE` models) you must choose **either**:

- one or more of its `[SIDEARM]` weapons; **or**
- one or more of its other ranged weapons.

Never both. Sidearms compete with the model's main armament rather than adding to it.

### `[SWEEP X]`

Each time you gather attack dice for this weapon, if you selected only **one** target for
all of that weapon's attacks, add **X** additional dice for every five models that were in
the target unit at the Select Targets step, rounding down.

### `[UNSTABLE]`

Each time a unit is selected to shoot or to fight, after it has resolved all of its
attacks, make one [backfire roll](06-visibility-and-damage.md#backfire-rolls) for that unit
per `[UNSTABLE]` weapon selected in the Select Weapons step.

---

## Unit abilities

### Aura

An ability that affects models or units within a stated range is an **aura**.

- A model with an aura is always within range of its own aura while on the board.
- A unit may be under several different auras at once.
- Being within range of the *same* aura more than once still applies it only once.

### Colossal Walker

Each time a unit with this ability makes a normal, advance or fall-back move:

- its models may move through models — including `MONSTER`/`VEHICLE` models, but excluding
  `COLOSSAL` models — and may move horizontally through sections of terrain features **4"
  or less** in height;
- before moving, you may give every model in the unit the `MOBILE` keyword until the move
  ends. If you do, roll one D6 when the move ends: on a **1** the unit becomes
  [suppressed](01-core-concepts.md#suppression).

`MOBILE` is what lets a non-infantry model move horizontally through
[dense terrain](13-terrain.md#terrain-and-movement) — so this is a gamble on nerve in
exchange for a straight line.

### Crippled X

While a model's remaining Wounds are **equal to or less than X**, that model is
**crippled**: its attacks suffer **−1 to hit rolls**.

### Detonation X

Each time a model in this unit is destroyed, roll one D6. On a **6** each unit within 6" of
that model suffers **X** points of
[piercing damage](06-visibility-and-damage.md#piercing-damage). If X is random, roll
separately for each affected unit.

### Elevated Fire

Not an ability but a standing rule, listed here because it behaves like one.

Each time a model makes a **ranged** attack against a visible unit containing one or more
models at **ground level**, improve the Ranged Skill of that attack by 1 if either:

- the attacking model is on a section of a terrain feature **3" or more** in height; or
- the attacking model has `TOWERING` and the target unit is within 12".

### Elusive

Unless part of a larger formation, this unit is **not visible** to enemy models unless they
are within **12"** of it, and it cannot be targeted by `[INDIRECT FIRE]` weapons unless the
attacking model is within 12".

`Elusive X"` substitutes X" for 12".

Where models in one unit have different Elusive ranges, the **highest** among the unit's
models on the board applies to the whole unit.

### Infiltrate

During deployment, if **every** model in a unit has this ability, it may be set up anywhere
on the board that is more than **8" horizontally** from your opponent's deployment zone and
from all enemy units.

### Shrug X+

Each time a model with this ability would lose a Wound, roll one D6: on an **X+** that
Wound is not lost.

It applies to every source of Wound loss — ordinary damage and piercing damage alike.

### Stealth

If **every** model in a unit has this ability, each ranged attack targeting that unit is
resolved as though the unit had [cover](13-terrain.md#cover).

### Strikes First

While **every** model in a unit has this ability, that unit is a **Strikes First unit**, and
is selected to fight in the first sub-step of the
[Fight step](12-fight-phase.md#fight-step).

Every model in a unit that made a charge move gains this ability until the end of that
turn.

### Vanguard X"

In the Resolve Pre-battle Rules step, if **every** model in a unit has this ability and the
unit is wholly within your deployment zone, it may make a **vanguard move**.

| | |
|---|---|
| **Maximum distance** | The X" in `Vanguard X"`. |
| **Eligible if** | It is the Resolve Pre-battle Rules step and the unit is wholly within your deployment zone. |
| **Effect** | The unit moves as described in [Moving](03-moving.md). |
| **After moving** | The unit must be more than **8" horizontally** from all enemy units. |

Where models in one unit have different Vanguard values, select the **lowest** value not
shared by every model in the unit.

---

## Abilities depending on out-of-scope systems

These exist in the wider game but hang off systems this specification
[does not cover](README.md#deliberately-out-of-scope). They are named here so their absence
is deliberate, not an omission:

| Ability | Depends on |
|---|---|
| **Insertion** | Strategic reserves — set up anywhere more than 8" horizontally from all enemy units on arrival, even inside the opponent's deployment zone. |
| **Hover** | Flight — arriving without the movement penalty for taking to the skies. |
| **Gun Ports X** | Transports — firing the weapons of embarked models. |
| **Leader** / **Support** | Attached units — joining a bodyguard unit before the battle. |
