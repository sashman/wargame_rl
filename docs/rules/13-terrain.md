# 13 — Terrain

Terrain is what makes position mean something. It restricts movement, breaks line of
sight, and hands out defensive benefits.

Two words do different jobs throughout this chapter:

- a **terrain feature** is a physical obstacle — a wall, a building, a wood;
- a **terrain area** is a bounded region of the board containing one or more features.

**Most terrain rules key off the area, not the feature.** Cover, hidden and obscuring are
all area rules; movement and solid are feature rules.

## Placing terrain

Before the battle, set up terrain by any mix of these:

- mark a boundary (a base or a mat) and place one or more features wholly within it;
- place a single feature directly on the board;
- place two or more features directly on the board so that together they enclose a
  region.

In each case the region occupied by the boundary or the feature is a **terrain area**. A
mission's layout may define where areas go and how big they are; otherwise the players
agree before the battle.

Features sharing one terrain area may belong to different categories.

## Terrain categories

Every feature belongs to one of three categories. The category decides how the feature
affects movement and visibility.

| Category | Character | Typical features |
|---|---|---|
| **Exposed** | Crossed without hindrance; offers no protection. | Craters, wire, scattered rubble. |
| **Light** | Can grant cover, but does not slow anyone down or block sight. | Barricades, low walls, statuary. |
| **Dense** | An obstacle even to large models; can hide a whole unit. | Buildings, ruins, containers, woods. |

Dense features carry the [solid](#solid) rule.

Board balance is a function of dense features. Too few and shooting armies dominate;
enough space must also be left around them for large models to manoeuvre.

## Terrain and movement

| Category | Who may pass through |
|---|---|
| **Exposed** / **Light** | Every model, horizontally and vertically. |
| **Dense** | `INFANTRY`/`BEASTS`/`SWARM`/`MOBILE` models may pass horizontally. `INFANTRY`/`BEASTS`/`SWARM` models may also pass vertically. |

Any **other** model may pass horizontally through a dense feature only where every section
its path crosses is **2" or less in height**. Where a section is taller, the model must
climb it — moving vertically to ascend and descend. Such a model may not pass through
floors or ceilings, and may not end its move on any surface above ground level.

### Moving vertically

While a model ascends or descends a feature:

- it must stay within ½" horizontally of that feature;
- the distance travelled up **and** the distance travelled down both count against its
  move.

### Ending a move on terrain

Any model may be set up or end a move on the **ground level** of a feature.

A model may be set up or end a move on a surface **above** ground level only if:

- it has one of `INFANTRY`/`BEASTS`/`SWARM`/`FLY`/`MONSTER`; **and**
- it ends up stable, with no part of it overhanging the outer edge of that surface.

### Solid features and movement

A model may not end a move with any part of it inside an enclosed part of a
[solid](#solid) feature that is **3" or less from ground level** — not through a door, not
through a window. Without this, a protruding part of a model could be used to see out of
somewhere the solid rule says it cannot see out of.

## Terrain and visibility

Four rules layer on top of raw [line of sight](06-visibility-and-damage.md#visibility):
**cover**, **hidden**, **obscuring** and **solid**.

### Cover

Each time a **ranged** attack targets a unit, that unit has **cover** against that attack
if **every** model in it satisfies at least one of:

- the model has `INFANTRY`/`BEASTS`/`SWARM` and is **within a terrain area**;
- the model is **not fully visible** to the attacking model, because of one or more
  intervening terrain features and/or one or more intervening
  [obscuring](#obscuring) terrain areas.

**Effect: worsen the Ranged Skill of that attack by 1.**

Two consequences worth internalising:

- Cover is all-or-nothing at the **unit** level. One model of a unit standing in the open
  denies cover to the whole unit.
- The first condition needs only *within* a terrain area, not *wholly within* — but every
  model must satisfy it.

### Hidden

A model is **hidden** while all of:

- it has `INFANTRY`/`BEASTS`/`SWARM` and is within a terrain area containing one or more
  **light or dense** features;
- its unit made **no ranged attacks this turn or the previous turn**.

While a model is hidden it is visible **only** to enemy models within its **detection
range**. Unless stated otherwise, detection range is **15"**.

During the first turn, "did not happen during the previous turn" is true by default — so a
qualifying unit starts the battle hidden.

Hidden is what makes shooting a commitment: firing strips the concealment for this turn
and the next.

#### Gone to ground

A hidden model inside a solid feature gets more. A model has **gone to ground** while all
of:

- it is hidden;
- it is not fully visible to the attacking model because of one or more intervening
  **solid** features;
- its unit made no ranged attacks this turn or the previous turn.

While a model has gone to ground, **subtract 3" from its detection range** — 12" by
default.

A unit that shot this turn or last cannot go to ground, whatever other abilities it has.

### Obscuring

A terrain area containing one or more **light or dense** features is an **obscuring**
terrain area.

If **every** line of sight between two models crosses one or more obscuring terrain areas,
those two models are **not visible to each other**. Obscuring areas that one or both of
the models are *inside* are excluded from that test — you can see out of, and into, the
area you are standing in.

### Solid

Dense features are **solid**. Line of sight cannot be drawn across any enclosed gap in the
surface of a solid feature that is **3" or less from ground level**.

That is: doors, windows, shell holes and the small gaps between adjacent features do not
let you see a model sheltering at ground level. Above 3", line of sight is traced
normally — a model on an upper floor can be seen and can see out.

Missions may adjust the height at which this rule applies.
