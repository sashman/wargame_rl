# 01 — Core concepts

The vocabulary and the primitives every other chapter builds on.

## Armies, units and models

Each player commands an **army** of **units**. A unit holds one or more **models**, and
those models act as one: they are set up together, move together, shoot together and
fight together.

A model's physical extent matters. Distances are measured to and from the closest part
of a model unless a rule says otherwise, so a large model reaches further than its
centre suggests.

**Friendly** means "belongs to your army"; **enemy** means "belongs to your opponent's".
A rule that says neither word applies to both. When a rule applies to a *unit*, it
applies to every model in that unit.

### Descriptors

A rule may qualify a unit — a *suppressed* unit, a *hidden* unit, a *visible* unit. The
qualifier attaches to the unit as soon as **one** of its models satisfies it. That does
not push the underlying property onto the other models; a unit containing one hidden
model is a hidden unit, but its other models are not thereby hidden.

### "A" means "one or more"

When a rule refers to *a* unit, *a* model or *a* objective without a count, read it as
"one or more". A condition of "within range of an objective" is satisfied by being
within range of two.

### "Other"

When a rule excludes the model or unit using it by saying *other*, two separate units
built from the same profile still count as other units to each other.

## Unit strength

A unit's **starting strength** is the number of models it contained at the start of the
first battle round. Several rules key off how far a unit has fallen from it. The
thresholds differ depending on whether the unit is one model or many:

| | Starting strength 1 | Starting strength 2+ |
|---|---|---|
| **Below starting strength** | Remaining Wounds are fewer than its W characteristic | Fewer models remain than its starting strength |
| **At half-strength** | Remaining Wounds are exactly half its W characteristic | Exactly half the starting number of models remain |
| **Below half-strength** | Remaining Wounds are fewer than half its W characteristic | Fewer than half the starting number of models remain |

## Sides

At every moment one player is the **active player** and the other is the **opposing
player**. The roles swap as the battle progresses.

- Between turns — at the start or end of a battle round — the player who takes the first
  turn each round is the active player.
- During a player's turn, that player is the active player, with two exceptions that
  hand the role temporarily to whoever owns the acting unit:
  - while a unit is moving, its controlling player is the active player until that move
    ends;
  - while a unit is shooting or fighting, its controlling player is the active player
    until those attacks are resolved.

### Whose rule is it

A rule belongs to the player whose army supplies it: army-wide rules, anything on a
unit's profile, and anything a mission grants them. Restrictions like "once per phase"
bind only the player who owns the rule, not both players.

A mission rule that is not used by either player — one that simply takes effect — is
resolved before any of the active player's rules, in whatever order the active player
picks.

### Resolving simultaneous rules

When several rules want to fire at the same moment, work through them in this order:

1. The active player's compulsory rules, in an order they choose.
2. The active player's optional rules that they choose to use, in an order they choose.
3. The opposing player's compulsory rules, in an order they choose.
4. The opposing player's optional rules that they choose to use, in an order they choose.

If resolving one rule creates the opportunity to use another at the same timing, that new
rule waits until everything already queued at that timing has resolved.

## Measuring distances

Distances are in inches. Either player may measure anything at any time.

Unless a rule says otherwise, measure to and from the closest part of the model in
question.

### Within and wholly within

These two phrases mean different things and the difference is load-bearing.

- **Within X"** — no more than X" away. A model is within X" if *any* part of it is. A
  unit is within X" if *any one* of its models is.
- **Wholly within X"** — *every* part of the model is within X". A unit is wholly within
  X" only if *every* model in it is wholly within X".

So a unit straddling the edge of a terrain area is *within* that area but not *wholly
within* it.

### Closest, and as close as possible

The **closest** model or unit is measured from whichever model or unit is using the rule.
If two are equally close, the controlling player of the model using the rule picks.

A rule that moves a model **as close as possible** to something requires the model to end
in **base contact** with it if its move is long enough to get there without breaking any
other restriction — coherency included. If the move is too short, get as close as the
move allows. A model already as close as possible does not move, but still counts as
having made the move.

Moving as close as possible to an *objective* means ending within range of it if the move
allows. A model already within range may move up to its full distance but must still end
within range.

### Base contact

Two models whose physical footprints touch are in **base contact**, and are as close as
possible to each other.

## Dice

Six-sided dice, written **D6**.

| Notation | Meaning |
|---|---|
| `2+`, `3+`, … | That result or higher succeeds |
| `1-3`, `4-6`, … | Any result inside the range triggers the rule |
| `2D6`, `3D6` | Roll that many dice and sum them |
| `D3` | Roll one D6, halve it, round up |
| `D6+1`, `2D6+3` | Roll, then add the stated value |

### Automatic success

When a roll is automatically successful, do not roll — advance straight to the next step
as though the needed result had come up. Nothing that keys off a specific die result
fires, so an automatic hit is never a critical hit.

### Re-rolls

- Re-rolling a summed roll (`2D6`, `3D6`) means re-rolling all of its dice.
- No die is ever re-rolled more than once.
- Re-rolls happen **before** modifiers.
- A re-rolled die is still a die roll, so anything triggered by a roll can be triggered by
  the re-roll.

### Rolling off

Both players roll one D6; the higher result wins. Re-roll ties.

### Reading multiple dice

- A **double** is any two dice showing the same result in one roll; a **triple** is any
  three.
- If a rule wants the highest or lowest die and several tie for it, the active player
  chooses which one counts.
- When a result is *treated as* or *set to* a value, anything that would trigger on
  actually rolling that value triggers — and the value may exceed 6.

## Resolve rolls

To make a **resolve roll** for a unit, its controlling player rolls 2D6. The roll
succeeds if the result is at least the unit's Resolve (Rv) characteristic. Otherwise it
fails. Whatever called for the roll says what success and failure mean.

## Suppression

A **suppression roll** is a resolve roll with a fixed consequence:

- On a success, the unit does not become suppressed.
- On a failure, the unit — and every model in it — becomes **suppressed**.

While a unit is suppressed:

- The Control Value of all its models becomes `-`, so it contributes nothing to holding
  objectives.
- It cannot be the target of its controlling player's tactics, where those exist.

Suppression represents a unit's nerve giving out under casualties or pressure. It is
checked most often in the [Command phase](08-command-phase.md), and it persists until a
later suppression roll succeeds.
