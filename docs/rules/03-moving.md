# 03 — Moving

How any move is physically executed. The [Movement phase](09-movement-phase.md) chapter
covers *which* moves are available and when; this chapter covers what a move is.

## Making a move

Every move type states who may make it, how far, and what conditions apply. When a unit
moves, its models are moved **one at a time**. Each model may be moved in a straight line
and/or rotated, as many times as its controlling player wants.

Unless a rule says otherwise, while a model moves:

- it may pass through friendly models;
- it may pass through any gap it physically fits through;
- it may **not** pass through enemy models;
- it may **not** cross the board edge;
- every stated *while moving* condition must hold.

### Straight lines and rotation

Moving in a **straight line** means moving horizontally across the board. Measure from
the same point on the model at the start and the end, and add that distance to everything
it has already travelled since its unit began the move. The running total may never
exceed the move type's maximum distance.

**Rotating** turns the model around its own centre, keeping it upright. Rotation costs
nothing — it does not count towards the distance moved.

Splitting a move into several legs is free; only the summed length matters. Two 3" legs
and one 6" leg both consume 6" of a 6" move.

### Different Move characteristics

If models in one unit have different M characteristics, each model gets its own maximum
distance for a move type sized by M. Every other restriction still applies to all of
them.

### Moving through models

An ability that lets a model "move through models" lets it pass through enemy models as
well as friendly ones. Every other restriction on the move it is making still applies.

Passing through an enemy unit's engagement range during a move does **not** make the
moving unit engaged. Only where it *ends* matters.

## Ending a move

Once every model that is going to move has moved, check all of the following:

- if the unit is on the board, it is in **coherency**;
- no model is left on top of another model, or partway through a surface of a terrain
  feature such as a wall or a ceiling;
- every stated *after moving* condition holds.

If any check fails, the move cannot be made: return every model to where it started. If
they all pass, resolve anything else the move type's *after moving* section says, and the
move ends.

Because coherency is checked at the end, a requirement to move "if able" never forces a
unit out of coherency — if the only way to satisfy the requirement breaks coherency, the
requirement was not able to be met.

## Setting up

Some rules **set up** a unit rather than moving it — deployment being the common case.
Place the unit's models so that:

- the unit is in coherency;
- the unit is unengaged;
- every other stated requirement holds.

If not every model can be placed legally, the unit is removed from the board and returned
to wherever it came from.

When setting up is itself a move type and the placement fails, the unit **has not been
selected to move**. It may therefore be selected again later in the phase — to try again,
or to remain stationary instead.

### Redeployment

A rule that redeploys a unit removes it from the board and deploys it again from scratch,
using every deployment rule that unit has.

## Coherency

A unit of more than one model must be set up in, and end every move in, **coherency**.
A unit is in coherency while every model in it is:

- within **2" horizontally and 5" vertically** of at least one *other* model in the unit,
  **and**
- within **9" horizontally and 5" vertically** of *every other* model in the unit.

The first condition chains models together; the second caps the unit's overall spread.
A unit must form a single connected group — it may not split into two clusters that each
satisfy the 2" rule internally.

### Regaining coherency

In the End of Turn step of each player's turn, any unit on the board that is out of
coherency loses models — its controlling player removes them one at a time until
coherency is restored. Models removed this way are **destroyed**, but they do not trigger
anything that fires when a model is destroyed.

## Engagement

A model's **engagement range** is everything within 2" horizontally and 5" vertically of
it.

While a friendly model is within engagement range of one or more enemy models, those
models — and their units — are **engaged** with each other. A unit containing no engaged
models is **unengaged**.

Engagement is what melee keys off. It also gates most shooting: an engaged unit generally
cannot shoot, and generally cannot be shot at.
