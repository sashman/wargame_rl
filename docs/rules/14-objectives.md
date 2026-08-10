# 14 — Objectives

**Objectives** are the locations both sides are trying to hold. Holding them is the
primary way to score.

## What an objective is

A mission states where objectives sit. Each location should coincide with a
[terrain area](13-terrain.md#placing-terrain) — that area *is* the objective, and is called
a **terrain objective**.

Where a location does not coincide with a terrain area, mark it instead with a flat
circular **objective marker**, 40 mm across, centred on the point.

| | Within range means | Measure |
|---|---|---|
| **Terrain objective** | The model is **within the terrain area**. | To and from the closest part of the area. |
| **Objective marker** | The model is within **3" horizontally and 5" vertically** of the marker. | To and from the closest part of the marker. |

Models may move through objective markers and may end a move on top of one.

## Level of control

At the start of the battle no objective is controlled by anyone.

At the **end of each phase and each turn**, for each objective, each player sums the
**Control Value (CV)** of all their models within range of it. That sum is that player's
**level of control**.

- The player with the **higher** level of control controls the objective.
- If the two are **equal**, the objective is not controlled by either player — unless it is
  [secured](#secured-objectives).

To gain control at all, a player needs at least one model with CV of 1 or more within
range. A model whose CV is `-` contributes nothing — which is exactly what
[suppression](01-core-concepts.md#suppression) does to a unit.

While one or more of a player's units are within range of an objective that player
controls, each such unit that contains a model with CV 1 or more is said to be
**controlling** that objective.

Note that control is re-evaluated at the end of *every* phase, not just when scoring. A
unit that walks off an objective in the Movement phase has already lost it by the end of
that phase.

## Secured objectives

Some rules let a player's army **secure** an objective. A secured objective stays under
that player's control **even after they have no units within range** — right up until the
end of a phase at which their opponent's level of control over it is *greater* than
theirs.

Equal levels of control are therefore not enough to break a secured objective; the
opponent must actually exceed the securing player.
