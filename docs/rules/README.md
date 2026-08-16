# Rules Reference

This directory is the rules specification for the game `wargame_rl` simulates: a
two-player tabletop miniatures battle fought on a rectangular board, resolved with
six-sided dice, and scored by holding ground.

It is a **self-contained specification**, written for this project. It is not a
transcription of any published product, it names no product, publisher, faction or
model, and it is the only rules authority the repo recognises. Where the environment
disagrees with it, the environment is either behind or deliberately simplified — and
[`implementation-status.md`](implementation-status.md) says which.

## Reading order

| # | File | Covers |
|---|---|---|
| — | [Glossary](00-glossary.md) | Every term and characteristic used below |
| 01 | [Core concepts](01-core-concepts.md) | Armies, units, models, sides, measuring, dice, resolve and suppression |
| 02 | [Unit profiles](02-unit-profiles.md) | Characteristics, weapon profiles, modifiers, keywords |
| 03 | [Moving](03-moving.md) | How a move works, setting up, coherency, engagement range |
| 04 | [Making attacks](04-making-attacks.md) | Selecting weapons and targets, gathering attack dice |
| 05 | [Attack sequence](05-attack-sequence.md) | Hit → wound → save → damage |
| 06 | [Visibility and damage](06-visibility-and-damage.md) | Line of sight, piercing damage, backfire rolls |
| 07 | [Battle round](07-battle-round.md) | Round and turn structure, ordering of simultaneous rules |
| 08 | [Command phase](08-command-phase.md) | Suppression checks |
| 09 | [Movement phase](09-movement-phase.md) | Remain stationary, normal, advance, fall back |
| 10 | [Shooting phase](10-shooting-phase.md) | The four shooting types |
| 11 | [Charge phase](11-charge-phase.md) | Declaring and making charges |
| 12 | [Fight phase](12-fight-phase.md) | Pile in, fight, consolidate |
| 13 | [Terrain](13-terrain.md) | Terrain categories, movement, cover, hidden, obscuring, solid |
| 14 | [Objectives](14-objectives.md) | Level of control, securing |
| 15 | [Missions and scoring](15-missions-and-scoring.md) | Setup, deployment, victory points, ending the battle |
| 16 | [Ability reference](16-ability-reference.md) | Every weapon and unit ability, alphabetical |

Three companion files sit alongside them:

- [`constants.yaml`](constants.yaml) — every number in one place, in inches, so tests can
  assert against the rules rather than against a magic literal.
- [`implementation-status.md`](implementation-status.md) — what the environment
  implements, what it diverges on, and what is absent. This is the roadmap for the next
  implementation phase.
- [`primitives.md`](primitives.md) — the chapters above read *across* rather than down: the
  six mechanisms every named rule is composed from, and the grammar that composes them, so
  a rule can be authored as data instead of implemented as a function. Read it before
  building a mechanic, and apply its four fragility tests to the design.

## Conventions

- **Distances are in inches (`"`).** The environment plays on a discrete grid of cells;
  translating between the two is an environment concern, recorded in the gap map.
- **`within` and `wholly within` are not interchangeable.** *Within* X" means "no further
  than X"" — one model of a unit, or one part of a model, being close enough is enough.
  *Wholly within* means every part, or every model, is close enough. See
  [Core concepts](01-core-concepts.md#within-and-wholly-within).
- **`unmodified`** describes a die result before any modifier is applied (but after any
  re-roll). Rules that trigger on a specific result almost always trigger on the
  unmodified one.
- Move and shoot types are specified as a block — *eligible if*, *effect*, *before*,
  *while*, *after*. Every condition in a block must hold; if one cannot, the type cannot
  be selected.
- Keywords are written in `SMALL CAPS`, weapon abilities in `[BRACKETS]`.

## Deliberately out of scope

The following are part of the wider game but are not modelled here, because the
environment cannot reach them in this phase. They are listed so the omission reads as a
decision rather than an oversight:

- **Transport models** — carrying units, embarking, disembarking.
- **Aircraft** — units that only ever arrive and leave.
- **Attached units** — leaders and support models joining a bodyguard unit.
- **Strategic reserves** — units held off the board and arriving later.
- **Flight and surge moves** — moving over models and terrain, and reactive moves.
- **Force building** — points, rosters, army-wide rules, unit limits.
- **The command-resource economy** — command points and the tactics bought with them.
- **Battlefield actions** — units spending a turn completing a task instead of fighting.

If any of these becomes reachable, add the chapter here first and the implementation
second.
