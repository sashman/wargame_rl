# 15 — Missions and scoring

How a battle is set up, scored and ended.

## Setting up a battle

Resolve these steps in order before the first battle round:

1. **Build forces.** Each player assembles an army. (Force building itself is
   [out of scope](README.md#deliberately-out-of-scope) — for this project, an army is
   whatever the environment config declares.)
2. **Determine mission.** Fixes the objectives, how they score, and any extra rules.
3. **Determine layout.** Fixes where terrain areas and objectives go, and where the
   deployment zones are.
4. **Create the battlefield.** Set up the terrain areas, then the terrain features on
   them, as the layout shows. The standard board is **44" × 60"**.
5. **Determine attacker and defender.** Agree which board edges correspond to the
   attacker's and defender's edges on the layout, then roll off; the winner chooses which
   role to take.
6. **Select secondary objectives**, if the mission uses them.
7. **Declare battle formations.** Both players secretly note any pre-battle decisions,
   then reveal.
8. **Deploy armies.** Players alternate setting up units one at a time, each **wholly
   within** their own deployment zone, starting with the defender. When one player has
   finished, the other sets up everything they have left.
9. **Redeploy units.** Any rule that repositions a unit after both armies are deployed
   resolves here. Players alternate, starting with the attacker.
10. **Determine first turn.** Roll off; the winner takes the first turn — and takes the
    first turn in **every** battle round.
11. **Resolve pre-battle rules.** Players alternate resolving any pre-battle abilities,
    starting with whoever takes the first turn.
12. **Begin the battle.** The first battle round starts.

### Deployment zones

Each player has a deployment zone defined by the layout. Units deployed in step 8 must be
set up **wholly within** it — one model of a unit hanging outside is not enough.

## Scoring

Score is measured in **victory points (VP)**.

VP are scored at points the mission names. The two that matter structurally are the **end
of the Command phase** and the **end of the turn** — both have a mission sub-step that
runs after all non-mission rules at that timing. See
[The battle round](07-battle-round.md#end-of-turn-step-ordering).

Objective scoring reads the **level of control** as defined in
[Objectives](14-objectives.md#level-of-control).

### Caps

VP from each source is capped. Anything scored above a cap is discarded, not carried:

| Source | Per battle round | Total |
|---|---|---|
| Primary objectives | 15 VP | 45 VP |
| Secondary objectives | 15 VP | 45 VP |

A mission that awards more than the per-round cap in a single round simply loses the
excess — which makes holding objectives *consistently* worth more than holding them
overwhelmingly once.

## Ending the battle

The battle ends after **five battle rounds** have been completed, unless the mission says
otherwise.

**A player who has no models left does not lose immediately.** Both players keep taking
their turns until the battle ends — so an army that has been wiped out still cedes
objectives for the remaining rounds, and the surviving player keeps scoring.

## Determining the victor

At the end of the battle, the player with the most VP wins. Equal VP is a draw.
