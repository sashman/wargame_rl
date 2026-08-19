# The two seats are not the same game

*2026-08-19. Found while building the Elo rating subsystem (phase 04, waves 1–2).*

## The question

A rating puts policies on one scale by playing them against each other. That is
only sound if the two seats are equivalent — otherwise every rating measures the
seat as much as the policy. Nothing in this repo had ever checked, because every
score here is quoted with the policy on the **player** seat.

So `just measure-seat-parity` plays one policy from **both** seats over the
balanced four legs (zone × first mover). Its rating difference is zero by
construction, so whatever margin survives is the seat itself. On a fair
scenario the aggregate is zero.

## The answer: no, by 24.6 vp

`configs/golden/25v25_shooting_opponent.yaml`, `squad_march_shoot` on both
seats, 30 layouts × 4 legs:

| Leg | mean margin | wins |
|---|---|---|
| zone 1, A first | **−40.8** | 0.40 |
| zone 1, B first | −27.5 | 0.40 |
| zone 2, A first | −15.2 | 0.50 |
| zone 2, B first | −14.8 | 0.57 |
| **aggregate** | **−24.6 ± 9.4 (1 se)** | |

Same policy, same board, same seeds, and the player seat loses by 24.6 vp at
2.6 standard errors. That is larger than most effects this project has ever
measured, and larger than the entire gap the agent spent months closing.

**Every number in this repo is measured from the player seat**, including the
bar `squad_march_shoot` at +38.0 and the agent's +2.6 against it.

## What it is not

Each of these was ruled out by measurement, not by reading.

| Ruled out | How |
|---|---|
| **Movement, placement, VP mechanics in general** | `squad_march`, which does not shoot, is **fair on the same config**: +7.2 ± 9.1. The asymmetry needs shooting to appear. |
| **Terrain, line of sight, cover** | Deleting terrain entirely makes it **worse and more consistent** (all four legs negative, ≈ −40). |
| **The known LOS asymmetry bug** | Real but **unbiased**: over 18 750 army-pair sightlines on 30 layouts, 52 disagree (0.28%) — 29 favouring the player, 23 the opponent. |
| **Policy identity / the mirror** | `contest_and_spread` loses from the player seat in **all four legs** (≈ −37). Two unrelated shooting policies, same direction. |
| **Weapons, action masks** | Both armies carry the identical `*rifle` anchor; `_opponent_action_mask` is a faithful mirror with `shoots` correctly derived. |

## What it is, part one: the second mover shoots after both armies have closed

The clock is IGOUGO — each side takes all its phases, then the other. So within
a round the first mover shoots at a gap that is one enemy move wider than the
gap the second mover shoots at.

In a *closing* engagement that is a large opening-rounds advantage. Player as
first mover, per round, 8 layouts:

| round | player shots | opponent shots | player alive | opponent alive |
|---|---|---|---|---|
| 3 | 1.69 | **4.25** | 22.62 | 24.00 |
| 4 | 5.62 | 6.44 | 19.00 | 20.75 |

Flip `turn_order` so the player moves second and it reverses:

| round | player shots | opponent shots | player alive | opponent alive |
|---|---|---|---|---|
| 3 | 4.00 | 5.31 | 21.50 | 22.50 |
| 4 | **6.19** | 5.25 | 18.62 | 18.12 |
| 8 | 3.33 | 2.13 | **10.53** | 9.07 |

The deficit is paid in the first three rounds and compounds through the whole
firefight.

**This is a property of the game, not a bug** — and it is exactly what the Elo
model's `h_turn` term exists to absorb. Measured on a movement-only scenario,
where nobody shoots, going first is instead worth **+59.2 Elo**: whoever
reaches an objective first holds it. So the first-mover advantage **changes
sign** when shooting is switched on. That alone is worth knowing before
designing a scenario.

## What it is, part two: the player out-fights and under-scores

Averaging over both turn orders cancels the effect above. What is left:

| | player | opponent | difference |
|---|---|---|---|
| shots fired | 62.90 | 59.83 | **+3.07** |
| kills | 19.05 | 16.90 | **+2.15** |
| fraction alive at end | 0.32 | 0.24 | **+0.09** |
| **victory points** | **108.00** | **117.38** | **−9.38** |

**The player wins the firefight on every combat measure and still loses on
victory points.** More shots, more kills, more survivors — fewer points. In the
per-round trace the player holds 1.00 objectives at the end against the
opponent's 1.67, with more models alive.

So the residual is positional or in scoring, not in combat resolution.

## What it is, part three: each side is scored at a different point in the round

VP is awarded in `_on_before_advance` when a side leaves its **command** phase —
the first phase of its own turn. Tracing the clock on the golden config with the
player moving first:

```
step 0: player movement (r1)      SCORE plr r1
step 1: player shooting (r1)      SCORE opp r1
                                  [opponent moves, opponent shoots]
                                  SCORE plr r2
```

So each side is scored **before its own move and immediately after the enemy's**
— structurally symmetric, and the event *counts* are equal (17.8 against 17.5
per episode; 20.0 against 20.0 on another leg). But the two scoring instants sit
at different points in the move sequence: the player is scored at move-count
`(N, N)`, a fully contested board where the enemy has just arrived, while the
opponent is scored at `(N, N-1)`, before it has committed that round.

Measured — what each side controls **at its own scoring instant**, 8 layouts:

| leg | player controls, at player's scoring | opponent controls, at opponent's scoring |
|---|---|---|
| zone 1, A first | 1.085 | **1.314** |
| zone 1, B first | 1.168 | 1.273 |
| zone 2, A first | 1.194 | 1.337 |
| zone 2, B first | **1.381** | 1.136 |

**This closes the arithmetic.** The player earns 5.37 VP per scoring event and
the opponent 6.54 — a gap of 1.17 VP, which at 5 VP per objective is 0.23
objectives, exactly the 1.085 against 1.314 difference on that leg. Over ~18
events an episode that integrates to roughly the seat gap observed.

It also predicts the *ordering*. Zone 2 with the opponent moving first is the
only leg where the player controls more at its own scoring instant (1.381
against 1.136) — and it is the only leg the player wins (+16.0). The worst leg
on this measure, zone 1 with the player first, is the worst leg on margin
(−40.8).

**So the residual is the scoring cadence, not positioning.** Both sides are
measured at the moment least favourable to them, but "least favourable" is not
the same board for the two seats, because the command phases are half a turn
apart in the interleaving. The player is measured on a fully contested board;
the opponent on one it has not yet committed to.

⚠ **This is n=8 on one config and the leg-level numbers are noisy** (the
ordering agrees with the margins on the extremes and swaps the middle two). The
mechanism and the arithmetic agree, but it should be confirmed at higher n
before anything is changed on the strength of it.

## Incidental finding

`WargameModel.advanced_this_turn` is **never set to `True` anywhere in the
codebase**. It is initialised `False`, reset to `False`, serialised, restored —
and read by both shooting masks as `player_advanced`, where it gates whether a
model that advanced may fire. The rule it implements is therefore inert. This
is the "settable but inert" failure `CLAUDE.md` warns about, sitting in the
shooting path.

## What this does and does not support

**Supported.** On this scenario the two seats are not equivalent; the gap is
shooting-dependent; it is not terrain, sight, cover, weapons, masks or policy
identity; a large part of it is a genuine second-mover shooting advantage that
reverses the movement-only first-mover advantage; and after averaging that out,
the player seat wins the firefight and loses the scoreboard.

**Not supported.** Any claim about *why* the player under-scores given a better
board. Any claim that the published baselines are wrong by a specific amount —
they were all measured from the same seat against the same kind of opponent, so
they remain comparable *to each other*; what is now in question is whether that
common frame is neutral.

**Not measured.** Whether the effect survives on other configs, on the real map
pool, or against a trained agent rather than a script.

## Consequence

Rating `configs/golden/25v25_shooting_opponent.yaml` is **blocked** until this
is explained — a rating table there would silently encode 24.6 vp of seat. The
gate is:

```
just measure-seat-parity <env_config> [policy] [n_layouts]
```

It should be run on any config before it is rated, and it costs four legs.
