# Pre-registration: the execution phase of the commitment layer — the soldiers paid by the plan alone (EX1 on the half-step, and the A3 check)

**Written 2026-09-22 22:51, before any training number exists.** Parent
#384; goal set by Sash 2026-09-22 ("make the soldiers execute a given
plan"); build #399; arms #400 (EX1, the five-objective half-step) and #401
(the A3 check). Branch `feature/commitment-revision` (PR #396, stacked on
#388); never merged to `main`.

## The question

Every per-model setting on the five-objective half-step shares one census:
the soldiers arrive and walk off, standing still on 0–2% of their
decisions on an objective. The Stage 1 read and the diagnosis with Sash
put two causes on it. The reward paid for approaching and never for
keeping — the staying term we ran paid 0.015 for ending inside against
0.018 for moving around inside and −0.003 for leaving, no reason to
prefer standing — and the success bonus of 5.0 at the end sat in the same
reward as that 0.015, so PPO's normalisation made the keep signal noise.

The execution phase pays a soldier by its squad's plan and by nothing
else. The plan is the committed objective plus a weight per task,
`(approach, hold)`, set by the environment's rule: `(1, 0)` until the squad
first has a member inside its objective (read at a turn close), then
`(0, 1)`, reset to `(1, 0)` on a new target. A soldier's per-decision
reward is approach × the travel term plus hold × the staying term, each
keyed to the committed objective. In hold mode the travel term is silent
and leaving pays exactly zero; in approach mode standing inside pays
nothing until the plan flips. The close's outcome terms (coverage, the
success bonus) go to the planning stream, which no soldier sees. **Do the
soldiers now keep their objectives?**

## The one change per arm

| arm | config | the one change | comparator |
|---|---|---|---|
| **EX1** (#400) | `a5_points_plan.yaml` (= `a5_points_cm_r1` + `execution: plan` + `objective_stay` 1.0) | the plan-weighted execution | CM1-R1 (#394): the same plan under the keyed execution; the staying arms beside |
| **A3 check** (#401) | `a3_plan.yaml` (= `a3_cm` + the same) | the same | `a3_cm`: the same plan under the keyed execution |

Both from scratch, three seeds (1 / 2 / 3), 122,880 rounds at 128 rounds
per update (`--num-rollout-envs 4 --rollout-rounds 32`), `--ent-coef
0.003`, eval and checkpoint every 512, a greedy episode recorded at every
checkpoint, Wandb group `curriculum-cm-ex`. Scored greedy with the passive
fingerprint, n=100 on seeds 700000+, sampled beside greedy at the end. Every
read gated on each seed's log carrying its round line, the final on zero
trainers; run directories resolved from the config stem. Not extended past
the cap.

## The comparators' rows

| | 40,960 | 81,920 | 122,880 | leave at the cap | follow | persist |
|---|---|---|---|---|---|---|
| CM1-R1 (`a5_points_cm_r1`) | 0.040 / 0.140 / 0.030 | 0.300 / 0.110 / 0.090 | 0.340 / 0.100 / 0.060 | 0.56 / 0.60 / 0.60 | 0.73–0.74 | 0.91–0.92 |
| S (`objective_stay` 0.5, one stream) | 0.040 / 0.010 / 0.020 | 0.170 / 0.150 / 0.190 | 0.210 / 0.190 / 0.120 | 0.48–0.67 (walk-off probe) | — | — |
| `a3_cm` | 0.520 / 0.350 / 0.240 | 0.890 / 0.650 / 0.750 | 0.820 / 0.850 / 0.930 | 0.41 / 0.49 / 0.40 | 0.90–0.94 | 0.94–0.97 |

## The bars, measured first (n=100, seeds 700000+, per-model facade)

| config | `squad_march_take` | success | turns | held | persist | claim | follow | leave | hold mode / hold declaration |
|---|---|---|---|---|---|---|---|---|---|
| `a5_points_plan.yaml` | the bar | **1.000** | 6.71 | 5.00 | 0.97 | 1.21 / max 3 | 0.97 | **0.14** (370) | **0.38 / 0.67** (1042) |
| `a3_plan.yaml` | the bar | **1.000** | 5.28 | 4.00 | 0.99 | 1.02 / max 3 | 1.00 | **0.00** (258) | 0.33 / 0.32 (311) |

The bar's rows are read against the environment's assignment, so its
leave share on the half-step is 0.14 (its own re-plan occasionally moves a
member off the objective the environment assigned); the pass bound below
is that number. The hold-mode share is what the arrival rule gives the
script's own walk; the hold-declaration share is how often the script
stands a whole squad once it is holding.

## Criteria

**EX1 (#400), the phase's gate, from the goal.**

- **PASSES** — the leave share on the committed objective ≤ the bar's 0.14
  on 3/3 at 122,880 AND success ahead of CM1-R1's seed by more than two
  binomial SE on 3/3.
- **KEEPS** — the leave clause on 3/3 without the success clause.
- **NULL** — otherwise.

Primary readouts before success: leave; the hold-mode share and the
hold-declaration share (`hold a / decl b`); follow-through; persistence.
Then success at 40,960 / 81,920 / 122,880 against CM1-R1 at matched rounds;
the flag ablation on every read; the walk-off probe (stayed / moved-and-left
and the step pay, which under the plan must read leaving at 0.000); the
census; the member panel; sampled beside greedy at the end. At n=100 a
leave-share difference of 0.05 clears two SE at ~700 moves from inside; a
success difference of 0.13 clears two SE near 0.2.

**The A3 check (#401).** HOLDS if success at the cap is not below `a3_cm`'s
seed by more than two SE on any seed AND the leave share is below
`a3_cm`'s on 3/3; COSTS if success is below by more than two SE on any
seed; NULL otherwise. Turns reported.

## What follows each reading (D9)

- EX1 **PASSES**: the soldiers execute a given plan; the planner is built
  next on this reward, on A3's shape first, with the B6 counterfactual
  credit proposed in the Stage 1 amendment.
- EX1 **KEEPS**: keeping is bought and does not add up to success on the
  half-step — the diagnosis reads the census for which objective stays
  empty and the approach phase for whether the walk slowed.
- EX1 **NULL**: keeping is not a reward problem on this trainer at any
  scale the plan can express; the diagnosis reads the hold-declaration
  share (was standing ever chosen?) and the walk-off probe's pay (was
  leaving really zero and staying really paid?), and names the action
  side (one STAY column among fifty) before anything else.
- The A3 check **COSTS**: the hold weight is too high for the approach
  phase, read from turns; the diagnosis names the ratio.

## What I expect (a guess, written so it can be wrong)

EX1 reads **KEEPS** with success in 0.3–0.5: the leave share falls under
0.14 on 3/3 because leaving is finally worth less than staying by a margin
the normalised advantage can see, the hold-declaration share rises to
0.3–0.6, and success is bounded by the approach phase — squads that arrive
late or share an objective — rather than by the walk-off. The A3 check
reads **HOLDS** at 0.85–0.95 with turns within 0.3 of `a3_cm`.
