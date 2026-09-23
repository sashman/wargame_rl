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

## Amendment 1 — written 2026-09-23 10:16: both arms read at the cap — EX1 NULL, the A3 check COSTS; keeping improves, arrival collapses

Read after every seed's log carried its `rounds 122880` line and no trainer
remained (chain `ex_reads.sh`; directories resolved from the config stem);
n=100, seeds 700000+, greedy, `last.pt` after exit. Launched 22:52 on
2026-09-22; the last trainer exited before 03:00. Wandb `curriculum-cm-ex`:
ex1 5fh8r1r8 / 9phybw9r / 2ybt8e53, ex1a3 i7sljhl9 / 79pb3imk / c5z14hpv.
Every run recorded a greedy episode at every checkpoint.

### EX1, the half-step (#400): NULL

| rounds | EX1 s1 / s2 / s3 | CM1-R1 | leave | hold mode / hold declaration | follow | complete |
|---|---|---|---|---|---|---|
| 40,960 | 0.000 / 0.000 / 0.010 | 0.040 / 0.140 / 0.030 | 0.71 / 0.53 / 0.53 | 0.23 / 0.39 / 0.28 · 0.00 / 0.00 / 0.18 | 0.70 / 0.68 / 0.74 | 0.28 / 0.20 / 0.36 |
| 81,920 | 0.030 / 0.000 / 0.000 | 0.300 / 0.110 / 0.090 | 0.61 / 0.56 / 0.38 | 0.31 / 0.27 / 0.32 · 0.00 / 0.00 / 0.23 | 0.75 / 0.76 / 0.79 | 0.47 / 0.35 / 0.46 |
| 122,880 | **0.100 / 0.030 / 0.000** | 0.340 / 0.100 / 0.060 | **0.38 / 0.60 / 0.52** (bar 0.14) | 0.33 / 0.28 / 0.32 · **0.00 / 0.00 / 0.29** (bar 0.38 · 0.67) | 0.74 / 0.71 / 0.76 | 0.56 / 0.41 / 0.44 |

PASSES needed the leave share ≤ 0.14 on 3/3 and success ahead of CM1-R1
on 3/3; KEEPS the leave clause alone. Neither: **NULL**. Success is behind
CM1-R1 on s1 by 3.7 SE and level on the other two; held 3.04 / 2.05 / 2.41,
turns 9.9–10.0 of ten. Persistence 0.92–0.94, no unit without a plan.
Sampled within 4 vp of greedy. Panels: member explained variance
**0.95–0.96**, clip 0.35–0.37, displacement entropy 1.7–1.8 nats,
declaration entropy 0.05 nats. The flag ablation: s1 0.10 → 0.02 under
BLANK (held 3.04 → 2.49), a first drop on the half-step; s2 and s3 at the
floor.

### The A3 check (#401): COSTS

| rounds | A3 check s1 / s2 / s3 | `a3_cm` | leave | hold / decl | turns |
|---|---|---|---|---|---|
| 40,960 | 0.430 / 0.280 / 0.450 | 0.520 / 0.350 / 0.240 | 0.48 / 0.16 / 0.44 | 0.23–0.28 · 0.00–0.04 | 7.1 / 7.8 / 7.1 |
| 81,920 | 0.500 / 0.300 / 0.690 | 0.890 / 0.650 / 0.750 | 0.11 / 0.32 / 0.37 | 0.18–0.29 · 0.00 / 0.33 / 0.00 | 7.5 / 7.5 / 7.6 |
| 122,880 | **0.420 / 0.270 / 0.660** | 0.820 / 0.850 / 0.930 | **0.18 / 0.15 / 0.22** (`a3_cm` 0.41 / 0.49 / 0.40) | 0.20 / 0.27 / 0.26 · 0.00 ×3 | **7.88 / 7.90 / 7.93** (`a3_cm` 6.02 / 5.65 / 5.59) |

Below `a3_cm` on every seed by 6–9 SE: **COSTS**. The leave clause it
would have passed (below on 3/3, at about half). Held 3.37 / 3.10 / 3.62
against 3.72–3.92. The ablation: s3 0.66 → **0.03** under BLANK (held 3.62
→ 2.37) and 0.34 under MISDIRECT — a policy that reads the marking on A3's
shape, the second on the ladder; s1 0.42 → 0.42, s2 flat. Panels: member
explained variance 0.95–0.97.

### The mode-aware walk-off probe (n=10, the scoring rule's "inside")

`drafts/walkoff_probe_mode.py`, written for this read: every movement
decision of a body inside its unit's COMMITTED objective while the unit
is in HOLD mode, and what the step paid.

| policy | decisions | stood still | moved, kept | left | pay: stood · kept · **left** |
|---|---|---|---|---|---|
| bar, half-step | 52 | 0.35 | 0.54 | 0.12 | +0.017 · +0.021 · **+0.000** |
| EX1 s1 / s2 / s3 | 204 / 120 / 131 | 0.04 / 0.00 / 0.00 | 0.62 / 0.48 / 0.50 | **0.33 / 0.52 / 0.50** | +0.031 · +0.024–0.026 · **+0.000 ×3** |
| bar, A3 | 22 | 0.36 | 0.64 | 0.00 | +0.031 · +0.039 · n/a |
| A3 check s1 / s2 / s3 | 59 / 96 / 94 | 0.00 / 0.00 / 0.01 | 0.88 / 0.88 / 0.91 | **0.12 / 0.12 / 0.07** | n/a · +0.033–0.044 · **+0.000 ×3** |

**The reward is as designed**: leaving in hold mode paid exactly zero on
every seed of both arms, staying inside paid 0.024–0.044 a step. ⚠ The
first pass of this probe read leaving at +0.002 to +0.025 because it
tested "inside" by distance to the objective's centre while the reward
uses the scoring rule from the base edge; a body stepping to the rim reads
gone to one and inside to the other. One definition of "on an objective"
(2026-08-22) applies to probes too.

### Reading

1. **Keeping is bought where the plan is held.** In hold mode the leave
   share is 0.07–0.12 on A3 (`a3_cm` 0.40–0.49 on the same readout's
   all-mode count) and 0.33–0.52 on the half-step, against 0.56–0.60 for
   the same plan under the keyed reward. Standing still is still never
   chosen (0.00–0.04; the hold declaration 0.00 on five seeds of six):
   the policy shuffles inside, which the hold term pays the same, and on
   the half-step the shuffle leaks bodies out of the disc.
2. **Arrival collapsed, and the reason is in the design.** With the
   outcome terms on the planning stream and no planner to receive them,
   nothing in the soldiers' training says WHEN to arrive. The travel term
   is a potential — the same total whether the squad arrives in five turns
   or eight — and the success bonus that used to scale with the rounds
   left was the only speed signal the members had. A3's turns went 5.6–6.0
   → 7.9 and success 0.82–0.93 → 0.27–0.66 on a plan that was already good.
   The member critic's explained variance of 0.95–0.96 is the same fact:
   a return made only of potentials is easy to predict and says nothing
   about the outcome.
3. **On the half-step the squads that never complete are the empty
   objectives.** Complete 0.41–0.56, one or two objectives empty in every
   census episode on s2 and s3, persistence 0.92–0.94 — the plan is stable
   and the approach is not finished, not walked off.

### The revision this asks of #384 (proposed; Sash decides)

The plan needs its **completion inside it**: a one-off ARRIVAL pot paid to
the squad's members on the execution stream when its mode flips from
approach to hold (the approach task completed), sized by what the
objective is worth under the mission's VP rule (D7's "taken pot", on the
member stream because it is the unit's own completion, not the army's
outcome). Paid once at the flip and discounted per round, it is worth more
the sooner the squad arrives — the urgency the terminal bonus used to
carry, without the lump. The hold term stays; the leave share says it
works. Not proposed: putting the outcome terms back on the member stream
(the lump broke the critic twice) or a planner (the soldiers do not yet
keep on the half-step).
