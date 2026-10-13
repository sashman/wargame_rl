# Curriculum — the commitment layer, Stages 0 and 1 (#384)

The commitment layer before the plan question: Stage 0 (the environment writes each squad's objective), its revision (the legibility rung), Stage 1 (a learned commitment head) and the execution phase. Each file was a pre-registration with its reads appended as amendments; all four are kept word for word.

## Contents

- [Pre-registration: the commitment layer's Stage 0 on the ladder — the assignment given as an observation, A0 → A3 and the CM1 gate](#2026-09-21-curriculum-commitment-stage0-preregistration)
- [Pre-registration: the commitment layer's revision on the ladder — CM1 under the arrived-keeps rule, and the legibility rung](#2026-09-21-curriculum-commitment-revision-preregistration)
- [Pre-registration: the commitment head on A3's shape (CM3, #398) — Stage 1 of the commitment layer](#2026-09-22-curriculum-commitment-stage1-preregistration)
- [Pre-registration: the execution phase of the commitment layer — the soldiers paid by the plan alone (EX1 on the half-step, and the A3 check)](#2026-09-22-curriculum-commitment-execution-preregistration)

---

<a id="2026-09-21-curriculum-commitment-stage0-preregistration"></a>

## Pre-registration: the commitment layer's Stage 0 on the ladder — the assignment given as an observation, A0 → A3 and the CM1 gate

Written 2026-09-21 09:40, **before any training round on any `_cm` config**,
on branch `feature/commitment-stage-0` (build `ea62586`, stacked on PR #383).
Parent question #384 (decision D9); arm #387 (CM1). Set as a goal by Sash
2026-09-21: build Stage 0, test it on the ladder from A0 upward, stop at
the five-objective half-step.

### The question

Every reward-side lever on the five-objective half-step is closed (six arms;
`reports/2026-09-19-curriculum-a5-points.md#2026-09-20-curriculum-A5-points-staying-preregistration`,
amendment 1) and the shared defect is a body that leaves an objective it
holds. Two stories fit: the policy cannot work out which squad goes where
(PLANNING), or it cannot hold a squad on an objective once there
(EXECUTION). **Stage 0 hands the policy the plan**: the environment assigns
every squad an objective at deployment by the scripted bar's own rule and
holds it (sticky; re-solved only when the objective is held by us without
the unit, or the unit dies), each member's token flags its unit's objective
and every objective token carries its claimant count, and the travel and
staying terms are keyed to the committed objective. Nothing decides
commitments yet. If the trainer executes a plan it is handed, planning was
the wall and the commitment head (Stage 1) is the next build; if it still
walks off its own assigned objective, execution is, and the head waits.

### The one change

`commitments.assignment: greedy` on the rung's config (`<rung>_cm.yaml`),
everything else the rung's own recipe: 128 rounds per update
(`--num-rollout-envs 4 --rollout-rounds 32`), `--ent-coef 0.003`, `gamma`
0.9, `mean` credit, eval and checkpoint every 512 rounds, recording on.
Three seeds (1 / 2 / 3), Wandb group `curriculum-cm`.

### The ladder, in order, and what each rung is for

| rung | config | budget | comparator (the rung's own runs, greedy n=100 on 700000+, read 2026-09-21 from their checkpoints at matched rounds) | what it says |
|---|---|---|---|---|
| **A0** three lone models, one objective | `a0_cm.yaml` | 40,960 | A0 at 40,960: **1.000 / 0.350 / 1.000** (turns 4.95 / 6.95 / 4.96; bar 1.000 in 4.93) | plumbing: the assignment is trivial (one objective); the layer must not break a rung that passes |
| **A1** the same three as one squad | `a1_cm.yaml` | 40,960 | A1 at 40,960: **1.000 / 0.790 / 1.000** (bar 0.960 in 4.96) | plumbing, with the coherency chain |
| **A2** four squads, one objective | `a2_cm.yaml` | 40,960 | A2b at 40,960: **1.000 / 1.000 / 1.000** (turns 4.99 / 4.23 / 4.02; bar 0.990 in 4.55) | plumbing: four units assigned one objective |
| **A3** four squads, four objectives | `a3_cm.yaml` | 122,880 | A3 at 122,880: **0.700 / 0.800 / 0.770**, held 3.5–3.7 (its 40,960 / 81,920 checkpoints read at the same rounds when the arm is) | the first rung where the assignment means anything: scratch passes only at 2× the cap |
| **CM1** six squads, five objectives (the half-step) | `a5_points_cm.yaml` | 122,880 | the original A5-points: 0.040 / 0.140 / 0.090 at 40,960, 0.220 / 0.220 / 0.110 at 81,920, **0.030 / 0.060 / 0.330** at 122,880; the whole-army control 0.910 / 0.810 / 0.950 at 120 epochs; the bar 1.000 in 6.71 | the gate (#387) |

A4 is skipped (the half-step contains its spare-squad shape); A5 (six
objectives) runs only on a PLANNING read; no C rung until the combat slot
has a writer.

### Criteria

**Plumbing (A0, A1, A2), each rung on its own:** PASS if greedy success at
40,960 is not below the comparator's seed-for-seed read by more than two
binomial SE on any seed, AND member follow-through ≥ 0.90 and the leave
share on the assigned objective ≤ the bar's on 3/3 (`measure-commitments`),
AND the run completes with a normal panel. A plumbing FAIL stops the ladder
until diagnosed: it is a defect in the layer, not a finding about the
trainer.

**A3:** read at 40,960 / 81,920 / 122,880 against A3's own checkpoints at
the same rounds. AHEAD if ahead of A3 by more than two binomial SE on 3/3 at
the cap; PASS if success ≥ 0.95 on 3/3 at the cap (where scratch needed
245,760); otherwise NULL. A3 is a readout on the way up, not a gate: the
ladder continues to CM1 whatever it reads, and its census (held, the empty
column, the leave share) is reported.

**CM1 (#387), as pre-registered there:**

- **PLANNING** — success ahead of the original by more than two binomial
  SE on 3/3 at 122,880 AND the leave share on the assigned objective below
  the bar's 0.33 on 3/3.
- **EXECUTION** — the leave share on the assigned objective stays above
  0.50 on 2/3 or more, whatever the success.
- **MIXED** — otherwise; read as execution first.

At n=100 a difference of 0.13 or more clears two SE for success near 0.2.
Every read is gated on the trainer EXITING; the final read is at `last.pt`
after exit; each seed's run dir is resolved on its own.

**Readouts on every read, before success:** `just measure-commitments`
(persist, claim, complete, empty, follow, leave — the bar's row beside it),
the by-turn census (bodies on objectives, held, max stack, empty share by
objective), the walk-off probe on the assigned objective, explained
variance and clip fraction, sampled beside greedy at the end.

### On a non-PLANNING read (D9)

A written diagnosis before the next build: which members leave, when in the
episode, what the execution potential paid on the step they left, whether
the assignment was reachable (freezing, friendly bases), and whether the
marked-target relation was what the members conditioned on. The plan on
#384 is revised from that reading; the layer is not abandoned on one read.

### What I expect (a guess, written so it can be wrong)

A0–A2 pass plumbing at the comparator's level or above (the assignment
cannot hurt a one-objective rung). A3 reads AHEAD at the cap (0.85–0.95)
with the empty column gone from the census — the given assignment removes
the re-derivation that left one objective short. CM1 reads **MIXED**:
success 0.4–0.7 with held 4.0–4.5, follow-through 0.85–0.95, the leave share
0.2–0.4 — better than any per-model setting from scratch and short of the
control, because the assignment fixes where to go and the staying term at
0.5 was already measured not to fix keeping. If CM1 reads PLANNING the
commitment head is the build; if EXECUTION, the credit fix on the execution
stream is, and the record's walk-off is an execution wall.

### Amendment 1 — written 2026-09-21 10:35: the plumbing rungs read; A3 and CM1 launched

**A0, A1, A2 at 40,960 rounds, three seeds each (Wandb `curriculum-cm`),
greedy n=100 on 700000+ at `last.pt`, `measure-commitments` at n=100:**

| rung | success | turns (bar) | comparator at 40,960 | persist · claim · complete · empty | follow | leave (moves from inside) | bar's leave |
|---|---|---|---|---|---|---|---|
| A0 | **1.000 / 1.000 / 1.000** | 4.96 / 4.94 / 5.01 (4.93) | 1.000 / 0.350 / 1.000 | 1.00 · 3.00/3 · 1.00 · 0.00 | 1.00 / 1.00 / 0.99 | 0.00 (74) / 0.00 (73) / **0.17 (66)** | n/a (0) |
| A1 | **1.000 / 1.000 / 1.000** | 4.88 / 4.87 / 4.84 (4.96) | 1.000 / 0.790 / 1.000 | 1.00 · 1.00/1 · 1.00 · 0.00 | 1.00 ×3 | 0.00 (52) / 0.00 (67) / **0.07 (67)** | 0.00 (99) |
| A2 | **1.000 / 1.000 / 1.000** | 4.02 / 4.02 / 4.97 (4.55) | 1.000 / 1.000 / 1.000 | 1.00 · 4.00/4 · 1.00 · 0.00 | 1.00 / 1.00 / 0.99 | 0.00 (15) / 0.00 (2) / 0.00 (472) | 0.00 (170) |

Panels normal on all nine (explained variance 0.59–0.86, clip fraction
0.005–0.27, in-run success 98–100% in the last quarter). No seed is below
its comparator; two comparator seeds (A0 s2 at 0.350, A1 s2 at 0.790) are
below the arm. Follow-through 0.99–1.00 on 9/9. **Plumbing PASS on all
three rungs.** ⚠ The leave clause ("≤ the bar's on 3/3") is missed on one
seed each of A0 (0.17 on 66 moves — eleven moves) and A1 (0.07 on 67 —
five moves), against a bar whose leave share is 0.00 or undefined (zero
moves from inside): a clause written as "≤ 0" cannot be cleared by a
stochastic count, and with success 1.000 at the bar's speed and the walk
uninterrupted it is recorded as a defect in the clause, not in the layer.
On A3 and CM1 the bar leaves on 0.33, and the clause is live there.

⚠ Two defects in this morning's reading, both procedural: the first read
chain fired before the trainers had spawned (gate on the process count at
launch; the premature files are kept as `*-PREMATURE-gate.txt`), and the
second chain's run-directory pattern (`per-model-curriculum_a1_cm-*`)
did not match the real names (`per-model-a1_cm-*`), so it scored only the
bar. The table above is the third chain (`read-*cm-final-v3.txt`,
`commit-*cm-final-v3.txt` in the session drafts), read after every log
carried its `rounds 40960` line and no trainer remained.

**A3 (`a3_cm.yaml`) and CM1 (`a5_points_cm.yaml`) launched 10:30, three
seeds each, 122,880 rounds, the same recipe** (128 rounds per update,
`--ent-coef 0.003`, eval and checkpoint every 512). Wandb: a3cm-s1=6rayjo62 a3cm-s2=fjkf3iir a3cm-s3=v4po4k4x a5pcm-s1=p3isqjsr a5pcm-s2=q1tshoax a5pcm-s3=jiqjvfqf. Reads at
40,960 / 81,920 / the end as pre-registered, each gated on every seed's log
carrying the round line and the final on zero trainers; A3 against its own
checkpoints at matched rounds, CM1 against the original's rows. A first
launch of the same six misfired on a shell word-splitting error and died
before any round; nothing from it is read.

### Amendment 2 — written 2026-09-21 14:20: A3 and CM1 read at the cap; A3 NULL (ahead on two seeds), CM1 EXECUTION

Everything below was read after every seed's log carried its `rounds
122880` line and no trainer remained (chain `cm_stage_reads_v2.sh`, which
resolves each seed's run directory from the config stem and checks the
checkpoint file exists; the chain armed at launch resolved CM1's directory
from the tag and would have read nothing, and was replaced before its first
read). n=100, seeds 700000+, greedy, `last.pt` after exit; the same
checkpoints, seeds and mode for every row. Wandb `curriculum-cm`:
a3cm 6rayjo62 / fjkf3iir / v4po4k4x, a5pcm p3isqjsr / q1tshoax / jiqjvfqf.
Every run recorded a greedy episode at every checkpoint.

**The A3 comparator is A3's own checkpoints scored on the plain
`a3.yaml`**, through the worktree's code with the layer off (bit-identical
to `main`'s per-model facade off, as the build verified). ⚠ A pre-layer
checkpoint scored on the layer-ON config is a perturbed policy: the context
embedding is one `Linear(CONTEXT_DIM)` shared by every token type, so the
new unit and objective columns land on weights trained for model-token
semantics. Scored plain, the cap row reproduces the A3 report's
0.700 / 0.800 / 0.770 to the thousandth, which is the check that the
comparator is the original policy.

#### A3 with the layer (`a3_cm.yaml`)

| rounds | A3 with the layer | A3's own, plain config |
|---|---|---|
| 40,960 | 0.520 / 0.350 / 0.240 | 0.400 / 0.350 / 0.340 |
| 81,920 | 0.890 / 0.650 / 0.750 | 0.590 / 0.670 / 0.610 |
| 122,880 | **0.820 / 0.850 / 0.930** | 0.700 / 0.800 / 0.770 |

**A3: NULL on the letter.** AHEAD needs more than two binomial SE on 3/3
at the cap: s1 +2.0 SE, s2 +0.9 SE, s3 +3.3 SE. PASS needs 0.95 on 3/3:
no seed. Ahead on two seeds and behind on none at the cap, ahead on two at
81,920 as well; held **3.81 / 3.72 / 3.92** against A3's 3.49 / 3.60 /
3.65; turns 6.02 / 5.65 / 5.59 against 6.05 / 5.95 / 5.81 (bar 5.28);
coherency 0.24 / 0.45 / 0.50 (A3 0.44–0.57). The empty column A3 left is
nearly gone: on the n=20 census every objective is empty in at most 10%
of episodes on every seed (A3: one objective short in a quarter), with
7.5 / 8.8 / 9.3 of twelve bodies on objectives from turn 5 to the end.
Readouts (`measure-commitments`, the bar's row beside): persist 0.96 /
0.94 / 0.97 (bar 0.99), claimants per claimed objective 1.07 / 1.11 /
1.06 with max 4 (bar 1.00 / 1), complete 0.91 / 0.89 / 0.96, empty 0.00,
follow-through 0.90 / 0.92 / 0.94 (bar 1.00), **leave on the assigned
objective 0.41 / 0.49 / 0.40** (bar 0.00). The walk-off probe (n=10): a
body on its objective stands still on 0.00 of its decisions, leaves on
0.38 / 0.48 / 0.40 of its moves from inside, and is paid −0.011 / −0.004 /
−0.004 on the step it leaves. Sampled beside greedy (n=100 paired):
sampled 1.4–3.2 vp above greedy, held greedy 3.72–3.92 against sampled
3.60–3.76 — a converged policy, not a diffuse one. Panel over the last
quarter, the original A3's beside it:

| run | clip fraction | ratio p99 | explained variance | displacement entropy |
|---|---|---|---|---|
| A3 with the layer s1 / s2 / s3 | 0.34 / 0.34 / 0.30 | 1.99 / 2.02 / 1.98 | 0.36 / 0.37 / 0.36 | 1.94 / 1.77 / 1.55 |
| A3's own s1 / s2 / s3 | 0.22 / 0.32 / 0.32 | 1.72 / 2.02 / 1.97 | 0.34 / 0.45 / 0.43 | 1.75 / 1.73 / 1.81 |

**The members do not read the marked-target relation.** A play-time
ablation (`drafts/commit_ablation_probe.py`, run by the parallel
instance of this session: the relation writer swapped so the flag is
BLANK, or MISDIRECTED to a different objective than the one the
environment pays against; reward is not computed at play, so only the
observation changes; n=100, the same seeds): as trained 0.82 / 0.85 /
0.93 · held 3.81 / 3.72 / 3.92; BLANK **0.83 / 0.87 / 0.95** · 3.81 /
3.76 / 3.94; MISDIRECT **0.87 / 0.83 / 0.90** · 3.85 / 3.67 / 3.88.
Whatever moved A3 moved through the reward keying alone: a travel target
that is assigned once at deployment and not re-derived every step.

#### CM1: the five-objective half-step with the layer (`a5_points_cm.yaml`)

| rounds | CM1 | the original (its own pre-registrations' rows) |
|---|---|---|
| 40,960 | 0.050 / 0.030 / 0.010 | 0.040 / 0.140 / 0.090 |
| 81,920 | 0.110 / 0.080 / 0.110 | 0.220 / 0.220 / 0.110 |
| 122,880 | **0.150 / 0.280 / 0.190** | 0.030 / 0.060 / 0.330 |

**CM1: EXECUTION, as pre-registered.** The leave share on the assigned
objective at the cap is **0.58 / 0.56 / 0.50** (`measure-commitments`,
918 / 1,095 / 988 moves from inside; the bar 0.01 on 429): above 0.50 on
two seeds and at it on the third. PLANNING is missed on both clauses:
success is ahead of the original by 3.0 and 4.3 SE on s1 and s2 and
**behind by 2.3 SE on s3** — the staying arms' exact shape (S was ahead on
two, behind on the third) — and no seed's leave share is under 0.33.
⚠ The PLANNING clause quoted "the bar's 0.33" from the walk-off probe,
which counts a body's decisions on any objective; `measure-commitments`
counts member moves from inside the ASSIGNED objective and puts the bar
at 0.01. The two readouts agree on the arm (0.50–0.58 against 0.53–0.62)
and the verdict is the same under either, but a future clause names one.

Held 3.16 / 3.35 / 3.31 of five (the staying arms 3.05–3.43), turns
9.79 / 9.48 / 9.61 of ten (bar 6.71), coherency 0.14–0.17. Census (n=20):
3–5 of eighteen bodies on objectives at turn 3, 5–6 from turn 5 to the
end, an objective empty in 40–70% of episodes on every seed, max stack
2.3–3.4 (bar 5.8). Readouts: persist **0.87 / 0.88 / 0.88** (bar 0.94),
claimants per claimed objective 1.45 / 1.44 / 1.45 with **max 6 on one
objective** (bar 1.20 / 3), complete 0.55 / 0.61 / 0.60, empty 0.00,
follow-through 0.77 / 0.76 / 0.79 (bar 1.00). The walk-off probe: a body
on its objective stands still on 0.00–0.01 of its decisions (every arm
on the half-step: 0.00–0.02; the bar 0.47), leaves on 0.62 / 0.55 / 0.53
(the original 0.60–0.81, the staying arms 0.48–0.67), and is paid
**−0.002 / −0.004 / +0.000** on the step it leaves. Panel over the last
quarter: clip fraction 0.30 / 0.38 / 0.36, ratio p99 1.99 / 2.17 / 2.06,
explained variance 0.69 / 0.73 / 0.73, displacement entropy 1.87 / 1.94 /
2.02 — the original's over the same window is 0.30 / 0.31 / 0.38, 2.02 /
1.97 / 2.23, 0.66 / 0.70 / 0.63, 1.60 / 1.88 / 1.82; normal for the
half-step. The flag ablation on the CM1 finals (the same probe, n=100):
as trained 0.15 / 0.28 / 0.19 · held 3.16 / 3.35 / 3.31; BLANK 0.13 /
0.26 / 0.19 · 3.05 / 3.36 / 3.40; MISDIRECT 0.17 / 0.23 / 0.20 · 3.00 /
3.30 / 3.46. **The members do not read the relation on the half-step
either.**


Sampled beside greedy on the CM1 finals (n=100 paired): greedy **+4.8 /
+1.6 / +5.0 vp** over sampled, and sampled holds MORE objectives at the
end (3.53 / 3.62 / 3.65 against greedy's 3.16 / 3.35 / 3.31) with lower
coherency (0.05–0.08 against 0.14–0.17): the greedy argmax concentrates
the walk, the sampled policy spreads it. Read as the argmax of a diffuse
displacement head (1.9–2.0 nats), not as a second policy.

#### The D9 diagnosis

Written before the next build, on the five questions the clause asks.

1. **Which members leave, and when.** The count on objectives is flat from
   turn 5 (5–6 of eighteen bodies) while the leave share is 0.50–0.58: the
   aggregate holds by turnover, as it did under the staying term. A body on
   its assigned objective never stands still (0.00–0.01 of its decisions)
   and leaves on about six moves in ten. There is no phase of the episode in
   which the members stay; it is the same walk-off the half-step has shown
   under every per-model setting.
2. **What the execution potential paid on the step they left.** On the
   half-step −0.002 to +0.000; on A3, where the plan lifts the rung,
   −0.004 to −0.011. The half-step's leaving step is nearly free, and the
   mechanism is in the retirement rule, verified in `retire_and_reassign`:
   a unit's ground commitment retires when the objective is OURS (any body
   of ours inside, one or more) and none of THIS unit's members is inside.
   Six squads over five objectives share one objective from deployment, so
   the squad that walks off a shared objective while its neighbour stays
   is retired and re-assigned to the nearest free objective, and the travel
   potential re-anchors on the new target — a switch pays 0.0, the record's
   "abandoning a target is free". The readouts show it: persist 0.87–0.88
   against the bar's 0.94, claimants up to 6 on one objective. On A3 no
   objective is shared and the rule never fires; on CM1 the layer gives the
   walk-off a free re-assignment.
3. **Whether the assignment was reachable.** Not measured beyond the stack:
   max 2.3–3.4 against the scripts' 5.8, so friendly bases are not what
   blocks a body from its assigned objective. Freezing was not probed.
4. **Whether the marked-target relation was what the members conditioned
   on.** No, on both rungs: blanking or misdirecting the flag moves success
   by at most a few hundredths. After 122,880 rounds from scratch the
   policy has not learned to read a relation column that only the reward
   keying makes meaningful.
5. **So what did Stage 0 measure.** The keying alone: a per-unit travel
   target fixed at deployment instead of re-derived every step. On A3 that
   is worth two seeds of about +0.1 and the empty column; on the half-step
   it is a seventh reward-side setting with the staying arms' signature
   (two seeds ahead, one behind, census unchanged). A commitment the policy
   cannot see is a reward term, and reward terms on this half-step are a
   closed class.

**The plan revision this asks of #384** (proposed, for Sash to decide, on
the issue): close the retirement hole (retire on death, or when another
unit held the objective before this one first arrived — never on a
walk-off by the earlier claimant); and before Stage 1's sampled pointer
head, which presupposes that the members condition on the commitment, run
a legibility rung — the assignment deliberately NOT the nearest objective,
with travel and success keyed to it, so a policy that reads the flag
passes and one that does not cannot. That is the cheapest test of whether
the network can learn to read the relation at all, and it is a rung, not a
seventh reward arm on the half-step.

---

<a id="2026-09-21-curriculum-commitment-revision-preregistration"></a>

## Pre-registration: the commitment layer's revision on the ladder — CM1 under the arrived-keeps rule, and the legibility rung

**Written 2026-09-21 21:19, before any training number exists.** Parent
#384 (decisions R1–R3, taken 2026-09-21); builds #392 (R1) and #393 (R2/R3);
arms #394 (CM1 re-run) and #395 (LR1). Branch `feature/commitment-revision`,
stacked on `feature/commitment-stage-0` (PR #388); never merged to `main`.

### The question

Stage 0 ([its pre-registration and amendments](2026-09-21-curriculum-commitment-stages.md#2026-09-21-curriculum-commitment-stage0-preregistration))
read EXECUTION on the five-objective half-step and found, by a play-time
ablation, that the members never read the marked-target relation on either
rung: what moved was the reward keying. Two things follow, and each is one
arm.

1. **The retirement hole (R1, #392).** Stage 0 retired a unit's ground
   commitment when its objective was ours without any of its members
   inside, so on six squads over five objectives the squad sharing an
   objective from deployment was re-assigned for free the moment it walked
   off, and the leaving step paid −0.002 to 0.000 against −0.004 to −0.011
   on A3. Now a unit keeps its commitment once it has ever had a member
   inside; only a latecomer to an objective another unit holds is
   re-assigned. **Do the members stay?**
2. **Legibility (R2, #393).** Stage 1's pointer head presupposes members
   that condition on the commitment. On A3's shape each squad is assigned
   the objective the greedy rule gave the NEXT squad (`assignment:
   rotated`), and success is every squad on ITS assigned objective
   (`all_units_on_commitment`). A policy that walks to the nearest
   objective covers the points and fails; only one that reads the relation
   can pass. **Can the network learn to read the target it is handed?**

### The one change per arm

| arm | config | the one change | comparator |
|---|---|---|---|
| **CM1-R1** (#394) | `a5_points_cm_r1.yaml` (= `a5_points_cm.yaml` under the new rule; the yaml differs only by name) | the retirement rule in code | Stage 0's CM1 rows, same seeds, same rounds |
| **LR1** (#395) | `a3_legible.yaml` (A3's shape; `rotated`; `all_units_on_commitment`) | the rung | the bar `squad_march_committed`; `squad_march_take` as the must-fail check |

Both from scratch, three seeds (1 / 2 / 3), 122,880 rounds at 128 rounds
per update (`--num-rollout-envs 4 --rollout-rounds 32`), `--ent-coef 0.003`,
eval and checkpoint every 512, a greedy episode recorded at every checkpoint,
Wandb group `curriculum-cm-rev`. Scored greedy with the passive fingerprint,
n=100 on seeds 700000+, sampled beside greedy at the end. Not extended past
the cap.

⚠ **Scoring routes every spec on a writer config to the per-model facade**
(`scoring.evaluate_spec`): the phase facade has no commitment state, so a
scripted name scored there plays a different game (on the legibility rung
its success is undefined, and Stage 0's bar rows on the `_cm` configs were
phase-facade rows — the greedy rule coincides there, so nothing is voided,
but the bar rows below are per-model rows). A scripted seat no longer
writes its own plan into the state when the environment is the writer.

### The bars, measured first (n=100, seeds 700000+, per-model facade)

| config | policy | success | turns | held | on obj | coherent | persist | claim | complete | follow | leave |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `a3_legible.yaml` | **`squad_march_committed`** (the bar) | **1.000** | 5.78 | 4.00 | 0.923 | 0.850 | 1.00 | 1.00 / max 1 | 1.00 | 1.00 | 0.01 (201) |
| `a3_legible.yaml` | `squad_march_take` (must fail) | **0.000** | 8.00 | 4.00 | 0.995 | 0.923 | 0.92 | 1.50 / max 4 | 1.00 | 0.88 | 0.00 (66) |
| `a5_points_cm_r1.yaml` | `squad_march_take` (the bar) | **1.000** | 6.71 | 5.00 | 0.889 | 0.832 | 0.94 | 1.20 / max 3 | 1.00 | 1.00 | 0.01 (429) |

The rung separates what it must: `take` covers all four objectives with the
wrong squads and reads 0.000 on the criterion; the script that follows the
assignment reads 1.000 at the A3 bar's speed (5.28 for `take` on A3's own
criterion; the rotated targets are one objective further along the column).
The half-step's bar is the same row on either facade (1.000 in 6.71, held
5.00), the bridge check for the routing change. Under the new rule the
bar's persistence is unchanged (0.94): the script never walks off.

### Criteria

**CM1-R1 (#394).** Primary readout the leave share on the assigned objective
(`measure-commitments`), 3/3 at 122,880.

- **STAYS** — leave share below 0.33 on 3/3.
- **MOVES** — leave share below 0.50 on 3/3 AND success not behind Stage 0's
  CM1 (0.150 / 0.280 / 0.190) by more than two binomial SE on any seed.
- **NULL** — otherwise. Success is a secondary readout against the CM1 rows
  at 40,960 (0.050 / 0.030 / 0.010), 81,920 (0.110 / 0.080 / 0.110) and
  122,880. At n=100 a success difference of 0.13 clears two SE near 0.2; a
  leave-share difference of 0.05 clears two SE at ~900 moves from inside.

**LR1 (#395).** Success and the flag ablation (`measure-commitment-ablation`:
trained / blank / misdirect / nearest) at 40,960 / 81,920 / 122,880.

- **READS** — success ≥ 0.80 on 3/3 at 122,880 AND under BLANK and under
  NEAREST success drops by more than 0.20 on 3/3.
- **PARTIAL** — success ≥ 0.80 on 3/3 and the ablation moves it by less on
  any seed: the rung is being solved by something other than the marking,
  and the diagnosis names what.
- **BLIND** — success below 0.50 on 3/3 with BLANK and NEAREST within 0.10
  of trained on every seed.
- **NULL** — otherwise. At n=100 a drop of 0.20 clears two SE anywhere on
  the range.

**Readouts on every read, before the verdicts:** `measure-commitments`
(persist, claim, complete, empty, follow, leave — the bar's row beside), the
flag ablation table, the by-turn census, the walk-off probe, explained
variance and clip fraction over the last quarter, sampled beside greedy at
the end. Every read is gated on each seed's log carrying its round line, the
final on zero trainers; each seed's run directory is resolved from the
config stem.

### What follows each reading (D9)

- CM1-R1 **STAYS** or **MOVES**: the retirement hole was load-bearing; the
  half-step becomes the place to read Stage 1 once LR1 reads.
- CM1-R1 **NULL**: the price of leaving is not what drives the walk-off;
  the half-step is closed to further reward-side work and Stage 1 is read on
  LR1's shape first.
- LR1 **READS**: the members can learn to condition on the relation; the
  Stage 1 pointer head is built on this shape.
- LR1 **PARTIAL** or **BLIND**: the observation encoding of the commitment
  is the next build (a relation bias is not what the network reads at this
  budget), not the head. The diagnosis names which of the four ablation
  columns moved.

### What I expect (a guess, written so it can be wrong)

CM1-R1 reads **NULL**: the leave share falls to 0.4–0.5 (the free
re-assignment removed, the leaving step now paid about what A3 pays) and
success sits in Stage 0's band, because the walk-off probe says a body on
its objective is paid within a few thousandths either way and the price of
leaving has never moved it. LR1 reads **PARTIAL** or **BLIND**: I expect the
policy to reach 0.3–0.6 by covering the points and to be moved by the
ablation on at most one seed — the relation bias is a weak channel, and
Stage 0 saw nothing read through it in 122,880 rounds.

### Amendment 1 — written 2026-09-22 01:13: both arms read at the cap; CM1-R1 NULL, LR1 NULL on the letter with two seeds reading the flag

Read after every seed's log carried its `rounds 122880` line and no trainer
remained (chain `rev_reads.sh`, directories resolved from the config stem);
n=100, seeds 700000+, greedy, `last.pt` after exit; every row the same
checkpoints, seeds and mode. Wandb `curriculum-cm-rev`: a5r1 gy78n4mo /
8z8aaxa0 / abiiu9v6, lr1 bc49wjgx / gqbpjrqs / 9gi4xf19. Launched 21:20,
the last trainer exited 00:55; every run recorded a greedy episode at every
checkpoint.

#### LR1, the legibility rung (#395): NULL on the letter — two seeds READ, one is blind

| rounds | success s1 / s2 / s3 | trained → blank → misdirect → nearest |
|---|---|---|
| 40,960 | 0.050 / 0.290 / 0.030 | s2: 0.29 → 0.01 → 0.09 → 0.01; s1, s3 flat |
| 81,920 | 0.380 / 0.840 / 0.030 | s1: 0.38 → 0.00 → 0.02 → 0.03; s2: 0.84 → 0.00 → 0.00 → 0.01; s3 flat |
| 122,880 | **0.710 / 0.950 / 0.080** | s1: **0.71 → 0.00 → 0.01 → 0.00**; s2: **0.95 → 0.00 → 0.00 → 0.00**; s3: 0.08 → 0.10 → 0.06 → 0.04 |

The bar `squad_march_committed` 1.000 in 5.78 turns; plain `take` 0.000.
**On the letter: NULL.** READS needed success ≥ 0.80 on 3/3 with BLANK and
NEAREST each dropping it by more than 0.20 on 3/3; PARTIAL and BLIND each
needed 3/3 too, and the seeds split. **On the substance: s1 and s2 read the
marked-target relation and s3 never found it.** Blanking the flag takes s2
from 0.95 to 0.00 and s1 from 0.71 to 0.00; pointing it at each model's
nearest objective does the same; pointing it one objective along leaves
0.00–0.01. The members go where the flag points and nowhere else: held
falls from 3.94 to 1.94 (s2) and 3.48 to 2.35 (s1) under BLANK, so without
the flag these policies do not even cover the column. s2 holds all four in
6.11 turns (bar 5.78; every objective empty in 0% of census episodes),
persistence 1.00, follow-through 0.97, sampled within 0.5 vp of greedy and
the same held. s1 holds 3.48 in 7.12 turns with follow-through 0.92. s3
covers 3.3 of four with the wrong squads (claimants 1.30 per objective, the
far objective empty in 45% of episodes), success 0.08, every ablation
column within 0.04 of trained — a policy that solved A3's old criterion on
a rung that asks a different question. Panels normal on all three (clip
0.36–0.37, ratio p99 2.22–2.26, explained variance 0.63 / 0.67 / 0.89,
displacement entropy 1.2–1.5 nats).

**What this settles.** The set network CAN learn to condition on a
relation column whose only meaning is what the reward pays against: two
seeds of three did, from scratch, in 122,880 rounds, on a rung where
nothing else passes. Stage 0's "the members never read the pointer" was
true of the greedy assignment on A3 and the half-step, where the nearest
objective and the assigned objective coincide often enough that reading
the flag buys nothing over reading the geometry; make the flag the only
route to success and it is read. What it does not settle is the third
seed: one of three finds the geometry solution first and stays there
(explained variance 0.89 — the critic fits the wrong policy well). The
Stage 1 head has something to steer on two seeds and nothing on the third.

#### CM1-R1, the half-step under the arrived-keeps rule (#394): NULL, as expected

| rounds | success s1 / s2 / s3 | Stage 0 CM1 | leave on the assigned objective | Stage 0 leave |
|---|---|---|---|---|
| 40,960 | 0.040 / 0.140 / 0.030 | 0.050 / 0.030 / 0.010 | 0.60 / 0.46 / 0.70 | 0.72 / 0.68 / 0.60 |
| 81,920 | 0.300 / 0.110 / 0.090 | 0.110 / 0.080 / 0.110 | 0.50 / 0.58 / 0.63 | 0.60 / 0.57 / 0.66 |
| 122,880 | **0.340 / 0.100 / 0.060** | 0.150 / 0.280 / 0.190 | **0.56 / 0.60 / 0.60** | 0.58 / 0.56 / 0.50 |

**NULL on the letter**: STAYS needed the leave share below 0.33 on 3/3,
MOVES below 0.50 on 3/3; it is 0.56–0.60. Success is ahead of Stage 0 by
3.4 SE on s1 and behind by 3.4 and 2.9 SE on s2 and s3 — the third arm on
the half-step with one seed up and two down. Persistence 0.91–0.92 (Stage
0 0.87–0.88; the free re-assignment is gone), follow-through 0.73–0.74,
complete 0.52–0.72, claimants 1.39 with max 4–6 on one objective, held
3.90 / 2.91 / 2.95, turns 9.4–9.9 of ten. The walk-off probe: a body on its
objective stands still on 0.01–0.02 of its decisions and leaves on
0.54–0.65, paid −0.004 to +0.001 on the step it leaves — the same as
Stage 0. Under the new rule the leaving step is STILL nearly free: the
retirement hole was one way a switch went unpriced, and closing it left the
travel potential's own rule (progress re-anchors on the new target, a
switch pays 0.0) as the other. The flag ablation is flat on s2 and s3; s1
drifts 0.34 → 0.29 → 0.22 → 0.18 under blank / misdirect / nearest, a hint
of conditioning at 2–2.5 SE and not a read. Panels normal (clip 0.34–0.37,
explained variance 0.69 / 0.80 / 0.81). ⚠ The bar's commitment readouts
changed with the emit guard: `take` is now read against the environment's
greedy assignment rather than its own per-phase re-plan, and reads persist
0.97 / follow 0.97 / leave 0.14 where Stage 0's bar row read 0.94 / 1.00 /
0.01; Stage 0's arm rows were read against the environment's assignment
already, so the arm-to-arm comparison stands and only the bar's row moved.

**Reading, per the pre-registration's D9 clause.** The price of leaving is
not what drives the walk-off, and the half-step is closed to further
reward-side work (the seventh setting, counting Stage 0). Stage 1 is read
on LR1's shape first.

#### Sampled beside greedy (n=100 paired)

LR1: greedy − sampled +2.5 / −0.5 / +1.1 vp, held equal within 0.2 on
every seed. CM1-R1: +3.2 / −1.6 / +4.2 vp, sampled holding 3.70 / 3.27 /
3.00 against greedy's 3.90 / 2.91 / 2.95. Converged policies on both arms.

---

<a id="2026-09-22-curriculum-commitment-stage1-preregistration"></a>

## Pre-registration: the commitment head on A3's shape (CM3, #398) — Stage 1 of the commitment layer

**Written 2026-09-22 01:35, before any training number exists.** Parent
#384 (D1–D9 decided 2026-09-20; Stage 1 opened on Sash's "proceed as
recommended" 2026-09-22 after the legibility rung read); build #397; arm
#398. Branch `feature/commitment-revision` (PR #396, stacked on #388);
never merged to `main`.

### The question

The legibility rung ([the revision's pre-registration, amendment 1](2026-09-21-curriculum-commitment-stages.md#2026-09-21-curriculum-commitment-revision-preregistration))
showed two seeds of three learning to READ a commitment when reading it is
the only way to win. Stage 1 asks whether the network can CHOOSE it: at each
squad's first open of the turn the set network's commitment head draws KEEP
or an objective, the environment writes it, the members are paid the travel
potential keyed to that objective alone (the execution stream), and the
commitment decision is paid the close's outcome terms — coverage and the
rung's success bonus — on a planning stream with a semi-Markov return over
the squad's own commitment steps and its own value head (D1 with K = 1, D2
pure, D4 no switch price, D6 semi-Markov, D8 no supervision). On A3's shape
the head must learn an allocation (four squads to four distinct objectives)
and the members must follow it.

### The one change

`a3_head.yaml` = `a3.yaml` with `commitments.assignment: head`; success
A3's own `all_objectives_occupied`. Three seeds (1 / 2 / 3) from scratch,
122,880 rounds at 128 rounds per update (`--num-rollout-envs 4
--rollout-rounds 32`), `--ent-coef 0.003`, `--planning-gamma 0.99`,
eval and checkpoint every 512, a greedy episode recorded at every
checkpoint, Wandb group `curriculum-cm-s1`. Scored greedy with the passive
fingerprint, n=100 on seeds 700000+, sampled beside greedy at the end.
Every read gated on each seed's log carrying its round line, the final on
zero trainers; run directories resolved from the config stem.

### Comparators, fixed by name

| | 40,960 | 81,920 | 122,880 |
|---|---|---|---|
| A3's own runs (`a3.yaml`, no layer) | 0.400 / 0.350 / 0.340 | 0.590 / 0.670 / 0.610 | 0.700 / 0.800 / 0.770 |
| `a3_cm` (the environment's greedy assignment given) | 0.520 / 0.350 / 0.240 | 0.890 / 0.650 / 0.750 | 0.820 / 0.850 / 0.930 |

The bar `squad_march_take`, which writes its own greedy plan into the
state under `head`, measured first on the per-model facade (n=100, seeds
700000+): **success 1.000 in 5.28 turns**, held 4.00, coherent 0.927;
persist 0.99 (1712), claim 1.00 / max 1, complete 1.00, follow 1.00
(6000), leave 0.00 (263).

### Criteria — per seed, as the revision's reading asked

At 122,880, per seed:

- **PASS** — success ≥ 0.95.
- **AHEAD** — not PASS, and ahead of A3's own seed by more than two
  binomial SE (at n=100 a difference of 0.13 clears it near 0.75).
- **NULL** — otherwise.

And per seed, the head's legibility: **STEERS** if the flag ablation
(`measure-commitment-ablation`, trained / blank / misdirect / nearest)
drops success by more than 0.20 under BLANK — the members follow the head's
pointer rather than the geometry. The arm's verdict is the count, written
as PASS / AHEAD / NULL with the STEERS count beside it. Readouts before the
verdict, on every read: the commitment readouts (persist, claimants — does
the head allocate DISTINCT objectives?, complete, empty, follow, leave; the
bar's row beside), the ablation table, the census, the walk-off probe, the
member panel and the planning panel (`train/planning/*`), sampled beside
greedy at the end. A planning explained variance below 0.2 over the last
quarter is a defect named in the amendment, whatever the success reads.

### What follows each reading (D9)

- **≥ 2 seeds PASS or AHEAD with STEERS**: the head chooses and the members
  follow; Stage 1 stands, and the next arm is the D7 ablation on this arm
  (the planning stream reduced to the success bonus alone, paired) before
  the half-step is revisited with the head.
- **AHEAD/PASS without STEERS**: the head's choice is not what the members
  execute; the diagnosis reads the ablation columns and the claimants
  readout, and names whether the head learned KEEP-only (persist 1.00,
  claimants at the greedy allocation the members walk anyway).
- **NULL on ≥ 2 seeds**: the diagnosis reads the planning panel first (a
  critic that never fit → the return's construction; a critic that fit and a
  head that never moved → the surrogate's scale), then the claimants (a head
  that stacks squads on one objective is a head that has not learned to
  allocate), before any change to the design.

### What I expect (a guess, written so it can be wrong)

One PASS, one AHEAD, one NULL, with STEERS on the two that move: the head
learns KEEP early (persistence near 1.00 from the first quarter), the
allocation comes from where each squad's first open lands it, and a squad
that commits to a neighbour's objective is the failure mode on the NULL
seed. Planning explained variance 0.3–0.6: four commitment steps per turn
against a coverage signal that mostly moves at the end.

⚠ A first launch of the three seeds at 01:33 ran on code the hooks had
refused to commit (two type errors in a test file); it was stopped within a
minute, its run directories deleted, and nothing from it is read. The runs
below start after this file and the build (`be3fbed`) are on the branch.

### Amendment 1 — written 2026-09-22 03:24: CM3 read at the cap — NULL / NULL / NULL, STEERS 0; the head commits, keeps and stacks

Read after every seed's log carried its `rounds 122880` line and no trainer
remained (chain `cm3_reads.sh`; directories resolved from the config stem);
n=100, seeds 700000+, greedy, `last.pt` after exit. Launched 01:36, the
last trainer exited ~03:05. Wandb `curriculum-cm-s1`: g8azma3g / tbz5jz2i /
1l1qkjng. Every run recorded a greedy episode at every checkpoint.

| rounds | CM3 s1 / s2 / s3 | A3's own | `a3_cm` |
|---|---|---|---|
| 40,960 | 0.000 / 0.000 / 0.000 | 0.400 / 0.350 / 0.340 | 0.520 / 0.350 / 0.240 |
| 81,920 | 0.110 / 0.070 / 0.000 | 0.590 / 0.670 / 0.610 | 0.890 / 0.650 / 0.750 |
| 122,880 | **0.030 / 0.000 / 0.000** | 0.700 / 0.800 / 0.770 | 0.820 / 0.850 / 0.930 |

**Per seed: NULL, NULL, NULL; STEERS on none** (the flag ablation moves
success and held by hundredths on every seed). The arm is far BEHIND A3
from scratch at every read: a head that chooses badly costs more than no
head, because the travel keying carries its choice to the members.

Readouts at the cap, the bar's row beside (persist 0.99, claim 1.00 / max
1, complete 1.00, follow 1.00, leave 0.00):

| seed | held | persist | claimants per claimed objective | complete | empty | follow | leave | max stack |
|---|---|---|---|---|---|---|---|---|
| s1 | 2.17 | 0.53 | 1.64 / max 4 | 0.43 | 0.06 | 0.74 | 0.73 | 3.5 |
| s2 | 2.64 | 0.85 | **2.56** / max 4 | **0.95** | 0.09 | 0.87 | 0.19 | **5.3** |
| s3 | 1.72 | 0.91 | 2.05 / max 4 | 0.18 | 0.03 | 0.80 | 0.80 | 3.5 |

s2 is the clearest reading: the head keeps its choices (0.85), commits
every squad (empty 0.09), the members follow (0.87) and the plan is
COMPLETED (0.95) — 10.2 of twelve bodies on objectives by turn 7 — and it
holds 2.64 of four because two and a half squads claim each claimed
objective and one objective is empty in every census episode. The head
learned to commit, to keep and to be followed; it did not learn to
allocate. s1 switches (persist 0.53) and holds 2.17; s3 keeps (0.91) a
plan the members complete in 18% of unit-cases and holds 1.72. Uncommitted
unit-turns fell from 13–23% at 40,960 to 3–9%. Sampled play is 2–5 vp below
greedy, the same held.

**Panels over the last quarter.** Members: clip fraction 0.24–0.32,
explained variance 0.75–0.85, displacement entropy 1.75–1.95 nats. Planning:
explained variance **0.23 / 0.25 / 0.33** (above the 0.2 defect line, and
low), return mean 0.26–0.34, commitment entropy 0.90–0.99 nats of 1.61
(from 1.1–1.2 at 40,960), commitment clip fraction 0.27–0.32 — the head
moved throughout and settled on the wrong distribution.

#### The D9 diagnosis

1. **The planning panel.** The critic fit weakly and the head moved: this
   is not a return that never reached the head. It is a return the head
   cannot use to tell its choices apart.
2. **The claimants.** 1.6–2.6 squads per claimed objective at the cap, from
   2.3–3.0 at 40,960: the head has not learned to allocate. Every squad's
   commitment step is paid the SAME close — the army's coverage and the
   rung's success bonus, broadcast — so a squad that piles onto an objective
   a neighbour already claims and a squad that takes the empty one receive
   identical credit, and the only route to telling them apart is the
   planning critic's reading of the claimant counts on the objective tokens,
   which at explained variance 0.25 it barely does. The whole-army record's
   critic-probe finding is the same fact from the other side: the value can
   know the stack is wrong while the policy gradient cannot find the
   redistribution.
3. **The members.** The flag ablation is flat on every seed, as on Stage
   0's A3: on this shape the members walk the geometry, and the head's
   choice reaches them through the travel keying (follow-through 0.74–0.87),
   not through the relation. Stacked commitments therefore stack the walk.
4. **The keying without allocation is the harm.** A3 from scratch derives a
   per-squad target every step and reads 0.70–0.80; the environment's greedy
   assignment reads 0.82–0.93; the head's assignment reads 0.00–0.03. The
   order is: a good fixed plan > a re-derived plan > a learned plan that
   stacks. The head is not a null on this rung; it is a cost, until it
   allocates.

#### The revision this asks of #384 (proposed; Sash decides)

- **B6, the per-unit counterfactual on the planning stream** — the build the
  design held behind follow-through, and the reading calls for it now: pay
  each squad's commitment step the DIFFERENCE its bodies make to the
  outcome (coverage with the squad's models masked out, subtracted from
  coverage with them in; the success bonus likewise), so a squad stacking
  on a covered objective earns 0 and the squad taking the empty one earns
  the whole objective. A difference reward, not a switch price (D4 stands),
  computed at the close in the retimer from the same distance cache. One
  arm, three seeds, same shape, read against this one.
- Not proposed: a longer budget (flat from 81,920), an entropy change (the
  head is at 0.9 nats and moving), or the half-step (the head must allocate
  four before it allocates five).

---

<a id="2026-09-22-curriculum-commitment-execution-preregistration"></a>

## Pre-registration: the execution phase of the commitment layer — the soldiers paid by the plan alone (EX1 on the half-step, and the A3 check)

**Written 2026-09-22 22:51, before any training number exists.** Parent
#384; goal set by Sash 2026-09-22 ("make the soldiers execute a given
plan"); build #399; arms #400 (EX1, the five-objective half-step) and #401
(the A3 check). Branch `feature/commitment-revision` (PR #396, stacked on
#388); never merged to `main`.

### The question

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

### The one change per arm

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

### The comparators' rows

| | 40,960 | 81,920 | 122,880 | leave at the cap | follow | persist |
|---|---|---|---|---|---|---|
| CM1-R1 (`a5_points_cm_r1`) | 0.040 / 0.140 / 0.030 | 0.300 / 0.110 / 0.090 | 0.340 / 0.100 / 0.060 | 0.56 / 0.60 / 0.60 | 0.73–0.74 | 0.91–0.92 |
| S (`objective_stay` 0.5, one stream) | 0.040 / 0.010 / 0.020 | 0.170 / 0.150 / 0.190 | 0.210 / 0.190 / 0.120 | 0.48–0.67 (walk-off probe) | — | — |
| `a3_cm` | 0.520 / 0.350 / 0.240 | 0.890 / 0.650 / 0.750 | 0.820 / 0.850 / 0.930 | 0.41 / 0.49 / 0.40 | 0.90–0.94 | 0.94–0.97 |

### The bars, measured first (n=100, seeds 700000+, per-model facade)

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

### Criteria

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

### What follows each reading (D9)

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

### What I expect (a guess, written so it can be wrong)

EX1 reads **KEEPS** with success in 0.3–0.5: the leave share falls under
0.14 on 3/3 because leaving is finally worth less than staying by a margin
the normalised advantage can see, the hold-declaration share rises to
0.3–0.6, and success is bounded by the approach phase — squads that arrive
late or share an objective — rather than by the walk-off. The A3 check
reads **HOLDS** at 0.85–0.95 with turns within 0.3 of `a3_cm`.

### Amendment 1 — written 2026-09-23 10:16: both arms read at the cap — EX1 NULL, the A3 check COSTS; keeping improves, arrival collapses

Read after every seed's log carried its `rounds 122880` line and no trainer
remained (chain `ex_reads.sh`; directories resolved from the config stem);
n=100, seeds 700000+, greedy, `last.pt` after exit. Launched 22:52 on
2026-09-22; the last trainer exited before 03:00. Wandb `curriculum-cm-ex`:
ex1 5fh8r1r8 / 9phybw9r / 2ybt8e53, ex1a3 i7sljhl9 / 79pb3imk / c5z14hpv.
Every run recorded a greedy episode at every checkpoint.

#### EX1, the half-step (#400): NULL

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

#### The A3 check (#401): COSTS

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

#### The mode-aware walk-off probe (n=10, the scoring rule's "inside")

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

#### Reading

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

#### The revision this asks of #384 (proposed; Sash decides)

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
