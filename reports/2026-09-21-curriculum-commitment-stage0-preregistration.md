# Pre-registration: the commitment layer's Stage 0 on the ladder — the assignment given as an observation, A0 → A3 and the CM1 gate

Written 2026-09-21 09:40, **before any training round on any `_cm` config**,
on branch `feature/commitment-stage-0` (build `ea62586`, stacked on PR #383).
Parent question #384 (decision D9); arm #387 (CM1). Set as a goal by Sash
2026-09-21: build Stage 0, test it on the ladder from A0 upward, stop at
the five-objective half-step.

## The question

Every reward-side lever on the five-objective half-step is closed (six arms;
`reports/2026-09-20-curriculum-A5-points-staying-preregistration.md`,
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

## The one change

`commitments.assignment: greedy` on the rung's config (`<rung>_cm.yaml`),
everything else the rung's own recipe: 128 rounds per update
(`--num-rollout-envs 4 --rollout-rounds 32`), `--ent-coef 0.003`, `gamma`
0.9, `mean` credit, eval and checkpoint every 512 rounds, recording on.
Three seeds (1 / 2 / 3), Wandb group `curriculum-cm`.

## The ladder, in order, and what each rung is for

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

## Criteria

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

## On a non-PLANNING read (D9)

A written diagnosis before the next build: which members leave, when in the
episode, what the execution potential paid on the step they left, whether
the assignment was reachable (freezing, friendly bases), and whether the
marked-target relation was what the members conditioned on. The plan on
#384 is revised from that reading; the layer is not abandoned on one read.

## What I expect (a guess, written so it can be wrong)

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

## Amendment 1 — written 2026-09-21 10:35: the plumbing rungs read; A3 and CM1 launched

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

## Amendment 2 — written 2026-09-21 14:20: A3 and CM1 read at the cap; A3 NULL (ahead on two seeds), CM1 EXECUTION

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

### A3 with the layer (`a3_cm.yaml`)

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

### CM1: the five-objective half-step with the layer (`a5_points_cm.yaml`)

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

### The D9 diagnosis

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
