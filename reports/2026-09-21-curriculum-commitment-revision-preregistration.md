# Pre-registration: the commitment layer's revision on the ladder — CM1 under the arrived-keeps rule, and the legibility rung

**Written 2026-09-21 21:19, before any training number exists.** Parent
#384 (decisions R1–R3, taken 2026-09-21); builds #392 (R1) and #393 (R2/R3);
arms #394 (CM1 re-run) and #395 (LR1). Branch `feature/commitment-revision`,
stacked on `feature/commitment-stage-0` (PR #388); never merged to `main`.

## The question

Stage 0 ([its pre-registration and amendments](2026-09-21-curriculum-commitment-stage0-preregistration.md))
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

## The one change per arm

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

## The bars, measured first (n=100, seeds 700000+, per-model facade)

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

## Criteria

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

## What follows each reading (D9)

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

## What I expect (a guess, written so it can be wrong)

CM1-R1 reads **NULL**: the leave share falls to 0.4–0.5 (the free
re-assignment removed, the leaving step now paid about what A3 pays) and
success sits in Stage 0's band, because the walk-off probe says a body on
its objective is paid within a few thousandths either way and the price of
leaving has never moved it. LR1 reads **PARTIAL** or **BLIND**: I expect the
policy to reach 0.3–0.6 by covering the points and to be moved by the
ablation on at most one seed — the relation bias is a weak channel, and
Stage 0 saw nothing read through it in 122,880 rounds.

## Amendment 1 — written 2026-09-22 01:13: both arms read at the cap; CM1-R1 NULL, LR1 NULL on the letter with two seeds reading the flag

Read after every seed's log carried its `rounds 122880` line and no trainer
remained (chain `rev_reads.sh`, directories resolved from the config stem);
n=100, seeds 700000+, greedy, `last.pt` after exit; every row the same
checkpoints, seeds and mode. Wandb `curriculum-cm-rev`: a5r1 gy78n4mo /
8z8aaxa0 / abiiu9v6, lr1 bc49wjgx / gqbpjrqs / 9gi4xf19. Launched 21:20,
the last trainer exited 00:55; every run recorded a greedy episode at every
checkpoint.

### LR1, the legibility rung (#395): NULL on the letter — two seeds READ, one is blind

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

### CM1-R1, the half-step under the arrived-keeps rule (#394): NULL, as expected

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

### Sampled beside greedy (n=100 paired)

LR1: greedy − sampled +2.5 / −0.5 / +1.1 vp, held equal within 0.2 on
every seed. CM1-R1: +3.2 / −1.6 / +4.2 vp, sampled holding 3.70 / 3.27 /
3.00 against greedy's 3.90 / 2.91 / 2.95. Converged policies on both arms.
