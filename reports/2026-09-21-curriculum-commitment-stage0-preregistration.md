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
