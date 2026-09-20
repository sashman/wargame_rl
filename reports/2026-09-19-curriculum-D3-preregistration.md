# Pre-registration: curriculum rung D3 — anchored PPO from a clone of a WEAKER teacher

Written 2026-09-19 01:20, **before the clone exists and before any PPO
round from it**, on branch `feature/per-model-actor-credit` (PR #383).
Parent question #340; the D branch's third rung; no issue opened, by
the same decision as A5c–A5i — this file is its record. Set as a goal
by Sash on 2026-09-19 01:05.

## The question

Twice now anchored PPO has held a clone exactly where it started: D2c on
C3b (the escort clone 0.950 → 0.944 / 0.956 / 0.956) and A5f / A5i on A5
(the bar's clones 0.930 → 0.85–0.97 over twenty-four reads, 0.960 →
0.960 / 0.980 / 0.960). Both starts were already at their teacher's
level, so "no improvement" is consistent with two readings: the anchor
is a brake with no engine, or there was nothing left to gain. **D3
separates them by starting from a clone with something to gain.** Plain
`squad_march_take` on C3b reads 0.280 (the escort 0.990 on the same
seeds): it walks the same walk and lacks one thing — shoot the blockers
first. If anchored PPO on C3b's reward lifts a clone of `take` toward
the escort, the trainer can improve a start from reward and the anchor's
coefficient is what to sweep. If it holds 0.28, the D-route on this
facade is imitation with extra steps, and the record says so.

## The one change

The teacher. Everything else is D2c: `configs/experiments/curriculum/
c3b.yaml` (C3 with success read on the final board), the clone recipe of
D1b, the anchor of D2c. The clone is `checkpoints/per_model/clones/
take-c3b-1200-s0.pt` — `just behaviour-clone-per-model squad_march_take
configs/experiments/curriculum/c3b.yaml 1200 40 <out> 0`, seeds 800000+,
the last 240 games held out, fit seed 0 (launched 01:13, before this
file was written; no number from it existed at the time of writing).

## Reads

The clone: greedy at **n=180 on 700000+** (`just measure-rung` with
`scripted_escort` as the bar and `squad_march_take` beside it), and the
ordering census (n=180): the turn of the first blocker kill, the turn the
first unarmed body reaches the blockers' point, the share of episodes
where the kill precedes the arrival, the share where the blockers are
wiped. Expected ≈ the teacher's: success 0.28, kill before arrival low,
blockers alive at the end in most episodes.

## The arms

| | D3a | D3b |
|---|---|---|
| start | the `take` clone, `--warm-start-from` | the same |
| anchor | `--kl-ref-coef 10 --kl-ref-target 0.03` (D2c's, which held the escort clone) | `--kl-ref-coef 1 --kl-ref-target 0.10` (A5g's, which held the A5 clone) |
| everything else | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003`, the default `mean` credit (D2c's), from the clone's cold critic | the same |
| seeds | 1 / 2 / 3 | 1 / 2 / 3 |
| budget | 122,880 rounds at 128 per update | the same |
| logging | Wandb `curriculum-d3`, tag `d3a` | `curriculum-d3`, tag `d3b` |

Read greedy at n=180 on 700000+ at 20,480 / 40,960 / 61,440 and at the
end, the ordering census at the end, sampled beside greedy at n=100 on
900000+, the health panel over the last quarter. Three seeds off one
clone are that clone's band, not three samples; the comparator on every
row is the clone on identical seeds.

## Criteria (per arm)

- **IMPROVES:** at 122,880, success above the clone's on **all three
  seeds**, paired per episode at n=180, by at least two paired SE on each
  seed; and kill-before-arrival above the clone's on all three.
- **HOLDS:** every seed within two paired SE of the clone on success.
- **DESTROYS:** any seed below the clone by more than two paired SE.
- **Mixed** rows (one seed up, one down) are reported as the seeds read;
  the arm's verdict is the worst clause any seed triggers.

**The goal's decision:** IMPROVES on either arm — the anchor permits
improvement from reward on this trainer, and the coefficient is the
lever to sweep on D2 and A5. HOLDS on both — the D-route is recorded as
imitation only. DESTROYS on D3b with HOLDS on D3a — the coefficient has
no useful middle at these two settings and the next sweep goes between
them.

## What I expect (a guess, written so it can be wrong)

D3a holds: drift capped near 0.03 nats per decision cannot reorder a
plan, and A5i moved a tenth of a turn under it. D3b either finds the
order partially (0.40–0.60 with kill-before-arrival rising, the shape
C3b's arm-from-C2 reached at 0.20 by wiping the blockers and stopping)
or slides toward D2 plain's collapse. Honest guess: **D3b ends between
0.10 and 0.50, IMPROVES not shown on 3/3, D3a within the clone's band.**
If D3b improves on 3/3 I am wrong in the useful direction.

## Amendment 1 — written 2026-09-19 02:50, the clone read; D3a and D3b launched, no PPO number yet

**The clone is its teacher.** Greedy at n=180 on 700000+ (`just
measure-rung`, `scripted_escort` as the bar), with the ordering census
on the same seeds:

| policy | success | held of 4 | alive | vp | coherent | first kill (turn) | first unarmed arrival | kill before arrival | blockers wiped |
|---|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (bar) | 0.978 | 3.97 | 0.965 | 62.6 ± 1.6 | 0.839 | 6.0 (100%) | 12.9 (91%) | 1.00 | 0.99 |
| `squad_march_take` (teacher) | 0.339 | 2.46 | 0.525 | 59.5 ± 3.4 | 0.934 | 7.0 (93%) | 8.7 (56%) | 0.79 | 0.62 |
| **the clone** `take-c3b-1200-s0` | **0.361** | 2.51 | 0.531 | 60.8 ± 3.3 | 0.870 | 7.0 (94%) | 8.9 (62%) | 0.82 | 0.64 |

The clone matches its teacher on every column (success +0.02, held
+0.05, alive +0.006, the census within a few points); held-out per-head
match: declaration 0.956, displacement 0.807, unit-pointer 1.000, joint
0.569. The teacher's 0.280 on the C3b report was n=100; at n=180 it
reads 0.339, and the clone starts there. The gap to the escort is
**0.62 in success, 0.43 in alive, 0.35 in blockers wiped** — the
headroom D3 asks reward to close. Note what the census says the gap
is: the teacher's armed squad does fire first in most episodes (kill
before arrival 0.79) and does not finish the job (wiped 0.62), and its
unarmed squads walk in while the blockers live (alive 0.53). The
escort's order is "wipe, then walk"; the teacher's is "walk, shooting
on the way".

**Launched 02:42, from the clone**, as pre-registered: D3a (anchor 10 /
0.03) `vjgk8f5w` / `i6pri90x` / `kjppfvfs`, directories
`per-model-c3b-2026-09-19-02-42-*-s{1,2,3}d3a`; D3b (anchor 1 / 0.10)
`vv1w0k0f` / `sfk1u00s` / `k3u5dz2m`, `…-s{1,2,3}d3b`. Wandb
`curriculum-d3`. Beside E1's six per-model runs on the box.

## Amendment 2 — written 2026-09-19 05:30, at 60k of 122,880 on both arms

The 20k / 40k / 60k reads, greedy at n=180 on 700000+ (the clone
0.361, the escort 0.978):

| rounds | D3a s1 / s2 / s3 (anchor 10 / 0.03) | D3b s1 / s2 / s3 (anchor 1 / 0.10) |
|---|---|---|
| 20,480 | 0.372 / 0.339 / 0.350 | 0.322 / 0.383 / 0.406 |
| 40,960 | 0.367 / 0.361 / 0.367 | 0.361 / 0.367 / 0.372 |
| 61,440 | 0.378 / 0.367 / 0.389 | 0.328 / 0.350 / 0.344 |

Eighteen reads, all within 0.04 of the clone (binomial SE 0.036 at
n=180), no trend on either arm. The ordering census does not move
either: first kill at turn 6.8–7.1 in 90–94% of episodes, first unarmed
arrival at 8.4–9.0 in 56–62%, kill before arrival 0.79–0.85, blockers
wiped 0.62–0.66, alive 0.52–0.54, held 2.44–2.57 — the clone's row on
every column, at both anchor strengths. Half the budget in, neither
arm has taken a step toward the escort's order. Written before the
end read; the verdict is at 122,880 (~08:30).

## Amendment 3 — written 2026-09-19 07:50, both arms read at 122,880: HOLDS on both, IMPROVES on neither

Greedy at n=180 on 700000+ at `last.pt` (the clone 0.361, the escort
0.978), with the ordering census on the same seeds and sampled play at
n=100 on 900000+ paired against greedy:

| row | success | held of 4 | alive | coherent | first kill (turn) | kill before arrival | blockers wiped | sampled − greedy, vp |
|---|---|---|---|---|---|---|---|---|
| the clone | 0.361 | 2.51 | 0.531 | 0.870 | 7.0 (94%) | 0.82 | 0.64 | — |
| D3a s1 | 0.339 | 2.44 | 0.519 | 0.895 | 7.0 (93%) | 0.79 | 0.64 | +2.0 ± 2.4 |
| D3a s2 | 0.356 | 2.56 | 0.534 | 0.877 | 7.0 (94%) | 0.81 | 0.65 | +0.5 ± 2.8 |
| D3a s3 | 0.333 | 2.49 | 0.524 | 0.884 | 6.8 (94%) | 0.83 | 0.67 | +0.8 ± 2.1 |
| D3b s1 | 0.328 | 2.47 | 0.522 | 0.859 | 7.1 (96%) | 0.84 | 0.64 | +3.4 ± 2.3 |
| D3b s2 | 0.344 | 2.54 | 0.535 | 0.880 | 6.9 (93%) | 0.78 | 0.64 | −1.1 ± 2.5 |
| D3b s3 | 0.356 | 2.51 | 0.519 | 0.879 | 7.1 (94%) | 0.84 | 0.63 | +1.9 ± 1.9 |

Every seed of both arms reads the clone: success 0.328–0.356 against
0.361 (all within one binomial SE, none above), held 2.44–2.56, alive
0.52–0.54, the census the clone's on every column, sampled play the
same policy as greedy. Twenty-four reads over 20k–122,880 rounds and
not one moved. **Verdict per arm: HOLDS, IMPROVES not shown, DESTROYS
not triggered — on both anchors.** (The paired per-episode success
difference against the clone, the letter of the HOLDS / IMPROVES
clauses, is appended to the report when computed; no seed is above the
clone, so IMPROVES cannot be met by it.)

**The goal's decision, as pre-registered: HOLDS on both — the D-route
is recorded as imitation only.** From a start with 0.62 of success to
gain, on a reward whose signal (the escort's plan) is the same one the
arm from C2 half-learned from scratch (C3b: wipes the blockers in
75–79% of episodes), anchored PPO at two coefficients took no step in
122,880 rounds. The anchor is a brake with no engine on this trainer:
what it holds is what the clone was, and what the clone was is what
the rung reads. Consequence for the goal set on 2026-09-19: goal 1 is
answered in the negative; goal 3 (decompose the conjunction on
A5-points) was conditioned on goal 1 climbing and is not run — with
the caveat, recorded so it can be re-opened, that goal 3 asks about
reward from scratch and goal 1 answered about reward from a clone.

## Amendment 4 — written 2026-09-19 11:30, the mechanism (post hoc)

Read off the runs' logs after the verdict, not pre-registered: the
anchor's coefficient sat at its 10,000 cap on D3a from the first quarter
(drift 0.08 against a 0.03 target) with the gradient norm 14,000–17,000
clipped to 0.5 on every step — the update is the anchor; the clone's
heads are near-deterministic (declaration 0.03 nats, shooting 3×10⁻⁵),
so the rung's alternative is never rolled out; and C3b's reward,
discounted at gamma 0.9 per round, pays walking-now 2.55 against the
escort's 1.25 in rounds 1–6 and leaves the plan worth +0.7 per episode
at the decision point. The report's addendum carries the tables.
