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
