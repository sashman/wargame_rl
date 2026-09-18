# Pre-registration: A5e (the hold term under the actor credit) and A5f (the bar cloned, then anchored PPO)

Written 2026-09-18 01:15, **before any number for either exists**, on
branch `feature/per-model-actor-credit` (PR #383). Parent question #340,
rung **A5**. Goal set by Sash: make A5 pass. A5d (the scaled actor
credit on A5's own reward) is at ~20k of 245,760 rounds beside these.

## A5e — the state-credit half of the build, on A5

**The one change relative to A5d:** `objective_hold` at
`crowding_exponent` 1.0, weight 1.0 (`configs/experiments/curriculum/
a5_hold.yaml`), the term the A3 speed screen's S1 read NULL under the
mean credit. Under `--credit actor` the close pays each model its own
value of it on the step it acted, so a body standing on a point it
reached is paid for standing there and a point pays a pot. Everything
else is A5d: from scratch, seeds 1 / 2 / 3, 245,760 rounds, 128 rounds
per update, `--ent-coef 0.003`, Wandb `curriculum-a5`, tag `a5e`.

Comparators: A5d (same credit, no hold term) seed for seed at the same
rounds — the paired reading of the term; A5b (0.260 / 0.180 / 0.160) as
the rung's record. Criteria: **PASS** success ≥ 0.95 on all three seeds
at 245,760 (greedy, n=100 on 700000+) with no in-run dip below 0.80
after the first rolling pass; **MOVES** ahead of A5d on 3/3 at 245,760;
**FAIL** neither. Prediction: the hold term stops the walk-off (`held`
rises past A5d's) and arrival improves; a pass is unlikely, since the
last empty point is the allocation failure the control also has.

## A5f — the bar cloned into the set network, then anchored PPO

The D-route applied to A5: `squad_march_take` solves the rung at 1.000
in 6.77 turns, and the record says the set network holds a scripted plan
from imitation that reward could not teach it (D1b: 0.96 on C3b from
1,200 games) and that anchored PPO holds it (D2: 0.94–0.96).

1. **The clone:** `just behaviour-clone-per-model squad_march_take
   configs/experiments/curriculum/a5.yaml 1200 40
   checkpoints/per_model/clones/take-a5-1200-s0.pt 0`, seeds 800000+,
   the last 240 games held out. Scored greedy at n=100 on 700000+.
   **PASS (by imitation)** if success ≥ 0.95, `held` ≥ 5.8, turns within
   a round of the script's 6.77; reported as what it is — the "start"
   axis, not a reward result.
2. **Anchored PPO from it** (only if the clone passes): A5's reward,
   `--warm-start-from` the clone, `--kl-ref-coef 10 --kl-ref-target
   0.03`, `--credit actor`, 3 seeds, 122,880 rounds, tag `a5f`.
   **HOLDS** if success ≥ 0.95 on 3/3 at the end; **IMPROVES** if turns
   are ahead of the clone's paired on 3/3.

What a pass here would mean: the architecture can play A5; the per-model
trainer cannot yet learn it from reward. Both halves go in the ladder
row.

## Amendment 1 — written 2026-09-18 04:05, after the clone's read and before any PPO round from it

**The clone (A5f step 1).** `take-a5-1200-s0.pt`, 1,200 games × 40
epochs, held-out match displacement 0.791 / declaration 0.971 / selector
0.639 / joint 0.513 (the opening order at chance, as with the escort).
Greedy at n=100 on 700000+: success **0.930**, `held` 5.93, `on_obj`
0.831, turns 6.96 (+0.19 ± 0.12 paired against the script's 6.77), vp
+68.0, coherent 0.744, stationary 0.18. Two hundredths under the
imitation pass mark, within one binomial SE (0.026) of it: **FAIL on the
letter, and the best policy on the per-model facade on this rung by a
wide margin** (A5b's best seed 0.260; A5e's best in-run 12% at 40k).

**Step 2 launched regardless, under the goal, as two arms**, both
`--warm-start-from` the clone, three seeds, 122,880 rounds, 128 per
update, `--ent-coef 0.003`, `--credit actor`, Wandb `curriculum-a5`:

- **A5f** as pre-registered: A5's reward, the anchor at `--kl-ref-coef
  10 --kl-ref-target 0.03` (D2's setting, which held the escort clone
  and improved nothing). Criteria as above: HOLDS ≥ 0.95 on 3/3 at the
  end; IMPROVES if turns are ahead of the clone's paired on 3/3.
- **A5g**, one change on top of A5f's recipe in each of two places, so
  it is not one change and is read as a screen: A5e's reward (the hold
  term under the actor credit, which is paying — A5e reads 8–12% rolling
  at 40k on two seeds where A5b never left 6%) and a looser anchor
  (`--kl-ref-coef 1 --kl-ref-target 0.10`), so reward can move the
  policy where D2's anchor pinned it. Tag `a5g`. Criterion: **PASS** ≥
  0.95 on 3/3 at the end with no dip below 0.80 after the first pass;
  a pass here is a pass by imitation plus reward, reported as such.

Prediction: A5f holds at 0.92–0.94 and does not improve (D2's result
again); A5g moves — the hold term fixes the walk-off that is the
clone's residual — and lands 0.93–0.97, a coin flip on the letter.

## Amendment 2 — written 2026-09-18 08:40, at 40k of 122,880 rounds on A5f/A5g, before any number for A5h

**Reads so far**, greedy at n=100 on 700000+ (the measure that decides),
the clone at 0.930:

| rounds | A5f s1 / s2 / s3 | A5g s1 / s2 / s3 |
|---|---|---|
| 20,480 | 0.950 / 0.920 / 0.930 | 0.920 / 0.910 / 0.960 |
| 40,960 | **0.850** / 0.930 / 0.940 | 0.910 / **0.950** / 0.940 |

Both arms sit in the clone's noise band (binomial SE 0.026); no seed set
clears 0.95 on all three. In-run reads on 500000+ run higher (A5f s1 at
96% rolling where its 40k checkpoint reads 0.850 held-out), so the
in-run curve is not the read. A5d at 80k: 1–5%; A5e at 80k: 8–9% on all
three seeds with 3.4–3.8 held — the hold term is a real but small edge.

**A5h, a third arm from the clone, launched now**: A5g's recipe with the
anchor loosened an order of magnitude (`--kl-ref-coef 0.1
--kl-ref-target 0.3`), the D2 finding's own next step ("the IMPROVES
question is the anchor's coefficient"). Under A5g's coefficient of 1 the
drift sits at its 0.10 target and reward moves the policy inside the
clone's band; under 0.1 it has room to move the residual — the point
left empty in six episodes of a hundred — and room to fall apart, which
plain PPO did by 2,560 rounds on D2. Seeds 1 / 2 / 3, 122,880 rounds,
`--credit actor`, `a5_hold.yaml`, tag `a5h`. Criterion as A5g's.
Prediction: it moves further than A5g in both directions across seeds —
one seed above 0.95 and one below 0.90 by 40k.

## Amendment 3 — written 2026-09-18 11:05, before any number for the second clone

**The 60k reads**, greedy at n=100 on 700000+: A5f **0.880 / 0.910 /
0.950**, A5g **0.900 / 0.970 / 0.900** (20k: 0.950 / 0.920 / 0.930 and
0.920 / 0.910 / 0.960; 40k: 0.850 / 0.930 / 0.940 and 0.910 / 0.950 /
0.940). Half the budget in, both arms still sit in the clone's band, one
seed touching 0.95–0.97 at a time and never the set. The anchored PPO
neither fixes nor breaks the clone's residual.

**A5d stopped at ~103k rounds** (216 periodic checkpoints kept under
`per-model-a5-2026-09-18-00-57-2*-s{1,2,3}a5d`): 1–6% rolling in-run
with 3–4 held at 100k, flat like A5b, and the box needs its memory for
the next clone. Its reading against the pre-registered MOVES criterion
is a FAIL at 100k: the scaled actor credit alone does not move A5's
arrival. The build's state-credit half is read from A5e, still running.

**A5f step 1 again, bigger: `take-a5-2000-s0`.** D1 → D1b moved the
escort clone 0.83 → 0.96 by quadrupling the games; this clone's residual
is a coordination error at the start (two squads choosing the same
point, the opening order being unlearnable) that more demonstrations of
the script's assignment should narrow. 2,000 games × 60 epochs, seeds
800000+, the last 400 held out, fit seed 0. Demonstrations are now
recorded at half precision (`compact_tokens`, pinned by a test) so the
fit holds ~14 GB instead of the ~27 GB the float32 recording would.
Scored greedy at n=100 on 700000+: PASS (by imitation) at ≥ 0.95,
`held` ≥ 5.8; if it passes, anchored PPO from it replaces A5f's start.
Prediction: 0.95–0.97.

## Amendment 4 — written 2026-09-18 12:50, all nine PPO-from-clone runs stopped

**The 80k reads**, greedy at n=100 on 700000+: A5f **0.930 / 0.910 /
0.960**, A5g **0.960 / 0.900 / 0.920**. Five reads over 20k–80k rounds put
every seed of both arms in the clone's band (0.85–0.97, the clone
0.930, binomial SE 0.026), one seed at 0.95–0.96 on each read and the
set never above the mark. A5h under the tenfold-looser anchor drifted
down instead: 84 / 89 / 88% rolling in-run at 40k against A5g's 90–96%
at the same rounds with the same reward.

**Stopped by decision** at ~89k (A5f, A5g) and ~44k (A5h) rounds: the
chance of a 0.95 × 3 read at the end from a true rate near 0.93 is
about one in fifty and would not survive the next read, so the end
reads would have served the record's completeness and not the goal;
the box's memory goes to the 2,000-game clone fit and to what follows
it. **A5f and A5g are read at 80k as their final row: HOLDS (the clone's
level on every seed), IMPROVES not shown. A5h: the anchor an order of
magnitude looser loses the clone slowly — the coefficient question's
answer is that 1 holds and 0.1 does not, and neither improves.**
Checkpoints every 512 rounds kept under the run directories.

**A5e** (stopped at ~110k, 11:30, for the same memory) reads 1–14%
rolling in-run over 60k–110k with 3.4–4.2 of six held on every seed,
against A5d's 1–6% and 2.5–4.0 at the same rounds: the hold term under
the actor credit is a small, consistent edge in arrival and nowhere
near a pass. A5d (stopped at ~103k) FAILS its MOVES criterion. Together:
**the actor credit build does not move A5's arrival; the per-model
trainer's failure on this rung is not a credit-scale problem.**

## Amendment 5 — written 2026-09-18 18:10, the reboot, and the stopped arms' final reads

**The machine rebooted at about 15:20** (a crash) and killed the
2,000-game clone fit of amendment 3 four and a half hours in, before
its first epoch had been written. It was relaunched at 17:01 with the
same recipe (2,000 games of `squad_march_take` on a5.yaml, seeds
800000+, 60 epochs, fit seed 0, `compact_tokens`; 312,672 decision
steps to fit, 78,108 held out) and is expected to finish around 23:00.
Its read and the criteria of amendment 3 are unchanged.

**Final reads of the arms stopped under amendment 4**, greedy at n=100
on 700000+ at each run's last checkpoint, written before the second
clone exists: A5d (~103k) **0.050 / 0.070 / 0.010**, A5e (~108k)
**0.170 / 0.060 / 0.060**, A5h (at 40,960) **0.950 / 0.840 / 0.820**.
A5f and A5g stand at their 80k rows. **The 1,200-game clone's seven
failures in a hundred are one event**: in every one the script's
assigned squad went to a point another squad was already taking, so
two squads shared a point and one stayed empty — never a late
arrival. The teacher breaks that tie by a rule the observation does not
carry (its own assignment order), which is the clone residual that a
bigger fit can shrink only if the tie-break is a function of the board;
the second clone's read decides whether it is. Every number in this
amendment is at the stops named in amendment 4 and at the memory's
prices — none is a read at budget.

## Amendment 6 — written 2026-09-18 20:55, the 2,000-game clone read; A5i launched from it

**The 2,000-game clone PASSES by imitation: 0.960**, held 5.96 of six,
turns 6.97 (+0.20 ± 0.10 against the bar's 6.77, paired), `on_obj`
0.838, coherent 0.771 (bar 0.822), greedy at n=100 on 700000+, beside
the 1,200-game clone re-read on the same seeds at 0.930 / 5.93 / 6.96.
Both criteria of amendment 3 are met (≥ 0.95, held ≥ 5.8) and the
prediction (0.95–0.97) was right. Held-out per-head match: declaration
0.977, displacement **0.821** (the 1,200-game clone 0.72), selector
0.639, joint 0.532 (the opening order, at chance as D1 recorded). The
fit finished at 20:42, earlier than the ~23:00 estimate, at 15 GB.

The census (n=100): 20.1 of 24 bodies on points and all six held by
turn 7, max stack 5.8 (the script 5.8). Four failures in a hundred:
point 4 empty in two, points 1 and 5 in one each — the 1,200-game
clone's seven were six at point 4 and one at point 0. Fewer failures,
less concentrated on one point; the event (two squads on one point)
is unchanged and the read does not separate "the tie is on the board"
from "the fit averages the tie better". More games moved the read by
three hundredths, which is the whole of the residual the goal needs.

**A5i launched 20:46**: A5f's pre-registered step 2 with the start
replaced — anchored PPO from `clones/take-a5-2000-s0.pt` on a5.yaml,
`--kl-ref-coef 10 --kl-ref-target 0.03`, `--credit actor`, `--ent-coef
0.003`, seeds 1 / 2 / 3, 122,880 rounds at 128 per update, in-run eval
every 512 rounds at n=30 on 500000+, Wandb `curriculum-a5` runs
`cb7pmoks` / `7hvtp7ju` / `m2ujugw1`, run dirs
`per-model-a5-2026-09-18-20-46-09-s{1,2,3}a5i`. Criteria as A5f's:
**HOLDS** if success ≥ 0.95 on 3/3 at the end (greedy, n=100 on
700000+); **IMPROVES** if turns are ahead of the clone's paired on 3/3;
read at 20k / 40k / 60k as A5f was. A HOLDS here is three seeds at or
above the rung's mark from a start that is itself at 0.960 — the
rung's PASS on the letter, reported as what it is: the "start" axis,
the bar imitated and held, not learned from reward. Prediction, from
A5f/A5g's twenty-four reads in the clone's band: every seed in
0.93–0.99, HOLDS on 3/3 with probability near one half (binomial at
0.96, n=100, three draws), IMPROVES not shown.

## Amendment 7 — written 2026-09-18 22:10, A5i at 40k of 122,880

The 20k and 40k reads of A5i, greedy at n=100 on 700000+:

| rounds | s1 | s2 | s3 | turns | held of 6 | coherent |
|---|---|---|---|---|---|---|
| the clone | 0.960 | — | — | 6.97 | 5.96 | 0.771 |
| 20,480 | **0.980** | 0.940 | **0.970** | 6.87 / 7.09 / 6.90 | 5.98 / 5.94 / 5.97 | 0.795 ×3 |
| 40,960 | **0.970** | **0.970** | **0.970** | 6.94 / 6.94 / 6.89 | 5.97 / 5.97 / 5.97 | 0.78–0.79 |

At 40k every seed is at 0.970 — one to two hundredths above the start,
inside its binomial band (0.960 ± 0.020), the first read on this rung
with all three seeds at or above the mark. Turns are within a tenth of
the clone's on every seed and a tenth or two behind the bar; the
in-run curve (n=30 on 500000+) sits at 96–99% rolling, three to four
points above the held-out row as on every arm before. Nothing here is
a verdict: HOLDS is read at 122,880, and A5f's seeds crossed 0.95 and
came back down between reads from a 0.930 start. Written before the
60k read.

## Amendment 8 — written 2026-09-19 00:55, A5i read at 122,880: HOLDS 3/3, the rung's mark on every seed

**A5i HOLDS on every seed: 0.960 / 0.980 / 0.960**, greedy at n=100 on
700000+ at 122,880 rounds (`last.pt`; runs `cb7pmoks` / `7hvtp7ju` /
`m2ujugw1`):

| row | success | turns | vs bar, paired | vs clone, paired | held of 6 | on obj | coherent | stat |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.77 | — | — | 6.00 | 0.856 | 0.822 | — |
| the 2,000-game clone | 0.960 | 6.97 | +0.20 ± 0.10 | — | 5.96 | 0.838 | 0.771 | 0.17 |
| s1 | **0.960** | 6.88 | +0.11 ± 0.10 | −0.09 ± 0.10 | 5.96 | 0.835 | 0.78 | 0.16 |
| s2 | **0.980** | 6.81 | +0.04 ± 0.08 | **−0.16 ± 0.10** | 5.98 | 0.851 | 0.79 | 0.16 |
| s3 | **0.960** | 7.04 | +0.27 ± 0.10 | +0.07 ± 0.13 | 5.96 | 0.845 | 0.79 | 0.16 |

Every read from 40k on had all three seeds at or above the mark (40k
0.970 ×3; 60k 1.000 / 0.970 / 0.970; 80k 1.000 / 0.980 / 0.960; end
0.960 / 0.980 / 0.960). **The in-run dip clause holds**: the rolling
in-run success (n=30 on 500000+, eight-evaluation window) first reached
95% within the first two evaluations on every seed and never fell below
94.0 / 95.8 / 95.0 after; the lowest single evaluation was 87 / 87 / 90.
**IMPROVES is not shown**: turns are ahead of the clone's paired on two
seeds (−0.09, −0.16) and behind on one (+0.07), against a 3/3 clause.
Sampled play (n=100 on 900000+, paired against greedy) is the same
policy: −0.1 / −1.2 / +0.1 ± 1.2 vp, held 5.96–5.98, coherent
0.775–0.791 against greedy's 0.807–0.818 — no do-nothing fingerprint
(stationary 0.16 either way, the clone's 0.17).

The census (n=100 on 700000+): 20.1–20.4 of 24 bodies on points and all
six held by turn 7 on every seed (the clone 20.1 / 6.0, the script
20.1 / 6.0), max stack 5.7–5.8 (the script 5.8). Failures: 4 / 2 / 4 in
a hundred, each a single short point, different points on different
seeds — the clone's event, at the clone's rate.

**Verdict on the arm as pre-registered: HOLDS 3/3, IMPROVES not shown.
Verdict on the rung: A5 PASSES on the letter** — success ≥ 0.95 on all
three seeds at the end, greedy at n=100 on 700000+, no in-run dip below
0.80 after the first rolling pass — **on the start axis**: the bar
cloned from 2,000 games (0.960) and held by PPO under a KL anchor
(coef 10, target 0.03) with the actor credit, on A5's own reward. The
reward added at most a tenth of a turn on two seeds and nothing to the
success rate beyond the clone's band. What it is not: the per-model
trainer learning A5 from reward, which read 0.26 / 0.18 / 0.16 at twice
this budget from scratch (A5b), 0.05 / 0.07 / 0.01 with the actor credit
(A5d), and 0.03–0.33 one point down (A5-points). The report carries
both halves in the ladder row.
