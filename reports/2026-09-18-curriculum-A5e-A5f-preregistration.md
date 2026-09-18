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
