# A3 speed screen, arm S3: warm-starting the spread rung from the one-point rung is AHEAD on 3 of 3 — the transfer the ladder promised, one rung early, against the prediction

**Verdict first.** `a3.yaml` unchanged, each seed started from the A2b
checkpoint of the same seed (`--warm-start-from`, #361), three seeds,
122,880 rounds, read greedy at n=100 on 700000+: success **0.910 / 0.940
/ 0.960** against A3 from scratch at the same rounds and seeds, **0.700 /
0.800 / 0.770** — ahead by **+0.21 / +0.14 / +0.19**, every seed past the
+0.10 bound. Two seeds sit under 0.95, so it is not FASTER on the letter;
it is **AHEAD** on all three, the only arm of the screen to clear a
clause. In-run it reached 80% at **22k / 11k / 19k** rounds where scratch
took 91k, 102k and never, and two seeds crossed 95% rolling at 62k and
71k. The prediction on file — faster to the first bodies and then NULL on
success, because an A2b policy has to unlearn sending everyone to one
point — is wrong: the one-point policy carried over, and what it had to
learn was only the spreading.

## Provenance

| field | value |
|---|---|
| date | 2026-09-15 (launched 10:03, exited 15:37–15:40) |
| GPU / no-GPU | GPU (RTX 4090), shared with eleven other per-model runs (the S1, S2 and A5b arms) |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; **not paired on init** — the init is A2b's `last.pt` of the same seed (122,880 rounds at `ent_coef` 0.003, revision `c2bcf26`), so per-seed differences carry init variance the other two arms do not |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/a3.yaml`, unchanged; unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; per seed against A3's read at 122,880 (layouts and seeds shared, init not) |
| comparator | A3's per-model runs at 122,880 (Wandb `curriculum-a3`: `wzalbuic` / `4x0p1q4a` / `jzmpi6pl`): 0.700 / 0.800 / 0.770, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81; the bar `squad_march_take`: 1.000, held 4.00, 5.28 turns, coherent 0.927 |
| opponent | none |
| budget | 122,880 rounds at 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, fresh optimizer |
| warm start | `checkpoints/per_model/per-model-a2-2026-09-15-02-44-23-s{1,2,3}a2b/last.pt` (A2b, PASS with drift: 1.000 / 1.000 / 0.990 on one objective of radius 6); the set network loads across the objective count unchanged, the displacement head is the same |
| code revision | launched from the `bd15db9` working tree of `feature/curriculum-a3-speed` (PR #362), rebased the same day to `71c0f8f` with an identical tree |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-a3-2026-09-15-10-03-16-s{1,2,3}a3w` |
| coherency | greedy 0.451 / 0.445 / 0.438 at 700000+; 0.43 / 0.44 / 0.45 greedy against 0.26 / 0.29 / 0.22 sampled at 900000+ |
| Wandb | `curriculum-a3-speed`: `zgc92is1` s1 · `w55z85n1` s2 · `z4tr3mfh` s3 |
| pre-registration | `reports/2026-09-15-curriculum-A3-speed-preregistration.md` at `bd15db9` (10:00, before the launch; the first launch at 10:00 died on the shell and started no run, relaunched 10:03) |

## The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 | — |
| A3 s1 / s2 / s3 at 122,880 | 0.700 / 0.800 / 0.770 | 6.05 / 5.95 / 5.81 | +0.77 / +0.67 / +0.53 | 3.49 / 3.60 / 3.65 | 0.65–0.74 | 0.57 / 0.48 / 0.44 | 38k, never, never / 23k, 91k, never / 37k, 102k, never |
| **S3 s1** | **0.910** | 5.85 | +0.57 ± 0.11 | 3.88 | 0.741 | 0.451 | **8k / 22k** / never (peak 93; last ten 80–93) |
| **S3 s2** | **0.940** | 5.54 | +0.26 ± 0.12 | 3.91 | 0.763 | 0.445 | **7k / 11k / 62k**; one dip below 80 in 119 evaluations after |
| **S3 s3** | **0.960** | 5.17 | −0.11 ± 0.10 | 3.96 | 0.800 | 0.438 | **11k / 19k / 71k**; two dips below 80 in 102 after |

Seed 3 arrives faster than the script (5.17 against 5.28 turns) with 3.96
points held, the A3x read at 245,760 in half the rounds.

**Greedy against sampled** (900000+, n=100, paired): −1.8 ± 1.5 / +1.8 ± 1.4 / −2.0 ± 1.2 vp; `held` 3.90 / 3.90 / 3.99 greedy against 3.73 / 3.84 / 3.88 sampled; coherency 0.43 / 0.44 / 0.45 greedy against **0.26 / 0.29 / 0.22** sampled. The sampled policy holds the points nearly as well as the greedy one, as on A3x and A4x.

**Health panel, last quarter** (240 updates per seed): explained variance
0.53 / 0.56 / 0.47 (A3 at the cap: 0.34–0.45); clip fraction 0.24 / 0.27
/ 0.30; ratio p99 1.81 / 1.86 / 1.96; displacement entropy **1.18 / 1.43
/ 1.18** (A3 at the cap: 1.7–1.8, A3x at the end: 1.4); advantage std
0.48–0.57.

**The by-turn census** (n=20 on 700000+):

| checkpoint | success (n=20) | on objectives, turn 5 → 6 → 8 | held, turn 5 → 8 | squads split |
|---|---|---|---|---|
| s1 at 20,480 | 0.85 | 8.0 → 8.8 → 8.9 | 3.3 → 3.8 | 37.5% |
| s2 at 20,480 | 0.95 | 9.0 → 9.1 → 9.1 | 3.2 → 4.0 | 30.2% |
| s3 at 20,480 | 0.60 | 5.5 → 6.3 → 7.5 | 2.3 → 3.6 | 25.2% |
| s1 at 61,440 | 0.85 | 8.6 → 9.0 → 8.7 | 3.4 → 3.9 | 25.0% |
| s2 at 61,440 | 0.95 | 9.2 → 9.4 → 9.3 | 3.8 → 4.0 | 35.2% |
| s3 at 61,440 | 0.95 | 9.3 → 9.3 → 8.9 | 3.9 → 3.9 | 23.3% |
| s1 at 122,880 | 0.95 | 8.2 → 9.0 → 9.0 | 3.5 → 4.0 | 20.6% |
| s2 at 122,880 | 0.95 | 8.5 → 8.6 → 8.8 | 3.6 → 4.0 | 28.5% |
| s3 at 122,880 | 1.00 | 9.5 → 9.5 → 9.5 | 4.0 → 4.0 | 27.8% |

**No walk-off and no stacking, from 20k on.** Bodies on objectives are
8.0–9.0 at turn 5 and rise or hold through turn 8 on every seed and
every checkpoint, against scratch's 5.0 → 2.3 at 20k; points held are
3.3–3.8 at 20k where scratch had 1.9. The A2b policy brought "arrive and
stay" with it and the 20k rounds bought the fourth point. The travel
gates read as A3's (one unit owning 2+ points 45–59% of steps) and the
squads split on 21–38% of squad-steps — the highest on the screen, and
the one thing the warm start did not carry.

## What it says

- **What A2b learned transfers.** The one-point policy walks bodies onto
  a point and keeps them there; started from it, the spread rung needs
  only the assignment, and the in-run curve reaches 80% in 11k–22k rounds
  against scratch's 91k–never. The prediction that it would have to
  unlearn stacking was wrong: at 20k rounds there is no stacking to see
  (3.3–3.8 points held, 8–9 bodies on them, no walk-off), so whatever
  unlearning happened was over inside the first 20k.
- **Two seeds are within noise of the pass mark and one passes**: 0.94
  and 0.91 at n=100 (SE 0.024–0.029) against 0.95, with the in-run curve
  holding 80–96 over the last ten evaluations. At the A3x budget this
  arm would very likely read 3/3; that is not measured.
- **The transfer rung T1 is now backed by a measurement one rung
  earlier than planned.** #340's T1 asks whether A5 warm-started from A4
  beats A5 from scratch; A3 from A2b does, on the rung where scratch was
  slowest. T1 keeps its own pre-registration; this is the prior.
- **Unpaired on init, as pre-registered.** Three seeds each from their
  own A2b checkpoint agree at +0.14 to +0.21, so the init variance is not
  what carries the result; but the old rule stands that seeds off one
  warm start are not independent samples, and each of these is off a
  different one.
- **Coherency is A3's** (0.44–0.45 greedy against 0.44–0.57): the
  warm start neither helps nor hurts formation; nothing here pays for it.

## What was not done

- Not scored at 245,760, where the two 0.91–0.94 seeds would be expected
  to clear 0.95.
- No control that warm-starts from a *random* A2-shaped init or from an
  A1x checkpoint, which would separate "any trained start" from "the
  one-point skill". One arm, one change.
