# A3 speed screen, arm S1: paying for standing on a point moves two seeds of three — NULL on the pre-registered letter

**Verdict first.** `a3.yaml` plus `objective_hold` at `crowding_exponent`
1.0 (#359), three seeds, 122,880 rounds, read greedy at n=100 on
700000+: success **0.940 / 0.810 / 0.900** against A3's own runs at the
same rounds and seeds, **0.700 / 0.800 / 0.770**. Seeds 1 and 3 are ahead
by 0.24 and 0.13; seed 2 by 0.01. That misses FASTER (0.95 on all three)
and AHEAD (+0.10 on all three), so the pre-registered verdict is
**NULL** — and the shape of the miss matters: two seeds moved by two to
five binomial SE, one did not move at all. The prediction on file
("FASTER on at least two seeds, rounds to 80% under 60k on all three")
is wrong on both counts: no seed passes at n=100, and rounds to 80% were
92k / 80k / 62k.

## Provenance

| field | value |
|---|---|
| date | 2026-09-15 (launched 10:00, exited 15:16–15:22) |
| GPU / no-GPU | GPU (RTX 4090), shared with eleven other per-model runs (the S2, S3 and A5b arms) |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; paired on init with A3 (same seed, same head) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/a3_hold.yaml` — `a3.yaml` with the one added term; unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; per seed against A3's read at 122,880 |
| comparator | A3's per-model runs at 122,880 (Wandb `curriculum-a3`: `wzalbuic` / `4x0p1q4a` / `jzmpi6pl`): 0.700 / 0.800 / 0.770, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81; the bar `squad_march_take`: 1.000, held 4.00, 5.28 turns, coherent 0.927 (identical on `a3_hold.yaml`) |
| opponent | none |
| budget | 122,880 rounds at 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003` |
| code revision | launched from the `bd15db9` working tree of `feature/curriculum-a3-speed` (PR #362), rebased the same day to `71c0f8f` with an identical tree |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-a3_hold-2026-09-15-10-00-58-s{1,2,3}a3h` |
| coherency | greedy 0.453 / 0.429 / 0.280 at 700000+; 0.41 / 0.44 / 0.29 greedy against 0.23 / 0.22 / 0.16 sampled at 900000+ |
| Wandb | `curriculum-a3-speed`: `g55n5x9h` s1 · `1eygass8` s2 · `ud74ov2g` s3 |
| pre-registration | `reports/2026-09-15-curriculum-A3-speed-preregistration.md` at `bd15db9` (10:00, before the launch) |

## The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 | — |
| A3 s1 / s2 / s3 at 122,880 | 0.700 / 0.800 / 0.770 | 6.05 / 5.95 / 5.81 | +0.77 / +0.67 / +0.53 | 3.49 / 3.60 / 3.65 | 0.65–0.74 | 0.57 / 0.48 / 0.44 | 38k, never, never / 23k, 91k, never / 37k, 102k, never |
| **S1 s1** | **0.940** | 5.56 | +0.28 ± 0.11 | 3.93 | 0.738 | 0.453 | 22k / 92k / never (peak 97 at 100k) |
| **S1 s2** | **0.810** | 5.89 | +0.61 ± 0.13 | 3.77 | 0.686 | 0.429 | 18k / 80k / never (peak 93 at 95k) |
| **S1 s3** | **0.900** | 5.93 | +0.65 ± 0.13 | 3.87 | 0.664 | 0.280 | 49k / 62k / 80k, then 5 dips below 80 in 84 evaluations |

**Greedy against sampled** (900000+, n=100, paired): +0.9 ± 1.4 / +2.3 ± 1.5 / +0.3 ± 1.5 vp; `held` 3.77 / 3.86 / 3.83 greedy against 3.80 / 3.85 / 3.78 sampled; coherency 0.41 / 0.44 / 0.29 greedy against **0.23 / 0.22 / 0.16** sampled — the lowest sampled coherency on the ladder.

**Health panel, last quarter** (240 updates per seed): explained variance
**0.61 / 0.59 / 0.68** against A3's 0.34–0.45 at the same rounds; clip
fraction 0.29 / 0.31 / 0.33; ratio p99 1.90 / 2.01 / 2.13; displacement
entropy 1.63 / 1.53 / 1.49; advantage std 0.41–0.50.

**The by-turn census** (bodies on objectives of 12 and points held of 4,
turn 5 → turn 8, n=20 on 700000+; A3 seed 1 at 20k read 5.0 → 2.3 bodies
and 1.9 → 1.3 held, the walk-off the arm was aimed at):

| checkpoint | success (n=20) | on objectives, turn 5 → 6 → 8 | held, turn 5 → 8 |
|---|---|---|---|
| s1 at 20,480 | 0.55 | 5.1 → 7.8 → 7.3 | 2.2 → 3.1 |
| s2 at 20,480 | 0.35 | 7.2 → 9.0 → **3.2** | 2.9 → **1.4** |
| s3 at 20,480 | 0.15 | 6.8 → **2.4** → 3.7 | 2.2 → 1.4 |
| s1 at 61,440 | 0.55 | 7.5 → 8.4 → 7.4 | 2.9 → 3.1 |
| s2 at 61,440 | 0.50 | 4.9 → 6.4 → 6.2 | 2.5 → 3.2 |
| s3 at 61,440 | 1.00 | 8.6 → 9.2 → 9.4 | 3.5 → 4.0 |
| s1 at 122,880 | 1.00 | 8.5 → 9.0 → 9.0 | 3.5 → 4.0 |
| s2 at 122,880 | 0.90 | 8.9 → 9.3 → 8.7 | 3.8 → 3.9 |
| s3 at 122,880 | 0.85 | 5.7 → 7.9 → 8.2 | 3.0 → 3.9 |

**The walk-off is still there at 20k on two seeds** — seed 2 puts nine
bodies on points at turn 6 and has three by turn 8; seed 3 drops from
6.8 to 2.4 in one turn. The term aimed at it did not remove it. Travel
gates on this arm read as A3's (one unit owning 2+ points 49–62% of
steps, fallback 40–53%), as they must: the travel term is unchanged.

## What it says

- **The term does what it was built to do and it is not enough.** The
  critic values the hold-paid episode far better (explained variance
  0.59–0.68 against 0.34–0.45), two seeds gain 0.13–0.24 at the cap, and
  the in-run curves reach 80% on all three seeds where A3's reached it on
  two. But no seed holds 0.95 at n=100 and seed 2 does not move. Under
  the pre-registered rule, three seeds at n=100 with one flat is a null.
- **Seed 2 is the informative one.** It reaches 80% in-run at the same
  rounds as A3's seed 2 (80k against 91k) and reads 0.810 against 0.800.
  Paying for standing changed nothing on that init, which says the
  walk-off at 20k is not the whole of what A3's ~200k rounds buy.
- **Coherency fell** (0.28–0.45 greedy against A3's 0.44–0.57, and
  0.16–0.23 sampled): nothing here pays for a squad staying together,
  and whatever the added term changed about the path, it did not help
  formation. Read the next section before attributing it to the pot.
- **The prediction was wrong twice.** Written before the run: FASTER on
  two seeds, rounds to 80% under 60k on all three. Read: no seed passes,
  62k–92k to 80%.

## Why the term could not do what the pre-registration asked of it

Read in `envs/per_model/reward_timing.py` after the numbers, not before.
On the per-model facade a **per-model state term is paid at the turn
close as the MEAN over alive models** — one scalar on the close step
(`_pay_close`: `weight × Σ calc(i) / n_alive`). The pot-split's total
still moves (three on each of two points earn more than six on one), but
no body receives its own share: the thirteenth model on a point and the
first are paid the same scalar at the same step, which is exactly the
flat-term defect the exponent exists to fix on the phase facade. The only
per-decision credit this facade pays is the action-class travel term, on
the step, to the model that acted. So S1 tested "the pot, as a mean, at
the close", not per-model credit for standing — and that is what a NULL
with two seeds moving looks like. ⚠ The same holds for `group_cohesion`
and every other per-model state term on this facade; a per-model
curriculum lever built on one of them is a global lever here until the
retimer pays state terms to the model that produced them.

## What was not done

- Not scored at 245,760 (the A3x budget); the criterion was the cap.
- No weight sweep; 1.0 was the one value pre-registered.
