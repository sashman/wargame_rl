# A3 speed screen, arm S2: one objective per squad in the travel assignment makes the rung SLOWER — HARMFUL, as the pre-registered clause names it

**Verdict first.** `a3.yaml` with `closest_objective_v2`'s assignment
made a matching (`one_objective_per_group: true`, #360), three seeds,
122,880 rounds, read greedy at n=100 on 700000+: success **0.830 /
0.690 / 0.610** against A3's own runs at the same rounds and seeds,
**0.700 / 0.800 / 0.770**. Seed 1 is ahead by 0.13; seeds 2 and 3 are
behind by 0.11 and 0.16, and `held` is 3.77 / 3.64 / **3.25** of 4 against
A3's 3.49–3.65. Under the pre-registered clauses that is **HARMFUL**: a
seed more than 0.10 below its A3 read. The gate the change was aimed at
moved exactly as measured before launch — no squad owns two points on
any step, squads are paid toward one target — and the policy learned
the rung more slowly for it. The prediction on file ("AHEAD but not
FASTER, the larger effect on coherency") is wrong on the verdict and
half-right on coherency (0.29–0.43 greedy, no better than A3's).

## Provenance

| field | value |
|---|---|
| date | 2026-09-15 (launched 10:00, exited 15:16–15:22) |
| GPU / no-GPU | GPU (RTX 4090), shared with eleven other per-model runs (the S1, S3 and A5b arms) |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; paired on init with A3 (same seed, same head) |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/a3_match.yaml` — `a3.yaml` with the one flag; unrefereed by design |
| decode | none |
| paired | per episode against `squad_march_take`; per seed against A3's read at 122,880 |
| comparator | A3's per-model runs at 122,880 (Wandb `curriculum-a3`: `wzalbuic` / `4x0p1q4a` / `jzmpi6pl`): 0.700 / 0.800 / 0.770, held 3.49 / 3.60 / 3.65, turns 6.05 / 5.95 / 5.81; the bar `squad_march_take`: 1.000, held 4.00, 5.28 turns, coherent 0.927 (identical on `a3_match.yaml`) |
| opponent | none |
| budget | 122,880 rounds at 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003` |
| code revision | launched from the `bd15db9` working tree of `feature/curriculum-a3-speed` (PR #362), rebased the same day to `71c0f8f` with an identical tree; the flag ships on this branch with `tests/test_shaping_gates.py::TestTheAssignment::test_one_objective_per_group_makes_the_assignment_a_matching` |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-a3_match-2026-09-15-10-00-58-s{1,2,3}a3m` |
| coherency | greedy 0.339 / 0.294 / 0.425 at 700000+; 0.33 / 0.29 / 0.43 greedy against 0.18 / 0.15 / 0.13 sampled at 900000+ |
| Wandb | `curriculum-a3-speed`: `q432pqkj` s1 · `sgrae329` s2 · `u1943s80` s3 |
| pre-registration | `reports/2026-09-15-curriculum-A3-speed-preregistration.md` at `bd15db9` (10:00, before the launch) |

## The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 5.28 | — | 4.00 | 0.892 | 0.927 | — |
| A3 s1 / s2 / s3 at 122,880 | 0.700 / 0.800 / 0.770 | 6.05 / 5.95 / 5.81 | +0.77 / +0.67 / +0.53 | 3.49 / 3.60 / 3.65 | 0.65–0.74 | 0.57 / 0.48 / 0.44 | 38k, never, never / 23k, 91k, never / 37k, 102k, never |
| **S2 s1** | **0.830** | 6.10 | +0.82 ± 0.12 | 3.77 | 0.608 | 0.339 | 71k / 77k / never (peak 97 at 106k; last ten 93–70) |
| **S2 s2** | **0.690** | 6.52 | +1.24 ± 0.14 | 3.64 | 0.563 | 0.294 | 14k / 80k / never (peak 87 at 79k; **last ten 30–63**) |
| **S2 s3** | **0.610** | 6.96 | +1.68 ± 0.13 | 3.25 | 0.474 | 0.425 | 39k / never / never (peak 70 at 95k; last ten 30–70) |

**Greedy against sampled** (900000+, n=100, paired): −5.3 ± 1.7 / +0.8 ± 1.6 / +5.8 ± 1.3 vp; `held` 3.82 / 3.63 / 3.27 greedy against 3.67 / 3.47 / 2.98 sampled; coherency 0.33 / 0.29 / 0.43 greedy against **0.18 / 0.15 / 0.13** sampled; the sampled policy takes 104–119 decisions an episode against the greedy 96–111 — it is still wandering at the end.

**Health panel, last quarter** (240 updates per seed): explained variance
**0.26 / 0.28 / 0.38** — below A3's 0.34–0.45 at the same rounds; clip
fraction 0.26 / 0.35 / 0.35; ratio p99 1.79 / 2.06 / 2.05; displacement
entropy 1.75 / 1.86 / 1.88 (A3 at its cap: 1.7–1.8); advantage std
0.47–0.62.

**The by-turn census and the travel gates** (n=20 on 700000+):

| checkpoint | success (n=20) | on objectives, turn 5 → 6 → 8 | held, turn 5 → 8 | one unit owns 2+ | fallback | squads split |
|---|---|---|---|---|---|---|
| s1 at 20,480 | 0.35 | 7.6 → **3.4** → 3.3 | 2.7 → 1.5 | 0.0% | 19.0% | 5.5% |
| s2 at 20,480 | 0.55 | 6.7 → 6.4 → 4.6 | 3.1 → 2.3 | 0.0% | 21.7% | 13.4% |
| s3 at 20,480 | 0.65 | 6.4 → 9.2 → 6.1 | 3.0 → 2.7 | 0.0% | 22.3% | 8.7% |
| s1 at 61,440 | 0.45 | 5.9 → 5.1 → 5.0 | 2.8 → 2.8 | 0.0% | 29.1% | 10.3% |
| s2 at 61,440 | 0.50 | 6.4 → 4.5 → 4.8 | 3.0 → 2.8 | 0.0% | 26.9% | 7.6% |
| s3 at 61,440 | 0.45 | 6.0 → 6.2 → 6.5 | 2.6 → 3.0 | 0.0% | 28.0% | 11.9% |
| s1 at 122,880 | 0.90 | 6.6 → 7.6 → 7.7 | 3.2 → 3.9 | 0.0% | 29.6% | 15.5% |
| s2 at 122,880 | 0.80 | 5.6 → 6.7 → 7.1 | 3.0 → 3.7 | 0.0% | 33.1% | 14.6% |
| s3 at 122,880 | 0.60 | 5.2 → 5.6 → 5.3 | 2.7 → 3.2 | 0.0% | 31.5% | 13.7% |

The gate did what it was built to do (no unit owns two points on any
step, against 49–62% on the other arms). Two things the table adds: the
walk-off at 20k is here too (seed 1: 7.6 bodies at turn 5, 3.4 at turn
6), so it is not the assignment's doing either; and at the end the
policies put only **5.3–7.7 of 12 bodies** on points (A3 at the cap: 8.0;
A3x: 9.7; the bar: 10.7) while splitting squads on 14–16% of squad-steps
against the bar's 1.1% under the same rule — the learned policy sends
fewer bodies and sends them apart.

## What it says

- **The change is real and it hurts.** Before launch the bar's gate
  census on this config read one unit owning 2+ points on 0.0% of steps
  (49.1% under the per-objective rule) and squads split on 1.1% (11.3%).
  Trained under it, two seeds finish 0.11–0.16 behind A3, one point
  short on seed 3 (`held` 3.25), and the in-run curves of seeds 2 and 3
  fall back to 30–70% over the last ten evaluations after peaking at 87
  and 70.
- **A matching makes the target switch more often, and a switch is
  free.** Under the per-objective rule an objective's owner is whichever
  group has the closest model, which is stable while squads walk
  straight. Under a matching the assignment re-solves every step over
  all four squads, so one squad's move can flip two other squads' targets
  — and `closest_objective_v2` pays zero on the step a target changes and
  re-anchors, so a policy whose targets keep flipping is paid less for
  the same walking. The gate census before launch counted assignments,
  not switches; the switch rate is the statistic that would have caught
  this, and it was not on the list.
- **The critic is worse under it** (explained variance 0.26–0.38 against
  0.34–0.45): the travel income is less predictable from the state when
  the target it pays toward depends on everyone else's position.
- **Nothing about coherency improved** (0.29–0.43 greedy against A3's
  0.44–0.57), against the prediction. Paying a squad toward one target
  did not keep it together.
- **Stop nominating `closest_objective_v2`'s assignment rule.** This is
  the record's fifth empty-or-worse result on that term (`CLAUDE.md` § The
  travel reward, audited, lists four). A matching is the natural fix for
  the defect the gate census shows, and it lost; the flag stays, default
  off, because the census that motivated it is real, but the next lever
  on this rung is not this term.

## What was not done

- No measurement of the target-switch rate; the mechanism above is the
  reading that fits the numbers, not a measured cause.
- Not scored at 245,760.
