# Curriculum — the C, D and E rungs (an enemy, guns, cloning, both armies armed)

The rungs after A: an enemy on a point (C1), its guns (C2), ours (C3 / C3b), the escort cloned and then trained (D1–D3), and both armies armed (E1; E2 and E3 were pre-registered and never run). Each pre-registration is followed by its result, word for word as first written.

## Contents

- [Pre-registration: curriculum rung C1 — A3 with one enemy unit standing on a point, nobody shooting](#2026-09-15-curriculum-C1-preregistration)
- [Curriculum rung C1: with an enemy squad standing on a point, the per-model arm passes on two seeds of three — FAIL as pre-registered](#2026-09-16-curriculum-c1-two-of-three)
- [Pre-registration: curriculum rung C2 — the enemy squad on the point fires at whoever approaches](#2026-09-16-curriculum-C2-preregistration)
- [Curriculum rung C2: the enemy squad on the point fires back — two seeds clear the mark, one is a hundredth short, and the whole-army control fails on all three](#2026-09-17-curriculum-c2-one-seed-one-hundredth-short)
- [Pre-registration: curriculum rung C3 — one of our squads is armed, and the blockers have to die before the unarmed squads walk in](#2026-09-17-curriculum-C3-preregistration)
- [Curriculum rung C3: one armed squad has to clear the blockers before the unarmed squads walk in — the arm found the dash, not the plan; the run with no head start found the plan and not the walk](#2026-09-17-curriculum-c3-the-dash-not-the-plan)
- [Pre-registration: curriculum rung C3b — C3 with success read at the end of the game, so a dash has to hold the point](#2026-09-17-curriculum-C3b-preregistration)
- [Curriculum rung C3b: with success read at the end of the game, nobody holds the point — FAIL on every seed of every trainer, and the arm's failure is now the plan's second half](#2026-09-17-curriculum-c3b-nobody-holds-the-point)
- [Pre-registration: curriculum rung D1 — the escort cloned into the set network, scored on the final-board rung](#2026-09-17-curriculum-D1-preregistration)
- [Curriculum rung D1: the escort cloned into the set network — the clone holds the plan and misses the mark; FAIL on the letter, with a defect in the match clause](#2026-09-17-curriculum-d1-the-clone-holds-the-plan)
- [Pre-registration: curriculum rung D1b — the escort cloned from four times the demonstrations](#2026-09-17-curriculum-D1b-preregistration)
- [Curriculum rung D1b: the escort cloned from four times the demonstrations — the clone reaches the escort's standard on the rung, and the fidelity bound was set where no clone can reach it](#2026-09-17-curriculum-d1b-four-times-the-games)
- [Pre-registration: curriculum rung D2 — PPO from the escort clone](#2026-09-17-curriculum-D2-preregistration)
- [Pre-registration: curriculum rung D2b — PPO from the escort clone with its critic fitted first](#2026-09-17-curriculum-D2b-preregistration)
- [Pre-registration: curriculum rung D2c — PPO from the escort clone with the KL anchor, cold critic](#2026-09-17-curriculum-D2c-preregistration)
- [Pre-registration: curriculum rung D2d — PPO from the escort clone with its critic fitted and the KL anchor](#2026-09-17-curriculum-D2d-preregistration)
- [Curriculum rung D2: PPO from the escort clone — reward destroys a copied plan unless the weights are anchored to it, the critic is not the lever, and under the anchor reward improves nothing](#2026-09-17-curriculum-d2-the-anchor-not-the-critic)
- [Pre-registration: curriculum rung D3 — anchored PPO from a clone of a WEAKER teacher](#2026-09-19-curriculum-D3-preregistration)
- [Curriculum rung D3: anchored PPO holds a clone with two thirds of the rung still to gain, and takes no step toward it — the D-route is imitation only](#2026-09-19-curriculum-d3-the-anchor-is-a-brake-with-no-engine)
- [Pre-registration: curriculum rung E1 — one unit, one objective, no opponent](#2026-09-14-curriculum-E1-preregistration)
- [Pre-registration: curriculum rung E1 — two armed armies on a flat board](#2026-09-17-curriculum-E1-preregistration)
- [Curriculum rung E1: two armed armies on a flat board — the whole-army control learns the guns and loses the formation; the per-model arm learns neither, and A5i's walk does not survive contact](#2026-09-19-curriculum-e1-the-walk-does-not-survive-contact)
- [Pre-registration: curriculum rung E2 — three units, three objectives, no opponent](#2026-09-14-curriculum-E2-preregistration)
- [Pre-registration: curriculum rung E3 — eight units, six objectives, no opponent](#2026-09-14-curriculum-E3-preregistration)

---

<a id="2026-09-15-curriculum-C1-preregistration"></a>

## Pre-registration: curriculum rung C1 — A3 with one enemy unit standing on a point, nobody shooting

Written 2026-09-15, **before any training number on this rung exists**,
on branch `feature/curriculum-c1` (stacked on PR #364). Issue #365,
parent question #340, rung **C1** — the "enemy" axis: the first rung on
which an opposing army exists.

### The scenario

`configs/experiments/curriculum/c1.yaml` is `a3.yaml` — four squads of
three, four objectives in a column at x=35, radius 4, eight rounds, the
60×44 board, Move 6, success `all_objectives_occupied` — plus **one
enemy unit of three**, declared by `group_id`, deployed in a 6×6 band
around the point at (35, 27) and holding it (`scripted_baseline` wrapping
`hold_deployment`). Nobody shoots. The success criterion counts **our**
bodies on every point, so the rung asks whether the squad bound for the
enemy's point still gets onto the disc: a move may not end within
engagement range of an enemy, three enemy bases sit on a radius-4 disc,
and the point can be occupied but never held.

**The build the rung needed, shipped with it.** `WargameEnvConfig.validate_coherency`
derived every side's unit size as `count // max_groups`, so a declared
three-model enemy unit under the player's `max_groups: 4` was refused as
"every model in its own unit" — a config the env runs correctly, since
the battle factory honours declared `group_id`s. It now judges a declared
list by its own smallest unit and keeps the count rule for a side that
declares none; `tests/test_coherent_deployment.py` pins both.

### The bar, measured first

`squad_march_take`, seeds 700000+ at n=100, **both facades identical on
every shared field** (`just measure-bridge`): success **0.990**, `held`
3.00 (the enemy's point is contested 3 v 3, never held), turns 4.98,
`on_obj` 0.871, vp 8.2 ± 0.6, coherent 0.933, decisions 4.98 turns. The
script clears the pass mark; the rung is not sent back to design. The
1% it misses is the census's own friendly gridlock around a crowded disc.

### The two arms

| | whole-army control | per-model arm |
|---|---|---|
| trainer | `train.py` via `just train-curriculum-control` | `train_per_model.py` |
| budget | 60 epochs of 2048 steps, extended once to 120 if still rising | runs to **122,880 rounds** at 128 per update; **read at the checkpoint the control's budget names** (below); resumed to 245,760 if the 2×-cap rule applies (#346: cadences passed explicitly) |
| start | from scratch | `--warm-start-from checkpoints/per_model/per-model-a3-2026-09-15-04-12-26-s{1,2,3}a3/last.pt` (A3x, 245,760 rounds, PASS 3/3), seed for seed — the ladder's rule for the C rungs |
| seeds | 1, 2, 3 | 1, 2, 3 |
| flags | defaults (`ent_coef` 0.03) | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003` |
| in-run eval | the trainer's own, seeds 500000+ | every 512 rounds, n=30, 500000+ |
| logging | Wandb `curriculum-c1`, tag `c1-ctl` | Wandb `curriculum-c1`, tag `c1` |

**The budget rule** (A0–A4x): the per-model budget is 6× the control's
slowest rounds-to-pass (an epoch is 2048 rounds on a movement-only rung),
floor 20,480, cap 122,880; when 6× exceeds the cap the budget is 2× the
cap, read once. Both arms launch together to keep the GPU full; the
per-model arm's read is taken from `pm-<budget>.pt` at the budget the
control sets, and every checkpoint after it is unread unless the 2×-cap
rule fires. If the control fails its own criterion at 120 epochs, that is
a finding about the control (A5's clause), and the per-model arm is read
at 245,760 beside it.

**No per-model arm from scratch.** A3's own scratch runs are the
reference for what this network does on the empty table (0.70 / 0.80 /
0.77 at the cap, 0.96 / 0.96 / 0.97 at 245k); the C rungs climb from the
rung below by the ladder's design, and the S3 arm this morning is the
measurement behind that rule.

### Criteria

Read greedy, no decode, n=100 on 700000+.

- **PASS:** success ≥ 0.95 on **all three** seeds at the budget, and no
  in-run dip below 0.80 after the first rolling pass (A2b's drift
  clause; the n=30 wobble on a rung both trainers pass is not a dip).
- **FAIL:** any seed below 0.95 at the budget. The next step is a
  recording of the failing episodes: whether the short point is the
  enemy's, and where the squad bound for it ends.
- **NULL (scenario):** only if the script fails its own criterion. It
  does not (0.990).
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts, not criteria:** vp paired against the script per episode
  (bar 8.2 ± 0.6 — #340's "vp ≥ bar − 1 SE" is a readout on this rung,
  since nothing shoots and the contested point pays nobody); `held`
  (3.00 is the ceiling); turns against the script's 4.98 and the
  control's; `on_obj`; coherency greedy and sampled; rounds to rolling
  50 / 80 / 95%; the panel over the last quarter; the control beside
  every row.

Power: binomial SE 0.022 at p=0.95, n=100; the bar's 0.990 sits two SE
above the bound.

### What I expect (a guess, written so it can be wrong)

The per-model arm, starting from a policy that already spreads over
four points, passes inside 40k rounds on all three seeds — the enemy's
disc costs the bound squad a turn of shuffling, not a new skill — and
arrives at the script's speed where the control is a round slower. The
control passes at 60 epochs on all three (its A3 self needed 120, but it
is learning the same scenario plus three obstacles). If instead the
per-model arm reads under 0.95 on a seed, the short point is the
enemy's on most of the failing episodes and the bound squad ends within
a base of the disc — the engagement-range endpoint rule, not
allocation.

### Amendment 1 — written 2026-09-16 13:02, after the control's 60-epoch read and before any per-model number

**The crash.** The box hard-crashed at 16:38:51 on 2026-09-15, seconds
after this rung's six runs launched (journal stops mid-line, no
out-of-memory record; rebooted 09:54 the next morning). Both arms
relaunched fresh at 12:43 on 2026-09-16 with the same flags; the crashed
stubs are deleted and their Wandb runs are junk (named on #365).

**The control at 60 epochs**, n=100 on 700000+, `last.ckpt`: success
**0.920 / 0.960 / 0.970**, `held` 3.12 / 3.04 / 3.02 (it sometimes
out-numbers the enemy on its point), turns 6.92 on every seed against
the script's 4.98, `on_obj` 0.90–0.94, coherent 0.71–0.78, vp −1.4 to
−2.8 against the script's +8.2. In-run (n=30, rolling five): 80% at
epochs 28 / 29 / 27; 95% at epoch **55 / never / 51**, the last ten
evaluations wobbling 0.87–1.00 on all three. Wandb `curriculum-c1`:
`dt6uu1f4` / `fj4sybie` / `s30qxkjv`.

**The extension.** One seed under 0.95 at n=100 and one that never
crossed 95% in-run is the shape A3's and A5's controls were extended
for, so the control gets its once-only extension to **120 epochs**
(`--resume-ckpt-path` from `last.ckpt`, launched 13:01, run suffix
`-ctl-x2`). By the symmetric-cap rule (A3x) the per-model arm's budget
is therefore **245,760 rounds**, read once at the end: the arm running
to 122,880 will be resumed in place (`--resume-from`, cadences passed
explicitly per #346) when it gets there, and its 122,880 checkpoint is
a readout, not the read. The criteria are unchanged.

The control's curves were read from the runs' local `.wandb` files
(`scratchpad/read_wandb_local.py`), not the API, which stays untouched
while runs train.

---

<a id="2026-09-16-curriculum-c1-two-of-three"></a>

## Curriculum rung C1: with an enemy squad standing on a point, the per-model arm passes on two seeds of three — FAIL as pre-registered

**Verdict first.** On the enemy rung (#340, arm #365) — A3 plus one
enemy unit of three standing on one of the four points, nobody shooting
— the per-model arm warm-started from A3x reads greedy success **0.990 /
0.930 / 0.990** at 245,760 rounds, n=100, `held` 3.15 / 2.96 / 3.06 of
3 (the enemy's point can be occupied but never held), turns **5.10 / 5.37
/ 5.01** against the script's 4.98 and the whole-army control's
6.82–7.09. Seed 2 sits under 0.95 at the budget, so this is **FAIL** on
the pre-registered clause. It is a fail by one seed on a plateau: seed 2
crossed 95% in-run at 4,608 rounds, never dipped below 80 after, and sat
at 85–95 for the remaining 240k. The control passes 3 of 3 at 120 epochs
(0.980 / 0.990 / 0.990) two rounds slower than the script. The
prediction on file — a pass inside 40k rounds on all three — is wrong on
one seed and right on the speed.

### Provenance

| field | value |
|---|---|
| date | 2026-09-16 (launched 12:43 after the 2026-09-15 crash killed the first launch at 384 rounds; first leg exited 17:28–17:30, resumed 17:30, exited 22:00–22:02) |
| GPU / no-GPU | GPU (RTX 4090), shared with eight other per-model runs (A5b, T1) and, for the first hour, the control |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; **not paired on init** — each seed starts from A3x's `last.pt` of the same seed (245,760 rounds, PASS 3/3) |
| n | 100 at seed base 700000 (final and the 122,880 readout); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled) |
| config | `configs/experiments/curriculum/c1.yaml` — unrefereed by design; success `all_objectives_occupied` counts our bodies |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`, both facades identical: success 0.990, held 3.00, turns 4.98, `on_obj` 0.871, vp +8.2 ± 0.6, coherent 0.933 |
| opponent | `scripted_baseline` wrapping `hold_deployment`: three models, one declared unit, on the point at (35, 27), unarmed |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 (#346: cadences passed explicitly); control 60 epochs extended once to 120 |
| code revision | `2a33b50` on `feature/curriculum-c1` (PR #366) for the launch; docs-only commits after |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c1-2026-09-16-12-43-32-s{1,2,3}c1` |
| coherency | greedy 0.524 / 0.500 / 0.632 at 700000+; 0.54 / 0.56 / 0.64 greedy against 0.34 / 0.34 / 0.47 sampled at 900000+ |
| Wandb | `curriculum-c1`: per-model `s6t9rpw3` / `an6yas9o` / `sypr0o0v`, resumed halves `fk7v2efa` / `t3wak5rd` / `6kqsj7n2`; control `dt6uu1f4` / `fj4sybie` / `s30qxkjv`, extended `w69w9t6h` / `wbd13z9q` / `5yk8p1il` |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-15-curriculum-C1-preregistration` at `2a33b50` (2026-09-15 16:38), amendment 1 at `4dc1c5f` (2026-09-16 13:01, after the control's 60-epoch read and before any per-model number) |

### The read

| row | success | turns | vs bar, paired | held | on obj | coherent | rounds to rolling 50 / 80 / 95% |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 0.990 | 4.98 | — | 3.00 | 0.871 | 0.933 | — |
| **s1 at 245,760** | **0.990** | 5.10 | +0.12 ± 0.07 | 3.15 | 0.801 | 0.524 | 2.6k / 2.6k / 40k; last ten 95–100 |
| **s2 at 245,760** | **0.930** | 5.37 | +0.39 ± 0.10 | 2.96 | 0.843 | 0.500 | 2.6k / 2.6k / 4.6k; last ten 85–95 |
| **s3 at 245,760** | **0.990** | 5.01 | +0.03 ± 0.06 | 3.06 | 0.813 | 0.632 | 2.6k / 2.6k / 10k; last ten 90–100 |
| s1 / s2 / s3 at 122,880 (readout) | 0.870 / 0.980 / 0.940 | 5.56 / 5.37 / 5.43 | +0.58 / +0.39 / +0.45 | 2.86 / 3.09 / 3.04 | 0.73–0.81 | 0.47–0.52 | — |
| control at 60 epochs | 0.920 / 0.960 / 0.970 | 6.92 ×3 | +1.94 | 3.02–3.12 | 0.90–0.94 | 0.71–0.78 | 95% at epoch 55 / never / 51 |
| control at 120 epochs | 0.980 / 0.990 / 0.990 | 7.09 / 6.82 / 6.83 | +1.84 to +2.11 | 2.99–3.00 | 0.93–0.98 | 0.72–0.79 | 95% at epoch 64 / 75 / 65 |

**Greedy against sampled** (900000+, n=100, paired): +0.5 ± 0.8 / +1.0 ± 0.7 / +0.5 ± 0.6 vp; `held` 3.20 / 3.09 / 3.01 greedy against 3.13 / 3.00 / 3.05 sampled; coherency 0.54 / 0.56 / 0.64 greedy against **0.34 / 0.34 / 0.47** sampled. The sampled policy holds the points as well as the greedy one and walks the squads further apart, as on every rung.

**Health panel, last quarter** (480 updates per seed): explained variance
0.52 / 0.46 / 0.50, clip fraction 0.24 / 0.27 / 0.26, ratio p99
1.88–1.91, displacement entropy 0.85 / 1.10 / 1.18, declaration entropy
≤ 0.02, advantage std 0.34–0.40. In-run dips below 80 after the first
rolling pass: 3 / 0 / 8 of 402 / 472 / 461 evaluations.

**The by-turn census** (n=20 on 700000+, bodies on objectives of 12 and
points held of 3, turn 5 → 8, with which point is empty at the end):

| policy | success (n=20) | turns | on objectives, turn 5 → 8 | held, turn 5 → 8 | point empty at the end (y = 8 / 17 / **27, the enemy's** / 36) | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 0.95 | 5.20 | 9.4 → 10.0 | 3.0 → 3.0 | 0 / 0 / **0.05** / 0 | 2.8 |
| s1 | 1.00 | 5.00 | 10.0 → 10.0 | 3.2 → 3.3 | 0 / 0 / 0 / 0 | 3.6 |
| s2 | 0.95 | 5.40 | 10.6 → 10.4 | 3.1 → 3.0 | 0 / 0.05 / 0 / 0.05 | 3.9 |
| s3 | 1.00 | 4.90 | 9.5 → 9.5 | 3.1 → 3.1 | 0 / 0 / 0 / 0 | 3.5 |

**The short point is not the enemy's.** The script's own one miss in
twenty is at the enemy's disc — the friendly gridlock the pre-registration
named. The learned policies get onto that disc every time; seed 2's
misses are the neighbouring points at y=17 and y=36, i.e. allocation
between empty points, the A3 failure mode, not the engagement-range
endpoint rule. The arm also out-numbers the enemy on its point often
enough to *hold* it on some episodes (`held` 3.0–3.3 of a 3.0 ceiling
for the script), stacking up to 3.5–3.9 on one point against the
script's 2.8.

### What it says

- **The warm start carries the rung's skill and the enemy's disc costs a
  turn of shuffling, as predicted** — every seed is at 80% in-run by
  2,560 rounds and two pass at n=100 within 0.1 turns of the script.
  The whole-army control needed 131k–154k rounds to reach 95% in-run and
  arrives two rounds behind the script.
- **One seed plateaus at 0.93 and that is the verdict.** Seed 2 passed
  first (4.6k) and then sat at 85–95 in-run for 240k rounds without a
  dip and without a climb; its 122,880 readout was the best of the three
  (0.980) and its 245,760 read the worst (0.930). Binomial SE at n=100
  is 0.026 at that rate, so the two reads are two SE apart: a plateau
  around 0.95 read twice, not a drift. The rule reads it as a fail, as
  A1 and A3 were read at their caps, and the record says what happened
  when those runs were given more rounds — and this one had 245,760.
- **The per-model arm is faster than the control on every seed and
  slightly below it on the bar**: turns 5.0–5.4 against 6.8–7.1, success
  0.93–0.99 against 0.98–0.99. Both had 245,760 rounds on this scenario;
  the arm also had A3x's 245,760 on the empty table behind it, which is
  what the C rungs' warm-start rule spends. Read at equal rounds on this
  scenario, the arm was at 80% in-run by 2.6k where the control reached
  it at 55k–60k.
- **The enemy's disc is not the failure; allocation is.** The
  pre-registration's FAIL clause asked for a recording of which point is
  short: it is the neighbours of the enemy's point, never the enemy's.
  The engagement-range endpoint rule costs the bound squad nothing the
  policy cannot learn in a few thousand rounds; what remains on seed 2
  is the residual assignment error A3 carries at 0.96–0.97.
- **Coherency 0.50–0.63 greedy** — unpaid, as on every rung so far.

### What was not done

- No second budget: the symmetric cap already put the arm at 245,760.
- No recording of seed 2's failing seven episodes beyond the census.

---

<a id="2026-09-16-curriculum-C2-preregistration"></a>

## Pre-registration: curriculum rung C2 — the enemy squad on the point fires at whoever approaches

Written 2026-09-16, **before any training number on this rung exists**,
on branch `feature/curriculum-c2` (stacked on PR #366). Issue #368,
parent question #340, rung **C2** — the "guns" axis, their side first.

### The scenario

`configs/experiments/curriculum/c2.yaml` is `c1.yaml` — four unarmed
squads of three, four points in a column at x=35, radius 4, eight
rounds, one enemy unit of three standing on the point at (35, 27) —
with the enemy unit **armed and firing**: a rifle of range 12 and one
attack per model, the `hold_and_shoot` seat (holds its ground, fires at
the nearest valid unit under the env's own mask each shooting phase),
and the shooting phase **stepped** (`skip_phases: [command, charge,
fight]`; every rung below stepped movement only, and the default skips
shooting whatever the weapons say). Success is still
`all_objectives_occupied` over our bodies; the reward is C1's, so nothing
pays for a casualty avoided or taken.

**The build the rung needed, shipped with it.** `hold_deployment` shoots
at nothing on purpose; `hold_and_shoot` is that seat with
`squad_march_shoot`'s target rule, registered under its own name and
pinned by `tests/test_hold_and_shoot.py` (the blockers never move;
approaching squads lose bodies).

### Why only their guns — the design change from #340's letter

#340's C2 row reads "C1 with every unit armed, opponent shoots back".
Measured before this file was written, on that scenario: our twelve
rifles wipe the three blockers in two shooting phases, and **the phase
facade ends the battle on an opponent wipe** (#317 — against the rules,
which the per-model facade follows: a wiped army keeps ceding ground
until the battle ends). The scripted bar therefore reads **0.380** on the
phase facade and **1.000** on the per-model facade on the same seeds
(`just measure-bridge`: "BRIDGE DIVERGES"), with `alive` 0.95 on both —
the guns kill almost nobody of ours; the wipe rule ends the game before
the last squad arrives. No whole-army control could be read beside a
per-model arm on that scenario. So this rung arms **their** side only,
and our guns become C3's axis, where they are one squad's and the
blocker has to die before the unarmed squads arrive.

⚠ **Standing consequence for the ladder, to be written into CLAUDE.md
when this rung lands:** on any rung where the opponent can be wiped, the
whole-army control's episodes end early and its numbers are not
comparable to the per-model arm's until #317 is fixed. The E rungs are
scored on the phase facade's own evaluation family, so the control there
is compared against itself; the C and D rungs avoid wipes by design.

### The bar, measured first

`squad_march_take` (unarmed here, so it cannot fire), seeds 700000+ at
n=100, **both facades identical on every shared field**: success
**0.900**, `held` 2.92, `alive` **0.842**, `on_obj` 0.909, turns 9.60
phase-clock (two stepped phases per round, so about 4.8 rounds), vp −2.7
± 0.8, coherent 0.932. The script loses about two bodies of twelve on the
way and misses a point in one episode of ten. That is the pass mark's
own level: #340's C-rung mark is 0.90, and the bar sits on it.

### The two arms

| | whole-army control | per-model arm |
|---|---|---|
| trainer | `train.py` via `just train-curriculum-control` | `train_per_model.py` |
| budget | 60 epochs of 2048 steps, extended once to 120 if still rising | runs to **122,880 rounds** at 128 per update; read at the checkpoint the control's budget names (6× its slowest rounds-to-pass, floor 20,480, cap 122,880; 2× the cap, read once, when 6× exceeds it); resumed in place if the 2×-cap rule applies (#346: cadences passed explicitly) |
| start | from scratch | `--warm-start-from checkpoints/per_model/per-model-c1-2026-09-16-12-43-32-s{1,2,3}c1/last.pt` (C1 at 245,760: 0.990 / 0.930 / 0.990), seed for seed |
| seeds | 1, 2, 3 | 1, 2, 3 |
| flags | defaults (`ent_coef` 0.03) | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003` |
| in-run eval | the trainer's own, seeds 500000+ | every 512 rounds, n=30, 500000+ |
| logging | Wandb `curriculum-c2`, tag `c2-ctl` | Wandb `curriculum-c2`, tag `c2` |

Both arms launch together, beside T1's three runs and A5b's last
minutes. If the control fails its own criterion at 120 epochs, that is a
finding about the control (A5's clause) and the per-model arm is read at
245,760 beside it.

### Criteria

Read greedy, no decode, n=100 on 700000+.

- **PASS:** success ≥ **0.90** on **all three** seeds at the budget, and
  no in-run dip below 0.75 after the first rolling pass (the drift
  clause, set one notch under the mark as A2b's was).
- **FAIL:** any seed below 0.90 at the budget. The next step is the
  census: which point is short, and whether the bound squad dies on the
  way (the enemy's fire) or ends beside the disc (allocation, C1's
  residual).
- **NULL (scenario):** only if the script fails its own criterion. It
  reads 0.900, on the mark.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts, not criteria:** `alive` against the bar's 0.842 (#340's
  "alive ≥ bar − 0.1" is a readout: the reward pays nothing for a body);
  vp paired against the script; `held`; turns against the script's 9.60
  phase-clock and the control's; `on_obj`; coherency greedy and sampled;
  rounds to rolling 50 / 80 / 90%; the panel over the last quarter; the
  control beside every row.

Power: binomial SE 0.030 at p=0.90, n=100. The bar sits on the bound,
so a seed at 0.87 is one SE under it and still a fail on the letter;
this rung will be read with that in mind and the verdict will not be
softened for it.

### What I expect (a guess, written so it can be wrong)

The per-model arm passes on all three seeds inside 40k rounds at about
the bar's `alive` (0.80–0.86): the warm start carries C1's approach and
the enemy's fire costs a body or two, not the point. It may learn to
hang the bound squad back a turn and arrive later, which the success
criterion does not punish. The control passes at 60 epochs on two
seeds and needs its extension for the third, as on C1. If instead the
arm reads under 0.90 on a seed, the census says the bound squad is dead
or short on the enemy's point, not the neighbours — the opposite of C1.

### Amendment 1 — written 2026-09-16 22:50, after the control's 60-epoch read and before any per-model number

**The control at 60 epochs**, n=100 on 700000+, `last.ckpt`: success
**0.710 / 0.690 / 0.760**, `held` 2.73–2.76, turns 13.8–14.5 phase-clock
(about seven rounds of eight — it arrives with the game nearly over)
against the script's 9.60, `on_obj` 0.93–0.97, coherent 0.79–0.81, vp
−8.1 to −11.0 against the script's −2.7. In-run (n=30, rolling five):
80% at epoch never / never / 33; 90% never; the last ten evaluations
0.57–0.83 on all three. Wandb `curriculum-c2`: `pqkc7sp3` / `conri3cb`
/ `ews7lbi8`.

**The extension.** Under the mark on every seed with the curves still
moving, the control gets its once-only extension to **120 epochs**
(`--resume-ckpt-path` from `last.ckpt`, launched 22:46, suffix
`-ctl-x2`). By the symmetric-cap rule the per-model arm's budget is
**245,760 rounds**, read once at the end: the arm running to 122,880
will be resumed in place when it gets there (#346: cadences explicit),
and its 122,880 checkpoint is a readout. If the control fails its own
criterion at 120, that is a finding about the control (A5's clause) and
the arm is read at 245,760 beside it. The criteria are unchanged.

### Amendment 2 — written 2026-09-16 23:10, after the control's 120-epoch read and before any per-model number

**The control at 120 epochs**, n=100 on 700000+, `last.ckpt` of the
`-ctl-x2` runs: success **0.790 / 0.690 / 0.780**, `held` 2.80–2.84,
turns **14.39 / 14.81 / 14.18** phase-clock (the script's 9.60; the game
is 16), `on_obj` 0.96–0.97, coherent 0.74–0.80, vp −6.4 to −9.2 paired
+4.6 to +5.2 turns behind the script. In-run (n=30, rolling five): 80%
at epoch 68 / 96 / 64, **90% never**; the last ten evaluations 0.73–0.90
on every seed. Seed 2 read 0.690 at 60 and 0.690 at 120. Wandb
`curriculum-c2`: `o640jkhb` / `ivu8b09r` / `az2s5lpc`.

**What this is.** The whole-army control fails C2's own criterion on
all three seeds at its extended budget, on a rung the script passes at
0.900 — A5's clause: a finding about the control, not a NULL. Its
bodies reach the points (`on_obj` 0.96) but late: it arrives in round
seven of eight, two and a half rounds behind the script, and one point
is still short in a quarter of episodes when the clock runs out. Whether
the enemy's fire is what slows it (C1's control arrived at 6.8–7.1
against the same script's 4.98, so about two rounds of the gap were
already there without guns) is a readout for the report, not a
criterion. **The per-model arm is read at 245,760 rounds beside the
control's 120-epoch row, criteria unchanged**: PASS is ≥ 0.90 on all
three seeds; the control's row is reported as what it is.

---

<a id="2026-09-17-curriculum-c2-one-seed-one-hundredth-short"></a>

## Curriculum rung C2: the enemy squad on the point fires back — two seeds clear the mark, one is a hundredth short, and the whole-army control fails on all three

**Verdict first.** On the first guns rung (#340, arm #368) — C1's four
unarmed squads and four points with the enemy unit on its point armed
and firing (`hold_and_shoot`, range 12, one attack per model, the
shooting phase stepped) — the per-model arm warm-started from C1 reads
greedy success ****0.890 / 0.920 / 0.950**** at 245,760 rounds, n=100, `held` 2.87 / 2.88 / 3.00 of
3, turns 10.21 / 9.84 / 9.79 phase-clock against the script's 9.60 and the
whole-army control's 14.2–14.8. ****FAIL as pre-registered, by one seed at one hundredth under the 0.90 mark** (binomial SE 0.03; the pre-registration said the verdict would not be softened for a seed one SE under, and it is not). Seed 1 is also the one seed with an in-run dip under the drift clause (rolling 73% at 123k, from a raw 63% at 121k, recovered by 124k).** The control fails its
own criterion on every seed at its extended budget (**0.790 / 0.690 /
0.780** at 120 epochs) on a rung the script passes at 0.900: under A5's
clause that is a finding about the control, and it is the first rung on
the ladder where the per-model arm is the trainer that reaches the mark while the control does not: the arm is at the script's speed (turns 9.8–10.2 against 9.60) and the script's survival (`alive` 0.826 / 0.842 / 0.856 against the bar's 0.842), and the control arrives two and a half rounds late (14.2–14.8 of 16) and misses a point in a fifth to a third of episodes. The prediction on file — a pass on all three inside 40k rounds at about the bar's `alive` — is right on the speed and the survival and wrong by one seed and one hundredth on the mark.

### Provenance

| field | value |
|---|---|
| date | launched 2026-09-16 22:18 (control and arm together); first leg exited at 122,880 at 01:50–01:51 on 09-17, resumed in place 01:51, exited 03:43 / 03:44 / 03:43 |
| GPU / no-GPU | GPU (RTX 4090), beside A5b, T1 and the control's extension — twelve trainers at launch, six from 01:51 |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; each seed from C1's `last.pt` of the same seed (245,760 rounds on C1: 0.990 / 0.930 / 0.990), fresh optimizer — **not paired on init with the control** |
| n | 100 at seed base 700000 (final, the 122,880 readout, `alive`); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (census) |
| config | `configs/experiments/curriculum/c2.yaml` — unrefereed by design; success `all_objectives_occupied` counts our bodies; `skip_phases: [command, charge, fight]` so shooting is stepped (two agent phases per round: turns are phase-clock, 16 in an eight-round game) |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take` (unarmed here), both facades identical: success 0.900, held 2.92, `alive` 0.842, `on_obj` 0.909, turns 9.60, vp −2.7 ± 0.8, coherent 0.932 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one declared unit, on the point at (35, 27), a rifle of range 12 and one attack each, firing at the nearest valid unit every shooting phase |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 (#346: cadences explicit); control 60 epochs extended once to 120 (245,760 rounds) — the symmetric cap |
| code revision | `1b993eb` on `feature/curriculum-c2` (PR #369) at launch; the resumed leg on `236c69a` (docs only between them) |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c2-2026-09-16-22-18-5{9,9,8}-s{1,2,3}c2`; control `checkpoints/ppo-transformer-curriculum_c2-…-s{1,2,3}c2-ctl-x2/last.ckpt` |
| coherency | greedy 0.582 / 0.563 / 0.662 at 700000+; 0.63 / 0.56 / 0.65 greedy against **0.38 / 0.40 / 0.42** sampled at 900000+ |
| Wandb | `curriculum-c2`: per-model `3ek95n2i` / `1pam14x6` / `yzoqo4cw`, resumed `8fa0j8m6` / `m2rkug75` / `9khjcvpg`; control `pqkc7sp3` / `conri3cb` / `ews7lbi8` (to 60), `o640jkhb` / `ivu8b09r` / `az2s5lpc` (60–120) |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-16-curriculum-C2-preregistration` at `1b993eb` (2026-09-16 22:18); amendment 1 at `edfe44d` (the control's 60-epoch read, the extension and the 245,760 budget, 22:50, before any per-model number); amendment 2 at `236c69a` (the control's 120-epoch read, 23:10, before any per-model number) |

### The read

| row | success | turns (phase-clock, 16 max) | vs bar, paired | held of 3 | on obj | alive | coherent | in-run: rounds to rolling 50 / 80 / 95%; last 40 |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` (unarmed) | 0.900 | 9.60 | — | 2.92 | 0.909 | 0.842 | 0.932 | — |
| **s1 at 245,760** | **0.890** | 10.21 | +0.61 ± 0.31 | 2.87 | 0.782 | 0.826 | 0.582 | 2.6k / 2.6k / 81k; 88.2 |
| **s2 at 245,760** | **0.920** | 9.84 | +0.24 ± 0.30 | 2.88 | 0.873 | 0.842 | 0.563 | 2.6k / 2.6k / 8.2k; 93.9 |
| **s3 at 245,760** | **0.950** | 9.79 | +0.19 ± 0.29 | 3.00 | 0.807 | 0.856 | 0.662 | 2.6k / 2.6k / 8.2k; 91.8 |
| s1 / s2 / s3 at 122,880 (readout) | 0.810 / 0.960 / 0.900 | 11.15 / 9.66 / 9.96 | +1.55 / +0.06 / +0.36 | 2.88 / 2.99 / 2.91 | 0.78 / 0.87 / 0.76 | — | 0.55 / 0.62 / 0.58 | — |
| control at 60 epochs | 0.710 / 0.690 / 0.760 | 13.8–14.5 | +4.2 to +4.9 | 2.73–2.76 | 0.93–0.97 | — | 0.79–0.81 | 80% at epoch never / never / 33 |
| **control at 120 epochs** | **0.790 / 0.690 / 0.780** | 14.39 / 14.81 / 14.18 | +4.79 / +5.21 / +4.58 | 2.84 / 2.83 / 2.80 | 0.96 / 0.97 / 0.96 | — | 0.74 / 0.80 / 0.80 | 80% at epoch 68 / 96 / 64; 90% never; last ten 0.73–0.90 |

The drift clause (no rolling dip under 0.75 after the first rolling
pass): seed 1 once, at 123,392 rounds (raw 63 at 121,344, in the first
leg; rolling 73.2; back at 90 by 124,416); seeds 2 and 3 never. Raw
in-run dips under 80 after the first rolling 95%: 22 / 12 / 15 of 322 /
465 / 465 evaluations. ⚠ The in-run rows after the resume at 122,880
are **n=20**, not 30: a resume drops `--n-eval-episodes` with the
cadences (#346, noted there); the n=100 reads are unaffected.

**Greedy against sampled** (900000+, n=100, paired): vp +0.9 ± 1.0 / +0.7 ± 0.8 / +2.0 ± 0.8; `held` 2.84 / 2.97 / 3.08
greedy against 2.76 / 2.98 / 2.96 sampled; `alive` 0.82–0.86 either
way; coherency 0.63 / 0.56 / 0.65 greedy against **0.38 / 0.40 / 0.42**
sampled; 80–86 decisions per episode. The sampled policy holds the
points and keeps its bodies alive as well as the greedy one and walks
the squads apart, as on every rung.

**Health panel, last quarter** (~480 updates per seed): explained variance 0.54 / 0.52 / 0.50, clip fraction 0.24 / 0.23 /
0.22, ratio p99 1.83 / 1.81 / 1.73, displacement entropy 0.99 / 0.98 /
1.19, declaration entropy 0.01–0.02, advantage std 0.41–0.48. The
regime's normal range on every line (C1: 0.46–0.52, 0.24–0.27,
1.88–1.91).

**The by-turn census** (n=20 on 700000+, bodies on objectives of 12
and points held of 3 at the end; which points are empty at the end, by
index 0–3 with index 2 the enemy's point at (35, 27)):

| policy | success (n=20) | turns | on objectives at the end | held at the end | episodes with each point empty | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 0.80 | 10.40 | 10.97 | 2.90 | 0 / 0 / **0.20** / 0.10 | 2.9 |
| s1 | 0.90 | 10.10 | 9.19 | 2.85 | 0.05 / 0.05 / 0.10 / 0.05 | 2.9 |
| s2 | 1.00 | 9.50 | 10.27 | 3.00 | 0 / 0 / 0 / 0 | 3.2 |
| s3 | 0.95 | 9.55 | 9.76 | 3.10 | 0 / 0 / 0.05 / 0 | 3.1 |

**The script's misses are at the enemy's point; the arm's are spread.**
At n=20 the script leaves the enemy's disc empty in a fifth of episodes
— the bound squad, under fire, is the one that fails to get on — where
seed 1 misses each point once in twenty and seeds 2 and 3 miss the
enemy's once or never. The arm ends with 9.2–10.3 bodies on points of
about ten alive (`alive` 0.83–0.86 of twelve): nearly every survivor is
standing on a point.

### What it says

- **FAIL on the letter by one hundredth on one seed, and the rung's
  question is answered the other way.** The clause is 0.90 on all three;
  seed 1 reads 0.890, one SE under, at the end of the budget. Seeds 2 and
  3 clear it. The record reads the letter as written (A1, A3, A4 and C1
  were all called FAIL at a cap or a plateau within an SE or two), and
  the same record says what the arm did here: it reached the mark on two
  seeds and sat a hundredth under on the third, at the script's speed
  and the script's survival, while the whole-army control on the same
  scenario and the same rounds failed on every seed by 0.11–0.21.
- **This is the first rung where the per-model arm is ahead of the
  control at equal rounds.** On A0–A4 and C1 the control passed and the
  arm caught up; on A5 the arm was worse. Here the control arrives in
  round seven of eight (turns 14.2–14.8 of 16) and misses a point in a
  quarter of episodes, with its bodies on the points it reaches
  (`on_obj` 0.96) — it is slow, not lost. The arm warm-started from C1
  was at 80% in-run by 2,560 rounds on every seed and at 95% by 8,192
  on two. What the C1 warm start carries is the approach; what C2 adds
  (a squad under fire on the way to one of the four points) costs the
  arm nothing the script does not pay: `alive` 0.826 / 0.842 / 0.856
  against the bar's 0.842.
- **The enemy's fire is not where the arm loses; the residual is C1's.**
  The census puts the script's own misses at the enemy's disc (a fifth
  of episodes at n=20) and the arm's misses across all four points at
  one in twenty each on seed 1 — the A3 allocation residual that C1
  inherited (0.96–0.97 at A3x's best, 0.93 on C1's seed 2), not the
  guns. Seed 1's dip at 121k rounds (raw 63%) is the one drift event on
  the rung; it recovered within 3k rounds and the panel is green.
- **The control's failure is speed under fire, and it is new.** C1's
  control passed 3 of 3 at 120 epochs two rounds behind the script
  (6.8–7.1 against 4.98); C2's is five rounds behind (14.2–14.8 phase
  clock, i.e. about seven rounds, against the script's 4.8) and never
  reaches a rolling 90% in-run. The same trainer, the same scenario but
  for three rifles, and it went from a round slower than the script to
  arriving with the game nearly over. Nothing in this rung's reward pays
  for a body, so the control is not hiding; it is walking around.
  Whether that is the whole-army policy's known stacking (§ Holding
  pays) meeting an occupied point under fire is a census for the control
  that this report did not run.
- **Coherency 0.56–0.66 greedy, 0.38–0.42 sampled** — unpaid, as on
  every rung; the enemy's fire does not change it.

### Standing consequence for the ladder (#317)

The rung was redesigned before launch because the phase facade **ends
the battle when the opponent is wiped** (#317), against the rules the
per-model facade follows; with both sides armed our twelve rifles wipe
the three blockers in two shooting phases and the scripted bar reads
0.380 on the phase facade against 1.000 on the per-model one, same
seeds (`just measure-bridge`: BRIDGE DIVERGES). So this rung arms their
side only, our guns are C3's axis, and **no whole-army control can be
read beside a per-model arm on any rung where the opponent can be
wiped** until #317 is fixed. Written into `CLAUDE.md` with this report.

### What was not done

- No recording of the per-model arm's failing episodes beyond the
  census.
- No arm with both sides armed (the bridge diverges; see above).
- The control's `alive` was not measured separately; its `on_obj`
  0.96–0.97 with `held` 2.80–2.84 says its bodies stand on the points
  they reach.

---

<a id="2026-09-17-curriculum-C3-preregistration"></a>

## Pre-registration: curriculum rung C3 — one of our squads is armed, and the blockers have to die before the unarmed squads walk in

Written 2026-09-17 04:10, **before any training number on this rung
exists**, on branch `feature/curriculum-c3` (stacked on PR #369). Issue
#370, parent question #340, rung **C3** — the "guns" axis, our side,
mixed: the first rung that asks for a plan with an order to it.

### The scenario

`configs/experiments/curriculum/c3.yaml` is `c2.yaml` — four squads of
three, four points in a column at x=35, radius 4, one enemy unit of three
standing on the point at (35, 27) and firing — with three changes that
are one axis: **squad 0 carries a rifle that outranges the blockers**
(range 18, four attacks; squads 1–3 stay unarmed), **the blockers are
tough and lethal inside their reach** (four wounds each, eight attacks
each at range 12), and the game runs **twelve rounds** (24 phase-clock
turns) because the plan takes about six rounds and the walk after it
three. Success is still `all_objectives_occupied` over our bodies; the
reward is C2's; nothing pays for a kill or a body.

**Why those numbers.** #340's letter: "a plain `squad_march_take` must
FAIL this rung, or it is not testing sequencing." With the blockers as
C2 had them (one wound, one attack) plain `take` PASSES at 0.95–1.00
whatever our rifle: its armed squad shoots on the march and outranges
the blockers, so it clears the point just before its unarmed squads walk
in — sequencing by geometry, no plan needed. A sweep at n=20 on the
per-model facade (2026-09-17, twenty variants of wounds × attacks ×
range × rounds, `scratchpad/c3sweep/`) found the blockers have to
outlast that incidental fire: at four wounds and eight attacks `take`
reads 0.550 with 36% of its bodies dead and the blockers still alive in
two thirds of episodes, while the escort reads 1.000. That variant is
the config.

**The two builds the rung is first to need, shipped with it.**
`scripted_escort` (#330): every armed squad steers as one body at the
nearest live enemy unit and stops just inside its own reach; every
unarmed squad walks its `squad_march_take` vector but stops just outside
the enemy's reach and waits; once the last enemy is dead, or with no
armed squad at all, everyone plays `squad_march_take` unchanged
(`tests/test_scripted_escort.py`: no unarmed body inside the blockers'
reach while one lives, every blocker killed, and bit-identical to `take`
on C2). And the #317 fix: the phase facade ended the battle on an
opponent wipe unconditionally, against the rules the per-model facade
follows; it now does so only under `terminate_on_opponent_elimination`
(default `False`, the mirror of the player-side switch), pinned by
`tests/test_opponent_wipe_termination.py`. Without it no whole-army
control could be read on a rung whose point is the wipe.

### The bar, measured first

Seeds 700000+ at n=100, **both facades identical on every shared field**:

| policy | success | turns | held | alive | on obj | vp | coherent | first kill → blockers wiped → first unarmed body on their point (phase-clock turns); kill precedes arrival |
|---|---|---|---|---|---|---|---|---|
| **`scripted_escort`** (the bar) | **0.990** | 15.25 | 3.97 | **0.963** | 0.915 | +5.3 ± 1.1 | 0.855 | 6.0 → 11.0 (99%) → 12.9 (90%); **1.00** |
| `squad_march_take` (must fail) | **0.590** | 15.25 | 2.66 | 0.615 | 0.957 | +10.1 ± 3.0 | 0.937 | 6.7 → 12.5 (31%) → 8.9 (50%); 0.77 |

The escort loses one body in twenty; `take` loses four or five of
twelve, its armed squad walks into reach and dies, and its unarmed squad
reaches the blockers' point in half the episodes and is shot off it.

### The arms

| | whole-army control | per-model arm | per-model from scratch (companion readout) |
|---|---|---|---|
| trainer | `train.py` via `just train-curriculum-control` | `train_per_model.py` | `train_per_model.py` |
| budget | 60 epochs of 2048 steps, extended once to 120 if still rising | 122,880 rounds at 128 per update, read at the checkpoint the control's budget names (6× its slowest rounds-to-pass, floor 20,480, cap 122,880; 2× the cap, read once, when 6× exceeds it; resumed in place, cadences and `--n-eval-episodes` explicit, #346) | the same |
| start | from scratch | `--warm-start-from checkpoints/per_model/per-model-c2-2026-09-16-22-18-5{9,9,8}-s{1,2,3}c2/last.pt` (C2 at 245,760: 0.890 / 0.920 / 0.950), seed for seed — the C rungs' rule, which stands on S3 and C1 | from scratch — #340's consequence of T1's FAIL ("later rungs train from scratch"), run beside the warm-started arm so the rung reads both and the rule is measured on a rung with guns |
| seeds | 1, 2, 3 | 1, 2, 3 | 1, 2, 3 |
| flags | defaults (`ent_coef` 0.03) | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003` | the same, no warm start |
| in-run eval | the trainer's own, seeds 500000+ | every 512 rounds, n=30, 500000+ | the same |
| logging | Wandb `curriculum-c3`, tag `c3-ctl` | Wandb `curriculum-c3`, tag `c3` | Wandb `curriculum-c3`, tag `c3s` |

Nine trainers on the GPU, launched together; no CPU read runs beside
them until the first exits.

### Criteria

Read greedy, no decode, n=100 on 700000+, from `last.pt` at the budget.

- **PASS:** success ≥ **0.90** on **all three** seeds of the warm-started
  arm, **and** the first kill precedes the first unarmed arrival on the
  blockers' point in ≥ **0.90** of episodes on every seed (#340's C3
  clause; measured by the sequencing census, `scratchpad/escort_census.py`,
  at n=100), and no in-run dip below 0.75 after the first rolling pass.
- **FAIL:** any seed below 0.90 at the budget, or the ordering clause
  missed on any seed. The census says which: dead unarmed squads (walked
  in early), a live blocker (the armed squad never closed or never fired),
  or C1's spread residual.
- **NULL (scenario):** only if the script fails its own criterion. It
  reads 0.990 / ordering 1.00.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts, not criteria:** `alive` (bar 0.963; `take` 0.615); vp
  paired; `held`; turns against the bar's 15.25 and the control's;
  `on_obj`; coherency greedy and sampled; rounds to rolling 50 / 80 /
  90%; the panel; the control beside every row; **the from-scratch
  companion beside every row** (its own success at n=100 and its in-run
  curve — if it reads within one SE of the warm-started arm on every
  seed, the C-rung warm-start rule bought nothing here and is reported
  as such; if the warm-started arm is ahead 3/3, the rule stands with a
  guns rung behind it).

Power: binomial SE 0.030 at p=0.90, n=100. The ordering clause is a
binomial at n=100 too, SE 0.030 at 0.90.

### What I expect (a guess, written so it can be wrong)

The warm-started arm is at 80% within 10k rounds on the movement half
(it inherits C2's approach) and passes on two seeds by 122,880 with the
third under the mark, because the shooting decision is new and the
unarmed squads' waiting is the thing per-step credit has never had to
learn: nothing in the reward pays for standing outside a circle. The
from-scratch companion reads under 0.5 at 122,880. The control passes at
60 epochs on no seed and at 120 on one or two: it walks everyone in, as
`take` does, and its bodies die.

### Amendment 1 — written 2026-09-17 04:25, after the control's 60-epoch read and before any per-model number

**The control at 60 epochs** (eleven minutes: the scenario is small),
n=100 on 700000+, `last.ckpt`: success **0.790 / 0.560 / 0.870**, turns
19.15 / 19.34 / 18.15 phase-clock of 24 (the bar's 15.25: it arrives in
round nine or ten), `held` 3.56 / 2.66 / 3.61, `on_obj` 0.96 / 0.77 /
0.92, coherent 0.73–0.80, vp +12.9 / +9.6 / +9.4 against the bar's +5.3
(it kills more and holds less). In-run (n=30, rolling five): 80% at
epoch 35 / never / 56, 90% at 37 / never / never; the last ten
evaluations 0.57–0.90. Wandb `curriculum-c3`: `xt95et6x` / `b00ym3pi` /
`a9ubzqa7`.

**The extension.** Under the mark on every seed with the curves still
moving, the control gets its once-only extension to **120 epochs**
(`--resume-ckpt-path` from `last.ckpt`, launched 04:22, suffix
`-ctl-x2`). By the symmetric-cap rule the per-model arm's budget — and
the from-scratch companion's — is **245,760 rounds**, read once at the
end: both are resumed in place at 122,880 (#346: cadences and
`--n-eval-episodes` explicit) and the 122,880 checkpoint is a readout.
The criteria are unchanged.

### Amendment 2 — written 2026-09-17 04:36, after the control's 120-epoch read and before any per-model number

**The control at 120 epochs**, n=100 on 700000+, `last.ckpt` of the
`-ctl-x2` runs: success **0.870 / 0.830 / 0.840**, turns 18.27 / 17.31 /
17.66 of 24 (the bar's 15.25), `held` 3.61 / 3.29 / 3.48, `on_obj`
0.94–0.95, coherent 0.74–0.79, vp +6.2 / +7.0 / +6.9 against the bar's
+5.3. In-run (n=30, rolling five): 80% at epoch 64 / 96 / 64, 90% at
88 / never / never; the last ten evaluations 0.67–0.93. Wandb
`curriculum-c3`: `bpa07dn5` / `do7u35wa` / `ib9ymhgv`.

**What this is.** The whole-army control fails C3's own criterion on
all three seeds at its extended budget, on a rung the script passes at
0.990 — A5's clause, as on C2: a finding about the control, not a NULL.
It gets closer than it did on C2 (0.79 / 0.69 / 0.78 there) and it is
a round or more behind the escort; whether its bodies walk in early and
die, or the armed squad never clears the point, is the control's census,
a readout for the report. **The per-model arm and its companion are
read at 245,760 rounds beside this row, criteria unchanged.**

---

<a id="2026-09-17-curriculum-c3-the-dash-not-the-plan"></a>

## Curriculum rung C3: one armed squad has to clear the blockers before the unarmed squads walk in — the arm found the dash, not the plan; the run with no head start found the plan and not the walk

**Verdict first.** On the first rung that asks for a plan with an order
to it (#340, arm #370) — C2's four squads and four points with squad 0
armed (range 18, four attacks) against three blockers of four wounds and
eight attacks at range 12 on the point at (35, 27), twelve rounds — the
per-model arm warm-started from C2 reads greedy success ****0.660 / 0.920 / 0.830**** at
245,760 rounds, n=100, with the first kill preceding the first unarmed
arrival on the blockers' point in ****0.67 / 0.71 / 0.62**** of episodes, `alive`
0.616 / 0.741 / 0.695 against the bar's 0.963, turns 14.76 / 10.66 / 12.51 phase-clock of 24 against the
bar's 15.25. ****FAIL as pre-registered on both clauses** — two seeds under 0.90 at the budget, the ordering clause missed on all three, and the drift clause tripped on all three (rolling dips under 75% after the first pass: 53 / 10 / 17).** The from-scratch companion reads
**0.740 / 0.740 / 0.090** with the kill before the arrival in **1.00 / 0.97 / 0.95** of episodes. The whole-army control fails its own criterion on every seed
at its extended budget (**0.870 / 0.830 / 0.840** at 120 epochs) on a
rung the escort passes at 0.990 and a plain `squad_march_take` fails at
0.590. The prediction on file — two seeds at the mark by 122,880 with the waiting the thing that is hard to learn — is wrong about what the arm learned: it did not learn to wait, it learned that a body dashed onto the blockers' point the instant the other three are held ends the game before the blockers fire again. The scenario rewarded that, and the second arm on this rung (C3b, #372, in flight) removes the reward by reading success at the end of the game.

### Provenance

| field | value |
|---|---|
| date | launched 2026-09-17 04:15 (control, arm and companion together); control extended 04:22–04:33; first per-model leg to 122,880 exited 06:50–06:54, resumed in place, exited 09:19–09:40 |
| GPU / no-GPU | GPU (RTX 4090), nine trainers at launch, six from 04:33 |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; the arm from C2's `last.pt` of the same seed (245,760 rounds on C2: 0.890 / 0.920 / 0.950), fresh optimizer; the companion from scratch — **not paired on init** with either |
| n | 100 at seed base 700000 (final, the 122,880 readout, the ordering census); 30 at 500000 (in-run, every 512 rounds; n=20 after the resume, #346); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census) |
| config | `configs/experiments/curriculum/c3.yaml` — unrefereed by design; success `all_objectives_occupied` counts our bodies; movement and shooting stepped (two agent phases per round, 24 phase-clock turns) |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `scripted_escort` on identical seeds |
| comparator | `scripted_escort`, both facades identical: success 0.990, held 3.97, `alive` 0.963, `on_obj` 0.915, turns 15.25, vp +5.3 ± 1.1, coherent 0.855; first kill at turn 6.0, blockers wiped at 11.0, first unarmed body on their point at 12.9, the kill before the arrival in 100%. `squad_march_take` (must fail): 0.590, alive 0.615 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, on the point at (35, 27), four wounds, rifle range 12 with eight attacks |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 with cadences and `--n-eval-episodes` explicit; control 60 epochs extended once to 120 (245,760 rounds) — the symmetric cap |
| code revision | `a5e603e` on `feature/curriculum-c3` (PR #371) at launch; docs-only commits after |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c3-*-s{1,2,3}c3` (arm) and `-s{1,2,3}c3s` (companion); control `checkpoints/ppo-transformer-curriculum_c3-…-s{1,2,3}c3-ctl-x2/last.ckpt` |
| coherency | greedy 0.442 / 0.548 / 0.499 (arm), 0.417 / 0.399 / 0.385 (companion) at 700000+; against sampled 0.37 / 0.33 / 0.35 and 0.29 / 0.23 / 0.21 at 900000+ |
| Wandb | `curriculum-c3`: arm `gfb0z2og` / `myyyfmo5` / `dfg3hndb`, companion `of8yv8f7` / `rzzt1oqk` / `eei1nkse`, resumed halves arm `3jud60gm` / `03csrlxc` / `xe8isgoo`, companion `k7f1x99v` / `991fx1z2` / `24n2mi0y`; control `xt95et6x` / `b00ym3pi` / `a9ubzqa7` (to 60), `bpa07dn5` / `do7u35wa` / `ib9ymhgv` (60–120) |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-17-curriculum-C3-preregistration` at `a5e603e` (2026-09-17 04:10); amendment 1 at `13d07e4` (the control's 60-epoch read and the extension, 04:25) and amendment 2 at `d0c6a09` (the control's 120-epoch read, 04:36), both before any per-model number |

### The read

| row | success | turns (24 max) | vs bar, paired | held of 3 | on obj | alive | coherent | in-run: rounds to rolling 50 / 80 / 95%; last 40 |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (the bar) | 0.990 | 15.25 | — | 3.97 | 0.915 | 0.963 | 0.855 | — |
| `squad_march_take` (must fail) | 0.590 | 15.25 | — | 2.66 | 0.957 | 0.615 | 0.937 | — |
| **arm s1** (from C2) at 245,760 | **0.660** | 14.76 | −0.49 ± 0.71 | 2.77 | 0.699 | 0.616 | 0.442 | 8.2k / 24.6k / never; 75.6 |
| **arm s2** at 245,760 | **0.920** | 10.66 | −4.59 ± 0.42 | 3.35 | 0.777 | 0.741 | 0.548 | 2.6k / 2.6k / 106k; 83.8 |
| **arm s3** at 245,760 | **0.830** | 12.51 | −2.74 ± 0.58 | 3.14 | 0.716 | 0.695 | 0.499 | 2.6k / 5.6k / never; 84.2 |
| arm s1 / s2 / s3 at 122,880 (readout) | 0.740 / 0.930 / 0.810 | 13.78 / 10.93 / 12.23 | −1.47 / −4.32 / −3.02 | 2.77 / 3.33 / 2.97 | 0.71 / 0.85 / 0.73 | 0.60 / 0.74 / 0.70 | 0.53 / 0.69 / 0.48 | — |
| companion s1 (scratch) at 245,760 | 0.740 | 16.46 | +1.21 ± 0.52 | 3.34 | 0.778 | 0.724 | 0.417 | 92k / 118k / never; 70.7 |
| companion s2 at 245,760 | 0.740 | 15.76 | +0.51 ± 0.50 | 3.36 | 0.697 | 0.772 | 0.399 | 98k / never / never; 49.2 |
| companion s3 at 245,760 | 0.090 | 22.91 | +7.66 ± 0.39 | 1.48 | 0.376 | 0.581 | 0.385 | never; 19.2 |
| companion at 122,880 (readout) | 0.720 / 0.550 / 0.170 | 17.3 / 19.4 / 22.1 | +2.03 / +4.18 / +6.86 | 3.50 / 2.97 / 2.25 | 0.71 / 0.64 / 0.56 | 0.78 / 0.71 / 0.67 | 0.31 / 0.40 / 0.42 | — |
| control at 60 epochs | 0.790 / 0.560 / 0.870 | 18.2–19.3 | +2.9 to +4.1 | 2.66–3.61 | 0.77–0.96 | — | 0.73–0.80 | 80% at epoch 35 / never / 56 |
| **control at 120 epochs** | **0.870 / 0.830 / 0.840** | 18.27 / 17.31 / 17.66 | +2.06 to +3.02 | 3.29–3.61 | 0.94–0.95 | — | 0.74–0.79 | 80% at 64 / 96 / 64; 90% at 88 / never / never |

⚠ The arm is **faster than the bar** on two seeds (10.7 and 12.5 turns
against 15.25) with a third of its bodies dead: that is the dash, not a
better plan. The drift clause: after its first rolling pass the arm's
rolling mean fell under 75% on 53 / 10 / 17 evaluations (seed 1 read
0.740 at 122,880 and 0.660 at the end). ⚠ In-run rows after the resume
at 122,880 are n=20 (#346).

**The ordering clause** (the sequencing census at n=100 on 700000+: the
turn of the first kill, of the blockers' wipe, and of the first unarmed
body on their point; the share of episodes in which the kill came
first):

| policy | success (n=100) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | armed squad dead of 3 |
|---|---|---|---|---|---|---|
| `scripted_escort` | 0.990 | 6.0 | 11.0 (0.99) | 12.9 (0.90) | **1.00** | 0.05 |
| `squad_march_take` | 0.590 | 6.7 | 12.5 (0.31) | 8.9 (0.50) | 0.77 | 0.72 |
| arm s1 | 0.660 | 8.1 | 13.1 (**0.22**) | 8.7 (0.85) | **0.67** | 1.09 |
| arm s2 | 0.920 | 6.5 | 10.6 (**0.16**) | 8.5 (0.88) | **0.71** | 1.01 |
| arm s3 | 0.830 | 7.1 | 12.5 (**0.30**) | 8.5 (0.90) | **0.62** | 0.56 |
| companion s1 | 0.740 | 7.0 | 11.4 (0.65) | 12.0 (0.50) | **1.00** | 0.93 |
| companion s2 | 0.740 | 6.8 | 11.8 (0.78) | 10.7 (0.77) | **0.97** | 0.23 |
| companion s3 | 0.090 | 8.1 | 12.4 (0.48) | 10.5 (0.24) | **0.95** | 1.48 |

The arm's unarmed bodies reach the blockers' point at turn 8.5–8.7 —
*before* the escort's armed squad has even finished killing (11.0) and
with the blockers alive in 70–84% of its episodes. Its own armed squad
walks in too (about one of three dead). The companion never sends an
unarmed body in before the first kill on two seeds and almost never on
the third.

**Greedy against sampled** (900000+, n=100, paired): vp +6.2 ± 3.6 / −7.0 ± 2.5 / +0.7 ± 3.1 (arm), +4.5 / −1.6 / +1.3
(companion); `held` within 0.4 either way; coherency 0.50 / 0.55 / 0.53
greedy against 0.37 / 0.33 / 0.35 sampled (arm), 0.39–0.42 against
0.21–0.29 (companion); 91–109 decisions per episode (arm, the dash ends
the game) against 129–145 (companion, the game runs on). The sampled
policy plays the same game as the greedy one on both.

**Health panel, last quarter** (~480 updates per seed): arm — explained variance 0.5–0.6, clip fraction 0.21 / 0.19 / 0.21,
ratio p99 1.78 / 1.71 / 1.74, displacement entropy 1.0–1.2, declaration
entropy 0.03; companion — clip **0.37 / 0.33 / 0.26**, ratio p99
**2.27 / 2.08 / 1.87**, displacement entropy 1.5–1.7, declaration
entropy 0.06–0.12. The arm's panel is green while it drifts (the clip
line does not see a policy alternating between two solutions of similar
value); the companion's is A5b's red, still moving.

**The by-turn census** (n=20 on 700000+, bodies on objectives of 12 at
turn 7 → 9 → 12 and at the end, points held at the end, empty points by
index with index 2 the blockers'):

| policy | success (n=20) | turns | on objectives 7 → 9 → 12 → end | held at the end | empty points at the end | max stack |
|---|---|---|---|---|---|---|
| `scripted_escort` | 1.00 | 15.80 | 0.7 → 1.8 → 3.5 → 10.0 | 3.95 | 0 / 0 / 0 / 0 | 3.0 |
| arm s1 | 0.70 | 14.10 | 0.9 → **6.5** → 7.0 → 7.9 | 2.85 | 0.05 / 0.20 / 0.20 / 0.25 | 2.2 |
| arm s2 | 0.85 | 11.65 | 1.0 → **8.5** → 9.2 → 9.3 | 3.50 | 0.05 / 0.15 / 0.05 / 0.05 | 2.5 |
| arm s3 | 0.95 | 11.35 | 1.3 → **7.3** → 8.9 → 8.7 | 3.40 | 0 / 0.05 / 0.05 / 0.05 | 2.7 |
| companion s1 | 0.65 | 17.25 | 0.5 → 5.1 → 7.6 → 8.3 | 3.00 | 0.10 / 0.15 / 0.35 / 0.25 | 2.9 |
| companion s2 | 0.60 | 17.40 | 0.6 → 2.9 → 6.1 → 7.7 | 3.10 | 0.05 / 0.25 / 0.35 / 0.25 | 2.6 |
| companion s3 | 0.05 | 23.35 | 1.3 → 5.6 → 6.3 → 5.4 | 1.75 | 0.35 / 0.65 / 0.85 / 0.90 | 1.6 |

The escort has 1.8 bodies on points at turn 9 because it is waiting; the
arm has 6.5–8.5 there — everyone walks in at once, the C2 approach
unchanged. The companion's seed 3 walks in and dies, stack 1.6, three
points empty at the end.

### What it says

- **FAIL as pre-registered on both clauses, and the census says which
  way.** Success 0.660 / 0.920 / 0.830 with the kill before the
  unarmed arrival in 0.67 / 0.71 / 0.62 of episodes. The FAIL clause
  offered three readings — dead unarmed squads, a live blocker, C1's
  spread residual — and the answer is the second, by design: the arm
  leaves the blockers alive in 70–84% of episodes and wins anyway,
  because `terminate_on_success` ends the game the instant a last body
  lands on their point. A dash is a legal move in the scenario as
  written; it is not the plan the rung was built to test.
- **The head start carried the wrong habit.** The C2 policy rushes
  everyone in and survives because C2's blockers barely kill. On C3
  that habit lands at the blockers' point at turn 8.5 with a third of
  the army dead, and is paid for it with an instant win. The arm never
  had to discover waiting; the from-scratch companion, with nothing to
  carry, learned the order (kill first in 95–100% of episodes, the
  blockers wiped in half to three quarters) and is slow at the walk
  that follows (0.74 / 0.74 / 0.09). The two runs found the two halves
  of the escort's plan, and neither found both. The C rungs' warm-start
  rule now has its first counter-example: a warm start helps when the
  rung below's skill is the rung's skill (C1, C2) and hurts when it is
  the thing to unlearn.
- **The arm drifts between the dash and something else.** Rolling dips
  under 75% after the first pass on every seed (53 / 10 / 17), seed 1
  down from 0.740 to 0.660 across the second half, with the health
  panel green throughout. Two solutions of similar value under this
  reward — dash and wait — and per-step credit does not settle it.
- **The control fails on all three seeds at 120 epochs (0.870 / 0.830
  / 0.840)** a round or more behind the escort, as it did on C2 with
  fewer guns. Read beside the arm it is the better of the two on
  success and the worse on speed; neither trainer learned the plan.
- **The rung's scenario has a defect, and it is measured.** With
  success read at the end of the game (`terminate_on_success: false`)
  the escort still reads 0.990 and plain `take` falls from 0.590 to
  0.280, because a dashed body has to hold the point under fire. That
  is C3b (#372), pre-registered and launched with this report:
  the same arms, one flag. **Rule for the ladder: a success criterion
  that ends the episode the instant it holds rewards whatever reaches
  it first; on any rung where reaching a state and keeping it are
  different skills, judge success on the final board.**
- **Coherency 0.44–0.55 greedy, 0.33–0.37 sampled** — unpaid, as on
  every rung.

### What was not done

- No census of the control's own failure (which of walking in early or
  never clearing the point) beyond its turns and `held`.
- No second budget; the symmetric cap already put the arm at 245,760.
- No arm warm-started from C3's own checkpoints into C3b: they carry the
  dash, so C3b starts from C2 as C3 did.

---

<a id="2026-09-17-curriculum-C3b-preregistration"></a>

## Pre-registration: curriculum rung C3b — C3 with success read at the end of the game, so a dash has to hold the point

Written 2026-09-17 09:50, **before any training number on this arm
exists**, on branch `feature/curriculum-c3` (PR #371, beside C3's own
arm). Issue #372, parent question #340, rung **C3** — a second arm
on the rung, with one change to the scenario.

### The one change

`configs/experiments/curriculum/c3b.yaml` is `c3.yaml` with
**`terminate_on_success: false`**: the episode runs its twelve rounds
and `all_objectives_occupied` is judged on the final board, not the
instant every point first has a body on it. Everything else — the armed
squad at range 18 and four attacks, the blockers of four wounds and
eight attacks at range 12, the points, the reward — is C3's.

### Why

C3's per-model arm, warm-started from C2, read 0.660 / 0.920 / 0.830 at
245,760 with the first kill preceding the first unarmed arrival in only
0.67 / 0.71 / 0.62 of episodes: it found the **dash** — put a last body
onto the blockers' point the moment the other three are held and let
the success termination end the game before the blockers fire again,
with the blockers alive in four episodes of five and a third of its
bodies dead. The rules' own game punishes that: a body that dashes in
has to hold the point to the last round under eight attacks a blocker.
The from-scratch companion, with no rush-in habit to carry, learned the
order (0.95–1.00) and not the walk (0.74 / 0.74 / 0.09). The rung was
built to test sequencing, and instant termination let a policy skip it.

### The bar, measured first

Seeds 700000+ at n=100, **both facades identical on every shared
field**: `scripted_escort` success **0.990**, `held` 3.97, `alive`
0.963, `on_obj` 0.993, vp +63.5 ± 2.0 (the game now runs to the end, so
vp accrues for twelve rounds), coherent 0.858, turns 24 (no early end);
first kill at turn 6.0, blockers wiped at 11.0, first unarmed body on
their point at 12.9, the kill before the arrival in 100%.
`squad_march_take` (must fail): **0.280**, `held` 2.33, `alive` 0.507 —
the dash and the walk-in are both punished now.

### The arms

As C3's, on `c3b.yaml`: the whole-army control (3 seeds, 60 epochs,
extended once to 120 if still rising), the per-model arm warm-started
from **C2's** `last.pt` seed for seed (the C-rung rule; the C3 arm's
checkpoints carry the dash, so they are not the start), and the
from-scratch companion; 122,880 rounds at 128 per update, `--ent-coef
0.003`, in-run eval every 512 rounds at n=30 on 500000+, resumed in
place to 245,760 if the symmetric cap applies. Wandb `curriculum-c3b`,
tags `c3b-ctl` / `c3b` / `c3bs`.

### Criteria

Read greedy, no decode, n=100 on 700000+, from `last.pt` at the budget.
C3's, unchanged:

- **PASS:** success ≥ **0.90** on all three seeds of the warm-started
  arm **and** the first kill precedes the first unarmed arrival on the
  blockers' point in ≥ **0.90** of episodes on every seed, and no in-run
  dip below 0.75 after the first rolling pass.
- **FAIL:** any seed below 0.90, or the ordering clause missed on any
  seed; the census says which.
- **NULL (scenario):** only if the script fails its own criterion. It
  reads 0.990 / ordering 1.00.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts:** `alive` (bar 0.963); vp paired; `held`; `on_obj`;
  coherency greedy and sampled; rounds to rolling 50 / 80 / 90%; the
  panel; the control and the companion beside every row; C3's rows
  beside every row, since the two arms differ by one flag.

Power: binomial SE 0.030 at p=0.90, n=100, on both clauses.

### What I expect (a guess, written so it can be wrong)

The warm-started arm unlearns the dash, because a dashed body now dies
on the point, and finds the order slowly: under 0.90 on two seeds at
122,880, one or two seeds at the mark at 245,760, ordering above 0.90
where success is. The companion reads about where C3's did on success
(0.7) with the order intact. The control passes on no seed.

### Amendment 1 — written 2026-09-17 10:05, after the control's 60-epoch read and before any per-model number

**The control at 60 epochs**, n=100 on 700000+, `last.ckpt`: success
**0.280 / 0.120 / 0.200**, `held` 3.09 / 2.35 / 2.18, `on_obj` 0.94 /
0.81 / 0.77, coherent 0.72–0.79, vp +38.2 / +31.8 / +25.7 against the
bar's +63.5 (every game runs the full 24 turns now). In-run (n=30,
rolling five): never 80%; the last ten evaluations 0.00–0.47 on seed 1
(rising), 0.03–0.17 on seed 2, 0.13–0.50 on seed 3. Wandb
`curriculum-c3b`: `wnle147m` / `qn1edzf8` / `uzsdlznj`.

**The extension.** Under the mark on every seed with the curves still
moving, the control gets its once-only extension to **120 epochs**
(`--resume-ckpt-path` from `last.ckpt`, launched 10:03, suffix
`-ctl-x2`). By the symmetric-cap rule the per-model arm's budget — and
the companion's — is **245,760 rounds**, read once at the end: both are
resumed in place at 122,880 with the cadences and `--n-eval-episodes`
explicit, and the 122,880 checkpoint is a readout. The criteria are
unchanged. Compared with C3's control at 60 (0.79 / 0.56 / 0.87), the
whole-army trainer is where the final-board rule bites hardest: its
bodies reach the points and do not hold them.

### Amendment 2 — written 2026-09-17 10:18, after the control's 120-epoch read and before any per-model number

**The control at 120 epochs**, n=100 on 700000+, `last.ckpt` of the
`-ctl-x2` runs: success **0.280 / 0.000 / 0.450**, `held` 2.77 / 2.15 /
3.14, `on_obj` 0.83–0.85, coherent 0.62–0.67, vp +34.9 / +28.7 / +39.8
against the bar's +63.5. In-run (n=30, rolling five): never 80%; the
last ten evaluations 0.10–0.50 / 0.00–0.40 / 0.37–0.63. Wandb
`curriculum-c3b`: `bfibh5va` / `ugs0hohp` / `i8sh0mm1`.

**What this is.** The whole-army control fails C3b's criterion on all
three seeds at its extended budget, by more than on any rung before it,
on a scenario the escort solves at 0.990: A5's clause, a finding about
the control. Its bodies are on points (`on_obj` 0.83–0.85) and the
points are not all held at the end — the walk-in that C3's instant win
paid is now paid for by the blockers. **The per-model arm and the
companion are read at 245,760 beside this row, criteria unchanged.**

---

<a id="2026-09-17-curriculum-c3b-nobody-holds-the-point"></a>

## Curriculum rung C3b: with success read at the end of the game, nobody holds the point — FAIL on every seed of every trainer, and the arm's failure is now the plan's second half

**Verdict first.** The second arm on the sequencing rung (#340, #372) —
C3 with one flag changed, `terminate_on_success: false`, so
`all_objectives_occupied` is judged on the final board after twelve
rounds instead of the instant it first holds — reads greedy success
****0.200 / 0.130 / 0.230**** on the per-model arm warm-started from C2 at 245,760 rounds,
n=100, and ****0.020 / 0.020 / 0.060**** on the from-scratch companion; the first kill
precedes the first unarmed arrival in **0.86 / 0.77 / 0.85** (mark 0.90) (arm) and 0.67 / 0.79 / 0.91
(companion). ****FAIL as pre-registered** — under 0.90 on every seed by a wide margin, the ordering clause missed on every seed of the arm by 0.04–0.13, no seed of any trainer past a rolling 51% in-run (peaks 37 / 42 / 51 for the arm, 22 / 14 / 12 for the companion).** The whole-army control fails at 120
epochs on every seed (**0.280 / 0.000 / 0.450**). The escort reads
0.990 with 96% alive on the same scenario; plain `squad_march_take`
0.280. With the instant win gone, the arm from C2 learned the first half of the escort's plan — it wipes the blockers in 75–79% of episodes, up from 16–30% on C3 — and stopped: from round five it holds about two and a half points with nine or ten bodies on them, stands still on 47% of its openings, and leaves the blockers' point and the far point empty in half the episodes at the end. Its vp margin (+60 to +71) is the escort's (+63.5); its success is not. The prediction on file — the dash unlearned, one or two seeds at the mark — is right about the unlearning and wrong about the mark.

### Provenance

| field | value |
|---|---|
| date | launched 2026-09-17 09:55 (control, arm and companion together); control extended 10:03–10:14; first per-model leg to 122,880 exited 12:10–12:23, resumed in place 12:25, exited 14:22–14:32 |
| GPU / no-GPU | GPU (RTX 4090), nine trainers at launch, six from 10:14 |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; the arm from C2's `last.pt` of the same seed (C3's checkpoints carry the dash and were not used), fresh optimizer; the companion from scratch — **not paired on init** |
| n | 100 at seed base 700000 (final, the 122,880 readout, the ordering census); 30 at 500000 (in-run, every 512 rounds; n=20 after the resume, #346); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census) |
| config | `configs/experiments/curriculum/c3b.yaml` — `c3.yaml` with `terminate_on_success: false`; unrefereed by design; movement and shooting stepped, 24 phase-clock turns, every episode runs to the end |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `scripted_escort` on identical seeds |
| comparator | `scripted_escort`, both facades identical: success 0.990, held 3.97, `alive` 0.963, `on_obj` 0.993, vp +63.5 ± 2.0, coherent 0.858; first kill at turn 6.0, blockers wiped at 11.0, first unarmed body on their point at 12.9, the kill before the arrival in 100%. `squad_march_take` (must fail): 0.280, alive 0.507. C3's own rows beside every row (one flag apart) |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, on the point at (35, 27), four wounds, rifle range 12 with eight attacks |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 with cadences and `--n-eval-episodes` explicit; control 60 epochs extended once to 120 |
| code revision | `8d456a8` on `feature/curriculum-c3` (PR #371) at launch; docs-only commits after |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c3b-*-s{1,2,3}c3b` (arm) and `-s{1,2,3}c3bs` (companion); control `checkpoints/ppo-transformer-curriculum_c3b-…-s{1,2,3}c3b-ctl-x2/last.ckpt` |
| coherency | greedy 0.660 / 0.543 / 0.641 (arm), 0.490 / 0.606 / 0.342 (companion) at 700000+; against sampled 0.45 / 0.36 / 0.37 and 0.23 / 0.37 / 0.20 at 900000+ |
| Wandb | `curriculum-c3b`: arm `u2fxw35y` / `7a709exu` / `7iz53zqn`, companion `zijgxv6m` / `720gc6a6` / `uf7bfm0x`, resumed halves arm `zr824rjk` / `ce82mijd` / `ydetbauh`, companion `cwir0s7g` / `ryk9o0lz` / `srw69nax`; control `wnle147m` / `qn1edzf8` / `uzsdlznj` (to 60), `bfibh5va` / `ugs0hohp` / `i8sh0mm1` (60–120) |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-17-curriculum-C3b-preregistration` at `8d456a8` (2026-09-17 09:50); amendment 1 at `ac5e5ff` (the control's 60-epoch read and the extension, 10:05) and amendment 2 at `a5a1b79` (the control's 120-epoch read, 10:18), both before any per-model number |

### The read

| row | success | held of 4 | on obj | alive | vp | coherent | stat | in-run: peak rolling; last 40 |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (the bar) | 0.990 | 3.97 | 0.993 | 0.963 | +63.5 ± 2.0 | 0.858 | — | — |
| `squad_march_take` (must fail) | 0.280 | 2.33 | 0.999 | 0.507 | +57.1 ± 4.5 | 0.939 | — | — |
| **arm s1** (from C2) at 245,760 | **0.200** | 2.50 | 0.793 | 0.586 | +62.2 ± 3.8 | 0.660 | 0.47 | 37% at 166k; 17.1 |
| **arm s2** at 245,760 | **0.130** | 2.36 | 0.695 | 0.566 | +60.2 ± 3.9 | 0.543 | 0.47 | 42% at 163k; 24.2 |
| **arm s3** at 245,760 | **0.230** | 2.68 | 0.768 | 0.601 | +71.2 ± 3.8 | 0.641 | 0.46 | 51% at 80k; 21.1 |
| arm at 122,880 (readout) | 0.080 / 0.220 / 0.220 | 1.80 / 2.65 / 2.62 | 0.46 / 0.62 / 0.75 | 0.56 / 0.58 / 0.56 | +32 / +62 / +65 | 0.37 / 0.48 / 0.50 | 0.45 / 0.18 / 0.47 | — |
| companion s1 (scratch) at 245,760 | 0.020 | 1.58 | 0.426 | 0.608 | +32.6 ± 5.1 | 0.490 | 0.41 | 22% at 225k; 11.2 |
| companion s2 at 245,760 | 0.020 | 1.55 | 0.528 | 0.522 | +25.2 ± 4.0 | 0.606 | 0.38 | 14% at 154k; 3.4 |
| companion s3 at 245,760 | 0.060 | 1.79 | 0.470 | 0.491 | +28.6 ± 4.0 | 0.342 | 0.34 | 12% at 208k; 5.4 |
| companion at 122,880 (readout) | 0.060 / 0.000 / 0.020 | 1.55 / 1.33 / 1.57 | 0.31 / 0.26 / 0.40 | 0.63 / 0.63 / 0.48 | +20 / +12 / +27 | 0.34 / 0.36 / 0.34 | 0.27 / 0.26 / 0.38 | — |
| control at 60 epochs | 0.280 / 0.120 / 0.200 | 3.09 / 2.35 / 2.18 | 0.94 / 0.81 / 0.77 | — | +38 / +32 / +26 | 0.72–0.79 | — | never 80% |
| **control at 120 epochs** | **0.280 / 0.000 / 0.450** | 2.77 / 2.15 / 3.14 | 0.83–0.85 | — | +35 / +29 / +40 | 0.62–0.67 | — | never 80% |
| C3's arm at 245,760 (one flag apart) | 0.660 / 0.920 / 0.830 | 2.77 / 3.35 / 3.14 | 0.70–0.78 | 0.62–0.74 | −0.5 to −4.6 | 0.44–0.55 | 0.02–0.06 | — |

Every game runs the full 24 turns now, so vp accrues for twelve rounds
and the paired vp column against the bar is meaningless (+0.00 on
every row: identical episode lengths); read the vp margin itself.
⚠ In-run rows after the resume at 122,880 are n=20 (#346).

**The ordering clause** (the sequencing census at n=100 on 700000+):

| policy | success (n=100) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | alive | armed squad dead of 3 |
|---|---|---|---|---|---|---|---|
| `scripted_escort` | 0.990 | 6.0 | 11.0 (0.99) | 12.9 (0.90) | **1.00** | 0.963 | 0.05 |
| `squad_march_take` | 0.280 | 6.8 | 12.3 (0.60) | 9.1 (0.52) | 0.77 | 0.507 | 0.92 |
| arm s1 | 0.200 | 6.1 | 11.7 (**0.75**) | 8.1 (0.75) | **0.86** | 0.586 | 0.98 |
| arm s2 | 0.130 | 7.1 | 12.3 (**0.77**) | 8.1 (0.65) | **0.77** | 0.566 | 0.81 |
| arm s3 | 0.230 | 6.3 | 11.6 (**0.79**) | 8.3 (0.62) | **0.85** | 0.601 | 1.02 |
| companion s1 | 0.020 | 7.2 | 11.1 (0.54) | 8.0 (0.62) | 0.67 | 0.608 | 0.83 |
| companion s2 | 0.020 | 7.3 | 11.4 (0.41) | 7.1 (0.32) | 0.79 | 0.522 | 1.00 |
| companion s3 | 0.060 | 8.8 | 13.8 (0.68) | 10.4 (0.31) | 0.91 | 0.491 | 1.04 |
| C3's arm (instant win) | 0.66 / 0.92 / 0.83 | 6.5–8.1 | 0.16–0.30 | 8.5–8.7 (0.85–0.90) | 0.62–0.71 | 0.62–0.74 | 0.56–1.09 |

The arm now fires first and wipes the blockers in three quarters of
episodes — the half of the plan the instant win let it skip — and
still sends an unarmed body in at turn 8.1, before the wipe at 11.7,
and loses two of its three riflemen doing it. The escort's armed squad
loses 0.05.

**Greedy against sampled** (900000+, n=100, paired): vp −10.4 ± 4.3 / +2.8 ± 3.9 / +9.5 ± 4.5 (arm), +4.7 / +14.4 / +23.6
(companion); `held` 2.44–2.76 greedy against 2.44–2.68 sampled on the
arm; win rate 82–94% on the arm either way; coherency 0.59–0.68 greedy
against 0.36–0.45 sampled (arm); 116–121 decisions per episode. The
sampled arm plays the same game; the sampled companion is worse than
its greedy self by 5–24 vp, a diffuse policy.

**Health panel, last quarter** (~480 updates per seed): arm — explained variance **0.25 / 0.20 / 0.24** (the ladder's lowest),
clip fraction 0.13 / 0.16 / 0.13, ratio p99 1.57 / 1.65 / 1.55,
displacement entropy 0.87 / 0.91 / 1.09, declaration entropy 0.04,
advantage std 1.0–1.2; companion — explained variance 0.29 / 0.51 /
0.38, clip 0.17 / 0.21 / 0.25, ratio p99 1.62–1.86. The policy is
settled (clip and tail calm) on a critic that cannot value the final
board it rarely reaches.

**The by-turn census** (n=20 on 700000+, points held of 4 and bodies on
points of 12 at turn 9 → 12 → 16 → 24; which points are empty at the
end, by index, with 2 the blockers' and 3 the far one at y=36):

| policy | success (n=20) | held / on points, 9 → 12 → 16 → 24 | empty points at the end | max stack |
|---|---|---|---|---|
| `scripted_escort` | 1.00 | 0.7/1.8 → 1.3/3.5 → 3.3/9.5 → 4.0/12.0 | 0 / 0 / 0 / 0 | 3.0 |
| arm s1 | 0.20 | 3.0/9.3 → 2.6/9.4 → 2.6/9.9 → 2.5/10.1 | 0 / 0.35 / **0.45** / **0.70** | 3.2 |
| arm s2 | 0.15 | 2.3/8.9 → 2.7/8.8 → 2.5/8.9 → 2.4/8.6 | 0.10 / 0.25 / **0.60** / **0.65** | 2.9 |
| arm s3 | 0.10 | 2.5/9.6 → 2.6/8.7 → 2.6/8.6 → 2.6/8.7 | 0.10 / 0.55 / 0.50 / 0.30 | 2.3 |
| companion s1 | 0.00 | 2.2/8.4 → 1.9/6.3 → 1.5/4.6 → 1.5/4.5 | 0.35 / 0.50 / 0.75 / 0.95 | 1.8 |
| companion s2 | 0.00 | 1.9/9.3 → 1.6/7.0 → 1.4/5.7 → 1.4/6.0 | 0.15 / 0.50 / 0.95 / 1.00 | 2.3 |
| companion s3 | 0.10 | 1.7/3.9 → 2.1/5.3 → 2.0/6.0 → 1.9/5.8 | 0.35 / 0.20 / 0.90 / 0.65 | 2.1 |

**The arm's board is static from round five.** Nine or ten bodies on
points and two and a half points held at turn 9, the same at turn 24;
the escort has 1.8 bodies on points at turn 9 (waiting) and twelve at
the end. What the arm holds it keeps — the near points — and what it
does not reach by round five it never reaches: the blockers' point in
45–60% of episodes and the far point in 30–70%. The companion's board
*decays*: bodies on points fall from 8–9 at turn 9 to 4–6 at the end as
the blockers shoot them off.

### What it says

- **FAIL as pre-registered, every seed of every trainer, and the
  answer to the rung's question is no.** The per-model arm from C2:
  0.20 / 0.13 / 0.23, ordering 0.77–0.86. From scratch: 0.02–0.06. The
  whole-army control: 0.28 / 0.00 / 0.45. The escort: 0.990. Nothing
  trained here learned to shoot the blockers off the point *and then
  walk onto it and stay*.
- **The instant win was the whole of C3's success, and removing it
  moved the arm halfway to the plan.** On C3 the arm from C2 wiped the
  blockers in 16–30% of episodes and won by the dash; here it wipes
  them in 75–79% and fires first in 77–86% — the killing is learned,
  from a start that never held a gun. What it does not learn is what
  to do afterwards: it holds the two near points it walked onto by
  round four and parks (stationary 47% of openings, the ladder's
  highest) while the far point and the cleared point stay empty. The
  reward pays the same for standing on a held point as for walking to
  an empty one (the travel term is largely inert, § The travel reward
  audit), and the critic — explained variance 0.20–0.25 — cannot see the
  final board from round five.
- **The head start still helps here, and for the same reason it hurt
  on C3.** The companion from scratch reads 0.02–0.06 with a decaying
  board; the arm's C2 approach gets bodies onto the near points early
  and keeps them. The habit that was the dash on C3 is the near-point
  hold on C3b. The C-rung rule stands with the caveat C3 wrote: read
  the companion beside it.
- **The control fails harder than on any rung** (0.28 / 0.00 / 0.45):
  it walks in as `take` does and the blockers take the points back. On
  C2 it was speed; here it is the plan, and it has none.
- **Where the ladder stands.** #340's answer for this axis: neither
  trainer learns an ordered plan from reward alone at this budget, and
  the thing the per-model trainer lacks is not the order (it fires
  first) but the *continuation* — a policy that keeps going after the
  half of the plan it was paid for. That is the D rungs' question by
  design: D1 clones the escort into the set network (#331), D2 asks
  whether PPO can improve a policy that already carries the whole plan.
  The ladder goes there next. **Rule: on a rung whose solution has
  more than one phase, read the census by phase** — "killed the
  blockers", "arrived", "held to the end" — before saying which half
  a trainer learned; success alone reads 0.20 for a policy that has
  half the plan and 0.02 for one that has none.
- **Coherency 0.54–0.66 greedy, 0.36–0.45 sampled** — unpaid, as on
  every rung.

### What was not done

- No census of the control's failure beyond its `held` and `on_obj`.
- No arm from C3's checkpoints (they carry the dash) and no second
  budget; the symmetric cap already put both per-model runs at 245,760.

---

<a id="2026-09-17-curriculum-D1-preregistration"></a>

## Pre-registration: curriculum rung D1 — the escort cloned into the set network, scored on the final-board rung

Written 2026-09-17 14:50, **before any clone of the escort exists at
house fidelity** (a 20-episode × 5-epoch smoke clone of the pipeline was
run before this file to check the plumbing; its numbers are not a
result), on branch `feature/curriculum-d1` (stacked on PR #371). Issue
#373, parent question #340, rung **D1** — the "start" axis, supervised.
Carries the per-model behaviour clone build (#331).

### The question

C3 and C3b (#370, #372) found that neither trainer learns the escort's
plan from reward at this budget: the per-model arm from C2 learned to
shoot the blockers off the point and then parked (0.20 / 0.13 / 0.23 on
C3b), the from-scratch run learned almost nothing, and the whole-army
control walked in and died. Before D2 asks whether PPO can improve a
policy that starts with the plan, D1 asks whether the set network can
**hold** one: copy the escort's decisions into the network by imitation
and see whether the copy reproduces them and plays the rung as the
teacher does. #340's row: per-head action match ≥ 0.95 held-out; the
clone's score not worse than the teacher, paired at n=100; ordering
within 5 pp.

### The build, shipped with the rung

`model/per_model/clone.py` + `scripts/behaviour_clone_per_model.py` +
`just behaviour-clone-per-model` (#331). `record_demonstrations` plays a
per-model chooser on the player seat and records every decision step as
the `Transition` the PPO update reads, with the teacher's model and the
head column that names its action; `action_to_column` is the inverse of
`SetAgent._decode`, and the recorder decodes every recorded pair back and
compares, so the map cannot drift silently. `fit_clone` minimises the
joint negative log-prob from `evaluate_transitions`; `match_report`
scores greedy agreement on held-out transitions (selector, each head,
joint); `save_clone` writes a per-model checkpoint at zero rounds that
the resolver plays and `--warm-start-from` accepts, with a `.clone.json`
of provenance beside it. Pinned by `tests/test_per_model_clone.py`. ⚠
**The critic is not fitted**: the clone's value head is at
initialisation. That is D2's question.

### The one change

The weights come from imitation, not reward. `just
behaviour-clone-per-model scripted_escort
configs/experiments/curriculum/c3b.yaml 300 40 <out> <seed>`: 300
demonstration episodes at seeds 800000+ (the last 60 held out), 40
epochs, batches of 64, Adam at 3e-4 — the house-fidelity lesson (the
whole-army clone at 120 × 8 under-fitted; at 1200 × 60 it carried its
teacher). **Two clones from identical data, fit seeds 0 and 1**: clone
twice before quoting a clone (the two whole-army clones that differed
only in initialisation measured 115.8 and 111.1). No PPO.

### Comparator, measured first

`scripted_escort` on `c3b.yaml`, seeds 700000+ at n=100, both facades
identical: success **0.990**, `held` 3.97, `alive` 0.963, `on_obj` 0.993,
vp +63.5 ± 2.0, coherent 0.858; first kill at turn 6.0, blockers wiped at
11.0 (99%), first unarmed body on their point at 12.9 (90%), the kill
before the arrival in **1.00**. Beside it, what reward produced on the
same scenario: the per-model arm from C2 0.200 / 0.130 / 0.230, from
scratch 0.02–0.06, the whole-army control 0.280 / 0.000 / 0.450.

### Criteria

Read greedy, no decode, n=100 on 700000+, from each clone's `.pt`.

- **PASS:** on **both** clones, held-out `joint` match ≥ **0.95**; success
  at n=100 ≥ **0.96** (the teacher's 0.990 less one binomial SE of 0.01,
  rounded down to the next hundredth — "not worse than the teacher" at
  this n); the kill-before-arrival share ≥ **0.95** (within 5 pp of the
  teacher's 1.00).
- **FAIL:** any clause missed on either clone. The census says which:
  a clone that matches and fails the rung is the whole-army lesson (a
  per-decision fit does not inherit a joint property — here, the
  unarmed squads' *waiting* is a joint property of twelve decisions per
  turn); a clone that fails to match is the network unable to represent
  the plan at this fidelity, and the loss curve says whether more epochs
  would.
- **NULL (scenario):** only if the teacher fails its own criterion on
  the scoring seeds. It reads 0.990.
- **Readouts:** the match by head; the loss per epoch; `held`, `alive`,
  `on_obj`, vp paired against the teacher; coherency greedy and sampled
  (`just measure-per-model-eval-mode`); decisions per episode; the
  by-turn census; C3b's rows beside every row.

Power: binomial SE 0.010 at p=0.99, n=100, on success; the ordering
share the same; the match is over ~5,000 held-out decisions, SE under
0.01 at 0.95.

### What I expect (a guess, written so it can be wrong)

Both clones match above 0.95 on the displacement head and the selector
and below it on the unit-pointer head (few shooting decisions per
episode, most of them "the only target"), for a `joint` near 0.93–0.96;
success 0.90–0.97, one clone passing and one a hair under; ordering
above 0.95 on both, since "wait" is the most common decision and the
easiest to copy. If instead both pass cleanly, D2 starts from the better
clone; if both fail on success while matching, the failure is the
whole-army lesson and the report says which joint property broke.

---

<a id="2026-09-17-curriculum-d1-the-clone-holds-the-plan"></a>

## Curriculum rung D1: the escort cloned into the set network — the clone holds the plan and misses the mark; FAIL on the letter, with a defect in the match clause

**Verdict first.** On the supervised rung (#340, arm #373) — the
escort's decisions on C3b's scenario copied into the set network by
maximum likelihood, 300 demonstration episodes × 40 epochs, two clones
from identical data differing only in the fit's seed — the clones read
held-out `joint` match ****0.441 / 0.440** (mark 0.95)**, greedy success ****0.830 / 0.850**** at n=100
against the escort's 0.990, and the kill before the unarmed arrival in
****0.99 / 1.00**** of episodes against the escort's 1.00. ****FAIL as pre-registered** on the match clause and the success clause (bound 0.96); PASS on the ordering clause.**
The two clones agree to a thousandth on every number, so nothing here is the fit's seed. What the numbers say is two things the pre-registration did not separate. The joint match counts a factor no clone can learn: which unit the escort opens next is not a function of the observation — the clones sit at chance on held-out opening decisions (0.24 for four units) and at 0.60 on their own training episodes. And the displacement head over-fits 240 episodes (0.95 on the training episodes, 0.61 held-out), a data-size question, not a representation one: the clone tracks the escort's board turn for turn, fires first in 99% of episodes, keeps 90% of its bodies, and ends a body short on one point in one episode of six. Reward alone reached 0.20 on this scenario; imitation reaches 0.85 from 240 games. D1b (#375) runs the one change the reading names — four times the demonstrations — with the match clause corrected.

### Provenance

| field | value |
|---|---|
| date | 2026-09-17, clones launched 14:55 on the CPU, read 15:12–15:14 |
| GPU / no-GPU | no GPU |
| seeds | fit seeds 0 and 1 on identical demonstrations (800000+, 300 episodes, the last 60 held out) — paired on data, not on init |
| n | 100 at seed base 700000 (final, the ordering census); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census); ~9,144 held-out decisions for the match |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against `scripted_escort` on identical seeds |
| comparator | `scripted_escort`, both facades identical: success 0.990, held 3.97, `alive` 0.963, `on_obj` 0.993, coherent 0.858; first kill 6.0, blockers wiped 11.0, first unarmed body on their point 12.9, kill before arrival 1.00. Reward-trained on the same scenario: the per-model arm from C2 0.200 / 0.130 / 0.230, from scratch 0.02–0.06, the control 0.280 / 0.000 / 0.450 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, on the point at (35, 27), four wounds, rifle range 12 with eight attacks |
| budget | 40 epochs of supervised fitting over ~36,700 decision steps, batches of 64, Adam 3e-4, grad-norm 1.0; no PPO |
| code revision | `245c48a` on `feature/curriculum-d1` (PR #374) |
| checkpoint | `checkpoints/per_model/clones/escort-c3b-s{0,1}.pt` (zero rounds) with `.clone.json` beside each |
| coherency | greedy 0.752 / 0.770 at 700000+; 0.78 / 0.76 greedy against 0.68 / 0.71 sampled at 900000+ |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-17-curriculum-D1-preregistration` at `245c48a` (2026-09-17 14:50), before any clone at house fidelity |

### The read

| row | success | held of 4 | on obj | alive | vp | coherent | stat | held-out match: joint · selector · declaration · displacement · unit |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (the teacher) | 0.990 | 3.97 | 0.993 | 0.963 | +63.5 ± 2.0 | 0.858 | — | — |
| **clone, fit seed 0** | **0.830** | 3.79 | 0.929 | 0.902 | +59.7 ± 2.4 | 0.752 | 0.38 | 0.441 · 0.631 · 0.935 · 0.606 · 1.000 |
| **clone, fit seed 1** | **0.850** | 3.79 | 0.938 | 0.903 | +58.8 ± 2.6 | 0.770 | 0.34 | 0.440 · 0.638 · 0.929 · 0.597 · 1.000 |
| the same, on their TRAINING episodes | — | — | — | — | — | — | — | 0.796 · 0.829 · 0.980 · **0.946** · 1.000 (seed 0); 0.774 · 0.812 · 0.973 · 0.939 · 1.000 (seed 1) |
| reward-trained on this scenario (C3b): arm from C2 · scratch · control | 0.20 / 0.13 / 0.23 · 0.02–0.06 · 0.28 / 0.00 / 0.45 | 2.4–2.7 · 1.5–1.8 · 2.2–3.1 | 0.70–0.79 · 0.43–0.53 · 0.83–0.85 | 0.57–0.60 · 0.49–0.61 · — | +60 to +71 · +25 to +33 · +29 to +40 | 0.54–0.66 | 0.47 | — |

**The match by decision kind** (train against held-out, seed 0; seed 1 within 0.02 everywhere):

| decisions | n held-out | selector, train → held-out | joint, train → held-out |
|---|---|---|---|
| `open` (which unit opens next) | 3,134 | 0.604 → **0.243** | 0.592 → 0.227 |
| `act` in movement (who moves, where) | 5,251 | 0.945 → 0.831 | 0.895 → **0.510** |
| `act` in shooting (who fires, at what) | 759 | 0.947 → 0.846 | 0.947 → 0.846 |

Chance on `open` is one in four (four units). The clone is at chance
held-out and at 0.60 on the episodes it was fitted on: the escort's
opening order is not in the observation. The movement selector
generalises (0.83); the displacement column does not (0.51 joint, 0.61
per head), from 0.95 on the training episodes.

**The ordering clause** (the sequencing census at n=100 on 700000+):

| policy | success (n=100) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | alive | armed squad dead of 3 |
|---|---|---|---|---|---|---|---|
| `scripted_escort` | 0.990 | 6.0 | 11.0 (0.99) | 12.9 (0.90) | **1.00** | 0.963 | 0.05 |
| clone, seed 0 | 0.830 | 6.1 | 11.3 (0.98) | 13.3 (0.87) | **0.99** | 0.902 | 0.08 |
| clone, seed 1 | 0.850 | 6.1 | 11.5 (0.98) | 13.5 (0.86) | **1.00** | 0.903 | 0.10 |
| C3b's arm from C2 (reward) | 0.13–0.23 | 6.1–7.1 | 0.75–0.79 | 8.1–8.3 | 0.77–0.86 | 0.57–0.60 | 0.81–1.02 |

The clone fires first, wipes the blockers in 98% of episodes, keeps its
riflemen alive, and sends the unarmed squads in after the wipe, half a
turn later than the escort does.

**Greedy against sampled** (900000+, n=100, paired): vp +0.1 ± 1.5 / −0.4 ± 1.6; `held` 3.88 / 3.86 greedy against 3.90 /
3.87 sampled; win rate 97–99% either way; coherency 0.78 / 0.76 greedy
against 0.68 / 0.71 sampled; 151–156 decisions per episode. The sampled
clone plays as well as the greedy one — the fitted distribution is sharp
where it matters.

**The by-turn census** (n=20 on 700000+, points held of 4 and bodies on
points of 12 at turn 9 → 12 → 16 → 24; empty points at the end by index,
2 the blockers'):

| policy | success (n=20) | held / on points, 9 → 12 → 16 → 24 | empty points at the end | max stack |
|---|---|---|---|---|
| `scripted_escort` | 1.00 | 0.7/1.8 → 1.3/3.5 → 3.3/9.5 → 4.0/12.0 | 0 / 0 / 0 / 0 | 3.0 |
| clone, seed 0 | 0.85 | 0.7/1.8 → 1.3/3.5 → 3.2/9.0 → 3.8/11.5 | 0 / 0.05 / 0.10 / 0.05 | 3.0 |
| clone, seed 1 | 0.80 | 0.8/2.0 → 1.4/3.8 → 2.8/7.9 → 3.6/11.4 | 0 / 0.10 / 0.15 / 0.15 | 3.2 |

The clone's board is the escort's to the first decimal through turn 12
— it waits — and lags it by half a body at turn 16 and half a point at
the end: a squad arriving a turn late, or a body short, on one point.

### What it says

- **The set network can hold an ordered plan it was never paid for.**
  From 240 demonstration games the clone fires first in 99% of
  episodes, wipes the blockers in 98%, keeps 90% of its bodies and puts
  a body on every point in 83–85% — on the scenario where 245,760
  rounds of reward-driven training reached 0.20 from a C2 start and
  0.02–0.06 from scratch, and the whole-army control 0.28. #340's D1
  consequence — "cannot reach the match: the architecture cannot hold a
  sequenced plan; stop and redesign" — does not fire on this reading,
  because the match it names is not the thing that failed.
- **The match clause has a defect: it counts a factor that is not in
  the observation.** Which unit the escort opens next is the script's
  internal plan order; the clones are at chance on it held-out (0.24)
  and at 0.60 on their own training episodes, so no clone of this
  teacher can reach a joint match of 0.95. Per head given the teacher's
  model, the declaration head is at 0.93 and the unit-pointer head at
  1.00. **Score a clone by per-head match given the teacher's model and
  by the rung's own criterion, never by a joint match that includes
  whose turn it is.** (D1's letter is read as written and the arm is
  a FAIL; the corrected clause is D1b's.)
- **The displacement head over-fits, and that is where the success
  gap lives.** 0.95 on the training episodes against 0.61 held-out;
  the escort's move is a continuous vector quantised into a column, and
  240 games do not cover the geometry. The by-turn census puts the miss
  at a body short on one point at the end, not at the plan. Four times
  the demonstrations (D1b, #375) is the one change that reading names; more
  epochs on the same games is not.
- **The clones agree to a thousandth**, so the two-clone rule is met
  and nothing here is the fit's seed.
- **D2 can start from this clone**: at 0.85 with the order intact it
  is a better start than anything reward produced, and the question of
  whether per-model PPO improves or destroys it (the whole-army record:
  a cold critic destroyed a clone at every entropy setting) is the next
  rung's. The critic is unfitted here by design.

### What was not done

- No PPO from the clone: that is D2.
- No third clone; the two agree to within a thousandth (0.441 / 0.440 joint, 0.830 / 0.850 success).

---

<a id="2026-09-17-curriculum-D1b-preregistration"></a>

## Pre-registration: curriculum rung D1b — the escort cloned from four times the demonstrations

Written 2026-09-17 15:20, **before any clone at 1,200 episodes exists**,
on branch `feature/curriculum-d1` (PR #374). Issue #375, parent
question #340, rung **D1** — a second arm, one change.

### The one change

1,200 demonstration episodes (seeds 800000+, the last 240 held out)
instead of 300; 40 epochs, batches of 64, Adam 3e-4, fit seeds 0 and 1,
as D1. `just behaviour-clone-per-model scripted_escort
configs/experiments/curriculum/c3b.yaml 1200 40 <out> <seed>`.

### Why

D1's clones (#373) read success 0.830 / 0.850 against the escort's
0.990 with the plan intact (kill before arrival 0.99 / 1.00, blockers
wiped 98%, alive 0.90), and their displacement head at 0.95 on the
training episodes against 0.61 held-out — an over-fit of 240 games of
continuous geometry, not a representation limit. The joint match D1
pre-registered counts the escort's unit-opening order, which is not in
the observation (chance held-out, 0.60 on the training set), so it is
replaced here by the displacement head's own held-out match.

### Comparator

`scripted_escort` on `c3b.yaml`, n=100 on 700000+: success 0.990, held
3.97, alive 0.963, kill before arrival 1.00. D1's clones beside every row.

### Criteria

Read greedy, no decode, n=100 on 700000+, both clones.

- **PASS:** success ≥ **0.96** on both; kill before arrival ≥ **0.95**
  on both; held-out **displacement** match ≥ **0.90** on both.
- **FAIL:** any clause missed on either clone. If the displacement match
  rises past 0.90 and success does not reach 0.96, the miss is not
  fidelity and the census says what it is.
- **NULL (scenario):** only if the script fails its own criterion (0.990).
- **Readouts:** the match by head, train against held-out; `held`,
  `alive`, `on_obj`, coherency greedy and sampled; the by-turn census;
  D1's rows.

Power: binomial SE 0.010 at p=0.99, n=100; the displacement match is
over ~26,000 held-out decisions.

### What I expect (a guess, written so it can be wrong)

Held-out displacement match 0.75–0.85 (up from 0.61, still under 0.90),
success 0.90–0.95: better, and short of both bounds — the geometry
needs more than four times the games, or a coarser target. If success
clears 0.96 the D rungs continue from this clone.

---

<a id="2026-09-17-curriculum-d1b-four-times-the-games"></a>

## Curriculum rung D1b: the escort cloned from four times the demonstrations — the clone reaches the escort's standard on the rung, and the fidelity bound was set where no clone can reach it

**Verdict first.** The second arm on the supervised rung (#340, #375):
D1's clone with one change, 1,200 demonstration games instead of 300,
two clones from identical data (fit seeds 0 and 1). At n=100 on
700000+ both read greedy success **0.960 / 0.960** (the bound 0.96; the
escort 0.990), `held` 3.93 / 3.92 of 4, `alive` 0.953 / 0.958 (the
escort 0.963), the kill before the unarmed arrival in **1.00 / 1.00** of
episodes, the blockers wiped in 98%. The held-out displacement match
rose from D1's 0.61 to **0.720 / 0.720** — under the 0.90 bound this arm
pre-registered for it. **FAIL as pre-registered, by the fidelity clause
alone; PASS on both behaviour clauses.** The prediction on file — displacement 0.75–0.85 and success 0.90–0.95, short of both bounds — was wrong in the direction that matters: success reached the bound, fidelity did not, and the second was the arm's own defect. The D rungs' fidelity clause is retired; a clone is scored by the rung's criterion and the ordering, with the per-head match a readout.

### Provenance

| field | value |
|---|---|
| date | 2026-09-17, clones launched 15:25 on the CPU (recording four minutes, the fit ~45), read 16:18–16:25 |
| GPU / no-GPU | no GPU |
| seeds | fit seeds 0 and 1 on identical demonstrations (800000+, 1,200 episodes, the last 240 held out) — paired on data, not on init |
| n | 100 at seed base 700000 (final, the ordering census); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census); 36,485 held-out decisions for the match |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against `scripted_escort` and against D1's clones on identical seeds |
| comparator | `scripted_escort`: success 0.990, held 3.97, `alive` 0.963, `on_obj` 0.993, vp +63.5 ± 2.0, coherent 0.858, kill before arrival 1.00. D1's clones (300 games): 0.830 / 0.850, held 3.79, alive 0.90, displacement 0.61 held-out |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, four wounds, eight attacks at range 12 |
| budget | 40 epochs over 146,870 decision steps, batches of 64, Adam 3e-4; no PPO |
| code revision | `0aaa555` on `feature/curriculum-d1` (PR #374) |
| checkpoint | `checkpoints/per_model/clones/escort-c3b-1200-s{0,1}.pt` (zero rounds) with `.clone.json` beside each |
| coherency | greedy 0.763 / 0.730 at 700000+; 0.79 / 0.75 greedy against 0.73 / 0.74 sampled at 900000+ |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-17-curriculum-D1b-preregistration` at `0aaa555` (2026-09-17 15:20), before any clone at 1,200 games |

### The read

| row | success | held of 4 | on obj | alive | vp | coherent | stat | held-out match: joint · selector · declaration · displacement · unit |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` (the teacher) | 0.990 | 3.97 | 0.993 | 0.963 | +63.5 ± 2.0 | 0.858 | — | — |
| **clone, 1,200 games, fit seed 0** | **0.960** | 3.93 | 0.965 | 0.953 | +59.6 ± 2.3 | 0.763 | 0.37 | 0.497 · 0.635 · 0.955 · **0.720** · 1.000 |
| **clone, 1,200 games, fit seed 1** | **0.960** | 3.92 | 0.972 | 0.958 | +59.6 ± 2.3 | 0.730 | 0.36 | 0.500 · 0.637 · 0.953 · **0.720** · 1.000 |
| D1's clones, 300 games | 0.830 / 0.850 | 3.79 | 0.93 | 0.90 | +59 to +60 | 0.75–0.77 | 0.34–0.38 | 0.44 · 0.63 · 0.93 · 0.61 · 1.00 |
| reward-trained on this scenario (C3b) | 0.20 / 0.13 / 0.23 (from C2) · 0.02–0.06 (scratch) · 0.28 / 0.00 / 0.45 (control) | | | | | | | |

**The match by decision kind** (train against held-out; seed 0, seed 1 within 0.03 everywhere):

| decisions | n held-out | selector, train → held-out | joint, train → held-out | D1 (300 games), joint held-out |
|---|---|---|---|---|
| `open` (which unit opens next) | 12,526 | 0.605 → **0.251** | 0.597 → 0.238 | 0.227 |
| `act` in movement (who moves, where) | 20,894 | 0.941 → 0.834 | 0.885 → **0.601** | 0.510 |
| `act` in shooting (who fires, at what) | 3,065 | 0.931 → 0.843 | 0.931 → 0.843 | 0.846 |
| displacement head alone | 20,894 | — | 0.941 → **0.720** | 0.606 |

The opening order is at chance held-out at four times the games, as
D1 found: it is not in the observation. The displacement head closed a
third of its gap to the training match (0.61 → 0.72 against 0.94) and
that was enough for the rung.

**The ordering clause** (n=100 on 700000+):

| policy | success | first kill | blockers wiped | first unarmed body on their point | **kill precedes arrival** | alive | armed squad dead of 3 |
|---|---|---|---|---|---|---|---|
| `scripted_escort` | 0.990 | 6.0 | 11.0 (0.99) | 12.9 (0.90) | 1.00 | 0.963 | 0.05 |
| clone, seed 0 | 0.960 | 6.1 | 11.4 (0.98) | 13.3 (0.89) | **1.00** | 0.953 | 0.06 |
| clone, seed 1 | 0.960 | 6.1 | 11.3 (0.98) | 13.3 (0.92) | **1.00** | 0.958 | 0.07 |

**Greedy against sampled** (900000+, n=100, paired): vp +0.3 ± 1.3 / +0.9 ± 1.0; `held` 3.97 / 3.99 greedy against 3.94 /
3.98 sampled; win rate 98–100% either way; coherency 0.79 / 0.75 greedy
against 0.73 / 0.74 sampled; 154–157 decisions per episode. Sampled
play equals greedy play.

**The by-turn census** (n=20 on 700000+, points held of 4 and bodies on
points of 12 at turn 9 → 12 → 16 → 24; empty points at the end by index):

| policy | success (n=20) | held / on points, 9 → 12 → 16 → 24 | empty points at the end | max stack |
|---|---|---|---|---|
| `scripted_escort` | 1.00 | 0.7/1.8 → 1.3/3.5 → 3.3/9.5 → 4.0/12.0 | 0 / 0 / 0 / 0 | 3.0 |
| clone, seed 0 | 0.90 | 0.6/1.6 → 1.3/3.4 → 3.3/9.1 → 3.8/11.4 | 0 / 0.05 / 0.05 / 0.10 | 3.1 |
| clone, seed 1 | 0.90 | 0.6/1.6 → 1.2/3.2 → 3.3/9.1 → 3.7/11.4 | 0 / 0.10 / 0.10 / 0.10 | 3.0 |

The clone's board is the escort's to the first decimal at every turn
but the last, where it ends half a body short on one of the three
points the unarmed squads walk to after the wipe (never the near one).

### What it says

- **The plan is held to the teacher's standard.** Four times the games
  took the clone from 0.83 / 0.85 to 0.96 / 0.96 on the rung, from 0.90
  to 0.95 alive, from 3.79 to 3.93 points held — a body short on one
  point in one episode of 25 instead of one in six — with the order
  intact at 1.00. The escort is at 0.99. D1's reading was right: the
  miss was fidelity on the displacement head, and games buy it.
- **The fidelity bound was set where no clone of this teacher reaches it,
  and that is the arm's own defect.** Displacement 0.61 → 0.72 held-out
  from 4× the games (0.94 on the training episodes); the escort's move is a continuous vector
  quantised into a column, and two moves a column apart are the same
  move on the table. The rung's criterion — a body on every point at the
  end — is the one that measures what the clone was for, and it passed.
  **Score a clone by the rung's criterion and the ordering; report the
  per-head match as a readout, and never set a bound on it that the
  teacher's own quantisation forbids.** Read with D1's rule (no joint
  match that includes whose turn it is), the D rungs' fidelity clause is
  retired.
- **The two clones agree to the third decimal on every number.**
- **D2 starts here.** The 1,200-game clone at 0.96 with the plan intact
  is the best policy on the per-model facade on any rung with guns; the
  next rung asks whether PPO improves it or destroys it. The critic is
  unfitted, by design, and the whole-army record says that matters.
- Coherency 0.73–0.76 greedy, unpaid as on every rung; the teacher's
  0.86 is the one number the clone does not reach.

### What was not done

- No third fidelity (the pre-registration's own guess of 0.75–0.85 on
  the displacement head was close; a further doubling was not run).
- No PPO: D2.

---

<a id="2026-09-17-curriculum-D2-preregistration"></a>

## Pre-registration: curriculum rung D2 — PPO from the escort clone

Written 2026-09-17 16:25, **before any PPO round from the clone exists**,
on branch `feature/curriculum-d1` (PR #374). Issue #376, parent
question #340, rung **D2** — the "start" axis, reward from a policy that
holds the plan.

### The question

D1b's clone (#375, fit seed 0, 1,200 games) holds the escort's plan on
C3b: success 0.960, kill before arrival 1.00, alive 0.953, held 3.93,
vp +59.6, against the escort's 0.990 / +63.5. On the same scenario,
PPO from a start without the plan reached 0.20 (C3b's arm from C2) and
0.02–0.06 from scratch. Does per-model PPO **improve** a policy that
starts with the plan, **hold** it, or **destroy** it — the whole-army
record's outcome, where PPO with a cold critic destroyed a clone at
every entropy setting (`CLAUDE.md` § Settled)?

### The one change

The starting weights. `train_per_model.py --env-config-path
configs/experiments/curriculum/c3b.yaml --rounds 122880 --seed {1,2,3}
--warm-start-from checkpoints/per_model/clones/escort-c3b-1200-s0.pt
--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512
--checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003`,
Wandb group `curriculum-d2`, tag `d2`. The value head starts at its
random initialisation: the clone fits no critic, by design, so this arm
measures PPO from a clone as the whole-army arm did. No KL anchor.

### Comparator, measured first

The clone itself, paired per episode on identical seeds: success 0.960,
held 3.93, alive 0.953, on_obj 0.965, vp +59.6 ± 2.3, coherent 0.763,
kill before arrival 1.00 (n=100 on 700000+; the n=180 read of the clone
is taken beside the arm's). `scripted_escort`: 0.990, +63.5 ± 2.0.

### Criteria

Read greedy, no decode, n=180 on 700000+, from `last.pt` at 122,880;
the in-run curve at n=30 on 500000+ every 512 rounds, its first point
being the clone's own score.

- **IMPROVES:** vp margin paired against the clone > 0 with t ≥ 2 on
  every seed, success ≥ 0.96 and kill before arrival ≥ 0.95 on every
  seed at the end.
- **HOLDS:** success ≥ 0.94 (one SE under the clone at n=180) with
  ordering ≥ 0.95 on every seed, and vp paired against the clone not
  significant on any seed.
- **DESTROYS:** success at the end < 0.90 on any seed. **Drift
  prediction, written first:** in-run success falls below 0.90 within
  20,480 rounds on every seed and does not recover by 122,880. If it
  holds, #332 (a per-model KL anchor to the clone) is the next build; if
  PPO holds or improves the clone without one, #332 stays on hold.
- **NULL (scenario):** only if the escort fails its own criterion.
- **Readouts:** the in-run curve; `alive`, `held`, `on_obj`, coherency
  greedy and sampled; the health panel (explained variance from a cold
  critic first); the census by phase at the end; the drift between the
  clone and the final policy — the final policy's held-out match against
  the escort's demonstrations, beside the clone's 0.72 on the
  displacement head.

Power: paired vp SE against the clone about 2 at n=180 (both near +60,
per-episode sd ~25 on this scenario); binomial SE 0.015 at p=0.96.

### What I expect (a guess, written so it can be wrong)

DESTROYS, by the drift prediction: the cold critic's first updates are
noise, the clone's success falls under 0.90 inside 10k rounds on every
seed, and the policy settles near C3b's arm (0.2–0.3) with the order
partly kept (0.7–0.9). If instead it HOLDS, the per-model trainer has
done what the whole-army one could not, and D2's second arm is the
anchor as a control rather than a rescue.

---

<a id="2026-09-17-curriculum-D2b-preregistration"></a>

## Pre-registration: curriculum rung D2b — PPO from the escort clone with its critic fitted first

Written 2026-09-17 16:55, **before any PPO round from the critic-fitted
clone exists**, on branch `feature/curriculum-d2` (PR #377). Issue #379,
parent question #340, rung **D2** — second arm, one change from #376.

### The one change

The starting checkpoint: `checkpoints/per_model/clones/escort-c3b-1200-s0-critic.pt`
— D1b's clone (fit seed 0) with **only its value head fitted** to the
escort's discounted returns (`just fit-per-model-critic`, 300 games at
800000+ × 20 epochs, gamma 0.9 per round, held-out explained variance
**−0.17 → 0.74**; the trunk and the policy heads bit-identical, checked
at save). Everything else is D2's: `--rounds 122880 --seed {1,2,3}
--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512
--checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003`,
no anchor, Wandb `curriculum-d2` tag `d2b`.

### Why

D2 (#376) reads 90 / 80 / 73% at its first evaluation and 6 / 0 / 46%
by 2,560 rounds: the whole-army record's outcome — PPO from a clone with
a cold critic destroys it — reproduced. The record's own diagnosis was
the critic: a value head at initialisation makes the first advantages
noise, and the whole-army clone that carried its teacher was fitted with
one. This arm changes that alone.

### Comparator

The clone, paired per episode: success 0.960, held 3.93, alive 0.953,
vp +59.6, kill before arrival 1.00 (n=100; the n=180 read is taken
beside the arm's). `scripted_escort` 0.990 / +63.5. D2's arm beside
every row.

### Criteria

D2's, unchanged, read greedy at n=180 on 700000+ from `last.pt` at
122,880 and in-run at n=30 on 500000+ every 512 rounds:

- **IMPROVES:** vp paired against the clone > 0 with t ≥ 2 on every
  seed, success ≥ 0.96 and kill before arrival ≥ 0.95 on every seed.
- **HOLDS:** success ≥ 0.94 with ordering ≥ 0.95 on every seed and vp
  paired not significant on any.
- **DESTROYS:** success < 0.90 at the end on any seed.
- **NULL (scenario):** only if the escort fails its own criterion.
- **Readouts:** the in-run curve; `train/explained_variance` from the
  first update; `alive`, `held`, coherency greedy and sampled; the
  panel; the census by phase; the final policy's held-out match against
  the escort's demonstrations beside the clone's.

Power: paired vp SE about 2 at n=180; binomial SE 0.015 at p=0.96.

### What I expect (a guess, written so it can be wrong)

HOLDS on two seeds and DESTROYS on one: a fitted critic removes the
first updates' noise, and what is left is PPO's own drift at 128 rounds
per update, slower but in the same direction. If it HOLDS 3/3 the cold
critic was the whole of D2's failure and the anchor is a refinement; if
it DESTROYS 3/3 the critic was not it and D2c is the arm that matters.

---

<a id="2026-09-17-curriculum-D2c-preregistration"></a>

## Pre-registration: curriculum rung D2c — PPO from the escort clone with the KL anchor, cold critic

Written 2026-09-17 16:55, **before any anchored PPO round from the clone
at this budget exists** (a 2,048-round smoke at these settings, seed 8,
was run to size the anchor and is not a result), on branch
`feature/curriculum-d2` (PR #377). Issue #378, parent question #340,
rung **D2** — third arm, one change from #376. Carries #332.

### The one change

`--kl-ref-coef 10 --kl-ref-target 0.03` on D2's recipe: the full
masked `KL(policy || clone)` over the selector and the step's head,
weighted by a coefficient the phase facade's rule adapts toward 0.03
nats per decision. Same clone (`escort-c3b-1200-s0.pt`, cold critic),
same seeds, same everything else. Wandb `curriculum-d2` tag `d2c`.

### Why

D2 collapsed from the first update. The whole-army anchor held a clone
where unanchored self-play destroyed it (`reports/2026-09-04-the-anchor-holds.md`),
and the per-model anchor's smoke from this clone held in-run success at
100% (n=10) through 2,048 rounds where D2 read 6 / 0 / 46% at 2,560. The
measured drift sat at ~0.09 nats per decision per update whatever the
coefficient — Adam normalises the penalty's gradient, so the coefficient
climbs to its cap without the drift falling — and the number that
matters is the success it protects.

### Comparator

As D2b's: the clone paired per episode (0.960 / +59.6 / ordering 1.00),
the escort (0.990 / +63.5), D2's arm beside every row.

### Criteria

D2's, unchanged (IMPROVES / HOLDS / DESTROYS at n=180 on 700000+; NULL
only if the escort fails). Readouts: the in-run curve; `train/kl_ref`
and `train/kl_ref_coef` over the run; the final policy's held-out match
against the escort beside the clone's; `alive`, `held`, coherency; the
panel; the census by phase.

### What I expect (a guess, written so it can be wrong)

HOLDS 3/3 — success 0.94–0.97 at the end with the order intact — and
IMPROVES on none: the anchor keeps the plan and the cold critic's noise
has nowhere to go, so the policy sits where it started. The fourth arm
(both changes) is where an improvement would first show.

---

<a id="2026-09-17-curriculum-D2d-preregistration"></a>

## Pre-registration: curriculum rung D2d — PPO from the escort clone with its critic fitted and the KL anchor

Written 2026-09-17 18:20, **before any PPO round with both changes
exists**, on branch `feature/curriculum-d2` (PR #377). Issue #380,
parent question #340, rung **D2** — fourth arm.

### The one change, per comparison

Relative to D2c (#378, the anchor alone): the starting checkpoint is
the critic-fitted clone `escort-c3b-1200-s0-critic.pt` (held-out
explained variance −0.17 → 0.74, policy bit-identical). Relative to D2b
(#379, the critic alone): `--kl-ref-coef 10 --kl-ref-target 0.03`.
Everything else is D2's recipe (three seeds, 122,880 rounds, 128 per
update, `--ent-coef 0.003`), Wandb `curriculum-d2` tag `d2d`.

### Why, and what is already known in-run

At 40k rounds, before any final read: D2 (neither) and D2b (critic
alone) sit at 2–17% in-run from the clone's 95%; D2c (anchor alone)
sits at 97–100%. The critic was not the mechanism of the collapse on
this trainer; the anchor is what keeps the plan. D2c's expected verdict
is HOLDS. The question that remains is IMPROVES — whether reward makes
the anchored policy better than its start — and a sound critic is the
obvious thing an anchored run needs to do that. This arm is the 2×2's
fourth cell, read beside the other three.

### Comparator

The clone paired per episode (0.960 / +59.6 / ordering 1.00); the escort
(0.990 / +63.5); D2, D2b and D2c beside every row.

### Criteria

D2's, unchanged: IMPROVES / HOLDS / DESTROYS at n=180 on 700000+, NULL
only if the escort fails. Readouts as D2b's and D2c's together.

### Launch

Three seeds launched by a detached script the moment D2's three exit
(the twelve-trainer limit), reads queued behind its own exit.

### What I expect (a guess, written so it can be wrong)

HOLDS 3/3 and IMPROVES on none, as D2c: the anchor at this coefficient
pins the policy to the clone and the critic has nothing to steer. If
it IMPROVES on any seed, the anchor's coefficient is the next sweep; if
D2c holds and D2d does not, the fitted critic is a liability under an
anchor, which would be worth a rule.

---

<a id="2026-09-17-curriculum-d2-the-anchor-not-the-critic"></a>

## Curriculum rung D2: PPO from the escort clone — reward destroys a copied plan unless the weights are anchored to it, the critic is not the lever, and under the anchor reward improves nothing

**Verdict first.** Four arms on the rung that asks whether the per-model
PPO improves a policy that starts with the plan (#340; #376, #379, #378,
#380), all from D1b's clone (success 0.960, kill before arrival
1.00 on C3b's final-board scenario), three seeds each, 122,880 rounds,
read greedy at n=180 on 700000+:

| arm | change from D2 | verdict | success at the end | ordering | vp paired against the clone |
|---|---|---|---|---|---|
| **D2** | — (cold critic, no anchor) | **DESTROYS** | 0.028 / 0.183 / 0.028 | 0.88 / 0.89 / 0.89 | +10.6 / +16.5 / +21.5 against +59.6 |
| **D2b** | the clone's critic fitted first (policy bit-identical) | **DESTROYS** | 0.056 / 0.006 / 0.028 | 0.97 / 0.97 / 0.87 | +43.1 / +33.8 / +28.7 against +59.6 |
| **D2c** | the KL anchor to the clone (coef 10, target 0.03), cold critic | **HOLDS** | 0.944 / 0.956 / 0.956 | 1.00 / 1.00 / 1.00 | +0.7 ± 0.8 (t 0.86) / −0.6 ± 0.9 (t −0.68) / −1.4 ± 1.0 (t −1.49) |
| **D2d** | both | **HOLDS on one seed, under the HOLDS bound by 0.03 on two; DESTROYS on none** | 0.939 / 0.906 / 0.911 | 1.00 / 1.00 / 1.00 | −0.1 ± 0.8 (t −0.10) / −1.9 ± 1.0 (t −1.90) / −1.3 ± 1.0 (t −1.23) |

**The pre-registered guesses were right on every arm:** D2 DESTROYS by
the drift prediction (under 0.90 within 20,480 rounds on every seed and
never back), D2b DESTROYS the same way, D2c HOLDS 3/3 and IMPROVES on
none, D2d HOLDS on one seed and sits 0.03 under the HOLDS bound on two
(0.906 / 0.911 against 0.94, binomial SE 0.02 at n=180), DESTROYS on none,
IMPROVES on none. The whole-army record's diagnosis — the cold critic —
is not the mechanism on this trainer; the anchor is the lever.

### Provenance

| field | value |
|---|---|
| date | 2026-09-17: D2 launched 16:30, D2b and D2c 16:56, D2d 20:16 (when D2's three exited); read D2 20:19, D2b 21:04, D2c 21:20, D2d 22:37–22:57 |
| GPU / no-GPU | GPU (RTX 4090), up to nine trainers at once |
| seeds | 1 / 2 / 3, every arm from the same clone (`escort-c3b-1200-s0.pt`, or its critic-fitted twin for D2b and D2d) — three seeds off one warm start are not three samples of a policy, so every read is against the start itself, paired |
| n | 180 at seed base 700000 (final, the ordering census); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census); 300 games at 800000+ for the drift-from-teacher match |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the clone and the escort on identical seeds; per seed across arms on the same layouts |
| comparator | the clone at n=180: success 0.950, held 3.92, `alive` 0.958, on_obj 0.963, vp +59.6 ± 1.7, coherent 0.747, kill before arrival 1.00, blockers wiped 0.98. `scripted_escort` at n=180: success 0.978, held 3.97, on_obj 0.993, vp +62.6 ± 1.6, coherent 0.839. Reward from a start without the plan (C3b): 0.20 / 0.13 / 0.23 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, four wounds, eight attacks at range 12 |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`; no extension |
| code revision | D2 on `fde8088` (PR #374's tree); D2b and D2c on `8d2e9ad` (PR #377); D2d on `4e20344` (PR #377) |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-c3b-*-s{1,2,3}d2{,b,c,d}` |
| coherency | greedy at 700000+: D2 0.41 / 0.40 / 0.42, D2b 0.49 / 0.58 / 0.38, D2c 0.78 / 0.78 / 0.75, D2d 0.77 / 0.75 / 0.76 (the clone 0.75); D2c against sampled 0.75–0.77 at 900000+ |
| Wandb | `curriculum-d2`: D2 `udqm4uzc` / `i36lut0r` / `gozwwmae`; D2b `5998xmnz` / `t8xm3yf2` / `j1zn8st7`; D2c `4n9vfyk1` / `le7mw85g` / `et51yp6f`; D2d `ziia57hp` / `jvcqlj5k` / `8qo65zat` |
| pre-registrations | D2 `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-17-curriculum-D2-preregistration` at `fde8088` (16:25); D2b and D2c at `8d2e9ad` (16:55); D2d at `4e20344` (PR #377) — each before its own numbers |

### The read

| row | success | held of 4 | on obj | alive | vp (± SE) | coherent | stat | vp paired against the clone |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` | 0.978 | 3.97 | 0.993 | 0.963 | +62.6 ± 1.6 | 0.839 | — | — |
| the clone (the start) | 0.950 | 3.92 | 0.963 | 0.958 | +59.6 ± 1.7 | 0.747 | 0.36 | — |
| **D2 s1 / s2 / s3** (cold critic, no anchor) | **0.028 / 0.183 / 0.028** | 1.44 / 1.88 / 1.81 | 0.43 / 0.48 / 0.52 | 0.55 / 0.56 / 0.60 | +10.6 / +16.5 / +21.5 | 0.41 / 0.40 / 0.42 | 0.04 / 0.07 / 0.26 | (not run: the success clause decides) |
| **D2b s1 / s2 / s3** (critic fitted) | **0.056 / 0.006 / 0.028** | 2.33 / 1.91 / 1.71 | 0.60 / 0.54 / 0.49 | 0.74 / 0.70 / 0.53 | +43.1 / +33.8 / +28.7 | 0.49 / 0.58 / 0.38 | 0.31 / 0.30 / 0.21 | (not run) |
| **D2c s1 / s2 / s3** (anchor) | **0.944 / 0.956 / 0.956** | 3.92 / 3.92 / 3.93 | 0.97 / 0.97 / 0.96 | 0.95 / 0.96 / 0.96 | +60.3 / +59.1 / +58.2 | 0.78 / 0.78 / 0.75 | 0.37 ×3 | +0.7 ± 0.8 (t 0.86, 47 of 88 differing) / −0.6 ± 0.9 (t −0.68, 40 of 92) / −1.4 ± 1.0 (t −1.49, 37 of 91); 88–92 of 180 episodes identical |
| **D2d s1 / s2 / s3** (both) | **0.939 / 0.906 / 0.911** | 3.92 / 3.84 / 3.87 | 0.96 / 0.95 / 0.96 | 0.96 / 0.94 / 0.94 | +59.6 / +57.7 / +58.4 | 0.77 / 0.75 / 0.76 | 0.36–0.37 | −0.1 ± 0.8 (t −0.10, 45 of 89 differing) / −1.9 ± 1.0 (t −1.90, 37 of 85) / −1.3 ± 1.0 (t −1.23, 43 of 91); 89–95 of 180 identical |

Every game runs the full 24 turns, so the vp column against the bar in
`measure-rung` is void (identical lengths); the paired vp here is
`just measure-paired` against the clone on the same 180 seeds.

**The ordering clause and the plan by phase** (n=180 on 700000+):

| policy | success (n=180) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | alive | armed squad dead of 3 |
|---|---|---|---|---|---|---|---|
| the clone | 0.950 | 6.1 | 11.3 (0.98) | 13.3 (0.90) | 1.00 | 0.958 | 0.07 |
| D2 s1 / s2 / s3 | 0.03 / 0.18 / 0.03 | 7.5 / 10.7 / 9.4 | 0.39 / 0.74 / 0.60 | 0.46 / 0.44 / 0.29 | 0.88 / 0.89 / 0.89 | 0.55 / 0.56 / 0.60 | 1.37 / 0.87 / 0.55 |
| D2b s1 / s2 / s3 | 0.06 / 0.01 / 0.03 | 7.8 / 7.0 / 8.0 | 0.93 / 0.82 / 0.67 | 0.37 / 0.69 / 0.49 | 0.97 / 0.97 / 0.87 | 0.74 / 0.70 / 0.53 | 0.43 / 0.81 / 1.23 |
| D2c s1 / s2 / s3 | 0.94 / 0.96 / 0.96 | 6.1 / 6.0 / 6.1 | 0.99 / 0.99 / 0.99 | 0.82 / 0.87 / 0.89 | **1.00 / 1.00 / 1.00** | 0.95 / 0.96 / 0.96 | 0.08 / 0.07 / 0.05 |
| D2d s1 / s2 / s3 | 0.94 / 0.91 / 0.91 | 6.1 / 6.0 / 6.1 | 0.99 / 0.98 / 0.99 | 0.87 / 0.86 / 0.83 | **1.00 / 1.00 / 1.00** | 0.96 / 0.94 / 0.94 | 0.04 / 0.08 / 0.08 |

The destroyed arms do not lose the order first: D2b still fires first
in 87–97% of episodes and wipes the blockers in two thirds to nine
tenths. What they lose is the walk after it — bodies on points, and the
points held to the end (D2's census: 1.3–2.4 held at turn 24).

**In-run** (rolling five of n=30 on 500000+; the first point is the clone's own score):

| arm | first evaluation (512 rounds) | 2,560 | 5k | 20k | 60k | last forty, mean |
|---|---|---|---|---|---|---|
| D2 | 90 / 80 / 73 | 6 / 0 / 46 | 8 / 4 / 2 | 8 / 4 / 2 | 14 / 7 / 17 | under 20 on every seed |
| D2b | ~95 ×3 | 27 / 13 / 7 (6k) | — | 4 / 9 / 13 | 4 / 9 / 13 | 6–20 |
| D2c | ~95 ×3 | 98 / 99 / 99 (5k) | 98 / 99 / 99 | 97 / 97 / 96 | 99 / 100 / 99 | 97–100 |
| D2d | ~95 ×3 | 98 / 99 / 99 (8k) | — | 99 / 97 / 97 | 96 / 99 / 98 | 98–99 |

Both collapses are complete inside the first five thousand rounds, and
neither recovers by 122,880. Both holds are flat from the first
evaluation to the last.

**Drift from the teacher** (the final policy's held-out match against the escort's demonstrations, beside the clone's 0.72 on the displacement head):

| arm | selector | declaration | displacement | unit-pointer | joint |
|---|---|---|---|---|---|
| the clone, on the same 300 games | 0.83 | 0.99 | **0.94** | 1.00 | 0.79 |
| D2 s1 / s2 / s3 | 0.63 / 0.62 / 0.62 | 0.77 / 0.81 / 0.87 | **0.11 / 0.06 / 0.05** | 0.97 / 1.00 / 1.00 | 0.18 / 0.15 / 0.15 |
| D2b s1 / s2 / s3 | 0.62 / 0.64 / 0.62 | 0.84 / 0.86 / 0.85 | **0.08 / 0.06 / 0.06** | 1.00 ×3 | 0.16 / 0.17 / 0.17 |
| D2c s1 / s2 / s3 | 0.77 / 0.77 / 0.77 | 0.97 / 0.98 / 0.98 | **0.88 / 0.87 / 0.86** | 1.00 ×3 | 0.70 / 0.70 / 0.69 |
| D2d s1 / s2 / s3 | 0.77 / 0.77 / 0.77 | 0.98 / 0.97 / 0.98 | **0.88 / 0.88 / 0.87** | 1.00 ×3 | 0.70 / 0.70 / 0.70 |

Scored on the first 300 of the escort's demonstration games — games the
1,200-game clone was fitted on (its own match there is 0.94 on the
displacement head), so this is drift from the START, not held-out
fidelity. The destroyed arms' displacement heads are unrecognisable
(0.05–0.11); the anchored arms moved from 0.94 to 0.87 and stayed there.

**Greedy against sampled**: D2c: vp −1.6 ± 1.3 / +0.7 ± 1.0 / +0.3 ± 0.9, `held` 3.93–3.98 either way,
coherency 0.77–0.81 greedy against 0.75–0.77 sampled — the anchored policy
is sharp (displacement entropy 0.30). D2d: −1.4 ± 1.5 / +0.8 ± 1.4 / +0.7 ±
0.8, `held` 3.8–3.9 either way, D2c's twin. D2: +1.6 / +4.3 / −0.3, `held`
1.3–2.2. D2b: **+12.8 ± 4.0 / +2.8 ± 3.1 / +12.6 ± 4.8** — a diffuse policy
whose argmax plays better than its samples, the do-nothing fingerprint's
neighbour (stationary share 0.21–0.32).

**Health panel, last quarter**: D2 — explained variance 0.01 / (n/a) / 0.23, clip 0.34 / 0.37 / 0.41,
ratio p99 1.9–2.4, displacement entropy 1.5–2.3; D2b — explained variance
0.10 / 0.07 / 0.01, clip 0.33–0.37, entropy 1.8–2.2; D2c — explained
variance **0.66 / 0.68 / 0.67**, clip 0.28–0.29, ratio p99 **2.9**,
displacement entropy **0.30**; D2d — 0.69 / 0.68 / 0.67, clip 0.28–0.29,
p99 2.9, entropy 0.30, D2c's panel to the second decimal from a critic
that started at 0.74 instead of cold. Under the anchor the cold critic
learned (0.66 from initialisation) and the policy stayed sharp; the
ratio's 99th percentile at 2.9 is the anchor's gradient distorting the
importance ratio, not a trust region failing — read it beside `kl_ref`.

**The by-turn census** (n=20 on 700000+, points held of 4 and bodies on
points of 12 at turn 9 → 12 → 16 → 24):

| policy | success (n=20) | held / on points, 9 → 12 → 16 → 24 | empty points at the end (index; 2 the blockers') |
|---|---|---|---|
| `scripted_escort` | 1.00 | 0.7/1.8 → 1.3/3.5 → 3.3/9.5 → 4.0/12.0 | 0 / 0 / 0 / 0 |
| the clone | 0.90 | 0.6/1.6 → 1.3/3.4 → 3.3/9.1 → 3.8/11.4 | 0 / 0.05 / 0.05 / 0.10 |
| D2 s1 / s2 / s3 | 0.00 / 0.15 / 0.00 | 1.2/3.8 → 1.4/3.7 → 1.4/3.8 → 1.5/4.5 · 0.4/0.9 → 0.9/2.6 → 2.1/5.4 → 2.4/7.0 · 0.9/2.2 → 1.2/4.1 → 1.3/5.2 → 1.3/5.5 | 0.3 / 0.55 / 0.8 / 0.9 · 0.3 / 0.3 / 0.45 / 0.55 · 0.3 / 0.8 / 0.9 / 0.7 |
| D2b s1 / s2 / s3 | 0.00 / 0.05 / 0.10 | 1.0/2.7 → 1.8/4.8 → 2.2/7.1 → 2.1/7.4 · 0.6/0.8 → 1.8/5.1 → 2.0/6.5 → 1.9/6.1 · 1.3/3.2 → 2.1/5.6 → 2.0/5.9 → 1.9/6.2 | 0.6 / 0.2 / 0.6 / 0.55 · 0.9 / 0.65 / 0.4 / 0.15 · 0.4 / 0.35 / 0.7 / 0.65 |
| D2c s1 / s2 / s3 | 0.90 / 0.95 / 0.95 | 0.8/2.0 → 1.4/3.8 → 3.2/8.6 → 3.8/11.2 · 0.7/1.9 → 1.2/3.3 → 3.3/9.2 → 3.9/11.5 · 0.7/1.7 → 1.3/3.4 → 3.0/8.5 → 3.9/11.3 | 0 / 0.05 / 0.05 / 0.10 · 0 / 0.05 / 0.05 / 0.05 · 0 / 0.05 / 0.05 / 0.05 |
| D2d s1 / s2 / s3 | 0.95 / 0.90 / 0.90 | 0.7/2.0 → 1.4/3.6 → 3.5/9.5 → 3.9/11.2 · 0.7/1.9 → 1.3/3.5 → 3.3/8.8 → 3.9/11.4 · 0.7/1.8 → 1.4/3.8 → 3.3/8.7 → 3.7/11.2 | 0 / 0.05 / 0.05 / 0.05 · 0 / 0 / 0.10 / 0.05 · 0 / 0.10 / 0.10 / 0.10 |

The anchored arms' boards are the clone's to the first decimal at every
turn. The destroyed arms walk bodies in early (1–4 on points at turn 9
against the clone's 1.6, waiting) and end with half the board empty.

### What it says

- **Per-model PPO from a clone that holds the plan destroys it in two
  thousand rounds, and a fitted critic does not stop that.** D2 and D2b
  read 0.03–0.18 at the end from a 0.95 start, both gone by 5k rounds,
  both flat after. The whole-army record's diagnosis — a cold critic's
  noisy first advantages — was the obvious mechanism and it is not the
  one here: with the value head fitted to the teacher's returns
  (explained variance 0.74) the collapse is the same shape at the same
  speed. What the two collapses share is the update itself: at 128 rounds
  per update, five epochs over the rollout with `eps_clip` 0.2 move a
  sharp policy far from a start the critic cannot yet value, and once
  the unarmed squads walk in early the plan's payoff is gone from the
  data. The destroyed policies keep the order (they still fire first in
  87–97% of episodes) and lose the walk.
- **The KL anchor holds the plan, and reward under it changes nothing
  measurable.** D2c: 0.944 / 0.956 / 0.956 against the clone's 0.950,
  paired vp −1.4 to +0.7 with half the episodes identical move for move,
  the board the clone's to the first decimal at every turn, the policy
  0.87 from the teacher on the displacement head where the clone is
  0.94. HOLDS 3/3, IMPROVES on none — the pre-registered guess. The
  anchor's coefficient climbed to its cap on every seed (the measured
  drift sat at ~0.09 nats per decision per update whatever the
  coefficient, Adam normalising the penalty's gradient), and the critic
  it protected learned to 0.66 explained variance from a cold start.
  This is #332's answer: build it, and the per-model trainer can carry a
  clone the way the whole-army one carried the KL-anchored clone.
- **Both changes together are the anchor alone, a hair worse.** D2d
  reads 0.939 / 0.906 / 0.911 with the order intact, the board the
  clone's, the panel D2c's to the second decimal (explained variance
  0.68 either way — the cold critic under the anchor reached what the
  fitted one started at), paired vp −0.1 / −1.9 / −1.3 against the
  clone. Two seeds sit 0.03 under the HOLDS bound at a binomial SE of
  0.02, none under 0.90, none improved. The fitted critic under the
  anchor is worth nothing; whether it costs the 0.03 is inside the
  noise. The 2×2 is clean: the anchor's row holds, the critic's column
  does not matter.
- **Where the ladder stands.** The D rungs asked whether the set
  network can hold an ordered plan (yes, from imitation: 0.96 at 1,200
  games) and whether PPO can improve one it starts with. The answer at
  this budget is that PPO from the plan keeps it only under an anchor,
  and under that anchor it does not improve it. The improvement question
  is now a coefficient sweep — an anchor loose enough to let reward move
  the policy and tight enough to keep the plan — and a budget question
  the record already names for the whole-army anchor (`--kl-ref-target`
  0.03 held there too, and won two of four cells). **Rule: warm-start the
  per-model PPO from a clone only under a KL anchor to it; the critic
  is not the lever; and read the 2×2, not the diagnosis on file, before
  naming a mechanism.**

### What was not done

- No longer budget: every arm that held was flat by 20k rounds, and
  every arm that collapsed was gone by 5k.
- No sweep of the anchor's coefficient or target; the smoke's setting
  was the one run.

---

<a id="2026-09-19-curriculum-D3-preregistration"></a>

## Pre-registration: curriculum rung D3 — anchored PPO from a clone of a WEAKER teacher

Written 2026-09-19 01:20, **before the clone exists and before any PPO
round from it**, on branch `feature/per-model-actor-credit` (PR #383).
Parent question #340; the D branch's third rung; no issue opened, by
the same decision as A5c–A5i — this file is its record. Set as a goal
by Sash on 2026-09-19 01:05.

### The question

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

### The one change

The teacher. Everything else is D2c: `configs/experiments/curriculum/
c3b.yaml` (C3 with success read on the final board), the clone recipe of
D1b, the anchor of D2c. The clone is `checkpoints/per_model/clones/
take-c3b-1200-s0.pt` — `just behaviour-clone-per-model squad_march_take
configs/experiments/curriculum/c3b.yaml 1200 40 <out> 0`, seeds 800000+,
the last 240 games held out, fit seed 0 (launched 01:13, before this
file was written; no number from it existed at the time of writing).

### Reads

The clone: greedy at **n=180 on 700000+** (`just measure-rung` with
`scripted_escort` as the bar and `squad_march_take` beside it), and the
ordering census (n=180): the turn of the first blocker kill, the turn the
first unarmed body reaches the blockers' point, the share of episodes
where the kill precedes the arrival, the share where the blockers are
wiped. Expected ≈ the teacher's: success 0.28, kill before arrival low,
blockers alive at the end in most episodes.

### The arms

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

### Criteria (per arm)

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

### What I expect (a guess, written so it can be wrong)

D3a holds: drift capped near 0.03 nats per decision cannot reorder a
plan, and A5i moved a tenth of a turn under it. D3b either finds the
order partially (0.40–0.60 with kill-before-arrival rising, the shape
C3b's arm-from-C2 reached at 0.20 by wiping the blockers and stopping)
or slides toward D2 plain's collapse. Honest guess: **D3b ends between
0.10 and 0.50, IMPROVES not shown on 3/3, D3a within the clone's band.**
If D3b improves on 3/3 I am wrong in the useful direction.

### Amendment 1 — written 2026-09-19 02:50, the clone read; D3a and D3b launched, no PPO number yet

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

### Amendment 2 — written 2026-09-19 05:30, at 60k of 122,880 on both arms

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

### Amendment 3 — written 2026-09-19 07:50, both arms read at 122,880: HOLDS on both, IMPROVES on neither

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

### Amendment 4 — written 2026-09-19 11:30, the mechanism (post hoc)

Read off the runs' logs after the verdict, not pre-registered: the
anchor's coefficient sat at its 10,000 cap on D3a from the first quarter
(drift 0.08 against a 0.03 target) with the gradient norm 14,000–17,000
clipped to 0.5 on every step — the update is the anchor; the clone's
heads are near-deterministic (declaration 0.03 nats, shooting 3×10⁻⁵),
so the rung's alternative is never rolled out; and C3b's reward,
discounted at gamma 0.9 per round, pays walking-now 2.55 against the
escort's 1.25 in rounds 1–6 and leaves the plan worth +0.7 per episode
at the decision point. The report's addendum carries the tables.

---

<a id="2026-09-19-curriculum-d3-the-anchor-is-a-brake-with-no-engine"></a>

## Curriculum rung D3: anchored PPO holds a clone with two thirds of the rung still to gain, and takes no step toward it — the D-route is imitation only

**Verdict first.** Two arms of anchored PPO from a clone of plain
`squad_march_take` on C3b (#340; the rung where the clone reads 0.361
and the escort 0.978), three seeds each, 122,880 rounds, read greedy at
n=180 on 700000+:

| arm | anchor | verdict | success at the end | paired v the clone | ordering (kill before arrival / blockers wiped) |
|---|---|---|---|---|---|
| **D3a** | coef 10, target 0.03 (D2c's) | **HOLDS; IMPROVES not shown** | 0.339 / 0.356 / 0.333 | −0.022 ± 0.026 / −0.006 ± 0.026 / −0.028 ± 0.024 | 0.79–0.83 / 0.64–0.67 (the clone 0.82 / 0.64) |
| **D3b** | coef 1, target 0.10 (A5g's) | **HOLDS; IMPROVES not shown** | 0.328 / 0.344 / 0.356 | −0.033 ± 0.024 / −0.017 ± 0.023 / −0.006 ± 0.026 | 0.78–0.84 / 0.63–0.64 |

Twenty-four reads over 20k–122,880 rounds, every one within a binomial
SE of the clone's 0.361 and none above it; the by-phase census the
clone's on every column; sampled play the same policy as greedy. The
question this rung was built to separate — is the anchor a brake with
no engine, or was there nothing left to gain on D2 and A5 — is
answered: **with 0.62 of success to gain and the teacher's plan as the
reward's own signal, anchored PPO on this trainer moved nothing at
either coefficient.** The D-route on the per-model facade is
imitation: what the anchor holds is what the clone was, and what the
clone was is what the rung reads. The goal set on 2026-09-19 is
answered in the negative on its first item, and its third (decompose
the conjunction on A5-points) is not run, by the goal's own ordering.

### Provenance

| field | value |
|---|---|
| date | 2026-09-19: clone fitted 01:13–02:19, read 02:42; arms launched 02:42, finished 07:25–07:35; final reads 07:35–07:55 |
| GPU / no-GPU | GPU (RTX 4090), six D3 trainers beside E1's six per-model runs |
| seeds | 1 / 2 / 3 per arm, every seed from the same clone (fit seed 0) — three seeds off one warm start are that clone's band, not three samples |
| n | 180 at seed base 700000 (every read, the ordering census); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 1,200 games at 800000+ for the clone, the last 240 held out |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the clone on identical seeds (success, turns); per seed across arms on the same layouts |
| comparator | the clone `checkpoints/per_model/clones/take-c3b-1200-s0.pt` at n=180: success 0.361, held 2.51, alive 0.531, vp 60.8 ± 3.3, coherent 0.870, first kill 7.0, kill before arrival 0.82, blockers wiped 0.64 (its teacher `squad_march_take` 0.339 / 2.46 / 0.525 / 0.79 / 0.62); `scripted_escort` 0.978 / 3.97 / 0.965 / 62.6 ± 1.6 / 0.839 / 1.00 / 0.99 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, four wounds, eight attacks at range 12 |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, the default `mean` credit, from the clone's cold critic; no extension |
| code revision | pre-registered at `10eeb1d`; runs on `d8a1059`; the branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | `last.pt` at 122,880 and every 512 rounds under `checkpoints/per_model/per-model-c3b-2026-09-19-02-42-*-s{1,2,3}d3{a,b}` |
| coherency | greedy at 700000+: D3a 0.895 / 0.877 / 0.884, D3b 0.859 / 0.880 / 0.879 (the clone 0.870, the escort 0.839, the teacher 0.934); sampled at 900000+ 0.81–0.85 |
| Wandb | `curriculum-d3`: D3a `vjgk8f5w` / `i6pri90x` / `kjppfvfs`; D3b `vv1w0k0f` / `sfk1u00s` / `k3u5dz2m` |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-19-curriculum-D3-preregistration` (+3 amendments) |

### The read

#### The clone is its teacher, with the escort's plan out of reach

`squad_march_take` cloned into the set network from 1,200 games on
C3b (D1b's recipe) reads **0.361** against its teacher's 0.339 on the
same 180 seeds; held 2.51 against 2.46, alive 0.531 against 0.525;
held-out per-head match declaration 0.956, displacement 0.807,
unit-pointer 1.000, joint 0.569 (the opening order, at chance as D1
recorded). The census says what the gap to the escort is: the teacher's
armed squad does fire first in most episodes (kill before arrival 0.79)
and does not finish the job (blockers wiped 0.62), and its unarmed
squads walk in while the blockers live (alive 0.53). The escort's
order is "wipe, then walk"; the teacher's is "walk, shooting on the
way". That order is the whole of the 0.62 of success between them, and
it is the order the arm from C2 half-learned from reward on C3b (wipes
the blockers in 75–79% of episodes before stopping).

#### Both anchors hold the clone, at every read

| rounds | D3a s1 / s2 / s3 | D3b s1 / s2 / s3 |
|---|---|---|
| the clone | 0.361 | 0.361 |
| 20,480 | 0.372 / 0.339 / 0.350 | 0.322 / 0.383 / 0.406 |
| 40,960 | 0.367 / 0.361 / 0.367 | 0.361 / 0.367 / 0.372 |
| 61,440 | 0.378 / 0.367 / 0.389 | 0.328 / 0.350 / 0.344 |
| **122,880** | **0.339 / 0.356 / 0.333** | **0.328 / 0.344 / 0.356** |

Twenty-four reads in 0.32–0.41, the clone's binomial band (0.361 ±
0.036 at n=180), no trend on either arm, and at the end no seed above
the start. The census at the end:

| row | success | held of 4 | alive | first kill (turn) | first unarmed arrival | kill before arrival | blockers wiped |
|---|---|---|---|---|---|---|---|
| the clone | 0.361 | 2.51 | 0.531 | 7.0 (94%) | 8.9 (62%) | 0.82 | 0.64 |
| D3a s1 / s2 / s3 | 0.339 / 0.356 / 0.333 | 2.44 / 2.56 / 2.49 | 0.519 / 0.534 / 0.524 | 7.0 / 7.0 / 6.8 | 8.6 / 8.7 / 8.9 | 0.79 / 0.81 / 0.83 | 0.64 / 0.65 / 0.67 |
| D3b s1 / s2 / s3 | 0.328 / 0.344 / 0.356 | 2.47 / 2.54 / 2.51 | 0.522 / 0.535 / 0.519 | 7.1 / 6.9 / 7.1 | 8.8 / 8.6 / 8.8 | 0.84 / 0.78 / 0.84 | 0.64 / 0.64 / 0.63 |

Not a column moved: the first kill at the same turn, the unarmed
arrival at the same turn, the blockers wiped in the same share of
episodes, the same third of the army dead. Sampled play (n=100 on
900000+, paired against greedy) is the same policy: −3.4 to +1.1 ± 2.5
vp, held within 0.06, coherency 0.81–0.85 against greedy's 0.86–0.90 —
no do-nothing fingerprint (stationary 0.45–0.48, the clone's 0.44).

**The letter of the clauses**, the paired per-episode success difference
against the clone on the same 180 seeds:

| seed | D3a: diff, SE, t, episodes differing | D3b: diff, SE, t, episodes differing |
|---|---|---|
| s1 | −0.022 ± 0.026, t −0.85, 22 of 180 | −0.033 ± 0.024, t −1.42, 18 of 180 |
| s2 | −0.006 ± 0.026, t −0.22, 21 | −0.017 ± 0.023, t −0.73, 17 |
| s3 | −0.028 ± 0.024, t −1.15, 19 | −0.006 ± 0.026, t −0.22, 21 |

Every seed within two paired SE of the clone (HOLDS), none above it
(IMPROVES not shown), none below by two SE (DESTROYS not triggered).
Only 17–22 episodes in 180 end differently from the clone at all: after
122,880 rounds of reward the policy is the clone on nine tenths of the
episodes and a coin-flip on the rest.

#### What the two coefficients say

D2c's setting (10 / 0.03) and A5g's (1 / 0.10) differ by an order of
magnitude in the coefficient and threefold in the target, and they read
the same. The looser anchor was the one the pre-registration guessed
might "find the order partially or slide toward D2 plain's collapse";
it did neither. A5h (0.1 / 0.3) lost the A5 clone slowly and D2 plain
(no anchor) lost the escort clone in 2,560 rounds, so the coefficient
has a floor below which the clone is destroyed and, above it, a plateau
on which nothing is gained — there is no setting on file at which
reward both keeps the plan and adds to it.

### What it says

- **The D-route on the per-model facade is imitation only, and the
  ladder's D and start-axis rows should be read as such.** D2c held the
  escort clone (0.95 → 0.94–0.96), A5f and A5i held the bar's clones
  (0.93 → 0.93; 0.96 → 0.96–0.98), and each could be read as "nothing
  left to gain". D3 removes that reading: from 0.361 with 0.62 to gain
  and the gap a plan the same reward has taught from scratch, the
  anchor held the clone at every read and every coefficient that holds.
  A rung passed from a clone is passed by the clone.
- **The anchor's coefficient is not the IMPROVES lever, and no more of
  them should be run.** Four settings across three rungs: 10 and 1
  hold, 0.1 loses slowly, 0 destroys. Between "held" and "lost" there
  is no measured "lifted". Whatever lifts a clone here is not the
  strength of the tether to it.
- **What is left for reward on this trainer is the from-scratch route,
  and the record says what it can and cannot learn there**: a walk to
  the bar's speed (A0–A4x), a mask-free engagement rule (C1), the first
  half of an ordered plan (C3b's 0.20), and not a conjunction over five
  or six points (A5-points 0.03–0.33) nor the second half of the plan.
  The goal's third item — decompose the conjunction on A5-points — asks
  about reward from scratch, which this rung did not test; it is
  recorded as not run by the goal's ordering, not as answered.
- **Three seeds off one clone read as one policy.** Six runs, two
  coefficients, twenty-four reads, all within one SE of one number.
  The next test of "can reward improve a start" needs a different
  kind of start (a partially trained scratch policy, or the C2 arm
  that half-learned the plan) rather than another clone.

### What was not done

- One fit seed for the clone; D1's two fit seeds agreed to a thousandth
  and the read here is the teacher's to two hundredths, so a second fit
  was not spent.
- No critic pre-fit (D2b) and no unanchored arm (D2): both destroyed
  the escort clone within 2,560 rounds on this scenario and were not
  re-run on a weaker one.
- No whole-army control: the rung asks about the per-model trainer's
  anchor, which has no whole-army counterpart.
- Goal 3 (per-point terminal bonus or unique-coverage pay on A5-points)
  was not built or run.

### Addendum — written 2026-09-19 11:30: why training does not improve the clone, measured

Three causes, each read off the runs' own logs (`wandb/run-*/run-*.wandb`,
by quarter of the run) and one measurement of the reward.

**1. The anchor owns the gradient.** The adaptive coefficient
(Schulman's rule, doubling above 1.5× the target, capped at 10,000)
saturated at the cap inside the first quarter on D3a because the
measured drift sat at 0.075–0.083 nats per decision and never reached
the 0.03 target. The pre-clip gradient norm read 14,000–17,000 with
`max_grad_norm` 0.5 and the clip binding on 100% of steps, so the
update direction is the anchor's; the policy loss (0.05) is six
millionths of the loss. D3b's coefficient hovered at 5–9 with the
drift at its 0.10 target, gradient norm 8–17, still clipped every step;
its loss is 96% anchor. The same saturation is in every anchored run
on file — D2c (coef 10,000, gradient 24,000–29,000) and A5i (10,000,
20,000–23,000): **"the anchor holds" has meant "the anchor is the
whole update", at every coefficient that holds.**

**2. There is nothing to explore from.** The clone's heads are
near-deterministic: entropy 0.03 nats on the declaration head, 0.24 on
displacement, 3×10⁻⁵ on the shooting head, ~0 on the unit pointer; a
from-scratch policy at the same stage (A5-points) reads 0.11 and
1.6–2.3. Sampled rollouts are the greedy policy (the paired
sampled − greedy difference was −3 to +1 vp), so the alternative the
rung asks for — the unarmed squads wait while the rifles work — is
never rolled out and its advantage is never measured. The anchor
forbids the cure: entropy added to a near-deterministic reference is
KL to it, so `ent_coef` 0.003 pushes against a coefficient of 10,000.

**3. The reward pays the wrong order first.** The re-timed C3b reward
stream, n=40 on 700000+, gamma 0.9 per round as PPO discounts it:

| policy | success | episode reward | discounted from the start | rounds 1–6 | rounds 7–12 |
|---|---|---|---|---|---|
| `scripted_escort` | 1.00 | 9.79 | 3.96 | 1.25 | 8.54 |
| `squad_march_take` | 0.30 | 6.23 | 3.24 | **2.55** | 3.69 |
| the clone | 0.30 | 6.26 | 3.22 | 2.47 | 3.79 |

Walking in at once earns twice the escort's reward in the first six
rounds (the travel term pays per inch closed, now); the escort's
payoff — coverage of the cleared point and the terminal bonus — lands
in rounds 7–12 and at the end. Undiscounted the plan is worth +3.6 per
episode; discounted to the rounds where the decision is made it is
worth **+0.7**, spread over ~120 decisions, against a per-decision
return sd of 1.7 and a critic at explained variance 0.45–0.58. The
advantage of waiting is a fraction of the noise on any one decision,
and negative on the decisions that matter most (the first moves).

Any one of the three would slow improvement; together they forbid it.
The escort clone (D2c) and the bar's clones (A5f, A5i) sat under the
same three, which is why HOLDS was the ceiling there too. **What the
record now says about the D-route:** the anchor as built (an adaptive
coefficient capped at 10,000 on a clipped gradient) is a freeze, not a
tether; a clone of a deterministic teacher has no entropy to explore
with; and a plan whose payoff is late loses to a travel term that pays
now under gamma 0.9. A tether that could lift a clone would need a
bounded coefficient (or a trust region on the policy step rather than
a penalty), an entropy floor the anchor does not tax, and a reward or
horizon under which the plan's payoff reaches the decisions that
choose it — three builds, and the third is the same wall the
conjunction rungs hit.

---

<a id="2026-09-14-curriculum-E1-preregistration"></a>

## Pre-registration: curriculum rung E1 — one unit, one objective, no opponent

Written 2026-09-14, **before any number from the arm exists**, on branch
`feature/per-model-pipeline-health` (the curriculum runs on one PR while it
is exploratory; the commit timestamp of this file is what orders it before
the numbers). Issue #333, parent question #328.

### The one change

Training on the **E1** rung: 1 unit of 3 models, 1 objective 25" from the deployment band, no opponent, movement only, 8 rounds, 60x44, Move 6 (`configs/experiments/curriculum/e1.yaml`). Two
arms of the same scenario, three seeds each: the **per-model** trainer
(`train_per_model.py`, the thing under test) and the **whole-army** trainer
(`train.py`, the pipeline control). Nothing else varies.

### Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/e1.yaml 100`), bridge identical on every shared
field: success **1.000**, turns-to-success **5.55 ± 0.07** (sd 0.67, range 4-7), `held` 1.00, vp 10.2 ± 0.4. In-run band: success 0.967, turns 5.40 (n=30, seeds 500000+ — the in-run eval band, quoted so the curve has a bar drawn on it).

### Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success rate ≥ 0.95 and mean
  turns-to-success ≤ 6.55 (the script's 5.55 + 1), on **every seed** of the per-model
  arm, within 20,480 rounds.
- **FAIL:** any seed misses either bound at budget while the whole-army
  control passes on ≥ 2 of 3 seeds — the fault is optimiser / reward plumbing on the per-model side if the whole-army control passes; a misconfigured scenario if neither does.
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds —
  the rung is misconfigured; redesign, do not read the per-model result.
- **Pass with a defect:** the bounds hold but the health panel
  (`docs/metrics.md` § The per-model health panel) is red over the last
  quarter of the run — recorded as such, never as a clean pass.

Power: success at n=100 has binomial SE 0.022 at p=0.95, so a true 0.90
fails the bound at ~2.3 SE; turns-to-success at sd ≈ 0.7 has SE 0.07 at
n=100, so the +1 allowance is fourteen SE wide — this rung is decided by the
success rate, and the turn bound is there to catch a policy that idles until
the last round.

### Budget, regime, seeds

| | per-model arm | whole-army control |
|---|---|---|
| budget | 20,480 rounds | 10 epochs of 2048 steps = 20,480 rounds |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update**, 160 updates | 2048 steps per update (the shipped loop) |
| seeds | 1, 2, 3 (`--seed`); rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `evaluate_spec` | same |
| decode | none (K=1) | K=1, `verify_moves` default — undecoded, the same regime |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4), `--no-wandb` | defaults, `--no-wandb` |
| compute | CPU-sized (episodes are 20-28 decisions); run on the idle GPU box for throughput, six processes concurrent | |

Launch: `just train-per-model-arm <rounds> 3 curriculum-e1 e1 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --no-wandb" configs/experiments/curriculum/e1.yaml`
and `just train-curriculum-control <epochs> 3 curriculum-e1 e1-ctl "" configs/experiments/curriculum/e1.yaml`.

### Primary readouts

Success rate and turns-to-success (paired against the script per episode),
`held`, `on_obj`; the passive pair and `train/rounds_per_update` on every
per-model row; the health panel over the last quarter. Reported with the
provenance table in the arm issue at close.

### What I expect (a guess, written so it can be wrong)

The whole-army trainer passes E1 within budget. The per-model trainer
either passes at a comparable round count — in which case the rung says the
plumbing works and nothing about the architecture — or its declaration head
collapses to `stationary` early (the do-nothing fingerprint) and the greedy
eval never leaves the floor while the sampled score climbs, which
`measure-per-model-eval-mode` will show. I do not know which.

---

<a id="2026-09-17-curriculum-E1-preregistration"></a>

## Pre-registration: curriculum rung E1 — two armed armies on a flat board

Written 2026-09-17 23:20, **before any training number on this rung
exists**, on branch `feature/curriculum-e1` (stacked on PR #377). Issue #381,
parent question #340, rung **E1** — the join of the bodies branch (A5's army)
and the guns branch (C2's rifles), the first rung of the road to the
real game.

### The scenario

`configs/experiments/curriculum/e1.yaml` is
`configs/experiments/24v24_maps_spare_squads.yaml` — the golden family's
twenty-round game, random turn order, the win reward phase
(`closest_objective_v2`, `objective_hold` at `crowding_exponent` 1.0,
`model_kills`, `group_cohesion`, `vp_gain`, `objective_coverage`),
`player_ahead_on_vp` as success, `squad_march_take` on the opponent seat
— with the map pool removed: a flat 60×44 board, six objectives of
radius 3 fixed symmetrically at (18, 12), (18, 32), (30, 8), (30, 36),
(42, 12), (42, 32), the two deployment bands 20" deep at either end,
every one of the twenty-four models on each side in eight squads of
three carrying the family's rifle (range 12, one attack). The observation
budgets stay at the family's values (six objectives, sixteen terrain
slots, all padding here) so a checkpoint trained on E1 loads on E2
(generated terrain) and E3 (the real tables), each of which changes one
thing on the way to the refereed evaluation family.

E1 is not "A5 with guns": A5's reward was the curriculum's occupancy
reward and its success `all_objectives_occupied`; E1's reward and
success are the golden family's, because that is what E3 is scored on
and the E rungs change the board one step at a time toward it. The join
is on the army — the same twenty-four bodies in eight squads.

### The bridge diverges, under a recorded rule — not a design fault

`just measure-bridge` on E1 reads BRIDGE DIVERGES with every scenario
field E1 adds bisected away (the padding budgets, exposure tracking,
turn order, three rounds instead of twenty), and the shooting golden
config `25v25_shooting_opponent.yaml` diverges the same way. The cause is
the per-model facade's own recorded rule: with several shooting units a
side, it judges a unit's targets and cover after the casualties an
earlier unit inflicted, which the whole-army step cannot
(`FacadeDivergence`, `shooting.targets_judged_after_casualties`) — first
in **every** episode, by round 3, on eight of eight. The C and D rungs
never met it because our side had at most one shooting unit. So the
#317 rule ("a divergence there is a design fault") is bounded, with this
rung: **a divergence under a recorded rule is the per-model facade
playing the more rules-faithful game; the rung stands, and each trainer
is read against the bar on its own facade.** `measure-bridge` now names
the recorded rules and exits 2 only when nothing was recorded (a build
this rung is first to need, shipped in its PR with a test on the
verdict).

### The bar, measured first

`squad_march_take` against `squad_march_take`, seeds 700000+ at n=100,
turn order random:

| facade | vp | ±SE | win (`player_ahead_on_vp`) | held | on_obj | alive | coherent | stat / hold |
|---|---|---|---|---|---|---|---|---|
| phase (the control's) | **+14.2** | 9.5 | 0.550 | 2.86 | 0.936 | 0.273 | 0.878 | — |
| per-model (the arm's) | **+20.3** | 9.5 | 0.620 | 2.88 | 0.942 | 0.283 | 0.871 | 0.56 / 0.05 |

A mirror script against itself is not zero: the seat carries an
advantage on this layout (the player deploys on the left band; on
`25v25_maps_two_mode` the same gate read +6.5 ± 6.1). Both trainers
train the player seat, so the bar they are read against is the player
seat's, on their own facade. Per-episode sd is about 95, so at the n=180
the E criterion names the SE of a single row is about 7.1 and the paired
SE smaller. `alive` 0.27 says the rifles decide this game: three quarters
of both armies die in twenty rounds.

The script fails no criterion of its own (NULL needs it to), and the
two facades' rows differ by 6 vp on identical seeds — inside one SE,
the recorded rule's whole effect.

### The arms

| | whole-army control | per-model arm | per-model companion |
|---|---|---|---|
| trainer | `train.py` via `just train-curriculum-control` | `train_per_model.py` | `train_per_model.py` |
| start | from scratch | **from scratch** | `--warm-start-from` A5b's `last.pt`, seed for seed (`per-model-a5-2026-09-15-10-07-*-s{1,2,3}a5b`, the rung below on the bodies branch, 0.260 / 0.180 / 0.160 on its own rung) |
| budget | 60 epochs of 2048 steps, extended once to 120 if still rising | **122,880 rounds** at 128 per update (the cap; the E criterion is a vp bound, so the control's rounds-to-pass is not a signal to scale from); 2× the cap, read once, if the control is extended | the same |
| seeds | 1, 2, 3 | 1, 2, 3 | 1, 2, 3 |
| flags | defaults (`ent_coef` 0.03) | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003` | the same plus the warm start |
| logging | Wandb `curriculum-e1`, tag `e1-ctl` | `curriculum-e1`, tag `e1` | `curriculum-e1`, tag `e1w` |

Nine trainers on the box at once, nothing else. A5b's runs took 37
hours for 122,880 rounds without a shooting phase; E1 steps two phases a
round with twenty-four shooters, so the per-model runs are expected to
take two to three days.

### Criteria

Read greedy, no decode, paired per episode against `squad_march_take`
on the arm's own facade, seeds 700000+ at **n=180**.

- **PASS:** on all three seeds, the paired vp difference (arm − script,
  same seeds) has mean ≥ **−1 paired SE**; `held` ≥ bar − 0.2 (phase
  2.66, per-model 2.68); greedy coherency ≥ bar − 0.03 (0.85 / 0.84).
- **FAIL:** any seed misses any clause. The next step is the census:
  where the bodies are when the clock runs out, `alive` against the
  bar's 0.27 (hoarding or under-arrival), the objective split.
- **NULL (scenario):** only if the script fails its own criterion — a
  script is its own bar here, so never on the letter; recorded as the
  clause it is.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts:** win rate against 0.55 / 0.62; `alive`; `on_obj`; turns
  (every episode runs the clock, 40 phase-clock); coherency greedy and
  sampled; the passive fingerprint; the panel over the last quarter;
  the control and the companion beside every row; in-run vp and win
  rate (n=30, seeds 500000+).

Power: per-episode sd ≈ 95 unpaired; paired, a script and a policy that
plays a similar game share most of the layout noise, so the paired SE at
n=180 should sit under 5. The bound is one paired SE, so a policy exactly
level with the script fails a seed about one time in six and 3/3 about
four times in ten — the clause is written to #340's letter and this
rung will be read with that in mind, as C2 was.

### What I expect (a guess, written so it can be wrong)

The per-model arm from scratch repeats A5's under-arrival and sits well
under the bar on all three seeds (paired −20 or worse), with `alive`
above the bar's 0.27 — it hoards, as the whole-army lineage does on the
real tables. The companion from A5b starts ahead in-run and finishes in
the same place. The whole-army control, the golden lineage's own trainer
on its own reward, reaches the bar's neighbourhood on two of three seeds
by 120 epochs and fails the letter on the third. If the per-model arm
instead passes, the E rungs have inverted the C rungs and the ladder's
next question is why the toy rungs were harder than the real one.

### Amendment 1 — written 2026-09-17 23:20, minutes after launch, before any evaluation row

**A name collision on the record.** `configs/experiments/curriculum/e1.yaml`
replaced, in the same path, the Stage 0 rung E1 of PR #339 (issue #333,
parent #328: one unit walking onto one objective — the rung that became
A1 when #340 superseded #328). That older rung's pre-registration is
`reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-14-curriculum-E1-preregistration`, and the older
`e2.yaml` / `e3.yaml` beside this file are #334 / #335's, likewise
superseded; they are left untouched until E2 and E3 replace them in
their own PRs. Two consequences for reading this rung: the Wandb group
`curriculum-e1` and the run suffixes `s{1,2,3}e1` are shared with the
2026-09-14 runs of that older rung, which are void
(`reports/2026-09-14-one-episode-per-update.md`); this rung's runs are
the ones dated 2026-09-17 23:13 (checkpoint directories
`per-model-e1-2026-09-17-23-13-*-s{1,2,3}e1{,w}`, run ids on #381), and
every read of E1 names them by directory.

**Wall-clock, corrected.** The per-model runs update every 25–30 s at
128 rounds (rollout 20–26 s, update 4–10 s), so the 122,880-round budget
is about half a day, not the two to three days written above; A5b's 37
hours were measured with twelve trainers on the box.

### Amendment 2 — written 2026-09-19 01:20, before the relaunch, no number from any E1 run on the record

**The 2026-09-17 23:13 runs were stopped by decision** within the hour
of launch (Sash, 2026-09-17: not to run E1 while A5's per-model arm was
failing, since E1 is A5's army with guns) and none was read; their
directories are stale and every read of this rung names the runs by the
2026-09-19 timestamps below. #381 held since.

**A5 has since passed on the start axis** (A5i 0.960 / 0.980 / 0.960 at
122,880 — the bar cloned from 2,000 games and held by anchored PPO;
`reports/2026-09-15-curriculum-a-rungs.md#2026-09-19-curriculum-a5-passes-on-the-start-axis`), so the
rung below on the bodies branch now has a policy at the bar's level, and
**the companion's warm start changes from A5b's `last.pt` to A5i's**,
seed for seed (`per-model-a5-2026-09-18-20-46-09-s{1,2,3}a5i/last.pt`).
Loading check, three episodes on e1.yaml, not a read: the checkpoint
loads and plays (held 1.33, −101.7 vp) — a policy that has never seen an
enemy or a gun, as expected. The companion runs **without the anchor**
(the question is whether reward can teach shooting on top of a held
walk; anchoring to a policy that cannot shoot would forbid the answer);
the scratch arm and the whole-army control are unchanged. Criteria,
n, seeds and budget are unchanged. Set as a goal by Sash 2026-09-19
01:05: "carry the A5i policy up to E1 and read whether the allocation
survives guns" — pass on the pre-registered clauses, and on a fail the
census says whether the walk broke (`held`, bodies on points by turn)
or the shooting was never learned (`alive`, kills).

Tags: control `e1-ctl`, scratch `e1`, companion `e1w`; Wandb
`curriculum-e1`; run ids on #381 when they exist. Nine trainers and the
C3b clone fit (D3) on the box.

### Amendment 3 — written 2026-09-19 02:25, the whole-army control read at 120 epochs; no per-model number yet

The control finished its 60 epochs in about twenty minutes (the phase
facade steps a 24-v-24 shooting round in ~10 ms; the "two to three
days" of § The arms was the per-model estimate) with the in-run vp
still rising on every seed (by quarter −194 → −19, −196 → −21, −208 →
−69), so the pre-registered once-only extension to 120 ran, resumed
from each seed's 60-epoch `last.ckpt` (Wandb `pp3ep94b` / `rkdln3xc` /
`njr0mg5a`; directories `…-2026-09-19-01-41-2*-s{1,2,3}e1-ctl-x`).
Read greedy, K=1, on the phase facade, n=180 on 700000+, paired per
episode against `squad_march_take` on the same facade:

| seed | vp | script | arm − script, paired | t | held (bar 2.66) | coherent (bar 0.85) | alive | on obj | win |
|---|---|---|---|---|---|---|---|---|---|
| s1 | 4.7 | 20.2 | **−15.6 ± 9.2** | −1.68 | 2.47 | 0.860 | 0.577 | 0.877 | 0.43 |
| s2 | **93.6** | 20.2 | **+73.4 ± 10.3** | +7.10 | 3.05 | **0.780** | 0.502 | 0.781 | 0.79 |
| s3 | **65.4** | 20.2 | **+45.2 ± 10.5** | +4.31 | **1.98** | **0.773** | 0.466 | 0.808 | 0.76 |

**The control FAILS on the letter, on every seed by a different
clause**: s1 on vp (−15.6 against a −9.2 bound), s2 on coherency (0.780
against 0.82), s3 on held (1.98 against 2.46) and coherency. And two of
three seeds beat the script by 45–73 vp, t 4.3–7.1 — the largest
margins over a script on any rung of this ladder — by killing: opponent
VP 121–136 against the script's own ~200 (s1, which does not, concedes
227). What the control learned on E1 is the guns, at the cost of
formation (coherency 0.77–0.78 against the script's 0.85) and, on s3,
of the points. The E clause was written as a conjunction of three
readouts at the script's levels; a policy that wins the fight and
loses the formation fails it, and that is what the clause was for.
Reported as: the control beats the bar on vp on two seeds and fails
the rung. The per-model arm and the companion are read against the
same three clauses on their own facade at 122,880.

### Amendment 4 — written 2026-09-19 10:30, the per-model arm and the companion read at 122,880

Greedy, no decode, on the per-model facade, n=180 on 700000+, paired
per episode against `squad_march_take` on the same facade (the script:
vp +20.2 ± 7.6, win 0.567, held 2.93, on_obj 0.946, coherent 0.873):

| row | vp | win | held (bar 2.93; clause ≥ 2.68) | on obj | coherent (clause ≥ 0.84) | stat / hold |
|---|---|---|---|---|---|---|
| scratch s1 | **−158.1 ± 3.3** | 0.000 | 0.76 | 0.096 | 0.584 | 0.00 / 0.21 |
| scratch s2 | **−151.9 ± 4.8** | 0.028 | 0.71 | 0.169 | 0.383 | 0.49 / 0.09 |
| scratch s3 | **−159.8 ± 3.3** | 0.000 | 1.24 | 0.201 | 0.328 | 0.14 / 0.13 |
| from A5i s1 | **−67.9 ± 5.3** | 0.128 | 1.26 | 0.261 | 0.374 | 0.42 / 0.11 |
| from A5i s2 | **−179.7 ± 5.1** | 0.006 | 0.43 | 0.128 | 0.735 | 0.78 / 0.05 |
| from A5i s3 | **−159.1 ± 5.8** | 0.061 | 0.58 | 0.178 | 0.467 | 0.66 / 0.06 |

**FAIL as pre-registered on every seed of both arms, on every clause,
by a margin no clause was written for**: 170–200 vp behind the script
on five seeds (the sixth 88 behind), a fifth of the bar's points held,
a tenth to a quarter of the bodies on any point, coherency 0.33–0.73
against 0.87. The in-run curve never reached the script's vp on any
seed at any evaluation; the companion's best row (s1, −68) was its best
from 60k on.

The by-turn census (n=60 on 700000+; bodies on points of 24 and points
held of 6 after turns 3 → 5 → 7 → end of 20):

| policy | bodies on points | held | max stack |
|---|---|---|---|
| `squad_march_take` | 8.8 → 10.7 → 10.1 → 6.8 | 2.6 → 3.1 → 3.2 → 2.9 | 2.8 |
| scratch s1 / s2 / s3 | 1.5 → 1.6 → 0.8 → 1.3 / 2.0 → 2.4 → 2.1 → 1.0 / 2.3 → 2.0 → 1.2 → 2.4 | 0.9 → 0.8 / 1.2 → 0.7 / 1.2 → 1.2 | 0.9–1.9 |
| from A5i s1 / s2 / s3 | 2.7 → 3.6 → 3.7 → 2.5 / 2.9 → 3.8 → 3.3 → 0.9 / 2.7 → 2.5 → 2.1 → 0.9 | 1.3 → 1.2 / 1.3 → 0.4 / 1.5 → 0.6 | 0.8–1.9 |

**The walk broke and the shooting was not learned.** The script has
nine bodies on points by turn 3 and holds three through the game while
losing three quarters of both armies. The scratch arm never puts more
than two and a half bodies on a point at any turn. The companion
started from a policy that puts twenty of twenty-four bodies on six
points by turn 7 on A5; on E1 it has three to four on points by turn
5 — a fifth of A5i's walk — and by the end one to two and a half, the
points it reached lost (held 1.3–1.6 at turn 5, 0.4–1.2 at the end).
The A5i walk was overwritten within 10k rounds (stationary share
0.5–0.9 in-run from the first evaluations, 0.42–0.78 at the end), and
what replaced it neither walks nor shoots: vp 88–180 behind a script
that walks and shoots on the way. Sampled play and `alive` are read
with the report.

**Read against the whole-army control** (amendment 3): the control
learned the guns (+45 / +73 vp on two seeds by killing) and lost the
formation; the per-model arm learned neither. This is the first rung
where the per-model arm is not merely behind the control but plays a
different game from every comparator — the ladder's E branch stops
here, and the report says what the E1 census says about why.

---

<a id="2026-09-19-curriculum-e1-the-walk-does-not-survive-contact"></a>

## Curriculum rung E1: two armed armies on a flat board — the whole-army control learns the guns and loses the formation; the per-model arm learns neither, and A5i's walk does not survive contact

**Verdict first.** The join of the ladder's bodies and guns branches
(#340, #381): A5's army of twenty-four in squads of three, both sides
armed, a mirror `squad_march_take` opponent, twenty rounds on a flat
board. Three trainers, three seeds each, read against `squad_march_take`
on each trainer's own facade, paired per episode at n=180 on 700000+:

| row | vp v script, paired | held (clause) | coherent (clause) | alive (bar 0.27) | verdict |
|---|---|---|---|---|---|
| whole-army control, 120 epochs | −15.6 ± 9.2 / **+73.4 ± 10.3** / **+45.2 ± 10.5** | 2.47 / 3.05 / 1.98 (≥ 2.46) | 0.860 / 0.780 / 0.773 (≥ 0.82) | 0.58 / 0.50 / 0.47 | **FAIL on the letter on every seed, by a different clause each; beats the script by 45–73 vp on two** |
| per-model from scratch, 122,880 | **−158 / −152 / −160** (± 3–5) | 0.76 / 0.71 / 1.24 (≥ 2.68) | 0.58 / 0.38 / 0.33 (≥ 0.84) | 0.52 / 0.23 / 0.40 | **FAIL on every clause, every seed** |
| per-model from A5i, 122,880 | **−68 / −180 / −159** (± 5–6) | 1.26 / 0.43 / 0.58 | 0.37 / 0.74 / 0.47 | 0.33 / 0.14 / 0.18 | **FAIL on every clause, every seed** |

**FAIL as pre-registered, all nine runs.** The whole-army control found
the guns: on two seeds it kills its way to the largest margins over a
script on this ladder while dropping below the script's formation and,
on one, its points. The per-model arm found nothing: 150–160 vp behind
a script that walks and shoots on the way, a fifth of the bar's points,
a tenth to a fifth of its bodies on any point. **The companion from
A5i — the policy that puts twenty of twenty-four bodies on six points
by turn 7 on A5 — had that walk overwritten within 10,240 rounds** and
ended one seed at −68 and two at the scratch arm's level. The goal set
on 2026-09-19 asked whether the allocation survives guns: it does not,
and the census says which half broke — both. The E branch stops here
for the per-model trainer.

### Provenance

| field | value |
|---|---|
| date | 2026-09-19: control 01:15–01:35 (60 epochs) and 01:41–02:05 (extended once to 120), read 02:15; per-model scratch 01:15–09:40 and companion 01:17–09:05, read 09:45–10:20 |
| GPU / no-GPU | GPU (RTX 4090); nine E1 trainers, then six beside D3's six |
| seeds | 1 / 2 / 3 per trainer; the companion seed for seed from A5i's `last.pt` (three seeds off three A5i seeds, themselves off one clone of the bar) |
| n | 180 at 700000+ (final, paired per episode against the script on the same facade); 30 at 500000+ in-run every 512 rounds (per-model) and every epoch (control); 100 at 900000+ greedy against sampled; 60 at 700000+ by-turn census |
| config | `configs/experiments/curriculum/e1.yaml`: 24 v 24 in squads of three, both sides armed (range 12), flat board, fixed layout, six points, twenty rounds, unrefereed; opponent `scripted_baseline` wrapping `squad_march_take` |
| decode | none on the per-model facade; K=1 on the phase facade |
| paired | per episode against `squad_march_take` on the arm's own facade (`just measure-paired` for the control, `just measure-rung` for the per-model rows) |
| comparator | `squad_march_take` against itself, n=180 on 700000+: phase facade vp +20.2 (sd 102), per-model facade vp +20.2 ± 7.6, win 0.567, held 2.93, on_obj 0.946, alive 0.27–0.28, coherent 0.873 — the bridge diverges under a recorded rule (`shooting.targets_judged_after_casualties`), the two facades' rows within one SE |
| opponent | `squad_march_take`, the mirror |
| budget | control 60 epochs of 2048 steps extended once to 120 (the curve still rising at 60); per-model 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, the `mean` credit; the companion `--warm-start-from` A5i without an anchor |
| code revision | pre-registered `5566b7b` (+4 amendments to `bbb9df6`); runs on `10eeb1d`; branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | control `checkpoints/ppo-transformer-curriculum_e1-…-2026-09-19-01-41-2*-s{1,2,3}e1-ctl-x/last.ckpt` (epoch 120); per-model `checkpoints/per_model/per-model-e1-2026-09-19-01-15-10-s{1,2,3}e1` and `…-01-17-*-s{1,2,3}e1w`, `last.pt` at 122,880, every 512 rounds kept |
| coherency | greedy at 700000+: scratch 0.58 / 0.38 / 0.33, companion 0.37 / 0.74 / 0.47, control 0.86 / 0.78 / 0.77 (the script 0.87); sampled at 900000+ 0.27–0.50 |
| Wandb | `curriculum-e1`: control `iszzu87u` / `t7lm8pj3` / `a6zyzu3g` then `pp3ep94b` / `rkdln3xc` / `njr0mg5a`; scratch `n51yisrb` / `ronr0zl9` / `5jqvcdt5`; companion `qjpti8yr` / `5jaor2py` / `1d3rohef` |
| pre-registration | `reports/2026-09-15-curriculum-c-d-e-rungs.md#2026-09-17-curriculum-E1-preregistration` (+4 amendments); the 2026-09-17 23:13 runs were stopped by decision within the hour and never read |

### The read

#### The whole-army control learned the guns and lost the formation

The control finished sixty epochs in twenty minutes with its in-run vp
still rising on every seed (by quarter −194 → −19, −196 → −21, −208 →
−69) and ran the pre-registered once-only extension to 120. Read on the
phase facade, K=1, n=180, paired against the script:

| seed | vp | script | paired | t | held | coherent | alive | on obj | win |
|---|---|---|---|---|---|---|---|---|---|
| s1 | 4.7 | 20.2 | −15.6 ± 9.2 | −1.68 | 2.47 | 0.860 | 0.577 | 0.877 | 0.43 |
| s2 | **93.6** | 20.2 | **+73.4 ± 10.3** | +7.10 | 3.05 | **0.780** | 0.502 | 0.781 | 0.79 |
| s3 | **65.4** | 20.2 | **+45.2 ± 10.5** | +4.31 | **1.98** | **0.773** | 0.466 | 0.808 | 0.76 |

Each seed fails a different clause: s1 vp (−15.6 against a −9.2
bound), s2 coherency (0.780 against 0.82), s3 held (1.98 against 2.46)
and coherency. Two seeds beat the script by 45–73 vp at t 4.3–7.1 —
the largest margins over a script on any rung of this ladder — and they
do it by killing: opponent VP 121–136 against the ~200 the script
concedes to itself. The E clause was a conjunction of three readouts
at the script's levels, and a policy that wins the fight and loses the
formation fails it; that is what the clause was for, and the row
reports both facts.

#### The per-model arm from scratch plays a different game from every comparator

| seed | vp | win | held | on obj | coherent | alive | stat / hold |
|---|---|---|---|---|---|---|---|
| s1 | −158.1 ± 3.3 | 0.000 | 0.76 | 0.096 | 0.584 | 0.52 | 0.00 / 0.21 |
| s2 | −151.9 ± 4.8 | 0.028 | 0.71 | 0.169 | 0.383 | 0.23 | 0.49 / 0.09 |
| s3 | −159.8 ± 3.3 | 0.000 | 1.24 | 0.201 | 0.328 | 0.40 | 0.14 / 0.13 |

The script holds 2.93 with a tenth of a point of variance between the
facades; the scratch arm holds 0.7–1.2 with one or two bodies on any
point at any turn of the game (census: 1.5 → 1.6 → 0.8 → 1.3 bodies on
points on s1 at turns 3 → 5 → 7 → 20 against the script's 8.8 → 10.7 →
10.1 → 6.8). Its `alive` is at or above the bar's on two seeds — it is
not being shot off the points, it never reaches them — and its vp is
150–160 behind on every seed, more than the whole spread of any
comparator on this rung. On the health panel nothing is red: the
in-run curve rose from −250 to −150 and stopped.

#### The companion from A5i: the walk did not survive contact

| seed | vp | win | held | on obj | coherent | alive | stat / hold |
|---|---|---|---|---|---|---|---|
| s1 | −67.9 ± 5.3 | 0.128 | 1.26 | 0.261 | 0.374 | 0.33 | 0.42 / 0.11 |
| s2 | −179.7 ± 5.1 | 0.006 | 0.43 | 0.128 | 0.735 | 0.14 | 0.78 / 0.05 |
| s3 | −159.1 ± 5.8 | 0.061 | 0.58 | 0.178 | 0.467 | 0.18 | 0.66 / 0.06 |

A5i puts 20 of 24 bodies on six points by turn 7 on A5. Loaded on E1
untrained it plays that walk into twenty-four rifles (−102 vp on three
episodes at the loading check). Trained on E1's reward from there, the
walk was gone within 10,240 rounds: the in-run stationary share read
0.5–0.9 from the first evaluations on every seed, and the census at the
end has 2.7–2.9 bodies on points at turn 3 against A5i's seventeen —
one seed climbs to 3.7 by turn 7, the other two never pass 3.8 and end
at 0.9, the points they reached lost (held 1.3–1.6 at turn 5, 0.4–1.2
at the end). What replaced the walk shoots no better than the scratch
arm (alive 0.14–0.33, below the bar's 0.27 on two seeds; vp 88–180
behind). One seed (s1, −68) is the per-model trainer's best row on E1
and is still a point and a half and 88 vp short.

**Sampled play is worse than greedy on five of six seeds**, by 13–32 vp
paired (n=100 on 900000+): the policies are diffuse, and the greedy
argmax a score reports is the better half of what training rolled out.
No do-nothing fingerprint on the scratch arm (stationary 0.00–0.49);
the companion's 0.42–0.78 is the residue of A5i's walk being unlearned
into standing still.

### What it says

- **A warm start does not survive a change of game.** T1 (A5 from A4x)
  carried a walk onto a bigger board and did not rescue it; C1 (from
  A3x) carried a walk past an unarmed enemy and it held; E1 (from A5i)
  carried the ladder's best walk into rifles and lost it in 10k rounds
  with nothing learned in its place. The C rungs' rule — warm-start
  from the rung below — was measured on rungs that added an enemy who
  does not shoot back; it does not extend to a rung that adds guns on
  both sides. On this trainer the walk is unlearned before the shooting
  is learned, and the reward's signal in the first 10k rounds is the
  casualties, not the points.
- **The per-model trainer has not learned to shoot from reward on any
  rung, and E1 is where that bites.** C2 gave it guns against it (it
  walked at the script's speed under fire); C3 gave it one armed squad
  (it found the dash, not the plan); C3b (the plan's first half, then
  stopped); E1 gives both sides guns and asks for walk and fire together
  — 150–200 vp behind. The whole-army control learns to kill at the
  cost of formation; the per-model arm learns neither. **Before another
  E rung, the per-model trainer needs a guns-only rung the way C2 was
  the enemy-only rung**: the bar's walk held under an anchor while the
  shooting head learns on top of it is the untested arm this record
  names, and it is a D-route arm, so D3's finding applies — the anchor
  will hold the walk and may add nothing.
- **Read a whole-army control's turns and coherency beside its vp on
  every guns rung** (C2's rule, confirmed): a control that beats the
  script by 73 vp and fails the rung is measuring what the clause
  measures, and the ladder row carries both.
- **Sampled play is a diagnostic on a failing per-model row.** Greedy
  ahead of sampled by 13–32 vp on five seeds says the policy is
  diffuse, not converged; on A5i the two agreed within 1.2 vp. Read the
  gap before spending rounds.

### What was not done

- No anchored companion (A5i held under the KL anchor while learning to
  shoot): named above as the next arm; D3 says the anchor will hold the
  walk and may add nothing, so it is a one-arm question with a likely
  answer.
- No guns-only per-model rung between C2 and E1 (both sides armed, the
  walk given, the shooting to learn); the ladder's E branch assumed the
  C branch had taught the guns, and C3b said it had taught half.
- The control's own 60-epoch reads were not scored at n=180 (its 120
  rows are the comparator; its 60 rows are in-run only).
- E2 and E3 are not reached; their old Stage 0 configs stay untouched.

---

<a id="2026-09-14-curriculum-E2-preregistration"></a>

## Pre-registration: curriculum rung E2 — three units, three objectives, no opponent

Written 2026-09-14, **before any number from the arm exists**, on branch
`feature/per-model-pipeline-health` (the curriculum runs on one PR while it
is exploratory; the commit timestamp of this file is what orders it before
the numbers). Issue #334, parent question #328.

### The one change

Training on the **E2** rung: 3 units of 3, 3 objectives in a column at x=35, no opponent, movement only, 10 rounds (`configs/experiments/curriculum/e2.yaml`). Two
arms of the same scenario, three seeds each: the **per-model** trainer
(`train_per_model.py`, the thing under test) and the **whole-army** trainer
(`train.py`, the pipeline control). Nothing else varies.

### Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/e2.yaml 100`), bridge identical on every shared
field: success **1.000**, turns-to-success **6.38**, `held` **3.00**, `on_obj` 0.941, vp 37.5 ± 0.8. In-run band: not measured separately; the arm's in-run curve is read against the 700000+ bar.

### Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success rate ≥ 0.95 **and `held` ≥ 2.80** (the script's 3.00 − 0.2) and mean
  turns-to-success ≤ 7.38 (the script's 6.38 + 1), on **every seed** of the per-model
  arm, within 51,200 rounds.
- **FAIL:** any seed misses either bound at budget while the whole-army
  control passes on ≥ 2 of 3 seeds — the fault is allocation across units.
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds —
  the rung is misconfigured; redesign, do not read the per-model result.
- **Pass with a defect:** the bounds hold but the health panel
  (`docs/metrics.md` § The per-model health panel) is red over the last
  quarter of the run — recorded as such, never as a clean pass.

Power: success at n=100 has binomial SE 0.022 at p=0.95, so a true 0.90
fails the bound at ~2.3 SE; turns-to-success at sd ≈ 0.7 has SE 0.07 at
n=100, so the +1 allowance is fourteen SE wide — this rung is decided by the
success rate, and the turn bound is there to catch a policy that idles until
the last round.

### Budget, regime, seeds

| | per-model arm | whole-army control |
|---|---|---|
| budget | 51,200 rounds | 25 epochs = 51,200 rounds |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update**, 400 updates | 2048 steps per update (the shipped loop) |
| seeds | 1, 2, 3 (`--seed`); rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `evaluate_spec` | same |
| decode | none (K=1) | K=1, `verify_moves` default — undecoded, the same regime |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4), `--no-wandb` | defaults, `--no-wandb` |
| compute | as E1 | |

Launch: `just train-per-model-arm <rounds> 3 curriculum-e2 e2 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --no-wandb" configs/experiments/curriculum/e2.yaml`
and `just train-curriculum-control <epochs> 3 curriculum-e2 e2-ctl "" configs/experiments/curriculum/e2.yaml`.

### Primary readouts

Success rate and turns-to-success (paired against the script per episode),
`held`, `on_obj`; the passive pair and `train/rounds_per_update` on every
per-model row; the health panel over the last quarter. Reported with the
provenance table in the arm issue at close.

### What I expect (a guess, written so it can be wrong)

The whole-army trainer passes E2 within budget. The per-model trainer
either passes at a comparable round count — in which case the rung says the
plumbing works and nothing about the architecture — or its declaration head
collapses to `stationary` early (the do-nothing fingerprint) and the greedy
eval never leaves the floor while the sampled score climbs, which
`measure-per-model-eval-mode` will show. I do not know which.

---

<a id="2026-09-14-curriculum-E3-preregistration"></a>

## Pre-registration: curriculum rung E3 — eight units, six objectives, no opponent

Written 2026-09-14, **before any number from the arm exists**, on branch
`feature/per-model-pipeline-health` (the curriculum runs on one PR while it
is exploratory; the commit timestamp of this file is what orders it before
the numbers). Issue #335, parent question #328.

### The one change

Training on the **E3** rung: 8 units of 3 (24 models), 6 objectives in two columns at x=30 and x=45, no opponent, movement only, 12 rounds — the spare-squads shape (`configs/experiments/curriculum/e3.yaml`). Two
arms of the same scenario, three seeds each: the **per-model** trainer
(`train_per_model.py`, the thing under test) and the **whole-army** trainer
(`train.py`, the pipeline control). Nothing else varies.

### Comparator, measured first

`squad_march_take`, scored through both facades on seeds 700000+ at n=100
(`just measure-bridge configs/experiments/curriculum/e3.yaml 100`), bridge identical on every shared
field: success **0.990**, turns-to-success **8.78**, `held` **6.00**, `on_obj` 0.907, vp 89.8 ± 1.3. In-run band: not measured separately.

### Criteria

Read at the END of the budget from the last checkpoint (`last.pt` /
`last.ckpt`), greedy, no decode, n=100 on seeds 700000+ — the same seeds
the bar was measured on, so success and turns pair per episode.

- **PASS:** success rate ≥ 0.95 **and `held` ≥ 5.80** and mean
  turns-to-success ≤ 9.78 (the script's 8.78 + 1), on **every seed** of the per-model
  arm, within 153,600 rounds.
- **FAIL:** any seed misses either bound at budget while the whole-army
  control passes on ≥ 2 of 3 seeds — the fault is scaling of the set network (the whole-army control is the pipeline reference here too).
- **NULL (scenario):** the whole-army control also fails on ≥ 2 of 3 seeds —
  the rung is misconfigured; redesign, do not read the per-model result.
- **Pass with a defect:** the bounds hold but the health panel
  (`docs/metrics.md` § The per-model health panel) is red over the last
  quarter of the run — recorded as such, never as a clean pass.

Power: success at n=100 has binomial SE 0.022 at p=0.95, so a true 0.90
fails the bound at ~2.3 SE; turns-to-success at sd ≈ 0.7 has SE 0.07 at
n=100, so the +1 allowance is fourteen SE wide — this rung is decided by the
success rate, and the turn bound is there to catch a policy that idles until
the last round.

### Budget, regime, seeds

| | per-model arm | whole-army control |
|---|---|---|
| budget | 153,600 rounds | 75 epochs = 153,600 rounds |
| regime | `--rollout-rounds 32 --num-rollout-envs 4` = **128 rounds per update**, 1,200 updates | 2048 steps per update (the shipped loop) |
| seeds | 1, 2, 3 (`--seed`); rollout layouts at seed×100+ | 1, 2, 3 |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair logged | every epoch, n=30, seeds 500000+ |
| final score | n=100, seeds 700000+, `evaluate_spec` | same |
| decode | none (K=1) | K=1, `verify_moves` default — undecoded, the same regime |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4), `--no-wandb` | defaults, `--no-wandb` |
| compute | GPU for the per-model arm; the control on CPU | |

Launch: `just train-per-model-arm <rounds> 3 curriculum-e3 e3 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --no-wandb" configs/experiments/curriculum/e3.yaml`
and `just train-curriculum-control <epochs> 3 curriculum-e3 e3-ctl "" configs/experiments/curriculum/e3.yaml`.

### Primary readouts

Success rate and turns-to-success (paired against the script per episode),
`held`, `on_obj`; the passive pair and `train/rounds_per_update` on every
per-model row; the health panel over the last quarter. Reported with the
provenance table in the arm issue at close.

### What I expect (a guess, written so it can be wrong)

The whole-army trainer passes E3 within budget. The per-model trainer
either passes at a comparable round count — in which case the rung says the
plumbing works and nothing about the architecture — or its declaration head
collapses to `stationary` early (the do-nothing fingerprint) and the greedy
eval never leaves the floor while the sampled score climbs, which
`measure-per-model-eval-mode` will show. I do not know which.
