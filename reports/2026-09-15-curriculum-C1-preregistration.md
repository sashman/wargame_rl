# Pre-registration: curriculum rung C1 — A3 with one enemy unit standing on a point, nobody shooting

Written 2026-09-15, **before any training number on this rung exists**,
on branch `feature/curriculum-c1` (stacked on PR #364). Issue #365,
parent question #340, rung **C1** — the "enemy" axis: the first rung on
which an opposing army exists.

## The scenario

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

## The bar, measured first

`squad_march_take`, seeds 700000+ at n=100, **both facades identical on
every shared field** (`just measure-bridge`): success **0.990**, `held`
3.00 (the enemy's point is contested 3 v 3, never held), turns 4.98,
`on_obj` 0.871, vp 8.2 ± 0.6, coherent 0.933, decisions 4.98 turns. The
script clears the pass mark; the rung is not sent back to design. The
1% it misses is the census's own friendly gridlock around a crowded disc.

## The two arms

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

## Criteria

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

## What I expect (a guess, written so it can be wrong)

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

## Amendment 1 — written 2026-09-16 13:02, after the control's 60-epoch read and before any per-model number

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
