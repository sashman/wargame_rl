# 2026-08-09 — TF32 costs 8.5 vp_margin

**Question.** A fresh 1000-epoch run of `configs/golden/25v25_shooting_opponent.yaml`
scored **+21.2 / +19.9 vp_margin** where the published figure was **+28.4**. Same
config, same seeds, same scenario. What changed?

**Answer.** TF32, which shipped on by default on 2026-08-08 and had never been
measured against a trained result. It costs **~8.5 vp_margin** and buys 17.8% of
an epoch. It is now off by default.

---

## What was measured

Everything below is n=100 on identical layouts (seeds 700000–700099), scored via
`just measure-checkpoint`, against a `squad_march_shoot` bar of **+17.0** that
reproduced exactly.

### The effect, at matched epochs

Both sides at epoch 1000, so no checkpoint-selection difference:

| seed | TF32 off | TF32 on | delta |
|---|---|---|---|
| 1 | **+30.8** | +21.2 | **−9.6** |
| 2 | **+27.4** | +19.9 | **−7.5** |
| mean | **+29.1** | +20.6 | **−8.5** |

Against the bar: beating it by **12.1** becomes beating it by **3.6**. The result
direction survives — the agent clears the bar under either setting — but the
effect size is cut to a third.

### The control reproduced bit-identically

The `--no-tf32` run was compared to the original pre-TF32 run *at the weight
level*, not by score:

| comparison | tensors identical | max abs diff |
|---|---|---|
| no-TF32 s1 ep970 vs original s1 ep970 | **222 / 222** | **0.0** |
| no-TF32 s2 ep692 vs original s2 ep692 | **222 / 222** | **0.0** |

This is the load-bearing measurement. It establishes two things at once:

1. **TF32 is the whole of the difference.** Nothing else in the window between
   the two runs contributed — not the env hot-path memoisation (#139), the
   `last.ckpt` fix (#140), eval-every-N (#141), `--lr`/`--max-grad-norm` (#142),
   the `load_state` VP-delta fix (#143), the DQN removal (#144), or the config
   restructure (#146).
2. **"Training is deterministic given seed + config + code" holds**, across all
   of those changes, to the bit.

The control was also self-announcing: it landed on the same best-training-reward
epochs as the original (970 and 692), which is what prompted the weight
comparison ahead of any scoring.

### The speed it buys

| | s/epoch | 1000-epoch wall |
|---|---|---|
| TF32 on | 10.99 | 3 h 03 m |
| TF32 off | 12.95 | 3 h 36 m |

**17.8%**, not the 1.34x the update-only benchmark implies — the PPO update is
one part of an epoch. Both runs were two-concurrent on one RTX 4090.

## Why it was missed for a day

The claim in `performance.py` was:

> This lowers matmul mantissa precision from 24 bits to 11, so trained results
> move slightly. That is below every effect size this project can resolve — win
> rate cannot separate differences under ~7pp and `vp_margin` under ~10.

The reasoning is sound and the conclusion is wrong. It was derived from the
mantissa drop and a throughput benchmark; no trained result was ever compared.
8.5 vp is *at* the stated resolution limit, not beneath it, and it is signed the
same way on both seeds.

**Win rate really would have missed it.** 0.705 → 0.65 is 5.5pp, inside the
documented ~7pp win-rate limit, while `vp_margin` separated cleanly on both
seeds. The standing "prefer `vp_margin` for arm comparisons" rule is what made
this visible.

## What this changes

- **TF32 is off by default.** `--tf32` opts in; `--no-tf32` still parses and is
  now a no-op restating the default. Use `--tf32` for smoke, profiling and
  throughput runs, where the result is not the point.
- **Treat a precision or numerics setting as a reward-affecting change.** It
  deserves the same two-seed screen as a shaping term. Cheap to run: this took
  one 3.5-hour control to settle definitively.
- **Runs split into two regimes at 2026-08-08 23:36:50** (commit `14e6b2f`).
  Anything trained between then and this fix is not comparable to anything
  either side of it. In practice that is only the run this report is about.

## Corrections to earlier claims

- **`CLAUDE.md`'s "+28.4 ... on both seeds"** was loose in two ways. +28.4 is the
  *mean* of the two seeds; neither scored it (+30.1 and +26.6). The underlying
  [report](2026-08-08-paying-the-pot-beats-the-bar.md) states the per-seed
  figures correctly.
- **The epoch-970/692 selection turned out not to matter here.** Those
  checkpoints came from the old `last.ckpt` bug, and the initial hypothesis in
  this investigation was that they flattered the result. They did not: honest
  epoch-1000 scores under the same setting are **+30.8 / +27.4**, averaging
  +29.1 — slightly *above* +28.4. The selection bias is real and documented, but
  it accounts for none of this gap.

## Reproducing

```bash
# The two arms, 1000 epochs, two seeds each, in one window per arm.
just train-seed-flags 1000 1 <group> ""        ""           configs/golden/25v25_shooting_opponent.yaml &
just train-seed-flags 1000 1 <group> -notf32   "--tf32"     configs/golden/25v25_shooting_opponent.yaml &

# Score both against the same bar on the same layouts.
just measure-checkpoint <ckpt> configs/golden/25v25_shooting_opponent.yaml 100
just measure-baselines configs/golden/25v25_shooting_opponent.yaml 100 "" 700000
```

`train-seed-flags` was added for this experiment: `train-arm` is the only recipe
taking extra flags and it runs its seeds sequentially, which would have cost two
full windows per arm.

**Runs.** TF32 on: `k9ivraa3` (s1), `ubp1t8bw` (s2), group
`shooting-opponent-golden-2026-08-09-12-31-10`. TF32 off: `ba8oeadh` (s1),
`19vxegom` (s2), group `tf32-control-2026-08-09-15-43-41`.
