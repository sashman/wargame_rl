# Phase 5: The Opponent Turn — Context

**Measured 2026-09-05 on `configs/golden/25v25_maps_two_mode.yaml`**, 400 steps,
random play, `build_info=False` (what the rollout actually runs). Box: Ryzen
3950X, **CPU-only** — this machine's GTX 1080 Ti is sm_61 and torch 2.8 ships no
kernels for it, so `auto_device()` falls back. Every number below is CPU.

## Why this phase exists

`just measure-throughput` on the config that trains:

```
env.step()   8.757 ms
  opponent turn      3.755 ms   42.9%     <- the largest single line
  observation build  1.600 ms   18.3%
  reward             1.027 ms   11.7%
  line of sight      0.786 ms    9.0%
```

The opponent turn is the biggest item in the environment step, and the
environment step is what caps a training run: the rollout is **2048 sequential
`env.step()` calls in one thread** (`_collect_rollout_parallel`'s own docstring:
*"keeps env stepping in Python (single process) but batches the policy/value
forward pass"*), i.e. **19.2 s per epoch on one core** while the box has sixteen.

It needs no hardware, no dependency, and no re-baselining — and it pays out
beyond training. Every `measure-*` recipe, the Elo arena, the baselines and the
seat-parity gate step the same env, and on a scripted-vs-scripted leg **both
sides** pay this cost.

## The profile, one level down

Instrumented with `perf_counter` wrappers, not cProfile — the repo's own note
applies: cProfile's per-call overhead inflates this exact shape (~100 small
calls per step) by ~3× and points the optimisation at the wrong target.

**The opponent turn — 3.771 ms, fully accounted:**

| section | ms/step | % of turn | calls/step |
|---|---|---|---|
| `policy.select_action` | **1.205** | **31.9%** | 1.00 |
| `action_handler.apply` (movement) | 0.902 | 23.9% | 0.50 |
| `_resolve_shooting_action` (opponent's share) | 0.777 | 20.6% | 0.50 |
| `_opponent_action_mask` | 0.612 | 16.2% | 1.00 |
| [unaccounted] | 0.275 | 7.3% | |

⚠ `_resolve_shooting_action` is **shared with the player's turn**. A first pass
wrapped it naively and double-counted, which made the sections sum to a tidy and
wrong 100%. The row above is the opponent's share only, taken with a re-entrancy
flag set inside `_apply_opponent_action`.

**Inside `select_action` → `select_movement` (1.057 ms, 93.6% of it):**

| | ms/step | % of `select_movement` | calls/step |
|---|---|---|---|
| `step_toward_objective` | **0.748** | **70.8%** | 7.37 |
| `squad_objectives` | 0.141 | 13.4% | 0.50 |
| loop + centroid + rest | 0.167 | 15.8% | |

`step_toward_objective` runs **14.7 times per movement phase at ~101 µs a call**,
and each call does `area.contains(x, y)` and `area.distance_to_point(x, y)` —
scalar polygon queries, one point at a time, against a ruin outline. The repo
already has vectorised polygon primitives (`polygons_contain_points`), and every
member of a squad is tested against **the same objective**.

## ⚠ Measured and REJECTED as a target

`best_action_toward` — **0.079 ms/step, 2.1% of the opponent turn**, only 6.5
calls per movement phase (most squads have arrived and take the other branch).

It has every mark of the classic bug: it allocates a constant `np.linspace` on
**every** call, its angle search is numpy overhead on a 16-element array, and
**75.6% of its argument tuples are exact repeats** of one already computed —
because `dx`, `dy` and `max_step_length` are shared across a squad and only
`model_idx` varies.

**It is still not where the time is.** This was predicted to be the hot spot
before it was measured, on the analogy of the reward-calculator fix (two
calculators recomputing a model-independent quantity per model, ~80% of a step).
The analogy was wrong. Recorded so nobody re-derives it and spends a day there.

## The invariant that makes this phase safe

The opponent's actions move models, which changes state, reward and VP. So:

- `tests/test_reward_golden.py` pins per-step reward, per-model reward, the
  breakdown, VP and **every model position** with `assert_array_equal` — never
  `assert_allclose` — and is verified sensitive to a one-ULP perturbation.
- `tests/test_reproduce_recording.py` replays a recorded match.

Every change in this phase is therefore gated by an existing, *sensitive*,
bit-identical test. That is the ideal shape for a performance refactor: it can be
**proven** not to have changed the experiment.

⚠ **Vectorising reorders float reductions, which is exactly what breaks that
gate.** Batching a per-point boolean test is safe. Anything that sums or reduces
*across* points is not, and numpy may take a different code path (SIMD, pairwise
summation) for a batched call than for a scalar one even where the maths is
identical. **Run the gate; never reason that a change must be safe.** A change
that cannot be made bit-identical does not ship — it is not worth a re-baseline
of the whole corpus to save a millisecond.

## ⚠ The ceiling, stated before anyone budgets on it

The opponent turn is 42.9% of `env.step`. Deleting it entirely would be a **1.75×**
env-step speedup, and env stepping is ~19.2 s of a ~30 s CPU epoch (the rest is
evaluation, itself ~22% of a real epoch). So:

- A **realistic** target — halving the two biggest items — is ~20% off `env.step`
  and **~12–15% off epoch wall-clock**.
- The theoretical maximum, with the opponent turn free, is ~25% off the epoch.

This is worth doing because it is free and compounding, **not** because it
unblocks anything. Anyone hoping for a 2× should read this paragraph first.

## Decisions

### Instrument before optimising

The breakdown above came from a scratch script. **It lands in
`scripts/measure_throughput.py` first**, so the before/after is produced by a
committed tool rather than re-derived by hand — and so the double-counting
mistake above cannot be repeated silently by the next person.

### Ranked targets

1. **`step_toward_objective`** — 0.748 ms/step, 19.8% of the turn. Per-model
   scalar polygon queries against one shared objective. Batch per squad.
2. **`_opponent_action_mask`** — 0.612 ms/step, computed on **every** step
   (1.00 calls/step, against 0.50 for the movement and shooting paths). Establish
   what it costs in each phase before touching it.
3. **`_resolve_shooting_action`** — the opponent pays **0.777 ms against the
   player's 0.248** over the same 0.50 calls/step, a **3.1× asymmetry**.
   ⚠ **Explain it before optimising it.** The likely reason is workload, not code
   path — the script declares shots on every model while a random player under a
   mask often cannot — in which case there is nothing to fix and the row is a
   measurement artefact of profiling against random play. If it is *not* that, it
   is a defect and more interesting than the throughput.
4. **`action_handler.apply` (movement)** — 0.902 ms/step, 1.80 ms per movement
   phase. ⚠ **Highest risk, lowest priority.** `docs/movement.md` records two
   attempts at this solver both measured and **reverted**, and CLAUDE.md's
   standing instruction is *"do not attempt a fourth"*. That instruction is about
   changing the *behaviour*; a pure-performance change that holds the golden
   bit-identical is a different act — but the bar for touching this file is a
   proof of no behavioural change, not an argument that there should not be one.

### Claude's discretion

- Whether target 2 is worth doing at all after it is measured per phase.
- The batching shape for target 1, provided the gate holds.
- Whether to stop after target 1. **Stopping after one target is a success**, not
  a partial delivery, if the gate holds and the number moved.

## Canonical references

- [docs/training-throughput.md](../../../docs/training-throughput.md) — where an
  epoch goes, and the bit-identical invariant
- [reports/2026-08-08-throughput-review.md](../../../reports/2026-08-08-throughput-review.md)
  — the ranked plan and five premises it corrected
- [docs/movement.md](../../../docs/movement.md) — why the movement solver is
  dangerous ground
- `CLAUDE.md` § Performance and numerics
