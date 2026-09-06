# Where the opponent turn goes — 42.9% of `env.step`, and one function is a fifth of it

*2026-09-05. **No GPU.** `configs/golden/25v25_maps_two_mode.yaml`, 400 steps,
random play, `build_info=False` (what the rollout actually runs). Box: Ryzen
3950X, **CPU-only** — this machine's GTX 1080 Ti is sm_61 and torch 2.8 ships no
kernels for it, so `auto_device()` falls back. Every number below is CPU.*

*Extracted from `.planning/phases/05-opponent-turn-throughput/05-CONTEXT.md` when
that directory was retired. The plan it justified is a GitHub issue; the
measurements are here, because evidence belongs in git.*

## Why this was measured

`just measure-throughput` on the config that trains:

```
env.step()   8.757 ms
  opponent turn      3.755 ms   42.9%     <- the largest single line
  observation build  1.600 ms   18.3%
  reward             1.027 ms   11.7%
  line of sight      0.786 ms    9.0%
```

The environment step caps a training run: the rollout is **2048 sequential
`env.step()` calls in one thread** — `_collect_rollout_parallel`'s own docstring
says *"keeps env stepping in Python (single process) but batches the policy/value
forward pass"* — i.e. **19.2 s per epoch on one core** while the box has sixteen.

## The profile

Instrumented with `perf_counter` wrappers, not cProfile: the repo's own note
applies, since cProfile's per-call overhead inflates this exact shape (~100 small
calls per step) by ~3× and points the optimisation at the wrong target.

**The opponent turn — 3.771 ms, fully accounted:**

| section | ms/step | % of turn | calls/step |
|---|---|---|---|
| `policy.select_action` | **1.205** | **31.9%** | 1.00 |
| `action_handler.apply` (movement) | 0.902 | 23.9% | 0.50 |
| `_resolve_shooting_action` (opponent's share) | 0.777 | 20.6% | 0.50 |
| `_opponent_action_mask` | 0.612 | 16.2% | 1.00 |
| [unaccounted] | 0.275 | 7.3% | |

⚠ **`_resolve_shooting_action` is shared with the player's turn.** A first pass
wrapped it naively and double-counted, which made the sections sum to a tidy and
**wrong** 100%. The row above is the opponent's share only, taken with a
re-entrancy flag set inside `_apply_opponent_action`. For contrast the *player's*
share of the same resolver is **0.248 ms** over the same 0.50 calls/step — a
**3.1× asymmetry** that is unexplained and should be explained before it is
optimised. The likely reason is workload rather than code path (the script
declares shots on every model while a random player under a mask often cannot),
in which case it is an artefact of profiling against random play.

**Inside `select_action` → `select_movement` (1.057 ms, 93.6% of it):**

| | ms/step | % of `select_movement` | calls/step |
|---|---|---|---|
| `step_toward_objective` | **0.748** | **70.8%** | 7.37 |
| `squad_objectives` | 0.141 | 13.4% | 0.50 |
| loop + centroid + rest | 0.167 | 15.8% | |

`step_toward_objective` runs **14.7 times per movement phase at ~101 µs a call**,
and each call does `area.contains(x, y)` and `area.distance_to_point(x, y)` —
**scalar** polygon queries, one point at a time, against a ruin outline. Every
member of a squad is tested against **the same objective**, and the repo already
ships vectorised polygon primitives (`polygons_contain_points`).

## ⚠ Measured and REJECTED as a target

`best_action_toward` — **0.079 ms/step, 2.1% of the opponent turn**, only 6.5
calls per movement phase (most squads have arrived and take the other branch).

It has every mark of the classic bug: it allocates a constant `np.linspace` on
**every** call, its angle search is numpy overhead on a 16-element array, and
**75.6% of its argument tuples are exact repeats** of one already computed —
because `dx`, `dy` and `max_step_length` are shared across a squad and only
`model_idx` varies.

**It is still not where the time is.** This was predicted to be the hot spot
before it was measured, by analogy with the reward-calculator fix (two
calculators recomputing a model-independent quantity per model, ~80% of a step).
The analogy was wrong. Recorded so nobody re-derives it and spends a day there.

## ⚠ The ceiling, stated before anyone budgets on it

The opponent turn is 42.9% of `env.step`. Deleting it entirely would be a
**1.75×** env-step speedup, and env stepping is ~19.2 s of a ~30 s CPU epoch (the
rest is evaluation, itself ~22% of a real epoch). So:

- A **realistic** target — halving the two biggest items — is ~20% off `env.step`
  and **~12–15% off epoch wall-clock**.
- The theoretical maximum, with the opponent turn free, is ~25% off the epoch.

Worth doing because it is free and compounding — it speeds every `measure-*`
recipe, the Elo arena and the baselines, not only training — **not** because it
unblocks anything. Anyone hoping for a 2× should read this paragraph first.

## What this does not say

- **Nothing here is a verdict on a change**; no optimisation has been attempted.
  The bit-identity gate that would govern one is `tests/test_reward_golden.py`,
  which pins reward, VP and every model position with `assert_array_equal` and is
  verified sensitive to a one-ULP perturbation.
- **These are CPU numbers.** The environment/update balance is different on a
  machine where the update actually runs on a card.
