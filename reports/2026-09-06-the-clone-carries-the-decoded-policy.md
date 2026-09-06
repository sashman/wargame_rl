# House-fidelity distillation carries the decoded policy — the predicted failure did not happen

**Provenance.** 2026-09-06 · **GPU** · teacher = arm 4 seeds 1–3 `last.ckpt` at
**epoch 1000**, played with **K=3 + reallocation + charge decode** ·
students `checkpoints/decodeclone-s{1,2,3}.ckpt`, **1200 demonstrations x 60
epochs** (house fidelity), demonstrations on seed base **800000** ·
scored **n = 180**, seed base **700000**, configs
`configs/evaluation/25v25_maps_melee_approach_{refereed,vs_take,vs_deny,vs_shoot}.yaml`,
**refereed** · decode K=3, `verify_moves` on, charge decode on, reallocation on ·
**paired per scenario and per seed** (clone-sN is distilled from arm4-sN) ·
comparator **`squad_march_take_charge`** at the same n=180 on the same scenarios ·
code revision `1c8b3bd` · **no PPO training** — this is the clone as fitted.

Pre-registered in `reports/2026-09-06-distillation-preregistration.md`.

## ⚠ The pre-registered prediction is REFUTED

The registration predicted: *"The clone will not carry the joint properties.
The record's standing finding is that a per-model fit does not inherit a joint
property: a 98.3%-action-match clone of `squad_march_take` held **0.40** unit
coherency against its teacher's 0.95. I expect the same failure mode here."*

It did not happen. The fit reached **action-match 0.973, unit-match 0.933** —
whole-squad joint choices reproduced 93.3% of the time — and played at n=180 the
clone is **not worse than its teacher on any cell**.

## The result

Clone against its own teacher, paired on matched seeds and scenarios (the
tightest comparison available here):

| cell | teacher | clone | gain | t | seeds |
|---|---|---|---|---|---|
| `refereed` | −4.04 | **+2.83** | +6.87 | 0.99 | 2/3 |
| `vs_take` | +44.62 | **+47.29** | +2.67 | 0.38 | 2/3 |
| `vs_deny` | +45.71 | **+49.64** | +3.93 | 0.52 | 2/3 |
| `vs_shoot` | +60.92 | **+64.19** | +3.27 | 0.92 | 2/3 |

**Positive on 4 of 4 cells, t < 1 on all of them, and seed 3 negative on every
one.** Direction consistent; magnitude unresolved.

## ⚠ A comparison that would have overstated it, avoided

Against the bar the clone's gaps read **+12.78 / +12.48 / +12.83 / +7.71**
versus the six-seed trained agent's +4.21 / +6.73 / +7.45 / −0.19 — roughly
double to triple. That comparison is **not like-for-like**: the clone has three
seeds and that agent row has six. On the matched three-seed subset the teacher
reads −4.04 / +44.62 / +45.71 / +60.92, and the gain shrinks to the table above.
**Always difference against the same seed set**, not against the published row.

## What this establishes

- **Distillation at house fidelity carries a decoded policy.** `docs/melee-teaching-goal.md`
  §46 measured a clone that "loses the operator's gain and falls below the plain
  teacher by 3.9" — at **120 demonstrations x 8 epochs**, i.e. 10x fewer
  demonstrations and 7.5x fewer epochs than this repo's own stated house
  fidelity. **That result is a power artefact, not a property of distillation.**
- **The policy-improvement loop is viable here**: decode as the improvement
  operator, distillation as the projection back into the weights. The first step
  was believed blocked and is not. Whether iterating it compounds is untested.
- **What is NOT established**: that distillation improves on its teacher. Four
  positive cells at t < 1 with one seed negative throughout is a direction, not
  an effect.

## ⚠ A defect this arm had to fix first

`behaviour_clone.py` built its teacher **without `charge_decode`**, and the
demonstration cache key did not record it. On a melee config that distils a
teacher which **declares charges it cannot execute**. Fixed; default off, so
every existing clone and cached collection is unaffected. The cache key now
carries `-charge`, and this arm's collections are filed under it.

## Not run

The registered arm's second half — PPO from the clone basin, 3 seeds x 1000
epochs — was **disarmed before launch** when the melee-ladder goal was cleared.
The clone diagnostic above was allowed to finish because it generalises beyond
that goal. Re-arm with `launch_distil.real.sh`.
