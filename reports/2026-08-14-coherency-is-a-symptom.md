# Coherency is a symptom: the adrift models are defecting

2026-08-14. Real-maps scenario (`configs/experiments/25v25_maps_coherent.yaml`
and arms), post-#182 and post-#186.

## The headline

The agent does not hold unit coherency: `eval/coherency_rate` 0.52–0.68, ~3 of
25 models adrift. Four mechanisms were tried. The one that works is the one this
project spent the least time on.

| mechanism | `coherency_rate` | models adrift | `vp_margin` |
|---|---|---|---|
| nothing (control, epoch 1000) | 0.52–0.65 | 2.9–3.7 | +98.5 |
| observation only, no reward | 0.58–0.64 | 3.2–3.3 | +92.4 (in-run) |
| reward, pay gap 0.05 (epoch 1000) | 0.62–0.68 | 2.5–2.9 | +98.2 |
| reward, pay gap 0.10 (epoch 370) | 0.63–0.70 | 2.2–2.9 | ~+92 (in-run) |
| **enforcement, warm-started (epoch 10)** | **1.000 / 0.999** | **0.00 / 0.005** | 79.0 / 90.2 |

Scripted reference points on the same config: `squad_march_shoot` 0.837 with
0.99 adrift, `hold_deployment` (never moves) 1.000 with 0.00.

## What the reward did, and why it stopped

Measured at **matched epoch 175**, because the first read of this compared
different epochs and got the sign of the conclusion wrong:

| pay gap | `coherency_rate` |
|---|---|
| 0 (control) | 0.562 |
| 0.05 | 0.621 |
| 0.10 | 0.659 |

Monotone, and `vp_margin` unchanged throughout — the term is **free**. But the
increments diminish as the gradient doubles (+0.059, then +0.038), and by epoch
370 the steeper arm had only reached 0.674. Extrapolating to the scripted 0.837
needs roughly five more doublings, landing near 0.5/step — far outside the range
this repo has evidence for (`group_cohesion` at −0.2 inverted the baseline
ranking).

Diminishing returns as a gradient doubles is the signature of a **bonus fighting
a stronger term**, not of a missing gradient.

## What it was fighting

`scripts`-adjacent diagnostic (`scratchpad/defection_check.py`), trained
checkpoint, 10 episodes, **1399 adrift-model observations**:

| | |
|---|---|
| nearest objective **differs** from the unit body's | **82.4%** |
| same objective as the body (boundary flicker) | 17.6% |
| distance from the body | median **13.6**, p90 20.0, max 22.4 |
| spread cap | **9.0** |

The adrift models are not slipping out of formation. **They have left, and they
are walking to a different objective.**

`objective_hold`'s `crowding_exponent` splits a point's pot among its occupants,
so the marginal model's private gradient points at the **emptiest objective** —
and nothing makes that spread respect unit boundaries. The `unit_coherency`
term, at ±0.05, is a counterweight to it.

**This retracts a claim in the calculator's own docstring**, which argued the two
terms were compatible because one is between units and the other within. Five
coherent units on five objectives does satisfy both; the policy simply is not
doing that.

**And the defection does not work.** Control is a strict count comparison, so a
lone model cannot flip a point. The agent holds **fewer** objectives than the
bar (3.44 v 3.76) while spreading more, and keeps **96%** of its army alive
against the bar's 72%. It is trickling single models onto objectives instead of
arriving as a squad. That is consistent with the abandonment result already on
file — ~37% of objectives get zero models, invariant across five reward
weightings.

## Enforcement, and why it looked impossible

`revert_unit` trained from scratch scored **−75.3** vp_margin with `on_obj` 0.08
and `alive` 0.63: the policy learned not to move. That was read as "the
constraint is unlearnable".

It is not. A near-random policy under a revert has **every move cancelled**, and
correctly learns that moving does nothing — a credit-assignment pathology of
training *into* the constraint, not a property of the constraint. Warm-starting a
competent policy into `revert_model` instead:

- `coherency_rate` **1.000 / 0.999** at epoch 10
- `models_out_of_coherency` **0.00 / 0.005**
- `vp_margin` 79.0 / 90.2 against the control's ~92.5 — early, wide, and the open
  question

The reward-phase curriculum cannot anneal a constraint (a phase carries reward
calculators and success criteria, not env parameters), but a **warm start
approximates it in two stages with no new machinery** and is already supported.

## What this changes

1. **Coherency is a legality constraint, not a capability to teach.** Human
   players never evaluate an incoherent move; it is not a priced choice. A
   penalty teaches an exchange rate, and the policy will pay it whenever the
   objective is worth more — which is exactly the 0.67 plateau.
2. **Prefer enforcement, warm-started.** 1.000 compliance, no bribery, and the
   metric stops being gameable.
3. **The next lever is not about formation.** It is **unit-atomic objective
   income** — a unit's contribution counted once and split within the unit, so a
   lone defector adds nothing its squad did not already have. Two standing
   traps: an anti-concentration lever must **conserve total income**
   (`surplus_value` and the overstack penalty both failed by destroying it), and
   it must stay **per-model differentiated** (flat `objective_hold` failed by not
   being).

## Method notes, including two mistakes

- **A first read compared rung 2 at epoch 175 against rung 1 at epoch 1000** and
  concluded the reward was saturating. Re-measured at matched epochs, the
  response is monotone. The repo's own warning about comparing arms at different
  epochs applies to *in-run* metrics too, not just to top-k checkpoints.
- **`coherency_rate` is confounded with squad size** — a unit shot to one model
  is coherent by definition, so the rate climbs as an army dies. The bar's 0.837
  is measured at `alive` 0.72 against the agent's 0.96, so some of that gap is
  denominator. Normalised, the gap survives (2.72/24 adrift v 0.99/18, still ~2x)
  but the raw comparison overstates it. Always read `models_out_of_coherency`
  beside it.
- The eval metrics themselves are new (`env_components/coherency_tracker.py`,
  #189). Before them, runs trained under the coherency rule with no record of
  whether they held formation.
