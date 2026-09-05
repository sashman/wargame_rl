# Pre-registration — does the surplus-reallocation decode transfer to the KL-anchored lineage?

Written 2026-09-05 while arm 4 seeds 4–6 were at epoch ~374/1000, **before any
number in this design existed**. Committed before the scorer ran.

## Why this exists

`build_action_selector` carries a fourth play-time decode nobody has been passing:
`reallocate`, the surplus-reallocation operator
(`model/common/reallocation_decode.py`). Its docstring records **+8.3 ± 4.25 vp**
on six trained checkpoints at n=45, K=3, frozen weights
(`docs/melee-teaching-goal.md` §40c) — *the largest lever on file*. Every score in
the melee-ladder goal so far — §46 clone, the interpolation route, arms 1, 3 and 4 —
was taken with `reallocate=False`, because `/tmp/melee_distil/score_cd.py` never
offered the flag.

This matters now for a specific reason. The blocker on the goal is not a deficit,
it is a **trade-off**: `refereed` and `vs_shoot` move in opposite directions at
~1.5:1 under both levers tried (pool composition, training length), and the
frontier misses winning both by ~1.7 vp. Pool composition and training length
slide *along* that frontier. A play-time decode is the only lever on file that
could move it *outward*, because it is applied to all four cells identically and
costs no training.

## Design

Paired on everything except the operator. Same checkpoints (arm 4 seeds 1–3,
`last.ckpt` at epoch 1000), same layouts (seeds 700000+), same n=45, same K=3,
same charge decode, **same device**. Only `reallocate` differs.

⚠ **Both arms run on CPU.** The GPU is carrying three live trainers and an OOM
would cost 21 GPU-hours and void the six-seed table. CPU and GPU differ in
floating point, which can flip an argmax tie and therefore the decode, so the
`ra=0` numbers already on disk (GPU) are **not** the control — the control is
re-measured on CPU. Anything else would confound the operator with the device.

## Primary readout

Paired Δvp (`ra=1` − `ra=0`) per cell, reported two ways, because they answer
different questions:
- **within-seed, n=45 layouts** — does the operator help *this* policy? Pairing
  here is near-exact (same weights, same layouts, deterministic decode).
- **across-seed, n=3** — does it help policies of this lineage in general?

## What this measurement CAN and CANNOT settle

⚠ **This is a SCREEN, not a verdict, and the power check says so before the fact.**
§40c's ±4.25 at six seeds implies a per-seed sd of ~10.4, so the across-seed SE at
three seeds is ~6.0. Any threshold I could write below +12 is inside the
estimator's own noise. This project has recorded a per-seed bound tighter than its
own spread as a **defect in the test** once already; I am not repeating it by
pretending three seeds can decide.

So the screen's job is narrow and stated up front: **decide whether `reallocate`
is included in the six-seed scoring that runs when seeds 4–6 land tonight.**

- **INCLUDE** if the within-seed paired Δ is ≥ 0 on `vs_shoot` and `refereed` for
  3 of 3 seeds. (A sign test on near-exactly-paired within-seed data, not a
  magnitude claim.)
- **EXCLUDE** if Δ < 0 on 2+ of 3 seeds on any cell. The operator is then not
  adopted for this lineage — and it is **not** adopted cell-by-cell. Picking the
  cells where a decode happens to help is comparator selection by another name,
  which this project has already paid for twice.
- Either way the **six-seed run scores both arms**, so the verdict is taken at
  six seeds and this screen never becomes the published number.

## The fairness assumption, stated because it is the weakest link

`reallocate` is applied to the agent and not to the bar. That is not withheld
advantage but it does rest on evidence rather than symmetry, and the evidence is
second-hand here:
1. `_resolve_baseline` takes no decode arguments at all, so the operator is
   **structurally** inapplicable to a scripted policy; and
2. §29 measured the redirect on the bar at **exactly zero**, on the grounds that
   the bar already allocates globally by construction.

⚠ If (2) is ever re-measured and is not zero, every number produced under this
pre-registration is void. The same already holds for K=3 and the charge decode,
which the agent gets and the bar does not — the decode family exists to close the
factored-policy gap that a scripted policy does not have.

## Registered risks — written now, so they cannot be discovered later

- **The +8.3 is from a different lineage.** It was measured pre-KL-anchor. The
  anchored policy sits near its warm-start weights by construction and has the
  highest decode headroom ever recorded here (+80.67). That cuts both ways and I
  am not predicting which: more headroom could mean more room for an operator to
  add, or it could mean the joint decode is already extracting what the
  reallocation would have.
- **The operator could HURT.** It overwrites a squad's chosen movement with a
  rigid redirect. On a policy that has been explicitly held near a good
  warm start, overwriting its choices is a plausible loss. A negative result here
  is a real outcome, not a bug to debug.
- **`min_stack` is untuned on this lineage** and left at its default 4. No
  sweep — a sweep on the nine held-out tables would be tuning on the test set.
- **Adopting a fourth decode widens the gap between what trains and what is
  scored**, which is the exact pathology this goal's central finding is about
  (PPO spends the decode's headroom). Adopting it is a *product* decision that
  makes the trained policy's own play worse relative to its scored play.

## Prediction, recorded before the run

I expect the operator transfers with a **smaller** effect than +8.3, because the
anchored policy allocates better than the §40 lineage did, and because §40d
attributes the gain to denial and attrition rather than ground taken — the
anchored lineage's advantage is already denial. I am **not** predicting a sign
for `refereed`, where the opponent is a mirror that reallocates too.

---

# Screen result — three seeds, recorded before seeds 4–6 exist

Written immediately after the three-seed screen finished and **before** the
six-seed numbers existed, so nothing below can be retro-fitted to them. This is
the screen, not the verdict; the pre-registration above forbids it becoming the
published number.

## Validity check, passed

The `ra=0` CPU control reproduces the on-disk GPU table **exactly** on all four
cells — +8.77 / +32.63 / +40.87 / +55.70, every printed digit — so the device is
not confounding the operator, and the CPU arm is comparable to the record.

## The paired numbers

n=45, K=3, charge decode on, arm 4 seeds 1–3 at epoch 1000, CPU both arms.

| cell | ra=0 | ra=1 | Δ | per-seed Δ | signs | bar | gap at ra=1 |
|---|---|---|---|---|---|---|---|
| `refereed` | +8.77 | +9.33 | **+0.57** | +1.50, +0.80, −0.60 | 2/3 | −5.3 | +14.63 (5.05 SE) |
| `vs_take` | +32.63 | +51.53 | **+18.90** | +3.40, +39.10, +14.20 | 3/3 | +20.2 | +31.33 (4.40 SE) |
| `vs_deny` | +40.87 | +51.23 | **+10.37** | +1.50, +4.50, +25.10 | 3/3 | +11.8 | +39.43 (10.60 SE) |
| `vs_shoot` | +55.70 | **+73.97** | **+18.27** | +14.20, +16.70, +23.90 | 3/3 | +56.6 | +17.37 (4.60 SE) |

Coherency falls slightly everywhere (−0.006 to −0.018) — the redirect is rigid
but it is still overwriting a chosen move. At 0.94–0.97 the agent remains far
above the scripts' 0.86–0.91.

## ⚠ MY OWN SCREEN RULE HAS A HOLE, and it is a defect in the rule, not a result

The rule reads: **INCLUDE** if Δ ≥ 0 on `vs_shoot` **and** `refereed` for 3/3;
**EXCLUDE** if Δ < 0 on 2+ of 3 on any cell.

`refereed` came in at **2/3** (one seed at −0.60). So INCLUDE does not fire, and
EXCLUDE does not fire either. **The rule has an unhandled middle and I wrote
it.** Recorded here as a defect in the test, in the same terms this project has
recorded two before it — an underpowered replication criterion, and a per-seed
bound tighter than its own estimator.

What actually resolves the decision is a different clause of the same
pre-registration: *"either way the six-seed run scores both arms."* The design
was robust where the rule was not. The seeds-4–6 pass is armed for both arms.

## What must NOT be concluded yet

All four cells read WON at three seeds. **That is not the goal met.** The goal
is six seeds, and this project has twice read a three-seed screen as a result
and reversed it. The `vs_take` and `vs_deny` columns are visibly heavy-tailed
(per-seed Δ spanning +3.4 to +39.1 and +1.5 to +25.1, sd 18.3 and 12.9); only
`vs_shoot` is tight (sd 5.04).

## One pattern that is data, and one explanation that is NOT

**Data:** the operator is worth +10 to +19 vp against the three non-charging
scripts and **+0.57** against `squad_march_take_charge`. `refereed` is the only
cell whose opponent charges (verified in the four eval configs).

**Hypothesis, untested:** a redirected squad is walking, and a charging opponent
punishes walking. Equally consistent: the charging bar contests more points, so
the "weakest-held objective" the operator targets is not weakly held. Nothing
here distinguishes these, and `stood` moves in both directions across seeds. It
is recorded as an open question, not as a mechanism.
