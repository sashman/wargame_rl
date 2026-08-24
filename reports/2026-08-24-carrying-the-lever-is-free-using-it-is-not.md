# The advance lever: carrying it is free, using it is not

**2026-08-24.** Three seeds, 300 epochs, **paired**. `configs/experiments/25v25_maps_advance.yaml`
against `..._advance_dark.yaml` — identical but for `dark_action_slices`, so both are
152 actions with the same head shape and a bit-identical initialisation. Scored
refereed at K=3 on the held-out nine, n=30, seeds 700000+.

⚠ **The pre-registered verdict is FAIL, and the criterion was broken.** Both halves
of that sentence matter and neither cancels the other.

---

## 1. The question, and why it is not "does advance pay"

Advance is a **lever**, not an advantage: further, but the unit does not shoot. Its
use is niche and using it for its own sake is detrimental — three scripted rules
priced that at −78, −18.4 and −11.9 vp *to their own users*
([report](2026-08-23-three-prices-for-the-advance-move.md)). So the question is not
whether the lever pays. It is **whether carrying it costs the agent anything**, with
its use left to learned behaviour.

⚠ **No melee exists**, so a shooting army has no reason to close except to stand on
an objective. Everything here is provisional on a mechanic the environment does not
have.

## 2. The result

| seed | advance | dark control | paired | usage | `held` a/d | `alive` a/d | `coherent` a/d |
|---|---|---|---|---|---|---|---|
| s1 | +32.3 | +19.8 | **+12.5** | **0.0%** | 2.19/2.03 | 0.571/0.535 | 0.957/0.957 |
| s2 | −18.2 | −8.4 | **−9.8** | **10.9%** | 1.69/1.60 | 0.420/0.433 | 0.938/0.956 |
| s3 | +19.9 | +15.9 | **+4.0** | **0.6%** | 1.97/2.00 | 0.479/0.467 | 0.953/0.953 |

**Paired difference +2.2 ± 6.5, t = +0.34.** Signs flip. The old encoding cost
**−26.7** and never flipped.

**2 of 3 seeds cleared the −8 bound**, so by the rule written before the numbers
existed this is a **FAIL**.

## 3. ⚠ The accept criterion could not have passed reliably

The per-seed paired sd is **11.3**. If the lever cost *exactly zero*:

- P(one seed lands below −8) = **23.9%**
- P(all three clear it) = **44%**

**A genuinely free lever fails this criterion 56% of the time** — worse than a coin
flip. The bound was set tighter than the estimator's own noise.

This is a defect in the pre-registration, not evidence about the arm, and it is
recorded here as a criticism of the rule rather than as grounds to overturn the
verdict. The verdict stands. ⚠ **Power-check a per-seed bound against the spread you
expect before writing it down** — this project already had a rule for writing the
*right* reject criterion; this is the same lesson one level down, in the statistics.

## 4. What IS established: the cost is in USING the lever

Usage and score order perfectly across the three seeds — usage 0.0 / 0.6 / 10.9%
against differences +12.5 / +4.0 / −9.8. That ordering was **not** pre-registered,
so it was tested rather than told:

**Forbid advance at PLAY, same weights.** The dark config shares the 152-action
shape, so an advance-trained checkpoint can be scored with the lever masked.
Prediction written before running: *s2 recovers most of its −9.8; s1 and s3 barely
move.*

| seed | advance allowed | advance forbidden at play | gain | usage |
|---|---|---|---|---|
| s1 | +32.3 | +31.2 | **−1.1** | 0.0% |
| s2 | −18.2 | **+8.1** | **+26.3** | 10.9% |
| s3 | +19.9 | +22.6 | **+2.7** | 0.6% |

**Confirmed 3 of 3.** The seed that used the lever recovers **26.3 vp** when denied
it; the seeds that had already learned to decline move by ~1–3 vp, i.e. noise.

So: **carrying the lever is free; using it is what costs.** The encoding is not the
problem. What varies is whether a given seed has learned to leave it alone.

## 5. Two of three seeds learned to decline it entirely

s1 chose **0 advances in 7,227 unit-turns** — 2 forfeited shooting slots out of
28,800. s3, 0.6%. The old encoding used it on 9.4–15.4% of model-steps.

⚠ **Checked before celebrating**: the declaration is *reachable*. All 25 alive models
are offered the advance action in every command phase, and the scripted policies do
declare through the same slice. The 0.0% is learned refusal, not a mask.

## 6. ⚠ The result mixes converged and unconverged runs

s2's advance usage across its last 50 epochs: **7.9% → 4.8% → 7.8%**. Oscillating,
not decaying — so "train longer and it settles" is the indicated test, not a
finding. What it does show is that s2 had **not converged** at 300 epochs while s1
and s3 sat at their floor.

Nearly all the noise in +2.2 ± 6.5 is that one seed. Drop it and the other two agree
at +12.5 and +4.0.

**Proposed, and new:** **lever usage is a convergence signal.** When the right answer
is "rarely", a lever whose usage is still oscillating means the run has not settled,
whatever the reward curve says. It costs one inference run at two checkpoints, and it
would have flagged s2 as not-comparable before it entered the average. This
generalises to every move type the rules add.

## 7. ⚠ A silent tooling failure, found on the way

**`--resume-ckpt-path` was broken for every checkpoint this repo has ever written.**
PyTorch 2.6 flipped `torch.load`'s `weights_only` default to True; these checkpoints
pickle the whole `WargameEnv` as a Lightning hparam, so Lightning's internal restore
raised `UnpicklingError: Unsupported global ... WargameEnv`.

Every run died in ~6 seconds **and the launcher still exited 0**, printing "seed 1
done / seed 2 done / seed 3 done" — the same silent-success shape already on file for
`train-arm`. It was caught only by checking process count and GPU memory rather than
the exit code.

Fixed by scoping `weights_only=False` to the resume call and restoring it afterwards,
consistent with `TransformerNetwork.from_checkpoint`, which has always read these
files that way. Allowlisting globals was rejected: the list would have to name every
config and geometry type a checkpoint happens to contain, and would break on the next
one. `tests/test_train_resume.py` pins both the flag and the restoration.

## 8. Open

Whether 1000 epochs closes the seed split — running. It answers the epoch-budget
question and sharpens the equivalence test at the same time. ⚠ **A properly powered
bound must be re-derived from the 1000-epoch spread before the result is read**; more
epochs cuts noise but does not repair a criterion set tighter than the spread.

---

# Revised pre-registration for the 1000-epoch extension

⚠ **Written 2026-08-24, while the runs are at epoch ~810 and BEFORE any 1000-epoch
score exists.** The 300-epoch criterion ("paired difference >= -8 on 3 of 3 seeds")
was measured to be unpassable: with a per-seed paired sd of 11.3, a lever costing
exactly zero fails it 56% of the time. Replacing a broken rule is legitimate;
replacing it *after seeing the new numbers* would not be, hence the timing.

## The criterion

This is an **equivalence** test — the claim is "carrying the lever costs nothing",
so the burden is to **rule out a cost**, not to detect a difference.

Compute the mean paired difference and its one-sided 95% lower bound
(`mean - t(0.95, n-1) * SE`). Then:

- **PASS** — the lower bound excludes a cost of **10 vp**, i.e. `mean - t*SE > -10`.
  Carrying the lever demonstrably costs less than 10 vp.
- **FAIL** — the *upper* side shows a real cost: `mean + t*SE < 0`, i.e. the
  difference is significantly negative.
- **UNDERPOWERED** — neither holds. Report it as that, and NOT as a pass. A null
  result that cannot exclude the effect is not evidence of absence.

## ⚠ n=3 is very probably UNDERPOWERED, and that is known in advance

At the observed sd of 11.3, ruling out a 10 vp cost needs:

| seeds | SE | t*SE | can a mean of 0 exclude a 10 vp cost? |
|---|---|---|---|
| 3 | 6.52 | 19.05 | **no** |
| 5 | 5.05 | 10.76 | no |
| **6** | 4.61 | 9.32 | **yes** |
| 8 | 4.00 | 7.55 | yes (comfortably) |

So unless 1000 epochs shrinks the per-seed spread substantially, the honest verdict
will be **UNDERPOWERED**, and the remedy is **six seeds, not more epochs**. Recorded
now so that a mean near zero at n=3 is not read as a pass.

## What the extension CAN settle regardless of power

- **Does s2's advance usage converge?** It was 7.9/4.8/7.8% over epochs 247-299 --
  oscillating, not decaying, while s1 sat at 0.0% and s3 at 0.6%. If s2 falls to the
  others' floor by epoch 1000, the 300-epoch split was under-training. If it does
  not, the split is a real multi-modality in what these runs learn, which is a more
  interesting and more awkward result.
- **Does the epoch budget matter for a larger action space?** Compare each arm's own
  300 vs 1000 score. That is a within-seed comparison and needs no equivalence bound.

---

# ⚠ RETRACTION — the 1000-epoch result reverses this report's headline

**2026-08-24, later the same day.** All six runs resumed to epoch 1000 and rescored,
refereed at K=3 on the held-out nine, n=30. **The title of this report does not
survive.**

| seed | advance | dark control | paired @1000 | paired @300 | usage @1000 | usage @300 |
|---|---|---|---|---|---|---|
| s1 | +26.7 | +33.6 | **−6.9** | +12.5 | 0.3% | 0.0% |
| s2 | +9.9 | +17.9 | **−8.0** | −9.8 | 4.9% | 10.9% |
| s3 | +1.6 | **+35.6** | **−34.0** | +4.0 | 5.1% | 0.6% |

**Mean −16.3 ± 8.9, t = −1.84, all three seeds negative.** Against +2.2 ± 6.5 with
flipping signs at 300 epochs.

**Verdict against the criterion committed to git before these scores existed:
UNDERPOWERED.** The one-sided 95% lower bound is −42.2, so it can neither exclude a
10 vp cost nor demonstrate one. It is **not** reported as a pass.

## What is retracted

- ⚠ **"Carrying the lever is free."** At 300 epochs the paired difference was
  +2.2 ± 6.5; at 1000 it is −16.3 ± 8.9. **s1 and s3 both flipped sign.** The
  300-epoch reading was not a noisier version of this one — it pointed the other way.
- ⚠ **"Two of three seeds learned to decline it entirely."** True at 300 (0.0%,
  0.6%). At 1000 only s1 is still near zero (0.3%); s2 and s3 both sit near 5%.
  **More training made them worse at leaving the option alone, not better.**
- ⚠ **The prediction that s2 would fall to the others' floor by 1000 FAILED.** Its
  usage went 7.8 → 6.6 → 3.2 → 4.2 → 4.9% — it halved, then plateaued and drifted
  back up. Not under-training: a second mode that 700 extra epochs did not unlearn.

## What survives

- **The usage/score relationship, and it is now stronger.** The seed ending at 0.3%
  usage lost least (−6.9); the two ending near 5% lost most (−8.0, −34.0). This is
  the same direction as the forbid-at-play falsifier, which remains the cleanest
  result here: same weights, advance masked at play, **+26.3 recovered** on the seed
  that used it and −1.1/+2.7 on the seeds that did not.
- **The four structural criteria.** Nothing above bears on dominated actions,
  stationary semantics, the unit declaration or additive cost; those are verified in
  code and unaffected.
- ⚠ **The accept criterion at 300 epochs was unpassable** (a free lever fails
  "−8 on 3/3" 56% of the time). That remains true and is why the equivalence rule
  replaced it.

## The hypothesis this raises, NOT a finding

**The control gained more from the extra 700 epochs than the arm did, on every
seed** — s1 +13.8 v −5.6, s2 +26.3 v +28.1, s3 **+19.7 v −18.3**. That is consistent
with the extra option *slowing or destabilising* learning rather than costing at
convergence, which is a different mechanism from anything measured here. Three seeds
with sd 15.3 cannot establish it.

## ⚠ What this costs the method, not just the claim

**A three-seed screen was read as a result twice, and reversed both times.** The
per-seed paired difference is not stable across seeds *or* across epoch budgets. The
power table in the appendix said six seeds; it was written about the equivalence
bound but applies to every number in this report.

**Nothing here should move a design decision.** The honest state is: the encoding is
structurally correct, the lever's cost is unresolved between "free" and "−16", and
resolving it needs six seeds at 1000 epochs — not another three.

---

# VERDICT: REJECT — and the reject clause's own explanation is refuted

The goal's decision rule was:

> **Accept**: the re-encoded arm beats the current advance arm, and reaches the
> non-advance control's +23.4 or better, three seeds, refereed, K=3.
> **Reject**: it does not clear the control — in which case ~12 vp is the permanent
> cost of a larger action space and Advance needs a different design, not a better
> encoding.

At 1000 epochs, refereed, K=3, held-out nine, n=30:

| | s1 | s2 | s3 | mean |
|---|---|---|---|---|
| arm | +26.7 | +9.9 | +1.6 | **+12.7** |
| control | +33.6 | +17.9 | +35.6 | **+29.0** |

It **beats the old advance arm** (−3.3) but **does not clear the control**, on any
seed. **The gate returns REJECT.**

## ⚠ But "a permanent cost of a larger action space" is REFUTED by the same run

The reject clause carried an explanation. It is wrong, and the falsifier that shows
it was pre-predicted and confirmed 3/3 — advance-trained weights, lever masked at
**play**:

| quantity | mean | SE | 95% lower bound |
|---|---|---|---|
| as-scored (the gate) | −16.3 | 8.86 | −42.2 |
| **carrying the option** | **−2.9** | **0.67** | **−4.8** |
| using it at play | −13.4 | 9.43 | −40.9 |

Weights trained *with* the lever score within **1.8–4.1 vp** of the control once the
lever is masked (s1 +29.5 v +33.6, s2 +15.1 v +17.9, s3 **+33.8 v +35.6**). s3 alone
recovers **+32.2**.

So the larger action space does **not** impose ~12 vp. It imposes **−2.9 ± 0.67**,
and that estimate is tight where every other number here is not. **Advance does not
need a different design.** What costs is a policy *exercising* a move that does not
pay — 0.3% usage → −2.8; 4.9% → −5.2; 5.1% → **−32.2**.

## What the verdict does and does not license

- **REJECT stands** on the criterion as written. The arm does not clear the control.
- ⚠ **Do NOT carry forward "the encoding costs ~12 vp".** It is measured at −2.9 with
  a lower bound of −4.8, by a pre-registered falsifier.
- ⚠ **Do NOT re-open the encoding.** All four structural criteria are met and the
  decomposition says the structure is not what is losing.
- **The open problem is a POLICY problem**: two of three seeds drifted into using a
  move that costs them, and 700 extra epochs made that worse rather than better
  (usage 0.0→0.3%, 10.9→4.9%, 0.6→5.1%). More seeds would measure that; they would
  not explain it.
