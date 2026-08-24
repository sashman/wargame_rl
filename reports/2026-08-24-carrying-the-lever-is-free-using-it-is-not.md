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
