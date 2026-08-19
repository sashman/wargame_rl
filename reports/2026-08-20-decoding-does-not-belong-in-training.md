# Joint constrained decoding does not belong in training

**2026-08-20.** Joint constrained decoding is worth **+40.5 vp on 45 of 45
tables** at play time, for no weight change. The obvious next step was to fold
it into training, so the policy would stop spending probability mass on
combinations that never execute.

It fails. From scratch it costs **−51.8 vp**; warm-started onto a trained policy
it costs **−43.7 vp** and destroys its own initialisation. Annealing the
candidate width — the obvious fix — is impossible by arithmetic.

**The implementation was correct. The idea is wrong.** That distinction is the
point of this report, because a future reader who assumes the code was buggy
will try it again.

---

## 1. What was built

`pi_unit(a | s) = softmax over LEGAL combos of sum_i log pi_i(a_i | s)`, at
K=3 on a five-model unit — ≤243 terms, so the joint log-probability is exactly
computable.

The constraint is applied to the distribution **before** sampling, which is the
argument that makes action masking valid: the sampled action is the executed
action, so the stored log-probability is the density of what actually happened.
Applying the decoder as a *filter* after sampling would break exactly that, and
would look like it was working.

The implementation lives in the history of
`feature/joint-policy-distribution`. `git log` restores it.

## 2. It was correct, and that was verified before any GPU time

| check | result |
|---|---|
| `exp(new − old)` at epoch start | **exactly 1** |
| the gate, with a wrong stored density injected | **ratio 78,143** — caught |
| unit-level clip, at the recorded 0.0073 nats/model | `exp(5 × 0.0073)` = **1.037**, inside `eps_clip` |
| the same figure at army level | `exp(25 × 0.0073)` = **1.2002** — reproduces the recorded failure |
| flag off | exact no-op, pinned |

Three defects were found by *sensitivity-checking the tests rather than
running them*:

- Padding a member's candidate list to a rectangle repeats an action, so the
  repeated **combination** would carry double probability mass.
- The padding test was **insensitive** — on the state it drew, only 3 of 27
  combinations were legal and all shared the same first-member action, so even a
  varying pad cancelled. Rewritten synthetically.
- The first tests passed **vacuously**: randomly placed models start incoherent,
  no combination is legal, the sampler correctly declines, and every assertion
  holds while proving nothing.

## 3. From scratch: −51.8 vp

`25v25_maps_two_mode`, 300 epochs, seed 1. Paired — `seed_everything` precedes
model construction and the flag changes no parameter shape, so both arms start
from identical weights. Scored on nine held-out tables, n=30, verified top-3
decode at play for **both** arms.

| arm | vp margin | coherency | held | on_obj |
|---|---|---|---|---|
| control (`topk 0`) | **+33.8** | 0.951 | 2.29 | 0.627 |
| joint (`topk 3`) | **−18.0** | 0.929 | 1.40 | 0.404 |

Twice the 26 vp seed spread — and it bought **less** coherency, not more.
Training reward 21.3 against 9.1, flat from epoch ~150 while the control was
still climbing at 251.

### The first explanation, and why it is not enough

Top-K approximates "what the policy wants" only once the policy wants
something. Measured on `table_30`, 102 actions per model:

| policy | top-3 mass | entropy |
|---|---|---|
| untrained (epoch 0) | **0.071** | 4.509 nats |
| trained control (300 epochs) | **0.895** | 1.049 nats |

From scratch the candidate set is ~7% of the policy's own mass, chosen
arbitrarily, and it is self-trapping: the policy can only sharpen inside the
slice it was restricted to. That predicts the warm start would fix it.

It does not.

## 4. Warm-started onto a trained policy: −43.7 vp

Both arms warm-started from the same +33.8 checkpoint, where top-3 carries
**0.895** of the mass. Same seed, 150 further epochs.

| arm | vp margin | coherency | held | on_obj | v its own start |
|---|---|---|---|---|---|
| control (`topk 0`) | **+39.6** | 0.960 | 2.59 | 0.749 | **+5.8** |
| joint (`topk 3`) | **−4.1** | 0.933 | 1.91 | 0.493 | **−37.9** |

The joint arm **destroyed its own initialisation** by 37.9 vp. Its best
checkpoint was epoch 0 — nothing in 150 epochs beat the first one.

**The fresh-optimiser confound is ruled out by the control.** Both arms got the
same weights-only warm start and the same cold optimiser, and the control
*improved* by 5.8. Only the joint arm collapsed.

So top-K starvation is **not** the main cause: at 89.5% coverage the failure is
nearly as large as at 7%.

## 5. Annealing K is impossible — refuted by arithmetic, no GPU spent

The candidate count is `K**k` against a cap of 4096:

| K | combos (K⁵) | feasible | untrained top-K mass |
|---|---|---|---|
| 3 | 243 | yes | 0.055 |
| **5** | **3,125** | **the maximum** | **0.089** |
| 8 | 32,768 | no | 0.137 |
| 20 | 3,200,000 | no | 0.308 |

**K ≤ 5** on a five-model unit, where an untrained policy still has under 9% of
its mass inside the candidate set. Reaching ~30% needs K=20 — 780× over the cap,
and 3.2M coherency checks per unit per step.

The same `K^k` arithmetic that makes the decode cheap at play time makes a wide
candidate set impossible during training. **Do not re-run this.**

## 6. What actually explains it

What fits both arms is the **speed tax**, already on record here: rigid legal
formation moves travel slowly, independent sampling is 1–9% legal at full speed,
and constraining every sampled move to a legal combination forces slow, clumped
play. That play earns less, and the policy learns to play that way.

This is the same destination as
[enforcement is a referee](2026-08-16-enforcement-is-a-referee.md), reached by a
different route: **the constraint substitutes for the skill.** Enforcement hands
the policy a legal board it never had to produce; constrained sampling hands it a
legal *action set* it never had to find. Both train the policy out of the
competence they were meant to install.

## 7. What this settles

- **Decode at play, never in training** — now true of the decoder as well as the
  referee, and for the same underlying reason.
- **The post-hoc decode is the mechanism, not a stepping stone.** +40.5 vp on 45
  of 45 tables is what this line of work yields.
- Before applying a top-K method to an untrained network, **measure its top-K
  mass**. That desk check costs seconds and would have predicted §3.

## 8. What was shipped

**Nothing.** The flag, the sampler and the training-loop wiring were all reverted
rather than merged: a refuted path left behind as config surface is a standing
invitation to re-run it, and this project has cut such surface before. The
implementation remains in the branch history for anyone who wants to reproduce
these numbers.
