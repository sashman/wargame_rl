# Enforcement is a referee, not a teacher

**2026-08-16.** Supersedes
[coherency: the verdict](2026-08-15-coherency-the-verdict.md),
[what it costs](2026-08-14-coherency-what-it-costs.md) and
[coherency is a symptom](2026-08-14-coherency-is-a-symptom.md). All three are
kept for the reasoning and the retractions; **their conclusions and every number
in them are void**, for two independent reasons given below. Scenario: the real
tables. Ten trained runs, plus the checkpoints they warm-started from.

---

## The answer

**Train with the reward gate. Never train under enforcement. Switch enforcement
on for play.**

Measured with the referee **off**, so the numbers describe the *policy* rather
than the wrapper, on the rules' own 2"/9" predicate:

| | units coherent | held-out `vp_margin` |
|---|---|---|
| trained under enforcement, no penalty | 0.569 | 70.3 |
| trained under enforcement, `coherency_intervention` -0.1 | 0.783 | 68.0 |
| **`objective_hold.require_coherent` alone, never enforced** | **0.756–0.886** | **81.5** |

Held-out is nine tables the training pool excludes, n=10 each. Gate-only is six
seeds; each enforced arm is two.

`configs/golden/25v25_maps_coherency.yaml` is that configuration.

**Why enforcement cannot teach.** Under `enforce_move` every reverted action
produces the *identical* outcome, so all of them share a return and an
advantage, and the policy gradient inside that whole equivalence class is
exactly zero. Only the entropy bonus acts there, and it pushes toward uniform
over a set covering most of the action space. Enforcement makes the board legal
and leaves the policy no reason to prefer a legal move to an illegal one.

---

## The measurement error that produced three retracted reports

**Every compliance figure was sampled after the enforcement fixed point.** Under
`enforce_move`, `eval/coherency_rate` is 1.000 *by construction* whatever the
policy does. Three reports read that as the agent having learned formation.

The same weights, referee off, intend **0.630 with 5.37 of 25 models adrift**.

`eval/intended_coherency_rate` and `eval/intended_models_out_of_coherency` exist
for exactly this and are logged on every run. The rule that follows:

> **Always measure compliance with the referee off.** A metric sampled after a
> corrective wrapper describes the wrapper. If a mechanism guarantees an
> outcome, no measurement of that outcome can rank the policies under it.

Note the failure mode is not "a wrong number" but "a number that cannot vary".
The original report's own figure said *a coherency metric alone cannot tell a
good policy from a frozen one — both read 1.000*, two sections below where 1.000
was quoted as proof.

---

## The second reason the old numbers are void

A geometry fix ([#193](https://github.com/sashman/wargame_rl/pull/193))
restored the closing edge dropped from padded polygon outlines. It reads as a
tidy-up; on these maps it is a physics change. The tables carry `V_max=5` with
4-vertex pieces, so **every rectangular ruin was traced as an open polyline**
and the containment test over-reported — and that test is what fires the
see-out/see-into exemption, so terrain was far too transparent.

Measured on 8 real layouts with a fixed lattice, `geometry.py` reverted in place
so the environment is controlled:

| | before | after |
|---|---|---|
| grid points reading "inside a piece" | 666 | 516 (−22.5%) |
| LOS pairs visible | 219,379 | **169,789 (−22.6%)** |

Every eval-map baseline and agent score measured before it is void. The runs
in flight when it landed were stopped and restarted rather than reinterpreted.

---

## What each lever is actually worth

| lever | verdict |
|---|---|
| `coherency.enforce_at_deployment` | **Keep, on.** Cheap, and it pays: the bar moved +38.0 → +58.9 because a squad that starts concentrated concentrates its fire. |
| `objective_hold.require_coherent` | **The one that works.** An illegal position earns no objective income — not less, none. Formation 0.51 → 0.76–0.89, free. |
| `coherency.enforce_move` | **A referee. Play only.** Guarantees a legal board; erodes what the gate taught if trained under. |
| `coherency_intervention` | **Refuted and removed.** Raised formation reliably (0.569 → 0.783, both seeds tight) and finished **last** on held-out ground, below even bare enforcement. Tidiness that does not generalise. |
| `unit_coherency` bonus | Saturates. 0.56 → 0.67 and stops; a counterweight, not a fix. |
| `objective_hold.marginal_weight` | Refuted earlier, removed with the rest: it collapsed the term's income 0.176 → 0.014, deleting the term rather than redirecting it. |

The four rows that removed anything are gone from the code as of this date, and
the config surface for this rule went from ten knobs to five.

---

## The open question: the warm start decides the outcome

Gate-only results are **bimodal, and the split tracks the warm-start checkpoint
rather than the training seed.**

| lineage | formation of descendants |
|---|---|
| `s1-maps` | 0.659, 0.665, 0.695, 0.704 |
| `s2-maps` | 0.853, 0.866, 0.879, 0.903 |

Eight runs, no overlap, roughly **3% by chance**. Each individual run is stable
(sd 0.011–0.025 across 100 evaluations), so this is two different learned
solutions, not variance within one.

**And it is not inherited formation.** Both warm starts measure almost identically
on the predicate themselves — 0.524 and 0.504. Something else in those weights
decides which solution training settles into, under an identical reward.

The coherent lineage also **wins more**: +11 vp held-out (86.1 v 74.8) and +8.7
on the training pool. So formation and score go together. An earlier claim here
that they were independent came from the 30-episode in-run evaluation, where all
four runs read 89–91; at n=100 they separate cleanly. That is the in-run eval
being too noisy to compare arms, not a real null.

This is the most interesting unexplained result on the board, and it likely
generalises past coherency: if initialisation silently selects among equal-value
solutions, it affects every experiment run here.

---

## Reusable lessons

- **A metric sampled after a corrective wrapper measures the wrapper.** Ask
  whether the number *can* vary with the thing being studied.
- **Score on held-out layouts.** The training-pool ranking put the intervention
  penalty comfortably second; held-out put it last. The ordering flipped.
- **Two seeds is the floor, and sometimes not enough.** "Never enforce" was
  stated on one seed and dissolved on its second (0.853 → 0.659).
- **A geometry change can be a physics change.** Digest the environment before
  and after anything touching `domain/` or `types/geometry.py`; it costs
  minutes and it invalidated a day of runs here.
- **A device change is a numerics change.** `_cuda_is_usable` rejected a working
  4090 by exact-string match on compute capability, so every measurement script
  ran on CPU (36 s → over 10 min) while Lightning trained on the GPU. Fixed, and
  verified by comparing greedy decisions rather than assuming: 1000 of 1000
  identical.
- **Separate the rule from the training recipe.** "The game enforces coherency"
  and "we train with enforcement on" are different questions. Conflating them is
  what turned one rule into ten config knobs.
