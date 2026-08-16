# Enforcement is a referee, not a teacher

## In short, without the jargon

There is a rule in the game that soldiers in a squad have to stay near each
other. We wanted the AI to actually play that way. There were two ways to get
it: **pay the AI** for keeping its squads together, or **have a referee** undo
any move that breaks a squad apart.

**Paying works. The referee does not teach — and using it while the AI is
learning makes it worse.** An AI trained with the referee switched on gets
lazy: it stops trying to stay together, because the referee tidies up after it
anyway. Turn the referee off afterwards and it is *sloppier* than an AI that
was never forced at all. It also loses more games on maps it has not seen
before.

So: **pay during training, and switch the referee on only for real games.**
The referee still earns its place — it is the only thing that guarantees the
army is actually legal — but it is a rule of the game, not a teaching aid.

**We had also been measuring it wrong, and that is the more useful lesson.**
We were checking whether the squads were together *after* the referee had
already fixed them. Of course they were — that was the referee's doing, not the
AI's. It looked like a perfect score. Measured properly, the AI was much
sloppier than we had been reporting, and three earlier write-ups had to be
withdrawn. The general form of the mistake: **if something guarantees an
outcome, measuring that outcome tells you nothing about what caused it.**

**One more thing, which turned out to matter.** Training doesn't start from
scratch — it continues from an AI that already knows how to play. We used two
such starting points. One was very slightly better at keeping squads together;
the gap was small enough that we first measured it as nothing at all. After
training, that small head start had grown roughly threefold, and every AI grown
from the better starting point ended up clearly tidier — sixteen runs, no
overlap, and swapping which one each run started from swapped the result every
time.

So the lesson is about how we test things, not just about this rule: **a small
difference in where you start gets multiplied by training.** Two test runs
begun from the same place will agree closely with each other and still be
telling you about their starting point rather than about the change you were
testing.

---

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

**At play, use `revert_unit` — the spec's own mode.** Nine held-out tables, four
checkpoints, two per warm-start lineage:

| referee | held-out `vp_margin` |
|---|---|
| off (illegal boards) | 87.2 |
| `revert_model` (a divergence from the rule) | 80.4 |
| **`revert_unit` (the rule as written)** | **79.4** |

The spec mode and the divergence **tie** — 1 vp apart, and the sign flips across
checkpoints — so the divergence buys nothing and rules accuracy is free. The
referee costs ~7 vp, and a policy with better formation pays much less of it
(86.0/84.6 against 75.8/70.4), because it gives the referee less to correct.
Every prior comparison of these modes predates #193 and is void.

**Why enforcement cannot teach.** Under `enforce_move` every reverted action
produces the *identical* outcome, so all of them share a return and an
advantage, and the policy gradient inside that whole equivalence class is
exactly zero. Only the entropy bonus acts there, and it pushes toward uniform
over a set covering most of the action space. Enforcement makes the board legal
and leaves the policy no reason to prefer a legal move to an illegal one.

---

## Against the floor and the bar

Every figure above ranks arms against each other. Against the scripted ladder,
on the same nine held-out tables, n=10 each:

| policy | referee off | `revert_unit` | the rule costs |
|---|---|---|---|
| `hold_deployment` (floor) | — | −55.7 | — |
| `random` | — | −50.6 | — |
| `squad_march` | 82.2 | 81.1 | −1.1 |
| **`squad_march_shoot` (the bar)** | **114.8** | **101.4** | **−13.4** |
| agent, best two seeds | 89.2 | 84.4 | −4.8 |
| agent, all four | 87.2 | 79.4 | −7.8 |

**The agent does not beat the bar here — and coherency is not the reason.** The
gap is **25.6** vp with the referee off and **17.0** with it on: obeying the rule
*narrows* it. The bar pays **−13.4** to the referee against the agent's −4.8,
because a shooting policy must break formation to get firing angles while the
agent already holds it. `squad_march` pays just −1.1, since marching keeps
squads together by construction.

So the deficit is **generalisation**, and it predates every coherency change:
the same checkpoints score **95.0** on the 36 training tables against 79–89 held
out, and finish with **~94% of their force alive against the bar's 78–83%**
while holding **3.3 objectives against 4.1**. The agent preserves its army and
under-contests ground.

This table exists because it nearly did not. The whole investigation ranked arms
against each other for a day without a floor or a ceiling on current physics —
which is how a policy scoring 17% against an 80% heuristic once read as progress
in this project. Quote the agent against these rows, never on its own.

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

## The warm start decides the outcome, by amplifying a small head start

Gate-only results are **bimodal, and the split tracks the warm-start checkpoint
rather than the training seed.**

**Confirmed by crossover.** The first eight runs had seed and lineage
confounded — seeds 1 and 4 both used warm start A, seeds 2 and 3 both used B. A
full crossover, every seed handed the *other* lineage, flipped every one into
its new lineage's band:

| seed | formation with A | formation with B |
|---|---|---|
| 1 | 0.704 | **0.885** |
| 4 | 0.703 | **0.882** |
| 2 | **0.885** | 0.720 |
| 3 | **0.891** | 0.703 |

Sixteen runs, no overlap between the bands, and the seed explains none of it.
Held-out `vp_margin` follows the same split — B lineage 86.0–88.7 against A's
73.7–83.8, eight runs, again no overlap.

**And the mechanism is amplification, not mystery.** Measured through the same
tracker on five independent seed sets, n=30 each, B starts ahead on **all five**:

| seeds | A | B | B − A |
|---|---|---|---|
| 700000 | 0.505 | 0.524 | +0.019 |
| 500000 | 0.521 | 0.651 | +0.130 |
| 300000 | 0.473 | 0.554 | +0.081 |
| 900000 | 0.526 | 0.609 | +0.083 |
| 100000 | 0.535 | 0.559 | +0.024 |

Mean **+0.067** at the start, **+0.19** after 300 epochs. The gate roughly
**triples** a small initial advantage: it pays for legal positioning, so a
policy already marginally better at it collects more and reinforces, while one
slightly behind never gets going.

**Two earlier readings of this were wrong, and both from the same cause.** That
B "started marginally behind, so this is not inherited formation" was an n=20
measurement on a single seed set. That "two-thirds of the gap exists at epoch 0"
used the in-run metric on seeds 500000+, the one set where B's head start is
largest (+0.130 against a +0.067 mean) — a starting advantage read as an instant
training effect. **The same checkpoint reads 0.505 to 0.651 depending on which
layouts you draw**, so no claim about a warm start means anything without its
seed set attached.

**The consequence reaches past coherency.** If a modest initialisation
difference is amplified threefold by a reward term, then **an arm comparison
whose seeds share one warm start is partly measuring that warm start** — and
its seeds will agree tightly with each other while doing so, which is exactly
what a real effect looks like. Vary the warm start across seeds, and record
which checkpoint each run descended from.

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
