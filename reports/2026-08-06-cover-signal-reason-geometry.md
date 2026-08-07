# 2026-08-06 — Cover with all three blockers removed: signal, reason, geometry

## TL;DR (ELI5)

We wanted the little soldiers to hide behind walls. Three times now they have refused.

Last time we could not blame them. There was almost nothing to hide behind, they could not
see who was aiming at them, and dying cost them nothing. So this time we gave them all
three, and those three are what the arm names mean:

- **geometry** — lots more walls, so hiding is actually *possible*
- **signal** — a sense that says "three enemies can shoot you right now", so being in danger
  is *noticeable*
- **reason** — a penalty for losing soldiers, so staying safe is *worth something*

Everyone got the walls. Then we ran four versions: one with neither of the other two
(**control**), one with only the sense (**signal**), one with only the penalty
(**reason**), and one with both (**full**). Comparing them tells you which of the two
actually mattered.

**They still did not hide.** But one of the three changes made them win a lot more anyway —
the penalty for losing soldiers. And here is the strange part: after we started punishing
them for losing soldiers, **they lost more soldiers, not fewer.** They also stepped into
the open more. They just won harder while doing it.

The "see who is aiming at you" sense did nothing at all. Zero. Adding it on top of the
penalty made things very slightly worse.

So: one of our two ideas worked, the other was useless, and the one that worked did the
*opposite* of what we thought it would do. We do not yet know why, and we are not going to
pretend we do.

---

**Question.** Batch 1/2 established that the agent ignores terrain and manages range
instead ([report](2026-08-05-stochastic-terrain-and-cover.md)). That result was measured in
a world where cover could not work, could not be seen, and was not worth anything. Remove
all three blockers and re-ask: does the policy use cover?

**Answer: still no** — and this time the null is clean, because the alternative was
genuinely available. Exposure sits at 0.092–0.110 across all four arms on terrain with
19.8% of the board hidden, with a per-model line-of-sight input in the observation and
model losses priced against model kills.

Two side findings, both stronger than the headline:

- **Pricing losses works.** `models_lost` is worth roughly **+7 VP margin**, separating
  with no seed overlap. It is the only lever in the 2x2 that does anything.
- **The line-of-sight observation is null.** Not weak — null, on both seeds, on every
  metric. Adding it on top of the loss penalty makes the arm marginally worse.

> **CORRECTIONS (2026-08-06, same day).** Both bullets above overstate what was measured,
> and one headline claim further down is simply wrong. Read the inline corrections before
> using any number from this report:
>
> 1. **"All four arms clear the bar" is false.** Agent and baseline were scored on
>    different layout sets. On matched seeds the bar is 0.77 win / +39.4 VP and the best
>    arm is 0.67 / +16.3 — the agent is *below* the scripted heuristic, not above it.
> 2. **The `models_lost` +7 does not survive.** It is window-dependent and reverses on
>    held-out layouts. Its sign is unestablished.
> 3. **"Null" overstates the signal result.** The data bound the effect at roughly
>    |Δ| ≲ 5pp win / ≲ 2 VP with a ±6.5pp paired swing. That is "not detected", not "null".
>    Deleting the feature was a defensible engineering call presented as a finding.
>
> The follow-up that found all three, and what actually closes the gap, is
> [2026-08-06-beat-the-shooting-opponent.md](2026-08-06-beat-the-shooting-opponent.md).

## What had to be built first

The prior batch could not answer the question, for three measurable reasons. PR #136
removed each:

1. **The geometry could not cast a shadow.** Exposure is "at least one enemy sees me", so
   hiding means breaking *every* sightline at once. The batch-1/2 profile (7 near-square
   ruins, 9.6% coverage) leaves only **5.8%** of the board hidden from a squad in weapon
   range. New `just measure-terrain` quantifies exactly this. Tuning there — in seconds —
   found that **piece count dominates piece size**: 9 pieces of 3-13 scored *worse* than
   the old profile despite double the coverage, while **29 pieces of 3-7 reaches 19.8%**.
2. **The decision could not be seen.** The shooting mask is built only during the shooting
   phase and only masks logits; at *movement* time, when the choice is made, the policy had
   no LOS information at all. `observe_threat_count` added a per-model scalar — alive
   enemies with LOS and range to this model — computed inside `build_observation`, the only
   site that is decision-time by construction. (Removed after this batch; see below.)
3. **The trade had no cost side.** `model_kills` paid for kills; nothing charged for
   losses. New global `models_lost` calculator. It must be global: `phase_manager` iterates
   *alive* models, so with `max_wounds: 1` a per-model damage penalty is identically zero.

Calibration was verified empirically before training, not just arithmetically: on the new
config, `model_kills` contributes **+1.44/episode** against `models_lost` **−1.60/episode**,
a ratio of 1.11 — close enough that a 1:1 trade is near neutral and only favourable trades
pay.

**Correction to the prior report.** Arm F (weapon range 24) was described there as
corroboration that the agent ignores cover. At 5.8% hidden, cover was not an available
alternative in arm F either, so it shows only that range was the agent's single lever — not
that it ignored a working one. The prior report has been annotated.

## Setup

PPO + transformer, 1000 epochs, **two seeds per arm**. 25v25 on 60x44, 3 objectives with
`objective_min_separation: 6`, weapon range 12, opponent `scripted_advance_and_shoot`,
29 mirrored ruins of size 3-7 regenerated every episode, `track_exposure: true`.

### What the arm names mean

For an agent to take cover, three separate things all have to be true. It has to be
**possible**, it has to be **perceivable**, and it has to be **worth doing**. Batch 1/2
had none of them, which is why its null result could not be trusted. The arm names are
those three conditions:

| name | the condition | the question it answers | how it was supplied |
|---|---|---|---|
| **geometry** | Cover is *possible* | Is there anything to hide behind that actually breaks every sightline? | 29 ruins of 3-7, 19.8% of the board hidden (was 5.8%) |
| **signal** | Cover is *perceivable* | Can the policy tell, at the moment it chooses a move, that it is exposed? | `observe_threat_count` — a per-model count of enemies with LOS and range to it |
| **reason** | Cover is *worth doing* | Does anything bad happen if it walks into the open and dies? | `models_lost` — a reward penalty per model lost, priced against `model_kills` |

**Geometry is not an axis — it is a precondition.** Every arm gets the new terrain, because
an arm without it tests nothing: it cannot distinguish "declined to use cover" from "had no
cover available", which is exactly the confound that invalidated batch 1/2 and arm F.

Signal and reason *are* axes, crossed into a 2x2. The design question was which of the two
is the binding constraint — and, because a 2x2 gives the interaction for free, whether
either only works in the presence of the other. My pre-registered prior was that the signal
is necessary and the reason alone does nothing, on the reasoning that no reward can fix a
missing input.

| arm | config | signal | reason | reads as | run s1 | run s2 |
|---|---|---|---|---|---|---|
| H | `25v25_cover_control.yaml` | off | off | can see nothing, cares about nothing | `kra8wkr0` | `j10eupjr` |
| I | `25v25_cover_signal.yaml` | **on** | off | knows it is exposed, has no reason to mind | `0ezq9h2a` | `35uq0e25` |
| J | `25v25_cover_reason.yaml` | off | **on** | pays for dying, cannot see it coming | `ybztgjym` | `u1pf5n7y` |
| K | `25v25_cover_full.yaml` | **on** | **on** | "full" — both, the arm cover should be easiest for | `nhbk7wl5` | `25b0bmkr` |

Read as differences: **I − H** isolates the signal, **J − H** isolates the reason, and
**K − (I + J − H)** is the interaction. Batch 3 carries its own control (H) because the
terrain profile changed, so **no batch-3 number is comparable to batch 1 or 2**.

`25v25_cover_signal.yaml` and `25v25_cover_full.yaml` no longer exist — they were deleted
with the feature after it measured null. The rows are kept here because they are what was
run.

Wandb group `train-multi-2026-08-05-12-06-29`, which holds the eight runs above and nothing
else. It also held eight crashed runs — a seed-2 attempt killed at epoch ~840 by an
unrelated process teardown, and a redundant relaunch stopped at epoch ~90 — neither of which
contributed to any number here; both have since been deleted.

**Two seeds were mandatory, not cautious.** `just measure-noise-floor` (new) holds layouts
fixed and varies only `reset(options={"combat_seed": ...})`. On the batch-3 control,
`squad_march_shoot` shows a vp_margin sd of **50.6 within a layout** against **45.0 between
layouts** — the dice contribute more outcome spread than the scenario does.

## Results

Measured: rolling mean of the last 100 eval points per run.

| arm | win s1 | win s2 | **win mean** | vp s1 | vp s2 | **vp mean** | exposure | fraction alive |
|---|---|---|---|---|---|---|---|---|
| H control | 51.2 | 58.3 | **54.8** | +5.1 | +2.8 | **+4.0** | 0.092 | 0.686 |
| I signal | 57.3 | 51.3 | **54.3** | +4.1 | +3.0 | **+3.5** | 0.094 | 0.674 |
| J reason | 63.0 | 55.9 | **59.5** | +14.7 | +8.2 | **+11.5** | 0.110 | 0.636 |
| K full | 62.2 | 54.9 | **58.5** | +15.1 | +6.6 | **+10.8** | 0.103 | 0.677 |

Floor and bar on this terrain, measured with `just measure-baselines`: `random` **0.00**,
`squad_march_shoot` **0.45** (down from 0.63 on the batch-1/2 profile — changing the
terrain invalidated the old bar). All four arms clear the bar.

> **CORRECTION (2026-08-06, same day).** The two sentences above are wrong, and the
> conclusion they support — that the agent is ahead of the scripted heuristic — is the
> opposite of the truth.
>
> **The `0.45` has the wrong provenance.** It is not what `just measure-baselines
> examples/env_config/25v25_cover_control.yaml 30` returns; that command returns **0.53**.
> `0.45` is 9/20, the in-run `eval/baseline_squad_march_shoot_win_rate`, which uses
> `BASELINE_EPISODES = 20` on seeds 10000-10019. The Reproducing section below was never
> actually run — `recordings/` contains no batch-3 baseline trace.
>
> **The comparison was never apples-to-apples.** Each arm's `eval/win_rate` comes from the
> 10 fixed layouts at `EVAL_SEED_BASE` (`train-multi-seeds` passes no `--n-eval-episodes`,
> so PPO's default of 10 applies). The bar came from a different layout set entirely, and
> `docs/metrics.md` states the rule this breaks: a run and a differently-seeded measurement
> are not comparable.
>
> **The bar is not a number, it is a distribution over layout sets.** Same policy, same
> config:
>
> | seeds | n | `squad_march_shoot` win | vp_margin |
> |---|---|---|---|
> | 10000-10019 (the in-run baseline) | 20 | 0.45 | −15 |
> | 10000-10029 (`measure-baselines … 30`) | 30 | 0.53 | −4.3 |
> | 700000-700029 (`measure-checkpoint`'s held-out set) | 30 | **0.77** | **+39.4** |
>
> A 32-point swing on a deterministic scripted policy, purely from which maps you draw.
>
> **On matched layouts the agent is below the bar.** Scored through `measure-checkpoint`
> on seeds 700000-700029, the same 30 layouts as the bar's 0.77 row:
>
> | policy | on obj | win | vp margin |
> |---|---|---|---|
> | `squad_march_shoot` (bar) | **1.000** | **0.77** | **+39.4** |
> | H control s1 | 0.925 | 0.67 | +16.3 |
> | H control s2 | 0.924 | 0.57 | +11.2 |
> | J reason s1 | 0.877 | 0.57 | +8.7 |
> | J reason s2 | 0.933 | 0.57 | +4.5 |
>
> The agent trails by ~10pp on win rate and **23 VP** on margin. Note also that the control
> beats the treatment on both seeds here, which is the reverse of the `models_lost` finding
> below — see the correction attached to it.
>
> The occupancy shortfall is not models failing to arrive. It is models **piling onto one
> objective**: measured final counts from H, e.g. `player [0,15,0]` against
> `opp [6,0,14]`. See [the 2026-08-06 follow-up](2026-08-06-beat-the-shooting-opponent.md).

### Win rate cannot answer this; vp_margin can

The **within-arm seed spread on win rate is 6.0–7.3pp**, and the largest between-arm gap is
H→J at 4.7pp. Win rate is therefore unreadable here, exactly as the noise floor predicted.

An earlier single-seed reading of this batch put H→J at +8.9pp and treated it as a probable
effect. That was seed noise. This is the concrete value of the second seed.

vp_margin separates cleanly:

- J's **worse** seed (+8.2) beats H's **better** seed (+5.1).
- K's **worse** seed (+6.6) beats H's **better** seed (+5.1).
- I overlaps H completely: (+4.1, +3.0) against (+5.1, +2.8).

Inferred, not measured: with two seeds there is no confidence interval, only
non-overlapping ranges. The claim supported is "the loss penalty moved vp_margin by roughly
+7 in both replicates"; the claim *not* supported is any precise effect size.

> **CORRECTION (2026-08-06, same day).** The non-overlap above is an artefact of the window
> it was measured in, and the effect does not survive a held-out check. The honest status of
> `models_lost` is **unestablished — including its sign**.
>
> **It is window-dependent.** The same rolling mean over the *previous* 100-epoch block:
>
> | arm | last 100 | previous 100 |
> |---|---|---|
> | H control s1 / s2 | +5.12 / +2.85 | +4.29 / +1.12 |
> | I signal s1 / s2 | +4.06 / +3.02 | +6.24 / **+8.79** |
> | J reason s1 / s2 | +14.72 / +8.22 | +10.93 / +13.87 |
> | K full s1 / s2 | +15.06 / +6.63 | +12.34 / **−2.66** |
>
> In the preceding block the ranges overlap outright — a no-penalty run (I s2, +8.79) beats
> a penalty run (K s2, −2.66). Within-run block-to-block swings are ±6 VP, the same size as
> the claimed effect. Batch 2 verified its arms were flat across their final buckets; batch
> 3 skipped that check, and five of eight runs move by more than 3 VP between the last two
> blocks. **Whether 1000 epochs converges on this config is unestablished.**
>
> **It reverses on held-out layouts.** Scored through `measure-checkpoint` on seeds
> 700000-700029, the control beats the treatment on *both* seeds: H at +16.3 / +11.2 against
> J at +8.7 / +4.5. With a per-episode VP sd of ~45–50 the standard error on a 30-episode
> mean is ~8–9, so this reversal is itself inside noise — which is the point. Both the +7
> and its reversal are inside the measurement noise.
>
> **What the data does support** is weaker and was never stated: the runs are seed-paired
> (`train-multi-seeds` gives every config the same `--seed`), and paired J−H is +9.6 and
> +5.4, paired K−H is +10.0 and +3.8. Four positive paired differences is better evidence
> than the non-overlap argument actually made — but it bounds the magnitude only to
> somewhere in 4–10, on the training layouts, and it does not survive the held-out set.

## What the 2x2 says

Taking the three differences the design was built to give:

| contrast | isolates | result |
|---|---|---|
| **I − H** | the signal alone | nothing (vp +3.5 vs +4.0) |
| **J − H** | the reason alone | **+7.5 vp margin** |
| **K − (I + J − H)** | the interaction | absent, if anything slightly negative |

**The reason is real.** Both arms carrying `models_lost` (J, K) beat both arms without it
(H, I), on both seeds, on vp_margin.

**The signal is null.** I sits on top of H on every metric (win 54.3 vs 54.8, vp +3.5 vs
+4.0, exposure 0.094 vs 0.092). And it does not become useful once there is a reason to use
it: K is marginally *below* J (58.5 vs 59.5 win, +10.8 vs +11.5 vp). That second point is
what the fourth arm was for — without K, "the signal needs a motive to matter" would still
be open.

**The feature and its two arms have since been removed** — the observation column, the
config flag, and `25v25_cover_{signal,full}.yaml`. A measured-null feature kept "in case"
is dead configuration, and this one carried a live footgun: the column had to sit inside
`core` before `alive`, because `TransformerNetwork._alive_feature_index` counts backwards
from the last column. `compute_threat_counts` itself stays; the exposure metrics use it.

This refutes the pre-registered prior in both directions. The plan predicted "the signal is
necessary and the reason alone does nothing." The reason alone does everything; the signal
does nothing.

The honest reading is not "LOS information is useless" but "**this** LOS encoding, as one
normalised scalar per model, added nothing the policy could not already act on." A count of
threats says how many guns bear on a model but not from where, so it cannot support "step
two cells left and the wall covers me" — which is the decision cover actually requires.

## Two results that contradict the mechanism we assumed

Flagged rather than explained. Neither has a measurement behind it yet.

**1. Pricing losses increased losses.** J keeps **0.636** of its models alive against H's
**0.686**, and exposes more (0.110 vs 0.092). A penalty on losses was added to buy caution
under the risk/reward framing — expose yourself only when the shot is worth it. It bought
aggression that pays instead. Whatever `models_lost` does, "decline bad trades" is not it.

A plausible but **unmeasured** explanation: the penalty fires on almost every combat step,
where `vp_gain` is sparse and objective-driven, so it may be acting as reward *shaping*
that densifies the combat gradient rather than as a *price* the policy reasons about. This
is a hypothesis, not a finding. Testing it means varying `penalty_per_loss` and checking
whether the effect scales with the weight (a price) or saturates (shaping).

**2. Still no cover, third setup running.** Exposure 0.092–0.110 with 19.8% of the board
hidden, a LOS input available, and losses priced. Every blocker named in the batch-1/2
retrospective was removed and the behaviour did not appear.

## `firepower_advantage` does not work as specified

Recorded as `engaged_theirs − engaged_mine`, intended as the batch-3 headline metric. It
cannot carry a conclusion:

| policy | firepower_advantage | win |
|---|---|---|
| `random` | 1.78 | 0.00 |
| H control | 1.07 | 54.8 |
| K full | 1.03 | 58.5 |

A policy that wins **zero** games scores highest, and the best-equipped arm scores *below*
control. As a raw count difference it is dominated by how much engagement happens at all,
not by who wins the exchange. It needs to be a ratio, or normalised by total engagement,
before any batch can turn on it. **Do not read the batch-3 `eval/firepower_advantage`
values as evidence of anything.**

`exposure_rate` remains the wrong headline for the same reason as before: it counts only
our side of the exchange, so it falls both when a policy manoeuvres well and when it hides.

## What this does not support

- **No ranking of `penalty_per_loss`.** One weight was tested, calibrated to parity with
  `model_kills`. Whether the effect is monotone in the weight is unknown, and that is the
  experiment that would distinguish price from shaping.
- **No claim that LOS information cannot help** — only that a per-model threat *count*
  does not. A directional or per-sector encoding is untested.
- **No claim about cover in other scenarios.** Everything here is 25 one-wound models at
  range 12. The open question from the batch-1/2 report stands: whether individual LOS can
  matter at all when losing a model costs 1/25th of the force and no model can retreat and
  heal.
- **Two seeds, not a confidence interval.** Non-overlapping ranges on n=2.

## Reproducing

```bash
# Floor and bar on this terrain -- the batch-1/2 baselines do not transfer
just measure-baselines examples/env_config/25v25_cover_control.yaml 30 record

# The terrain profile: fraction of board hidden from a squad in weapon range
just measure-terrain examples/env_config/25v25_cover_control.yaml 200

# The noise floor that made two seeds mandatory
just measure-noise-floor examples/env_config/25v25_cover_control.yaml

# Rolling means for any run above (point readings are n-sample binomials)
just run-summary kra8wkr0 50

# Re-run one seed group into an existing wandb group
just train-seed 1000 2 <group> examples/env_config/25v25_cover_control.yaml ...
```
