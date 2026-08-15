# Coherency: what it costs, and the one lever that moved it

> ## ⚠ SUPERSEDED — 2026-08-16
>
> **Read [Enforcement is a referee, not a teacher](2026-08-16-enforcement-is-a-referee.md)
> instead. This report's conclusions and every number in it are void**, for two
> independent reasons:
>
> 1. **Every compliance figure was sampled after the enforcement fixed point.**
>    Under `coherency.enforce_move`, `eval/coherency_rate` is 1.000 by
>    construction whatever the policy does, so it measures the referee and not
>    the agent. Measured with the referee *off*, the same weights intend 0.630.
> 2. **A geometry fix (#193) then moved line of sight on these maps by 22.6%**,
>    voiding every eval-map number measured before it.
>
> The settled answer: train with `objective_hold.require_coherent`, never train
> under `enforce_move`, and switch enforcement on for play.
> `configs/golden/25v25_maps_coherency.yaml` is that configuration.
>
> **Kept, not deleted, because the reasoning and the retractions are the
> reusable part** — this is a record of how a wrong conclusion was reached, and
> rewriting it with the answer in hand would destroy that.

---


**2026-08-14.** Follow-up to
[coherency is a symptom](2026-08-14-coherency-is-a-symptom.md), which this report
partly retracts. Everything here is the real-tables scenario
(`configs/experiments/25v25_maps_*`), n=100 at seeds 700000-700099 on identical
layouts unless stated.

## The answer

**Obeying coherency is cheap. Training under it is not.**

| weights | environment | `vp_margin` | `held` | `alive` |
|---|---|---|---|---|
| control | unenforced | +98.6 | 3.45 | 0.957 |
| control | **enforced** | +89.9 / +86.5 | 3.01 / 3.04 | 0.963 / 0.961 |
| **trained under enforcement** | enforced | +75.6 / +71.2 | 2.79 / 2.73 | 0.984 / 0.983 |

Imposing the rule on a trained policy costs **−8.7**. Training under it for ~100
epochs costs a further **−14.3 / −15.3** (both seeds). So roughly **62% of
coherency's apparent price is caused by training under it, not by obeying it**,
and the mechanism is visible in the columns the repo says to read: `alive`
climbs while `held` falls. The policy does not learn to manoeuvre in formation.
It learns to stop contesting ground.

That is the same pathology as `revert_unit` from scratch (−75.3), softened by
warm-starting but not removed.

## The lever that worked: `objective_hold.require_coherent`

A model outside its unit's coherent body earns no objective income. Not less —
none. The argument is the user's and is better than the one it replaced: a
detached model is not a legal state of the game, so it should not be paid.

Two seeds, 300 epochs, warm-started, scored at n=100:

| | `coherency_rate` | models adrift | `vp_margin` | `held` |
|---|---|---|---|---|
| control | 0.52-0.65 | 2.9-3.7 | +98.6 | 3.45 |
| `unit_coherency` bonus (saturated) | 0.67 | 2.5-2.9 | — | — |
| **`require_coherent`** | **0.708 / 0.849** | **2.33 / 1.24** | **+95.8 / +96.2** | **3.41 / 3.41** |

Coherency rises from ~0.55 to a mean of 0.78, models adrift roughly halve, and
`held` is unchanged at 3.41 against the control's 3.45 with `vp_margin` 2.6 down
— inside the ~4.5 standard error at n=100. **It is the first lever to move the
metric through learning rather than through construction**, and it does not
suppress objective play, which is what killed `surplus_value` and the overstack
penalty.

Caveat: the two seeds disagree by 14 points (0.708 v 0.849). Two seeds rank an
effect; they do not size it.

**Why the reward route works here when the level of pay does not.** The
difference-reward arm (`marginal_weight: 1.0`) was meant to remove the incentive
to defect and moved coherency not at all — 0.478 at epoch 8, 0.555 at epoch 164.
Checking the components showed why it was not the test it claimed to be:
`objective_hold` income collapsed from **0.176 to 0.014**, because a difference
reward pays zero on any securely-held objective. It deleted the term rather than
redirecting it. The result still stands in a stronger form: whether
`objective_hold` pays *for* defection or pays *nothing*, coherency sits at 0.55.
**The level of objective pay is not what drives models to detach; making income
conditional on legality is.**

## What did not work

**The enforcement-probability sweep.** No knee. p=0.75 is dominated by p=1.0 on
both coherency and vp — the same move undone sometimes and not others is not a
rule anything can learn. Do not re-run.

**`clamp` from scratch.** A new enforcement mode that *shortens* a move along
its own segment instead of cancelling it, added specifically to remove the
credit-assignment pathology that produced −75.3. From scratch it plateaus around
**−25** from epoch 96 to 164, against a control at **+90 by epoch 40**. Better
than `revert_unit` (−75.3), nowhere near viable. Training from scratch under
enforcement fails even when moves are never cancelled.

**`gated_clamp` — the gate and the constraint together.** The stated mechanism
was that the gate removes the motive to detach, so the constraint should bind
less often. **Measured on the finished gated checkpoint, it does not**: reverts
per step 6.79 against the ungated 6.34 under `revert_model`, 6.35 against 7.90
under `clamp` — no consistent reduction. The gate reduces the *severity* of
breaches (models adrift 3.3 → 1.24) rather than their *frequency*, and
enforcement fires on any unit ending its move in breach however marginally.

Worse, the arm was ill-conceived: **under full enforcement the gate is
definitionally inert.** `coherency_rate` is 0.999, so no model is ever detached,
and `require_coherent` only bites on a detached model. It ran flat at 53-68 —
indistinguishable from plain enforcement — and was killed at epoch 146. The two
mechanisms are not complementary; the gate only does work when models *can*
detach.

## Retracted from earlier today

- **"Enforcement costs ~28-30 vp."** n=6, where per-episode sd is 45-50 so the
  standard error is ~19 — it could not resolve the effect. The real figure is
  −8.7 to impose, −14.8 to train under.
- **"A coherent policy can win here; the agent just is not finding it."** That
  used the scripted bar's *unenforced* +105.6 at a coherency rate of 0.837, and
  0.837 is not compliance. Under enforcement the bar pays a toll too (+93.5).
- **"The difference reward tests the defection incentive."** It deleted the term.

## Method notes worth keeping

**Seed-set sensitivity moved three conclusions in one session.** The same
deterministic scripted policy scores **+67.5** at seeds 10000+ and **+93.5** at
seeds 700000+ on one config. Score agent and baseline on identical layouts at
the same `seed_base`, or the comparison means nothing.

**Score enforced arms against an enforced bar.** `just measure-baselines` on the
*enforced* config, never the unenforced one.

**Grepping process lists for arm names is ambiguous and silently wrong.** A
watcher for `clamp` matched all four `gclamp` processes, and one for `gated`
matched all eight, because the config path `25v25_maps_gated_clamp.yaml` and the
group `coherency-gated-clamp` both contain other arms' tags. Match the extracted
`--run-suffix` token, anchored. Separately, killing `uv run train.py` leaves the
`python3 train.py` child running — match `train.py`, not the wrapper.

**Test a mechanism before describing it as the reason for an arm.** The revert
count that refuted `gated_clamp` cost one command against a checkpoint that
already existed, and was run after the arm had been launched and explained.
