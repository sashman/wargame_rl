# The ladder was measurement noise — at n=180 no cell is resolved, and `vs_shoot` is level

Measured 2026-09-06, the single look committed in
`reports/2026-09-06-resolution-preregistration.md`. **This retracts the
"all four cells WON" result reported earlier the same day.**

## The result

Both sides measured on the same 180 scenarios (seeds 700000+), K=3, charge
decode, reallocation on, arm 4 at epoch 1000, six seeds. Paired per scenario.

| cell | agent n=45 | agent n=180 | bar n=45 | bar n=180 | gap n=45 | **gap n=180** | t | seeds > bar |
|---|---|---|---|---|---|---|---|---|
| `refereed` | +4.20 | −5.74 | −5.3 | −9.94 | +9.50 | **+4.21** | 0.65 | 4/6 |
| `vs_take` | +49.22 | +41.53 | +20.2 | +34.81 | +29.02 | **+6.73** | 1.08 | 6/6 |
| `vs_deny` | +45.15 | +44.26 | +11.8 | +36.81 | +33.35 | **+7.45** | 1.12 | 6/6 |
| `vs_shoot` | +72.20 | +56.28 | +56.6 | +56.47 | +15.60 | **−0.19** | −0.03 | 2/6 |

**No cell is resolved. `vs_shoot` — reported as a +15.6 win — is dead level.**
The goal is **NOT met**.

## Two independent errors, both mine, both in the same direction

1. **The comparator was a single noisy estimate treated as a constant.** The bar
   at n=45 carries SE ≈ 12.7 (per-scenario sd 81–89). Its `vs_deny` value moved
   **+11.8 → +36.8** on remeasurement. Every ladder row ever published in this
   goal was quoted against a bar with an unpropagated ±12.7.
2. **The across-seed SE (n=6) omitted scenario noise entirely.** Every seed is
   scored on the *same* scenarios, so that noise is common-mode and does not
   shrink with seeds. It made a ±12 difference look like 2.8–8.1 SE.

## ⚠ My pre-registered prediction was wrong, and instructively

I predicted "`vs_shoot` clears (t ≈ 2.6), `refereed` does not". `vs_shoot` came
in at **t = −0.03**. The error: I assumed raising n would shrink the SE around a
*fixed* mean. It also **moved the mean**, by −15.9 on `vs_shoot`. A prediction
that only shrinks the interval silently assumes the small-n point estimate was
unbiased — which is the same mistake as trusting the comparator, one level up.

## The standard seed band is atypical, and by a lot

Same runs, split into the protocol's own first 45 episodes versus the other 135:

| cell | agent (0–44 minus 45–179) | bar (0–44 minus 45–179) |
|---|---|---|
| `refereed` | **+13.23** | +6.15 |
| `vs_take` | **+10.23** | **−19.44** |
| `vs_deny` | +1.21 | **−33.37** |
| `vs_shoot` | **+21.23** | +0.11 |

The agent scores higher on the first 45 in **4 of 4** cells while the bar scores
lower in two of them — both biases inflating the gap, by 30–50 vp on `vs_take`
and `vs_deny`. Individually most of these sit within a 14.6 SE, so this is not
established as a systematic property of seeds 700000–700044; the point is that
**n=45 cannot tell the difference**, and the record's standard evaluation size
is 30–45.

## What SURVIVES

- **The reallocation decode is real**: paired per scenario at n=180,
  **+9.57 ± 2.55, 6 of 6 seeds** (t = 3.75) on `vs_shoot`. Smaller than the
  n=45 estimate of +16.87 — regression to the mean — but it holds. It is the
  only claim from this session that survives its own remeasurement.
- **Seed variance is small; scenario variance dominates.** Per-seed means at
  n=180 span just +50.1 to +66.2 on `vs_shoot`. Six seeds was never the binding
  constraint — episodes were.
- **Direction is consistent**: the agent's point estimate is above the bar on
  three of four cells, 6/6 seeds on two of them.

## What this costs to fix, and whether it is worth paying

At gaps of +6.7 and +7.5 with SE 6.3 at n=180, `vs_take` and `vs_deny` need
**n ≈ 1450** for t = 2 — about 8x more scoring, no training. `refereed` at +4.21
needs n ≈ 2900. `vs_shoot` at −0.19 cannot be won by more n at all.

⚠ **Before spending that, note what the honest claim would be even if it
succeeded**: "the agent beats a scripted policy by about 7 points on two of four
matchups, is level on the other two". That is a materially weaker statement than
the one this goal was pursuing, and it should be decided deliberately rather
than by default.

## Standing rules this produces

- **Measure the comparator at the same n as the arm, and propagate its SE.** A
  deterministic policy is not a constant: it is a fixed policy sampled over
  scenarios, and it carries the same per-scenario noise the agent does.
- **Never quote an across-seed SE for a claim about the game** when all seeds
  share the evaluation scenarios. Report the paired across-scenario estimator,
  or both.
- **Raising n moves the mean as well as the interval.** Predict both.
