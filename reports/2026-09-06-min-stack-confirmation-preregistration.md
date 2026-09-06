# Pre-registration — confirming `min_stack: 2` on the evaluation band

Written 2026-09-06 **before the evaluation-band numbers exist**. The value was
chosen on a held-out band; this is its test.

## What was tuned, and where

`min_stack` gates the reallocation decode: an objective is a candidate source
only if it holds at least that many of our models. It ships at **4** while the
agent's top stack averages **4.90**, so the rule almost never fires.

Swept on seeds **900000+** (disjoint from evaluation 700000+, in-run eval
500000+, logged baselines 10000+, clone demonstrations 800000+), n=90, three
seeds, `vs_shoot` and `refereed`:

| `min_stack` | `vs_shoot` | `refereed` |
|---|---|---|
| **2** | **+71.83** | +2.20 |
| 3 | +70.73 | +2.07 |
| 4 (shipped) | +66.60 | +3.13 |

Paired per seed, 2 v 4 on `vs_shoot` is **+5.2, 3 of 3**.

## The test, fixed now

`min_stack=2` against `min_stack=4`, everything else identical: **n=180**, seeds
**700000+**, six seeds, K=3, charge decode on, reallocation on,
`max_redirects=1` (iteration is measured and rejected), **all four cells**,
paired per scenario.

## Bounds

- **CONFIRMED** if the paired effect is positive with **t > 2 across seeds** on
  `vs_shoot`, and **not negative beyond −2 SE on any other cell**. The second
  clause matters: a knob that buys one cell by giving up another is the
  pool-composition trade in a cheaper costume, and this goal already has one.
- **NOT CONFIRMED** otherwise, and `min_stack` stays at 4.

⚠ **Confirmation is not the goal.** Even at its tuned value this is +5.2 on
`vs_shoot` and ~0 on `refereed`, against the **+13 and +8** the ladder needs. It
is recorded as a component, and a component only.

## Prediction

Confirms on `vs_shoot` at roughly half the tuned effect (regression to the mean:
the reallocation decode itself went +16.87 → +9.57 between bands), so **+2 to
+5**, and moves `refereed` by nothing. I do **not** expect it to change any
cell's verdict.
