# Pre-registration — ARM 6: EXTRAPOLATING past the anchored arm

Written 2026-09-05, before any extrapolated checkpoint exists. Arm 4
(1000 epochs) is still running and its numbers are unknown.

## The idea, and why it is not the same as §50

§50 interpolated `clone -> plain PPO` and found the optimum at **α=0.1**, with
α=0.5 clearly worse. But that line ends at a **bad** policy (−25.58), so of
course a small step was best — it was mostly avoiding the endpoint.

The `clone -> arm 1` line is different in kind. Arm 1 is **better than the
clone on every cell**, and its direction is **headroom-PRESERVING**: drift only
0.03–0.05 nats, and decode headroom **+78.43 against the clone's +74.87** — it
went UP. Plain PPO spent headroom; the anchored direction does not.

If that direction is genuinely good rather than merely short, then
`clone + α·(arm1 − clone)` with **α > 1** should continue along it. Nothing in
this record has tested α > 1 on any line.

## The test

Per seed, `clone_s + α·(arm1_s − clone_s)` for **α ∈ {1.25, 1.5, 2.0}**, six
seeds, refereed cell first (the only unwon cell), n=45, seeds 700000+,
K=3 + charge decode. Grid fixed now; the whole grid is reported.

## Bounds, fixed now

- **PASS** (score the other three cells): some α gives refereed six-seed mean
  **≥ +6.9** — the level that clears the bar by 2 SE at the observed spread.
- **PARTIAL**: some α beats arm 1's +1.45 by ≥ +3 without collapsing (no cell
  below the bar when the other three are then scored).
- **FAIL**: no α beats +1.45.

⚠ **The honest prior is FAIL or PARTIAL, not PASS.** Extrapolation past a
trained point usually degrades: the direction is only locally good, and doubling
it doubles the drift, which the same-day drift/headroom curve says costs headroom
(at drift 0.096 headroom was +54.7 against the clone's +77.2). The reason to run
it anyway is that it costs **inference only** and the anchored direction is the
first one measured here that does not spend headroom.

⚠ If an α wins, it is **selected from a grid of three on the cell it was chosen
for**, which is exactly the winner-selection this repo prices at +1.4 to +2.9 vp.
A PASS therefore requires the other three cells scored at that same α, and the
result reported as selected-then-confirmed, never as a clean six-seed win.


---

# RESULT, 2026-09-05 — FAIL. The anchored direction is only LOCALLY good.

Refereed cell, six seeds, n=45, K=3 + charge decode, no training.

| α | mean | SE | gap v bar | SEs | v arm 1 |
|---|---|---|---|---|---|
| **1.00** (arm 1 itself) | **+1.45** | 6.07 | +6.75 | 1.11 | — |
| 1.25 | −5.85 | 5.81 | −0.55 | −0.09 | **−7.30** |
| 1.5 | −3.50 | 4.36 | +1.80 | 0.41 | **−4.95** |
| 2.0 | **−19.48** | 7.04 | −14.18 | −2.01 | **−20.93** |

**No α beats arm 1's +1.45** — the registered FAIL condition, met on all three
doses. Degradation is **monotone in α** and collapses at 2.0.

**What it answers.** The premise was that the anchored direction is different in
kind from plain PPO's: it *preserves* headroom (+78.43 against the clone's
+74.87) where plain PPO spends it, so walking further along it might keep
paying. **It does not.** The direction is good over exactly the distance
training took it and no further. Arm 1 sits at or near the best point on its own
line, not partway along a good one.

Read with §50: interpolation toward a **bad** endpoint peaked at α=0.1, and
extrapolation past a **good** one peaks at α=1.0. Both say the same thing —
**the useful region is a narrow band around where training actually stopped**,
and neither shrinking nor extending the step finds anything better.

**Forbidden next**: α > 1 on any line here. Three doses, monotone decline,
−20.93 at the end of it.
