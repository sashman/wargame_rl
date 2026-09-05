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
