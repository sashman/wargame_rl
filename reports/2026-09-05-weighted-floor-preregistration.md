# Pre-registration — ARM 7: a 2:1 pool floor (conditional on arm 4)

Written 2026-09-05 **before arm 4's 1000-epoch numbers exist**.

## The observation that motivates it

On point estimates alone — ignoring significance entirely — **no policy on file
beats all four cells**, and the two best miss on *different* cells by under 1 vp:

| | refereed (bar −5.3) | `vs_shoot` (bar +56.6) |
|---|---|---|
| arm 1 (floor: 100% charging) | **+1.45** ✓ | +56.03 ✗ by 0.57 |
| arm 3 (floor: 50/50) | −5.95 ✗ by 0.65 | **+65.55** ✓ |

The floor's composition moves the policy along an axis between those two cells.
Arm 1 sits one side of the goal, arm 3 the other, and **both by less than a
point**. A floor of **2 charging : 1 shooting** is the obvious intermediate.

⚠ **This is NOT the same as arm 5, which failed.** Arm 5 averaged the two arms'
*weights* and was therefore confined to the straight line between them — where
"between" won neither cell. A differently-trained policy is not on that line and
is free to land off it. That is the whole reason this is worth a run rather than
more arithmetic.

## The arm

Arm 3 with `--pool-anchor squad_march_take_charge,squad_march_take_charge,squad_march_shoot`
(duplication expresses the weight; verified the pool accepts it and keeps all
three as never-evicted anchors under uniform sampling). Everything else
identical. Six seeds, 300 epochs.

**Launched only if arm 4 fails to win refereed** — if longer training solves the
cell, this arm is unnecessary and must not be run for the sake of running it.

## Bounds, fixed now

- **PASS**: all four cells WON at six seeds. That is the goal.
- **PARTIAL**: refereed and `vs_shoot` both **ahead or better** (> 1 SE) — the
  first policy to be on the right side of both at once.
- **FAIL**: either cell lands below its bar on the point estimate, i.e. the 2:1
  floor buys nothing the 1:1 and 1:0 floors did not.

⚠ **The honest prior is PARTIAL at best.** The floor axis has moved refereed by
about 7 vp between arm 1 and arm 3 and `vs_shoot` by about 9. An intermediate
should land intermediate on both — near the bar on each, which wins neither.
**A 2:1 floor is a plausible way to be simultaneously adequate and an
implausible way to be simultaneously convincing.** Recorded now so that a
two-tie result is not written up as progress.

⚠ Six seeds at 300 epochs is ~8 GPU-hours. It is worth that only because it is
the last untested point on the one axis that demonstrably moves both cells.
