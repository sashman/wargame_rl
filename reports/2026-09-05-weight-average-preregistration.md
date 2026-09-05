# Pre-registration — ARM 5: averaging arm 1 and arm 3 (no training)

Written 2026-09-05 **before any averaged checkpoint exists**. Arm 4
(1000 epochs) is still running and its numbers are NOT known.

## Why this should work at all

Arm 1 (`take_charge` anchor) and arm 3 (`take_charge + shoot`) are, per seed,
warm-started from the **same clone** and each held near it by the same KL anchor
(drift ~0.03-0.05 nats). Two policies anchored to a common reference are the one
case where averaging weights is principled rather than a lottery: they have not
left the region the reference sits in, so the segment between them is expected
to be well behaved.

They also have **complementary** strengths, which is the whole reason to try:

| cell | arm 1 (6 seeds) | arm 3 (6 seeds) | bar |
|---|---|---|---|
| refereed | **+1.45** (ahead) | −5.95 (tie) | −5.3 |
| `vs_take` | **+34.60** WON | +30.62 WON | +20.2 |
| `vs_deny` | **+32.70** WON | +30.90 WON | +11.8 |
| `vs_shoot` | +56.03 (tie) | **+65.55** WON | +56.6 |

## The test

Per seed, average the two state dicts 50/50 (`0.5 * arm1 + 0.5 * arm3`), score
all four cells, n=45, seeds 700000+, K=3 + charge decode, six seeds. No
training.

## Bounds, fixed now

- **PASS**: all four cells WON (gap to bar > 2 SE) — the goal.
- **PARTIAL**: refereed WON and `vs_shoot` still WON — closer than either parent.
- **FAIL**: the average is a compromise, i.e. refereed below arm 1's +1.45 AND
  `vs_shoot` below arm 3's +65.55. That is the *expected* outcome if the two
  arms simply trade along a line.

⚠ **The honest prior is FAIL.** Averaging two points usually lands between them,
and "between" on these two cells means winning neither. A soup only helps if the
average is *better than both parents*, which is a real phenomenon but not the
default. I am recording that expectation now so a compromise result is not
written up as a near miss.

⚠ This is **not** a new training method and must never be quoted as one. It is
arithmetic on two existing checkpoints, exactly like §50's interpolation, and
§46's borrowed-geometry caveat governs it verbatim.


---

# RESULT, 2026-09-05 — FAIL, exactly as the prior said

Six seeds, n=45, K=3 + charge decode, no training.

| cell | soup | SE | bar | gap | SEs | read | arm 1 | arm 3 |
|---|---|---|---|---|---|---|---|---|
| refereed | −0.90 | 3.88 | −5.3 | +4.40 | 1.13 | ahead | +1.45 | −5.95 |
| `vs_take` | +31.68 | 6.39 | +20.2 | +11.48 | 1.80 | ahead | +34.60 | +30.62 |
| `vs_deny` | +29.95 | 4.26 | +11.8 | +18.15 | 4.26 | **WON** | +32.70 | +30.90 |
| `vs_shoot` | +55.35 | 5.47 | +56.6 | −1.25 | −0.23 | tie | +56.03 | **+65.55** |

**1 of 4 WON — worse than either parent (arm 1: 2, arm 3: 3).** Every cell landed
**between or below** the two, and `vs_shoot` fell below *both*. The registered
FAIL condition (refereed below +1.45 AND `vs_shoot` below +65.55) is met on both
clauses.

**The complementary strengths cancelled rather than combined.** Being anchored to
a common reference was enough to make the average *well behaved* — nothing blew
up, every cell is sane — but not enough to make it better than its parents. A
soup bonus is a real phenomenon and it did not occur here.

⚠ **One side-effect worth recording and NOT overclaiming**: averaging **cut the
variance**. Refereed SE 3.88 against arm 1's 6.07, `vs_deny` 4.26 against 6.30.
On refereed the soup reads **1.13 SE** ahead against arm 1's **1.11** — the same
significance from a *lower* mean and a tighter spread. That is consistent with
the same-day noise decomposition of that cell (per-episode sd **82.3**, so
measurement contributes ~12.3 of arm 1's 14.9 seed spread and genuine policy
variation ~8.4): averaging damps the policy-variation term and cannot touch the
measurement term. It is **one observation on one comparison** and is not
evidence that averaging is a variance-reduction technique worth adopting.

**Forbidden next**: averaging anchored arms in the hope of combining cells. It
was tested, it is a compromise, and the compromise wins fewer cells than either
input.
