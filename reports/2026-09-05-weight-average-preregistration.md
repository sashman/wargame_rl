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
