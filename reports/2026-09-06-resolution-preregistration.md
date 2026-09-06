# Pre-registration — resolving the two estimators, at n fixed in advance

Written 2026-09-06 immediately after the six-seed ladder was read at n=45 and
**before any n=180 number exists**.

## The problem this exists to settle

The six-seed ladder reads WON on all four cells by the criterion this project's
record has used all along — the gap to the bar over the **across-seed** SE
(n=6). A paired **across-scenario** analysis of the same data disagrees:

| cell | across-seed (n=6) | across-scenario (n=45, paired) |
|---|---|---|
| `refereed` | +9.50, 2.82 SE → WON | +9.52 ± 13.32, t=0.71 |
| `vs_shoot` | +15.60, 4.57 SE → WON | +15.65 ± 12.07, t=1.30 |

Both are correct about different questions. Across-seed asks whether another
training seed reproduces the result (it does: 6/6 seeds positive in both cells).
Across-scenario asks whether another draw of tables and dice reproduces it, and
**cannot answer at n=45** — per-scenario sd is 81–89, so the SE is ~12–13.

⚠ **The across-seed SE is anti-conservative for a claim about the game.** Every
seed is scored on the *same* 45 scenarios, so scenario noise is common-mode:
averaging seeds does not shrink it, and an n=6 SE omits it entirely. This was
not visible until the per-scenario dump existed.

⚠ **Sign counts do not discriminate here and must not be quoted as if they do.**
Observed 22/45 and 25/45; a *true* effect of the measured size predicts 24.4 and
25.9 against these sds. The counts are consistent with the effect, not evidence
against it. (The standing rule "quote a t and a sign count" assumes the sign
count is informative; at this noise level it is not.)

## The protocol, fixed now

**n = 180**, decided before any of it is run and **looked at once**. Not "raise n
until significant" — optional stopping is exactly the failure this section
exists to avoid. n=180 is chosen because it halves the SE (~12 → ~6), which is
the precision needed to resolve a +15 effect at t≈2.5, and because it costs
about an hour on an idle GPU.

Scored: all four cells × six seeds at `reallocate=1`; `vs_shoot` × six seeds
also at `reallocate=0` (the only cell where the two arms differ in verdict); the
bar once per cell (it is deterministic given layout and dice seed). Seeds
700000+, K=3, charge decode on — the ladder's own protocol, only longer.

## Decision rule, committed

A cell is **WON** only if **both** estimators agree: the across-seed gap exceeds
2 SE **and** the paired across-scenario t exceeds 2. A cell where they disagree
is reported as **unresolved**, not as a win.

⚠ This is a **stricter** rule than the one the record has used, and applying it
retroactively may unmake previously published "WON" rows on this ladder. That is
the correct direction for a rule change discovered this way, and the earlier
rows are not silently amended — they were taken under the criterion of their
time and are labelled with it.

## Prediction, recorded before the run

`vs_shoot` clears (+15.6 at SE ~6 → t ≈ 2.6). `refereed` does **not** (+9.5 at
SE ~6.5 → t ≈ 1.5), so the honest expected outcome is **three cells won and the
mirror unresolved** — which is *not* the goal as stated. I am recording that
before running it rather than after.
