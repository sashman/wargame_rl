# Pre-registration — two play-time knobs, measured on the instrument that works

Written 2026-09-06 after the n=180 retraction
(`reports/2026-09-06-the-ladder-was-measurement-noise.md`) and **before either
knob has been scored**.

## Why these two, and why not a training arm

At n=180 with both sides equally measured the gaps are `refereed` **+4.21**,
`vs_take` +6.73, `vs_deny` +7.45, `vs_shoot` **−0.19** — nothing resolved. To
win four cells at n=180 (SE ≈ 6.3) needs gaps near +13, so roughly **+10 vp of
real improvement**, not more scoring. `vs_shoot` at −0.19 cannot be fixed by n
at any size.

Every large gain in this goal has come from the **decode**, not from training:
the stack is worth ~+81 vp, and the reallocation operator survived its own
remeasurement at **+9.57 ± 2.55, 6/6 seeds**. Two knobs on that stack have
never been scored. They cost no GPU training, and unlike the pool-composition
arm they act on **all four cells at once** rather than trading two against each
other.

## Knob A — `decode_stay` (already in the code, never measured)

Stands a unit still when its top-K set contains no legal combination, instead of
handing it to the referee to revert. Documented mechanism: it fires on only
0.3–1.4% of unit-moves, but a revert additionally runs the overlap cascade that
drags *neighbouring* units back and accounts for **9.2–15.3% of all freezes**,
while a deliberate stay triggers nothing. The eval configs run
`enforce_move: revert_unit` **and `attrition: true`**, so a reverted unit can
also lose models.

No code change. `build_action_selector(..., decode_stay=True)`.

## Knob B — iterated reallocation (`max_redirects`)

The rule picks the single biggest stack and its single largest donor squad, and
returns one `(donor, target)`. One squad per movement phase, against a diagnosed
failure of **4.90 models on the top point** (script: 2.73), **55.3% of
objectives empty**, and a redistribution ceiling of **+2.20 objectives**. One
redirect barely dents that.

`choose_surplus_reallocation` gains `exclude_groups` / `exclude_targets`
(committed groups counted as already gone, so the next surplus test sees the
board as it will be); `apply_reallocation` gains `max_redirects`.

⚠ **Defaults are provably unchanged.** `max_redirects=1` with empty exclusions
is the old rule, verified end-to-end: seed 1 on `vs_shoot` scores **+79.4,
coherency 0.952, decl 11.13 / tried 9.40 / stood 6.82** either side of the
change — every printed digit. 4656 tests pass. This matters because every
reallocation figure on file was taken under the old path.

## Protocol, fixed now

n = **180**, seeds 700000+, six seeds, K=3, charge decode on, reallocation on,
**all four cells**, paired per scenario against the n=180 bar already measured
(the bar is unaffected by either knob — `_resolve_baseline` takes no decode
arguments, and §29 measured the redirect on it at exactly zero).

Grid: `decode_stay ∈ {off, on}` × `max_redirects ∈ {1, 3}`. Four cells × six
seeds × three new cells of the grid (1/off is already measured) = 72 runs.

## Decision rule, committed

- A knob is **adopted** only if its paired per-scenario effect is positive with
  **t > 2 across seeds** and **≥ 5 of 6 seeds** positive, averaged over the four
  cells. Per-cell cherry-picking is forbidden — that is comparator selection,
  which this project has paid for twice.
- **All four grid cells are reported** whatever they show. Reporting only the
  winner is how the n=45 ladder happened.
- The goal is met only if, with whatever survives, **all four ladder cells reach
  t > 2** against the n=180 bar.

## Predictions, recorded now

- **`decode_stay`: small, +0 to +3.** It fires on ~1% of unit-moves; the
  cascade argument is a mechanism for why it is not *zero*, not for why it
  would be large.
- **Iterated reallocation: the larger of the two, but well under 3x the single
  redirect.** The rule requires a genuine surplus, and after one squad leaves,
  the second-biggest stack often fails the `min_stack: 4` test. I expect
  **+3 to +8**, not +19.
- ⚠ **Most likely outcome: both positive, both too small.** +10 of headroom is
  needed on `refereed` and `vs_shoot`; two decode knobs plausibly deliver half
  that. Recorded now so a two-small-wins result is not written up as the goal.
