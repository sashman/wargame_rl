# Pre-registration: curriculum rung D1b — the escort cloned from four times the demonstrations

Written 2026-09-17 15:20, **before any clone at 1,200 episodes exists**,
on branch `feature/curriculum-d1` (PR #374). Issue #375, parent
question #340, rung **D1** — a second arm, one change.

## The one change

1,200 demonstration episodes (seeds 800000+, the last 240 held out)
instead of 300; 40 epochs, batches of 64, Adam 3e-4, fit seeds 0 and 1,
as D1. `just behaviour-clone-per-model scripted_escort
configs/experiments/curriculum/c3b.yaml 1200 40 <out> <seed>`.

## Why

D1's clones (#373) read success 0.830 / 0.850 against the escort's
0.990 with the plan intact (kill before arrival 0.99 / 1.00, blockers
wiped 98%, alive 0.90), and their displacement head at 0.95 on the
training episodes against 0.61 held-out — an over-fit of 240 games of
continuous geometry, not a representation limit. The joint match D1
pre-registered counts the escort's unit-opening order, which is not in
the observation (chance held-out, 0.60 on the training set), so it is
replaced here by the displacement head's own held-out match.

## Comparator

`scripted_escort` on `c3b.yaml`, n=100 on 700000+: success 0.990, held
3.97, alive 0.963, kill before arrival 1.00. D1's clones beside every row.

## Criteria

Read greedy, no decode, n=100 on 700000+, both clones.

- **PASS:** success ≥ **0.96** on both; kill before arrival ≥ **0.95**
  on both; held-out **displacement** match ≥ **0.90** on both.
- **FAIL:** any clause missed on either clone. If the displacement match
  rises past 0.90 and success does not reach 0.96, the miss is not
  fidelity and the census says what it is.
- **NULL (scenario):** only if the script fails its own criterion (0.990).
- **Readouts:** the match by head, train against held-out; `held`,
  `alive`, `on_obj`, coherency greedy and sampled; the by-turn census;
  D1's rows.

Power: binomial SE 0.010 at p=0.99, n=100; the displacement match is
over ~26,000 held-out decisions.

## What I expect (a guess, written so it can be wrong)

Held-out displacement match 0.75–0.85 (up from 0.61, still under 0.90),
success 0.90–0.95: better, and short of both bounds — the geometry
needs more than four times the games, or a coarser target. If success
clears 0.96 the D rungs continue from this clone.
