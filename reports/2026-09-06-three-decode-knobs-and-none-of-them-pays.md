# Three decode knobs, and none of them pays

**Provenance.** 2026-09-06 · **GPU** (grid) and **CPU** (tuning sweep) ·
arm 4 seeds 1–6, `last.ckpt` at **epoch 1000** ·
`checkpoints/*-s{1..6}-sp2a1kb/last.ckpt` ·
**n = 180**, seed base **700000** (grid) and **900000** (tuning) ·
configs `configs/evaluation/25v25_maps_melee_approach_{refereed,vs_take,vs_deny,vs_shoot}.yaml`,
**refereed** (`enforce_move: revert_unit`, `attrition: true`) ·
decode **K=3**, `verify_moves` on (the default), charge decode on, reallocation on ·
**paired per scenario** against the same seeds' `(decode_stay=0, max_redirects=1)` runs ·
comparator named: **`squad_march_take_charge`**, measured at the same n=180 on the
same scenarios · opponents per cell: `squad_march_take_charge` / `squad_march_take` /
`squad_march_deny` / `squad_march_shoot` · code revision `c611200` ·
coherency reported below · no Wandb run (scoring only, no training).

Pre-registered in `reports/2026-09-06-two-decode-knobs-preregistration.md`,
committed before any of these numbers existed.

## Why these were tried

After `reports/2026-09-06-the-ladder-was-measurement-noise.md` retracted the
ladder, the gaps at n=180 were `refereed` +4.21, `vs_take` +6.73,
`vs_deny` +7.45, `vs_shoot` −0.19 — nothing resolved, and about **+10 vp of
real improvement** needed. Every large gain in this goal has come from the
decode, not from training, so the decode's unmeasured knobs were the cheapest
place to look.

## Result — the pre-registered grid

Paired per scenario, six seeds, four cells. The rule committed in advance:
adopt only at **t > 2 and ≥ 5 of 6 seeds, averaged over the four cells**.

| knob | refereed | vs_take | vs_deny | vs_shoot | **average** | verdict |
|---|---|---|---|---|---|---|
| `decode_stay` | +0.74 | +0.61 | −0.27 | +1.00 | **+0.52 ± 0.31**, t=1.69, 13/24 | **not adopted** |
| `max_redirects=3` | −0.57 | −1.66 | **−4.33** | +1.42 | **−1.28 ± 0.76**, t=−1.69, 11/24 | **REJECTED** |
| both | +0.53 | −1.28 | **−3.99** | +2.55 | −0.55 ± 0.78, t=−0.71, 13/24 | not adopted |

**Iterating the reallocation is worse than not iterating it**, and significantly
so on `vs_deny`: **−4.33 ± 1.37, t=−3.17, 1 of 6 seeds positive** (−3.99 and
**0 of 6** with `decode_stay` on).

## ⚠ My own prediction was wrong, in the direction that matters

The pre-registration predicted `decode_stay` "+0 to +3" — **correct**, it landed
at +0.52 — and iterated reallocation as "**the larger of the two**, +3 to +8".
It is **negative**. I wrote the code, pre-registered it, and it lost. The
mechanism I assumed (more over-stacked points emptied per turn) is not what
happens; redirecting a second squad evidently gives up ground the first redirect
was relying on.

Two independent measurements agree, which is why this is a verdict and not a
noise reading: the held-out tuning band (seeds 900000+, n=90, 3 seeds) also put
`max_redirects=3` **below** `=1` at every threshold tried.

## The one thing that did move, and it is not enough

The tuning sweep (**held-out band 900000+**, so this is a hypothesis, not a
result — the evaluation-band confirmation is a separate, pre-registered run):

| `min_stack` | `vs_shoot` | `refereed` |
|---|---|---|
| 2 | **+71.83** | +2.20 |
| 3 | +70.73 | +2.07 |
| 4 (shipped) | +66.60 | +3.13 |

Lowering the threshold from 4 to 2 is worth **+5.2 on `vs_shoot`, 3 of 3 seeds**
paired — the rule fires far more often, which fits the diagnosis that the
agent's top stack averages **4.90** models and a threshold of 4 almost never
triggers. **It does nothing on `refereed`** (+3.13 → +2.20, i.e. slightly worse).

⚠ **+5.2 against the +13 `vs_shoot` needs and ~0 against `refereed`'s +8.** The
pre-registration's own stated expectation — "most likely both positive, both too
small" — is what happened, and lowering `min_stack` does not change it.

## What this establishes

**Three decode knobs have now been measured and the decode stack is at its
ceiling for this goal.** `decode_stay` is a null, iteration is negative, and the
threshold is worth +5 on one cell. Closing a +10 gap on two cells is not
available here; it needs a better policy, measured on the n≥180 instrument that
this day's earlier retraction established.

## What survives, and is worth keeping

- `apply_reallocation` gained `max_redirects` and
  `choose_surplus_reallocation` gained `exclude_groups`/`exclude_targets`.
  **Defaults are bit-identical** — verified end-to-end (seed 1 `vs_shoot`:
  +79.4, coherency 0.952, decl 11.13 / tried 9.40 / stood 6.82 either side).
  Kept as the measured-rejected control, the way
  `squad_march_take_charge_realloc` was.
- Coherency is essentially unmoved by every knob (agent 0.94–0.97 throughout,
  against the scripted 0.93).
