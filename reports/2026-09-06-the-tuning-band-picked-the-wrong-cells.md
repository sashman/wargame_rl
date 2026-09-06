# `min_stack` is NOT confirmed — the tuning band picked the wrong cells

**Provenance.** 2026-09-06 · **GPU** · arm 4 seeds 1–6, `last.ckpt` at **epoch
1000** · `checkpoints/*-s{1..6}-sp2a1kb/last.ckpt` · **n = 180**, seed base
**700000** · configs
`configs/evaluation/25v25_maps_melee_approach_{refereed,vs_take,vs_deny,vs_shoot}.yaml`,
**refereed** · decode **K=3**, `verify_moves` on, charge decode on, reallocation
on, `max_redirects=1` · **paired per scenario**, six seeds · comparator
**`squad_march_take_charge`** at the same n=180 on the same scenarios · code
revision `cf6f6fd` · coherency 0.94–0.97 throughout · no Wandb run (scoring only).

Pre-registered in `reports/2026-09-06-min-stack-confirmation-preregistration.md`.

## Verdict: NOT CONFIRMED

The criterion, committed before these numbers existed: **positive with t > 2 on
`vs_shoot`**, and no other cell worse than −2 SE.

| cell | `min_stack=4` | `min_stack=2` | Δ paired | t | signs |
|---|---|---|---|---|---|
| `refereed` | −5.74 | −4.22 | **+1.51** | 2.40 | 5/6 |
| `vs_take` | +41.53 | +41.99 | +0.46 | 0.49 | 4/6 |
| `vs_deny` | +44.26 | +46.78 | **+2.52** | 3.47 | 5/6 |
| `vs_shoot` | +56.28 | +56.79 | **+0.51** | **0.45** | 4/6 |

**`vs_shoot` — the registered primary — is a null.** `min_stack` stays at 4.

## ⚠ The tuning transferred in NEITHER magnitude NOR location

The held-out band (seeds 900000+, n=90, three seeds, paired) said:

| | tuning band said | evaluation band gave |
|---|---|---|
| `vs_shoot` | **+5.2**, 3/3, t≈2.9 | **+0.51**, 4/6, t=0.45 |
| `refereed` | **−0.93** (slightly worse) | **+1.51**, 5/6, t=2.40 |

It predicted a large effect on the cell where there is none, and a small
negative on a cell where the effect is real. Had `min_stack` been tuned and
adopted in one step — which is the normal way a threshold gets chosen — a false
positive would have entered the record as a +5.2 win on the goal's blocking
cell.

## What this establishes, and it is not about `min_stack`

**The held-out-tune-then-confirm protocol did its job**, and it is the only
reason this was caught. But the tuning band was itself under-powered: n=90 with
three seeds gave t≈2.9 on an effect that is really ~+0.5. ⚠ **A tuning sweep
needs the same n discipline as the arm it feeds** — at a per-scenario sd of
81–89, n=90 has SE ≈ 9.4 per seed, and three paired seeds are not enough to keep
a threshold sweep honest.

## The ladder after this arm

With `min_stack=2` applied anyway (it is *not* adopted; shown because it is the
most favourable configuration measured to date), against the n=180 bar:

| cell | gap | t |
|---|---|---|
| `refereed` | +5.72 | 0.89 |
| `vs_take` | +7.19 | 1.17 |
| `vs_deny` | +9.98 | 1.48 |
| `vs_shoot` | **+0.32** | 0.05 |

**Still nothing resolved.** No decode configuration measured reaches the bar
significantly on any cell.

## Open, not claimed

`min_stack=2` is non-negative on all four cells and significant on two
(`refereed` +1.51, `vs_deny` +2.52). That is an **unregistered** finding on
cells that were not the primary, so it is a hypothesis, not a result — and by
this report's own argument it would need its own properly-powered confirmation
before adoption. It changes no verdict either way.
