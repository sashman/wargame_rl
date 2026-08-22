# Freezing is friendly gridlock, and it is absorbing only for policies that mean it

**Date:** 2026-08-22 · **No GPU.** `just measure-freezing`, held-out nine,
`configs/golden/25v25_maps_two_mode.yaml`, n=5 per map, seeds 700000+, K=1.

## The finding

A model that asks to move and does not is invisible in every score this project
records. `vp_margin` sees the consequence, `coherent` sees the formation, and
nothing counts the order that evaporated.

| policy | ordered | frozen | truncated | delivered | P(f\|f) | P(f\|moved) | absorbing |
|---|---|---|---|---|---|---|---|
| `squad_march_take` | 4348 | 11.8% | 12.3% | 91.8% | **0.893** | 0.035 | **+0.858** |
| `squad_march_shoot` | 4705 | 10.3% | 12.9% | 92.0% | **0.888** | 0.030 | **+0.858** |
| `contest_and_spread` | 6776 | 11.2% | 12.3% | 92.2% | **0.892** | 0.028 | **+0.864** |
| `random` | 11570 | **3.7%** | **27.5%** | 86.3% | **0.121** | 0.035 | **+0.086** |

**The headline is not the 11%. It is the 0.89.** A frozen model stays frozen
**89% of the time**; a moving model freezes **3%** of the time. This is not
"moves are occasionally delayed" — it is a subset of the army getting stuck and
**staying** stuck.

Aggregate delivery of ~92% is why it went unnoticed: in total inches, movement
looks nearly fine. The loss is concentrated in a small population that never
recovers.

## `random` is the control that identifies the cause

`random` **truncates more than twice as often** (27.5% v 12.3%) and delivers
**less** of its ordered distance (86.3% v 91.8%) — yet it is the only policy
that does not get stuck: absorbing **+0.086** against +0.858.

So the collision system is not the fault. **A blocked random policy tries a
different direction next phase and escapes. A purposeful policy re-issues the
same blocked order forever**, because what it wants has not changed.

⚠ **Freezing is therefore a property of DETERMINISM meeting an obstacle, not of
the obstacle.** Any measurement that compares a trained policy to `random` on
movement delivery will read backwards.

## The obstacle is FRIENDLY

Frozen against moving model-steps, on three held-out tables:

| | frozen (n=49) | moved (n=566) |
|---|---|---|
| friendly bases in contact | **1.27** | 0.32 |
| enemy bases in contact | 0.22 | 0.03 |
| **≥ 1 friendly touching** | **91.8%** | 27.7% |

Friendly bases may be **crossed but not ended on** (`domain/movement.py`), so a
model whose destination is already taken backs off — and `_advance` returns
`start.copy()` when it cannot find room. Enemies barely feature.

**This is the stacking finding's mechanical consequence.** The agent puts
**4.90 models on its top point** against `squad_march_take`'s 2.73; stacking
produces contact, contact produces gridlock, gridlock removes models from play.
It also bears on **"the agent never stands still" (0.4% STAY** against the
scripts' 38–57%): a meaningful share of its army is not choosing to stand still,
it is stuck, and the STAY statistic cannot tell the two apart.

## What this does NOT say

- **Not measured: the effect on score.** Nothing here shows that fixing freezing
  is worth vp. A model frozen *on an objective it already holds* loses nothing.
- **Not measured: trained agents.** All five rows are scripted. The agent stacks
  harder than any script, so its rate is plausibly worse — but that is a
  prediction, not a result.
- **Not new physics.** The tangential-slide fix was tried in 2026-08-10 and
  measured **worse** (`squad_march_shoot` 0.70/+20.6 → 0.57/+1.0), because a
  fully blocked model spends its whole move sliding sideways into the open. Do
  not re-run it. That report's own conclusion — "the real fix is on the policy
  side, distinct target slots around an objective rather than aiming every model
  at the centre" — is consistent with everything above.

## Why it was measured now

The advance move is the longest move in the game, so it is the most likely to be
truncated or stopped. An advance arm could measure "the feature does not help"
when the moves never executed. **Read `absorbing` beside any movement feature's
result.**

## Reproduce

```
just measure-freezing squad_march_take configs/golden/25v25_maps_two_mode.yaml 5
just measure-freezing random            configs/golden/25v25_maps_two_mode.yaml 5
```
