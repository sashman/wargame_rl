# Five rounds does not rescue the agent, and cannot tell six of them apart

**Measured 2026-08-24. No GPU.** M4 of the five-round plan — the one step never run.
Pre-registered in
[the companion file](2026-08-24-five-round-screen-preregistration.md), committed
before any number below existed.

Held-out nine, n=30, seeds 700000+, **decode K=3**, both configs **refereed**
(`enforce_move: revert_unit` + `attrition`). Six `-newmaps` seeds at `last.ckpt`,
four scripts, two opponents, two horizons — 40 runs, none failed.

## Verdict against the pre-registered criteria: **MIXED**

CLOCK required the `held` shortfall to shrink by **≥ 50% on both** opponents. It
shrank by 37% on one and **grew by 67%** on the other.

| opponent | rounds | agent `held` | best script `held` | shortfall | t | ahead on |
|---|---|---|---|---|---|---|
| `squad_march_take` | 20 | 1.98 | 2.46 | **−0.49 ± 0.18** | −2.67 | 1/9 |
| `squad_march_take` | **5** | 1.80 | 2.61 | **−0.81 ± 0.04** | −20.61 | **0/9** |
| `advance_and_shoot` | 20 | 2.03 | 3.84 | **−1.81 ± 0.14** | −13.02 | 0/9 |
| `advance_and_shoot` | **5** | 2.14 | 3.28 | **−1.15 ± 0.11** | −10.50 | 0/9 |

The label is MIXED and it stays MIXED. But the criteria were written to answer one
question, and on that question the data is not mixed at all.

## What the data does say

**1. `held` is very nearly horizon-invariant, for everybody.** Quarter the game and
the agent goes 1.98 → 1.80 and 2.03 → 2.14; the scripts go 2.46 → 2.61 and 3.84 →
3.51. **The agent's shortfall survives a game with no time to walk anywhere.** It is
not failing to *arrive* — it is failing to *spread*, and it fails to spread just as
badly when spreading is a four-round problem. That is the reading the plan called the
research read-out, and it points at allocation-as-search, not at the clock.

**2. Shortening the game makes the agent's position WORSE where it currently wins.**
Against `squad_march_take` it is **+13.0 ahead on 7 of 9** tables at twenty rounds and
**−5.8 behind on 0 of 9** at five. Its entire edge is denial, and denial accrues per
scoring event: four events cannot pay for it. The `advance_and_shoot` gap does narrow
(−74.3 → −7.3 raw; **−1.42 → −0.66** in per-episode sd) — but that is the *scripts*
losing the room to run away with it, not the agent gaining anything. It is still
behind on 0 of 9 and still 1.15 objectives short.

⚠ Raw vp is **not** comparable across horizons and is quoted here only inside a
horizon. Per-episode outcome sd is **61.7 → 12.6** on one config and 52.2 → 11.2 on
the other; cross-horizon comparisons are given normalised.

**3. THE DECISIVE ONE — five rounds cannot tell six trained agents apart.**

| opponent | rounds | between-seed sd | of which measurement | **true policy spread** |
|---|---|---|---|---|
| `squad_march_take` | 20 | 12.79 | 3.59 | **12.27** |
| `squad_march_take` | **5** | 1.05 | 0.76 | **0.72** |
| `advance_and_shoot` | 20 | 23.92 | 3.22 | **23.70** |
| `advance_and_shoot` | **5** | 1.01 | 0.68 | **0.75** |

Six independently trained policies collapse from a genuine 12–24 vp spread to **0.7**.
Measurement noise falls only 4.7x over the same change, so this is **not** a resolution
artefact that more episodes could buy back — the policies genuinely differ less. The
plan's own "low resolution" criterion is met and then some: **do not train at five
rounds.** Whatever six seeds learned that distinguishes them, they learned it about the
long game.

This also re-frames the earlier five-round result on file. `hold_deployment` scoring
−33.1 against a marcher's −0.7 showed the horizon separates a *degenerate* policy from
a competent one. It does not follow that it separates two competent ones, and it does not.

## ⚠ Side finding: the headline table is stale on one row, and a voiding note under-scopes itself

The twenty-round rows were run as an instrument check. One reproduced and one did not.

| row | published 2026-08-21 | measured now | moved |
|---|---|---|---|
| `advance_and_shoot` | agent +61.4, best script +137.2, gap **−75.9** | agent +61.4, best script +135.6, gap **−74.3** | 1.6 |
| `squad_march_take` | agent +25.1, best script −1.1, gap **+26.1** | agent +19.4, best script +6.5, gap **+13.0** | **13.2** |

The differentiator is **which side the changed policy is on**. On the `take` config
`squad_march_take` is the *opponent*, wrapped in `scripted_baseline`; on the other the
opponent is `scripted_advance_and_shoot`, a different family. Two documented changes in
the window could do this — the **endpoint rule** ("a move must end unengaged", global to
every config) and the command-phase scoring change (episodes ending early by
elimination). Neither is asserted here as the sole cause; what is established is that
the row moved.

⚠ **CLAUDE.md's voiding entry says the movement rule "changed on every config" and then
scopes the voiding to "every scripted-bar figure on an advance config".** That is
narrower than its own premise, and this is the counter-example: a non-advance config,
scripts moved **+7.6 vp**, the published gap halved. The other three rows of the
"Where the agent stands" table were not re-measured here and should be assumed stale by
a similar amount until they are.

**The five-round comparison above is unaffected** — both horizons were measured today,
in one sweep, on one code revision.

## What this changes

- **Five rounds is not a training scenario.** Closed, on resolution, not on degeneracy.
- **The offence deficit is not the clock.** It survives at a quarter of the horizon, on
  both opponents, at t ≥ 10. The critic-probe conclusion — reward is right, critic is
  right, the *policy* does not search — stands without the horizon caveat it could have
  had.
- **Re-measure the headline table** before any new arm is compared against it.
