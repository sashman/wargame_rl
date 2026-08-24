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

---

# CORRECTIONS — 2026-08-24, same day, after two expert panels

Two uncoordinated panels (14 agents) audited this report. The instrument survived —
three panels tried to break `measure_maps` and failed, and every published figure was
recovered from the raw files to the second decimal. **Every defect below is in the
inference, not the measurement.** The originals above are left visible.

## ⚠ RETRACTED: "five rounds cannot tell six trained agents apart"

This was the report's decisive claim and it is **wrong twice over**.

1. **The noise term was the wrong one.** It used mean within-cell episode sd / sqrt(270),
   which omits the **seed x map interaction** entirely. A two-way seed x map ANOVA gives
   interaction sd 4.32 (take r20) and **18.32** (ashoot r20), and **0.00** at both
   five-round cells. The two biases therefore run in *opposite directions by horizon*:
   at twenty rounds the omitted interaction makes the published noise too small, at five
   the shared-layout component makes it too large. Both inflate the headline ratio.
   Corrected, the collapse is 12.13 → 0.81 and 22.88 → 0.82 — still large, ~13% of what
   was claimed being artefact.
2. **On `held` — the primary readout this very report designated, precisely because raw
   vp is not comparable across horizons — the seeds separate slightly BETTER at five
   rounds**: F(5,40) 7.96 → 9.70 (take) and 14.51 → 9.47 (ashoot). And the fixed-policy
   control that should have been run (four scripts, which do not train and so carry no
   transfer confound) separates **better** at five rounds too: F 33.68 → **57.54** and
   23.93 → **70.44**.

⚠ **The decisive table compared raw vp across horizons, which this report's own
pre-registration forbids in bold.** Normalised, the collapse is 3.2–5.9x, not 17x and 31x.

**What survives is the report's OTHER argument**, and it is a claim about *scoring*, not
resolution: the agent's entire edge is denial, denial accrues per scoring event, and five
rounds has four events against twenty's nineteen. Five rounds may still be the wrong
place to train. **It is not closed by the evidence in this report.**

## ⚠ The comparator switches identity between the two cells being compared

The "best script" is selected by **vp** while the readout is **held**, and its identity
changes between horizons on both configs (take: `deny` at r20, `take` at r5; ashoot:
`take` at r20, `deny` at r5). Fixed to `squad_march_take`, the shortfall reads
−0.728 → −0.815 (grew **12%**, not 67%) and −1.809 → −1.374 (shrank **24%**, not 37%).
Fixed to `deny`, the *sign of the change flips* on ashoot.

**The verdict is robust — no comparator rule yields a ≥50% shrink on both opponents, so
NOT CLOCK stands. Every quoted magnitude above is a comparator artefact.**

Selection bias, measured properly (bootstrap over 9 maps, 2000 resamples, plus an
exhaustive 126-split half-sample): **+1.4 to +2.9**, and it **inflates the script**, so it
makes the agent look *worse*. It explains 20% of the +13.0 and 1.8% of the −74.3.

## ⚠ A pre-registered voiding clause fired and was overridden after the data existed

The pre-registration said: *if the rounds=20 rows do not reproduce, neither horizon can be
read.* One row moved 13.2 vp. The rescue offered — both horizons measured in one sweep on
one revision — is substantively correct for the *within-sweep* comparison, but the clause
had **no tolerance band and no power check**, and it was reinterpreted once its trigger
was known. That is the second occurrence in five days of the meta-error this project
logged on the advance lever's −8 bound.

## The staleness is FOUR TIMES larger than reported, and one cause was never named

Re-measured independently twice — by the panel and by me, agreeing row for row. Six
seeds, held-out nine, n=30, K=3, refereed, today's code:

| opponent | agent | best script | gap now | t | signs | gap published | moved |
|---|---|---|---|---|---|---|---|
| `squad_march_deny` | +20.0 | −6.1 (`take`) | **+26.1** | +3.51 | 7/9 | +35.4 | **−9.3** |
| `squad_march_take` | +19.4 | +6.5 (`deny`) | **+13.0** | +1.44 | 7/9 | +26.1 | **−13.1** |
| `squad_march_shoot` | +33.2 | +27.7 (`deny`) | **+5.5** | **+0.58** | **3/9** | +16.2 | **−10.7** |
| `contest_and_spread` | +16.7 | +30.5 (`take`) | −13.8 | −1.61 | 4/9 | −9.5 | −4.3 |
| `advance_and_shoot` | +61.4 | +135.6 (`take`) | −74.3 | −6.98 | 0/9 | −75.9 | +1.6 |

The agent moved **−6.4 / −5.7 / −6.0 / −4.1** on the four `squad_march` opponents and
**exactly 0.0** on `advance_and_shoot` — a clean one-directional signature confirming the
"which side the changed policy is on" reading.

⚠ **THE AGENT NOW BEATS THE BEST SCRIPT SIGNIFICANTLY ON ONE OF FIVE OPPONENTS, NOT
THREE.** The `shoot` row is a null (t=+0.58, 3 of 9).

**Bisected, which nobody else thought to do.** `squad_march_deny` on the take config reads
**−1.1 at the publishing commit** — the published value, to the decimal — and +6.5 at HEAD,
in two steps: the endpoint rule **+5.0**, and **`d607561` "the wholly-within check was
unsound wherever a zone edge is not axis-aligned"**, a deployment-placement fix **nobody
named**, **+2.6**. The command-phase change contributes **0.0**. This report named two
causes; one contributes nothing and the second-largest was in no account.

⚠ **This project owns the strongest attribution instrument available — deterministic
scripted policies, fixed seeds, and git — and did not use it.** A worktree at an old
revision costs about a minute per point and resolved the whole question in six.

## Two further measurement facts, both of which resize the project's doctrine

- **Per-episode `vp_margin` sd is 51–83 on these map-pool configs** (scripts 80.9–83.1 on
  the take config, agents 62.3–67.1), against CLAUDE.md's "~45–50" that sizes every n and
  every gate here. Low by ~1.7x on the configs that matter.
- **The ~6 vp resolution floor is false for the SCRIPTS and true only for the AGENT.**
  Between-table sd is 0–6 for scripts against 8.5–22.0 for the agent (F 1.49–4.60). The
  learned lineage is table-dependent; the scripts are not. That is the inverse of the
  reason on file.

## The best new measurement in the packet, which nobody had

`held` and VP per scoring event, by round, take config. `squad_march_take`: held
2.28 / 2.61 / 2.81 / 2.74 / 2.73 / 2.70 at rounds 2/5/8/12/16/20. Agent s1:
1.92 / 1.93 / 2.06 / 2.19 / 2.13 / 2.10.

**Both plateau by round 8. Twelve of twenty rounds are a constant-rate replay of a frozen
board — and the agent's allocation is fixed by round 2**, gaining +0.18 objectives over
the remaining eighteen rounds against the script's +0.53 by round 8. That is the sharpest
statement of the search failure on file and it is worth more than the two-point horizon
comparison this report was built on.

## Rules earned

- **Fix the comparator by name before measuring, and select it on the statistic you will
  report.** A comparator chosen by argmax on the same data changes identity between cells
  and turns a magnitude into an artefact.
- **A variance decomposition across two designs needs the interaction term.** Omitting it
  biases in opposite directions in the two cells and inflates the ratio between them.
- **Bisect a staleness claim.** Scripted policies are deterministic and git is free.
- **Run the fixed-policy control.** Four scripts that do not train would have refuted the
  resolution claim in one sweep.
