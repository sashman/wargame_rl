# Expert panel scorecard

One row per round. The point is to keep an honest prior on what panels are good for.

| date | mode | nomination | verdict | audits that landed |
|---|---|---|---|---|
| 2026-08-22 | generate | the per-model return is gated on survival, so the agent over-prices staying alive; pay the dead / redistribute by pivotality | **REFUTED** by `measure-critic-probe` — the critic prefers spreading at t=+8.3, both directions | edge-to-edge objective distance (the γ argument rested on the wrong metric); `grad_clipped_fraction` = 1.0, so `max_grad_norm` is the step size and was never swept; `measure_paired_policies.py` reports `t = 0.0` for a zero-variance difference; r = +0.991 is an accounting identity |
| 2026-08-23 | audit | `closest_objective_v2` + `fallback_to_nearest` pulls models to an objective's centre point, dominating STAY | **REFUTED** by `measure-shaping-gates` — those objectives are areas, distance is to the outline, and 43.5% of paid model-steps are already inside | `measure_angle_collapse` had no movement-phase filter and decoded shooting actions as headings; the clone control (a factored clone of the winning script scores *below* the agent, so the statistic measured architecture); consensus decoding forced agreement and lost 3/3 seeds; both briefed confounds checked and correctly dismissed |

**Running total — nominations 0/2, audits 8/8.** Default to `audit` mode.

⚠ Two panel claims were confidently wrong and cheap to check: "pays toward the centre point"
(refuted by one docstring and one config read) and "the existing control pairs for free"
(refuted by one init comparison — 73 of 110 tensors differ). **Verify load-bearing claims
in code before acting on them.**

| 2026-08-23 | audit | PR #245 (scripts advance, opponent advances, move must end unengaged, stale columns zeroed) — is the bar trustworthy enough to train against? | **BLOCKED the merge.** Three real defects, all reproduced independently before being accepted | (1) the bar table had **no error bar on the delta** — redone paired at n=30, **two of four rows wrong in SIGN**, `deny` −20.0 (t=−2.82, 1/9) not +1.3; (2) the engagement figure was wrong the *other* way — a hardcoded 2.26" ring fractionally larger than the env predicate counted every rescued model as still engaged, true figure **7.52% → 0.00%**; (3) **a shipped movement bug** — the back-off walked endpoints into friendly bases, 0.18% of pairs overlapping, and **six unit tests covered the function with zero `env.step` calls** so none could see it |

**Running total — nominations 0/2, audits 11/11.** Audit mode found a live bug in shipped
code and two sign errors in a published table, at zero GPU. Both panels reached the `deny`
result independently (−18.7 and −20.0, both 1/9).

⚠ **New standing rule earned here: compute the error bar on the quantity you are claiming,
not on its parts.** And: a test that never calls `env.step` cannot see a composition defect —
this project has now paid for that twice.
