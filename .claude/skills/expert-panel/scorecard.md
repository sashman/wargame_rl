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
