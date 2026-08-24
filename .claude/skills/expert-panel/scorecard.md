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

**Second panel, same round, found the decisive one:** the 2×2 nobody had run. Advancing
costs its USER ~78 vp (−81.8 against −4.1); both-advance (−3.6) is indistinguishable from
both-walk (−4.1). **The published "+15.5 to the bar" was two self-inflicted wounds
cancelling.** Also: the OFF column had been measured on *different code*, and the
"gains most despite forgoing its shooting" highlight was false — it fires 3% MORE shots.

**Running total — nominations 0/2, audits 15/15.** Audit mode found a live bug in shipped
code and two sign errors in a published table, at zero GPU. Both panels reached the `deny`
result independently (−18.7 and −20.0, both 1/9).

⚠ **Three standing rules earned here: (1) compute the error bar on the quantity you are
claiming, not on its parts; (2) NEVER measure a symmetric change with both sides changed at
once — run the 2×2; (3) check both columns were measured on the same code.** And: a test that never calls `env.step` cannot see a composition defect —
this project has now paid for that twice.

## 2026-08-25 — melee implementation, AUDIT mode, two panels (8 agents each)

Target: the twelve-commit melee feature on `feature/melee-stage-0`, before anything was
measured. Panel A: rules fidelity / action space / architecture / exploitation / geometry /
tests. Panel B: measurement / optimisation dynamics / observability / throughput / scenario
design / this repo's own retraction history.

**The audits landed again — 4 of 4 acted on.** Score to date: audits ~12 for ~12; headline
nominations still 0 for 3.

| finding | verdict after I verified it | action |
|---|---|---|
| A corpse shields its whole unit from shooting | **CONFIRMED**, reproduced in 12 lines | fixed; `subject_alive` now required on the predicate |
| The 8.7-micro-inch premise is a MINIMUM read as a TYPICAL | **CONFIRMED**, and worse than stated: 0.0% of declarations within one speed bin, not 27.6% | retracted in 4 places |
| "Zero inches in the charge phase" is a vacuous control | **CONFIRMED** — the policy returns STAY for that phase regardless | retracted; the test now claims only what it proves |
| Register rows 62 and 92 are lying | **CONFIRMED** against the rule text and a live env | both rewritten |
| `_rolled_for` is never cleared by `reset()`, so `turn_order: player` leaves charge_roll at 0 | **DID NOT REPRODUCE** — rolls fire every episode under all three turn orders. The staleness is real; the failure it predicts is not | reported, not fixed |
| `_enforce_charge` trusts the action mask — 4 lenses rated FATAL | **OVERSTATED**; one lens rated it MINOR with the right reasoning, and no shipped actor takes that path | recorded, not fixed |

**What made the difference this round.**

- **The red team's dual mandate found the FATAL defect, and it was not in any proposal.**
  It came from running the gate the brief's own no-op proof could not: *melee ON with a
  policy that never charges must equal melee OFF*. 8 of 12 seeds differed. Naming specific
  instruments and telling it to distrust the brief is what produced this.
- **Naming my own suspected weak points in the brief paid.** I flagged the unverified
  0.02415 lethality target and the missing `charge_roll` observation up front. The target
  turned out **SOUND** — and three of five lenses "corrected" it WRONGLY by comparing
  per-round melee against a per-fight target. Volunteering the doubt got it checked properly
  instead of asserted.
- ⚠ **Correlated error appeared exactly where the skill says to look.** Three lenses
  converged on "melee is 1.92x too lethal" from one shared arithmetic slip. Two more
  converged on "the corpse bug IS the charge mechanism" — both measured it with a
  NON-CHARGING script, where live engagement is 0.0000% by construction, so the result is an
  identity. **Counting votes would have funded both.**
- ⚠ **Six hand-rolled charging scripts produced +6.5 to +88.8 vp for nominally the same
  measurement — a 14x spread.** Nobody measured "the value of melee"; each measured their own
  heuristic. Two chairs quoting +62.50 to the cent was *implementation* convergence, not
  independent confirmation. **Quote the ablation and the 2x2, never one arm's number.**
- Panel agents left 10 probe scripts and one config in the repo. Moved the scripts to the
  scratchpad; kept the config after verifying it myself — it is a genuinely pairable dark
  control, which this project rarely gets on an action-space change.

### Panel B (measurement / dynamics / cost / history) — same day

**Independently reproduced the corpse shield by a different route** (a live-episode
trace to a single wound at step 10 of seed 700001, vs Panel A's constructed state).
Two panels that never met, two methods, same defect — this is the one place the
convergence is *earned*.

| finding | verdict after I verified it | action |
|---|---|---|
| Schema 2.7's melee flags never reach a replay | **CONFIRMED, but not by their mechanism** — they said the flag was cleared before the exporter; it is actually absent from the DELTA codec. Full snapshots carry it fine | fixed; tautological test replaced |
| "Neutral" uses the wrong conditional — an engaged model is one that would certainly have shot | **CONFIRMED and it is the better critique.** ~10x, not ~1x | relabelled lethality-NEGLIGIBLE |
| My power figures are 50%-power CI half-widths, not MDEs | **CONFIRMED** — 25.97 at n=3, not 19.05 | corrected |
| `charge_roll` has no observation column; the mask is applied to final logits only | **CONFIRMED** at `net.py:711` — the trunk cannot condition on it, the critic cannot see it | recorded, not fixed (user's call) |
| The joint decoder is inert in the charge phase | **CONFIRMED** at `decoding.py:272` | recorded |
| Per-episode sd is 84-92 "on this config, not the 45-50 the rules assume" | **REJECTED by their own red team**: identical on the golden config, so it is the map-pool draw — and the wrong estimator anyway | not propagated |

**Lessons this round adds to the skill.**

- ⚠ **The two panels DISAGREED about my lethality figure and both were partly right.**
  Panel A's chair defended the arithmetic (correct); Panel B's red team attacked the
  conditional (also correct, and more important). Neither alone would have got me to the
  right answer. **Run both panels even when the first one looks conclusive.**
- ⚠ **A panel's mechanism can be wrong while its observation is right.** Panel B said the
  charge flag was cleared before the exporter ran; I measured it True in the snapshot, then
  found the real cause in the delta codec. **Verify the mechanism, not just the symptom —
  had I trusted the stated cause I would have "fixed" the clear ordering and broken it.**
- ⚠ **Five panels and I were mutating ONE working tree while measuring on it.** Panel B's red
  team caught this and re-ran everything against a pinned `git archive`. Every other number in
  both packets is dated to an unstated code state. **Pin the tree before a measuring panel.**
- **A tautological assertion is worse than no test**, because it occupies the slot where the
  real one would go. `assert flag in (True, False)` shipped and hid a live defect for a day.
