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

## 2026-08-24 — melee scope + five-round statistics (mode: audit, two panels, 14 agents)

**Nomination audited:** melee v1 (charge-as-target-slice, auto-resolved fight) and the
five-round screen published the same day.

**Verdict: WRONG SCOPE (design panel) / BUILD SOMETHING ELSE FIRST (spend panel).**
Both panels reached it by disjoint routes, and the red team agreed with the conclusion
while **rejecting most of the arguments** — the strongest convergence available, because
the verdict is robust to which argument you delete.

### Audits that landed (all verified by me before acting)

1. **The melee premise was factually wrong.** "Engagement suppresses shooting, so contact
   is a pure loss" — engagement is **0.0000%** and structurally unreachable. My own probe:
   60,520 model-pair observations, zero engaged, minimum edge-to-edge gap **exactly
   1.0000**. Melee does not rebalance a priced trade-off; it creates a state the env has
   never entered. ⚠ **The evidence was in my own brief** (I quoted 7.52% → 0.00% and did
   not draw the implication).
2. **Design claim 1 (a charge needs no direction) — FALSE on the rules alone.** "Engaged
   with all targets, not engaged with any non-target" are joint constraints on where the
   unit *ends*: they pin a radius, not a bearing. I read a constraint set as a solution.
   And whoever computes the bearing writes a **fourth movement solver**, forbidden by name.
3. **The headline table was stale on FOUR of five rows, not one**, and the agent now
   clears the best script on **one** opponent, not three. Reproduced by me row for row.
4. **The red team BISECTED it** — the one thing nobody else did. Two causes, one of which
   (`d607561`, the wholly-within deployment check, +2.6) was in **no** account.
5. **"Five rounds cannot tell six agents apart" — refuted on the pre-registration's own
   designated primary readout.** The noise term omitted the seed x map interaction, and
   on `held` the seeds separate *better* at five rounds.
6. **My melee gate could not have failed**: P(pass | melee does nothing) = **0.50**. That
   is verbatim the defect logged five days earlier on the advance lever's −8 bound.
7. Per-episode vp sd is **51–83** on the map-pool configs, not the doctrine's ~45–50.

### Correlated error the chairs caught — the rule working as designed

- **Every charge-geometry number in all four design-panel lenses** was computed on
  trajectories from policies **that never charge**, forced unengaged by the very solver
  melee would exempt. Four panels agreeing was **one invalid measurement repeated four
  times**. Biased low in the dangerous direction, which killed one panel's own central case.
- **"`normalized_round` is out of distribution"** — two lenses, full confidence, same
  upstream premise, **neither did the arithmetic, both wrong.** The four-experts-one-premise
  failure occurring *inside* the document that warns about it.
- **"Melee fires on ~1% of model-steps"** — two lenses quoting a stale figure while
  `_melee_census.py` sat unrun on disk. Real answer: 0.0000%.
- **"Melee costs +41% throughput"** — priced `skip_phases: []`, which the plan does not
  propose. Actual: **+19%**. The anti-melee case doubled its own strongest cost figure.

### Cost of the arguments I should NOT repeat

"It voids everything" is too broad — reward and observation goldens live on non-maps
configs and stay bit-identical, `ratings/` holds only a README. The real casualty is
narrow and permanent: **the checkpoint corpus can never warm-start or pair against a
melee config**, because a new slice changes the head shape.

### Running score

Generate mode: still 0 for 3. Audit mode: **~13 of 13.** Two of the seven landed audits
here were against claims I had published hours earlier, and one was against a claim I had
already tried to correct and got backwards.

## 2026-08-25 (second round, same day) — the charging bar and the melee pre-registration

Two uncoordinated panels, 8 agents each, against a **pinned worktree** (the fix from the
round above, applied). Target: the instrument I built to unblock melee, plus the gate I
pre-registered for it.

**Audits: 7 of 7 landed. My score for the round: I shipped four defects and the panels found
all four.** Nominations still 0 for 3.

| finding | verdict after I verified it | action |
|---|---|---|
| **The blocker was closed on the bar and left open on the OPPONENT** — both configs seat a non-charging policy, so the arm trains in the unilateral cell | **CONFIRMED** in one grep. Their 2×2 at n=100: unilateral **+23.95**, mutual **−13.10** | both configs reseated; pinned by a test |
| **My charge decoder was a CHOOSER, not a filter** — `stands` rejects the all-STAY combination, which the env never judges | **CONFIRMED**: K=3 forced **47 of 112** model-charges against a strictly-STAY argmax; K=1 forced 0 | fixed, and the exemption had to appear **twice** — the verify loop overwrites `best`, so my first fix changed nothing and the probe still read 47 |
| **My pre-registration's requirement 1 was unsatisfiable** — `exclude_engaged_targets` hardwired to `melee.enabled`, no config field | **CONFIRMED** | `melee.shield_engaged_targets` added; the ablation then ran for the first time |
| **RETRACTED: "28.2% incoherent, so it is the RESOLVER — do not attempt a fourth movement fix"** | **CONFIRMED wrong on all three clauses.** My own re-derivation: **82.2%** were already incoherent *before* charging; 8 of 135 broken by the move | one policy-side clause: standing fraction **0.628 → 0.887** |
| **My gates are cleared by a network that learned nothing** | **CONFIRMED, worse than stated.** Untrained at K=3: standing fraction reaches **0.815**; `declared` swings **7–15×** between K=1 and K=3 on identical weights | thresholds rewritten against the measured floor |
| **The observation columns forked the melee family off the entire checkpoint corpus** (61 v 63) | **CONFIRMED** — and their fix (make it 63 everywhere) would have orphaned every existing checkpoint instead | `melee.observe_charge` escape hatch; pinned |
| **The arm may measure the COORDINATION CEILING, not the mechanic** | **CONFIRMED**: perturbing **one of five** members to a different *legal* rung halves the standing fraction (0.842 → 0.472) and costs **27 vp** | the clone control is now the decisive next experiment, at ~1 GPU-hour against the arm's 49–74 |

### What the two panels disagreed about, and both were useful

Panel A said the gate is **unpowered**. Panel B said it **cannot fail**. Both are right about
different halves: the *vp* readout is underpowered, and the *mechanism* readouts are passed
by the machinery. I had designed against only the first.

### Corrected against me, in my favour

Doubt (c) — "the charging bar might be strictly bad, so using it as the bar flatters the
agent for free" — **died**, four ways. It is a *good* policy. But it died the opposite way
from the fear: it is good only against an opponent that cannot charge, which is the failure
mode the doubt was about, arriving by a route the doubt did not anticipate.

### Lessons this round adds

- ⚠ **A 2×2 run below its resolvable n is not a 2×2**, and the resolvable n is computable
  from the per-episode sd *before* you run it. I ran one at n=6 (SE ≈ 36 vp), recorded it as
  noise, and then reasoned about the mechanic from a single cell anyway. The standing rule
  "never measure a symmetric change with both sides changed at once" was on file; I followed
  its letter and not its point.
- ⚠ **Any behavioural readout that gates an arm must be floored on a random-init network
  through the arm's own selector path.** One inference run. It would have caught all three of
  my thresholded gates.
- ⚠ **"Run the within-policy control" applies to MECHANISM COUNTS, not just vp.** Third time
  a published explanation here was built against a missing control.
- ⚠ **A decoder that filters in one phase can CHOOSE in another.** `decode_topk=3` is a
  scoring convention everywhere else in this project; in the charge phase it was a treatment,
  applied to the arm and not to its dark control, landing inside the paired estimator.
- **Pinning the tree worked.** Both panels measured a fixed commit while I kept editing, and
  every number they reported reproduced.

## 2026-08-25 — melee implementation, AUDIT mode, two panels (8 agents each)

Target: the twelve-commit melee feature on `feature/melee-stage-0`, before anything was
measured. Panel A: rules fidelity / action space / architecture / exploitation / geometry /
tests. Panel B: measurement / optimisation dynamics / observability / throughput / scenario
design / this repo's own retraction history.

**The audits landed again — 4 of 4 acted on.** Score to date: audits **~17 of ~17**
(the seven from 2026-08-24 above, plus these four and the three below that were verified
and *rejected* — a rejection the panel earned is still an audit that worked); headline
nominations still **0 for 3**.

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

## 2026-08-25 — audit — "is the melee arm worth training, or does the GPU go to allocation?"

Two uncoordinated panels, 14 agents (5 experts + red team + chair each). Red team A: **FATAL**.
Chair A: DROP_MELEE. Chair B: MORE_FREE_EVIDENCE_FIRST. **Both chairs: do not train the arm as
specified.** Panel votes: 3 DROP / 2 SPEND_ON_ALLOCATION / 5 MORE_FREE_EVIDENCE_FIRST.

**Audits that landed — 5 of 5, every one verified by me in code before acting.**

1. ⚠ **The arm's estimator is MIS-AIMED, not underpowered.** `melee.enabled: false` degenerates
   the charging opponent to a walking one, so `arm − dark` is the **diagonal** of the 2×2. Range
   −29.4 to +2; **contains no PASS at any seed count**. The passing test and the config comment
   that certify the pair both *state the fact that breaks it*.
2. ⚠ **"The mechanic is worth about zero" read the diagonal** — a mirror match, near-zero by
   symmetry for any symmetric mechanic. Same-row it is +25.88 (t=3.99) and +31.33 (t=5.95).
3. ⚠ **The "declared unit may not STAY" rule is MASK-ONLY**, and `apply` takes no mask, so the
   bar could break a rule the agent cannot. `apply`'s own docstring names the hazard for the
   sibling rule. ⚠ **MAGNITUDE OVERSTATED ~8x: both panels said 14.7/14.8%, it is 1.8%** (2 of
   109 declared rows), and golden configs were clean at 0 of 2,660. **Two panels agreeing is
   not a measurement** — I published theirs before re-measuring, which is the correlated-error
   caution the chairs raised, applied to me. All four defects fixed; digest still 9 of 9.
4. ⚠ **`charge_progress` never implements its declaration gate.** The non-charging script earns
   **36% more** from the charge term than the charging one. Fixing the gate does not rescue it —
   `declared_charge` is not an observation column. It is also in no config.
5. ⚠ **Two of four gates are cleared by declaring NOTHING**, and the gate as written rejects
   policies that stand *more* charges — non-monotone in competence.

**Correlated error, caught by both chairs:** three DROP votes shared ONE upstream premise (the
diagonal framing) and were counted as three. Five agents "replicating" the 2×2 on identical seeds
with deterministic scripts is **arithmetic, not replication** — the same error CLAUDE.md already
records. One panel committed the config-field-for-behaviour error that a *different panel in the
same batch* was diagnosing.

**Running total: nominations 0 for 3. Audits ~13 of ~13.** The audit seat has still never been
wasted; the generate seat has still never landed.

**Rules earned:**
- **Certify a paired control on BEHAVIOUR, not a config field.** Diff the episodes, not the YAML.
- **Name the ESTIMAND before the n.** Every power calculation here — mine and three panels' —
  was arithmetic on the wrong quantity.
- **A mirror-match diagonal is near-zero by construction.** Only same-row means anything.
- **A gating readout needs a ceiling clause and a monotonicity check**, not just a floor.
- ⚠ **Re-measure a panel's number before publishing it, even when two panels agree.** Both
  overstated one figure 8x, and I carried it into two documents before checking.


## 2026-08-25/26 — follow-through: what the audits actually cost and bought

The two panels above led to a **rules-lawyer subagent** and a **game-AI generate subagent**, and
the follow-through is the part worth scoring.

**The rules lawyer landed 3 of 3**, and its findings were larger than either panel's:
the charge's *while-moving* condition implemented **nowhere** with no gap-map row; the
declaration and 2D6 cap enforced **only in the mask** while `apply` takes none; and a gap-map
row rated `implemented` that was mask-only. It also **refuted** three of my own suspicions
(no seat asymmetry across 2,580 mask rows; Strikes First correct 37/37; no stale
`previous_location`) — negative results I would otherwise have chased.

**The generate seat broke its 0-for-3 streak**, and the reason is instructive: it ran probes
instead of theorising, and **killed its own headline proposal** (a selective-charge veto,
−3.56 ± 6.47). Its best finding — that the binding constraint is the ANGLE, not coordination
— survived my check. ⚠ But it **overstated the STAY-exemption magnitude 8x** exactly as both
panels had, and I published their figure before re-measuring. **Two panels agreeing is not a
measurement.**

**Running total: nominations 1 for 4. Audits ~16 of ~16.**

### Rules earned in the follow-through

- ⚠ **A probe needs a known-answer row.** My `p^k` falsifier counted the referee's REVERT as a
  stand and reported the teacher at **0.079** against its own measured 0.947. The teacher row
  is what caught it within one run.
- ⚠ **Kill your own hypothesis with the falsifier you wrote for it.** `p^k` matched a measured
  0.083 at `0.448^3.47 = 0.062` and was wrong: collapsing k to 1 buys 2x on average, **nothing**
  on one seed, and tops out at 0.42 against 0.92.
- ⚠ **Re-read the primary source before implementing.** *"Fighting after death"* is a granted
  ABILITY, not a default; implementing it as one would have made the game incorrect. The
  register's stated reason was wrong too.
- ⚠ **Check the inventory against the code before working an item.** One listed as missing was
  already implemented and already tested. The inventory has now been wrong in **both**
  directions in one day.
- ⚠ **A monitor that greps for success is indistinguishable from silence.** One armed here
  watched for `Epoch 599` in a log Lightning writes with carriage returns; it would have sat
  quiet for hours and read as "still training".

## 2026-08-26 — melee: "the agent is learning the game and not the charge" (mode: audit)

**Target:** the mid-flight reading that would decide the next 20–40 GPU-hours, plus the four
measurements under it. Two uncoordinated panels, 7 lenses each, both with the dual-mandate red
team. Run deliberately *while* the arm was still training so neither panel saw the outcome.

**Verdict: the single most valuable round run here.** Panel B returned **FATAL** on the primary
readout and it was **real** — `scripts/measure_charges.py` counted pile-in, consolidate and the
whole opponent turn as evidence a charge had stood, inflating an untrained network **+300%** and
the scripted teacher only **+2.9%**. Anti-monotone in competence. Four further defects, all
verified in the code by hand before acting: `charge_progress` gated on a clock that had already
advanced so it could not fire on a charge at all (**the lever the next spend would have wired**);
"held-out nine, n=9" visits **five** tables (uniform draw *with replacement*); the PASS gate's
coherency clause never sees the charge phase; the comparator under-declares on a stale cap.

**The headline claim was UPHELD and every number used to argue it was wrong** — in the direction
that weakened the author's own case.

**The lesson that generalises, and it indicts the panel format itself:** six of Panel B's seats
came in *against* the headline, and all six were wrong for one shared reason — they trusted
`measure_charges.py`'s **docstring**, which lists three of its author's past measurement errors
and therefore reads as audited. One seat touched the instrument and still landed one checkpoint
away from the right answer. **Discount a panel's agreement to the number of seats that
independently validated the instrument.** Convergence among seats reading the same unvalidated
gauge is correlated error with extra steps.

**Process notes.** Naming the instruments *by path* in the brief and ordering the red team to
read them is what produced this; the same mandate produced this project's previous best round.
Two throttling defects in my own brief: "run at most one measurement at a time" bounds per-agent
load and not the aggregate (14 agents obeying it produced load average 160), and advising
`CUDA_VISIBLE_DEVICES=` breaks checkpoint loading outright, since checkpoints saved on CUDA need
`map_location`.

## 2026-08-26 — audit x2 (staggered): "is the melee approach right at a high level?"

**Mode:** audit, two uncoordinated panels of 6 seats + red team + chair (16 agents total).
Panel A: curriculum / mechanic-economics / action-space / rules-fidelity / strategy /
methodologist. Panel B: credit-assignment / incentives / abandon-adversary / statistician /
optimisation / architecture, fed Panel A's findings per §6.

**Both red teams returned FATAL. Both chairs answered the commissioned question identically:
the goal is right, the implementation HINDERS, keep building melee and stop training on it.**
6 of 6 Panel B seats said HINDERS.

**Audits that landed** (chair verified each in source or by running code):
- `_rolled_for` leaks across `reset()` (`wargame.py:193`, `:1260-1262`); `from_env`'s unseeded
  reset poisons episode 0 of every network-building measurement and none of a scripted one.
  Reproduced: opponent charge rolls 0.0 on all 25 models.
- `pile_in`/`consolidate` are **dead by construction** — `short_move_legality` offers the move
  only to engaged models and `pile_in.py:353` pins exactly those models, ONE PREDICATE USED
  TWICE IN OPPOSITION, at a radius 1.0" wider than `12-fight-phase.md:35`'s base contact.
  40 of 140 decision steps decide nothing.
- **`engaged` is nowhere in the observation tensor.** The model token carries `charge_roll`,
  `fell_back_this_turn`, `declared_charge` — the three states carrying none of the value —
  while the mechanic's entire measured worth is the shooting shield. The project's own
  cheapest standing desk check, failed for the central mechanic.
- `_record_coherency` fires only under `phase == BattlePhase.movement` (`wargame.py:1553`):
  the `coherent` PASS clause covers 0 of 3 melee phases.
- Episode length is **140 steps (7 phases x 20 rounds), 3.5x the golden 40** — not the 100 the
  brief claimed. A `skip_phases` change is a reward-shaping AND discount change, invisible in
  a config diff.
- The comparator's declaration gate still uses the retired `min(move, roll)` cap
  (`scripted_squad_march.py:307`) while aiming uses the full roll (`:497`).

**Nominations REFUTED (0 for 3 again, and one was the brief's own):**
- "The encoding costs 40 vp" — t=1.08 at n=9; 11 consecutive same-policy blocks span **63.9 vp**.
- "Rigid charge raises discovery 0.0006 -> 0.153" — numerator and denominator measured on
  different populations; same-population it is 7.0-15.5x. Direction real, ratio an artefact.
- "vp_gain's income share fell 0.38 -> 0.06" — measured 0.143 -> 0.121. Three Panel B seats
  "CONFIRMED" the compound claim by verifying only its true half.

**Correlated error, and the headline lesson:** "retune gamma" was proposed by **6 of 6 Panel A
seats and 3 of 6 Panel B seats — 9 of 12 across two panels that never met — and is wrong both
times for two different reasons.** Panel A also had 6 of 6 agree the ladder fix was worth
+3.5%, all lifting one stale line from one file (~4x wrong). ⚠ **Unanimity was worth LESS than
four-way agreement**, because the unanimous items came from a single shared source.

**Chair errors caught by the panels** (the brief was written by the audited engineer):
`beyond_move_ladder` was described as an open decision and had **already shipped** at
`5da54ed`; episode length was given as 100 and is 140; `docs/melee-teaching-goal.md` §10 is
stale on both counts.

**Panel B's own headline was WRONG and the chair caught it**: it read
`25v25_maps_melee_refereed.yaml`'s "do not launch an arm on this pair" as forbidding the
running arm. That warning forbids an **arm-vs-dark-control pairing**; no dark control is in
flight, and scoring against the scripted bar is what the pre-registration prescribes. **Verify
a panel's headline, not just its findings.**

**Cost:** ~3.0M subagent tokens, 939 tool calls, ~1h50m wall-clock staggered. Load stayed
<=3.5 on 24 cores; the trainers lost nothing.

## 2026-08-28 — audit — the v9 verdict (C1 null / C2 no-internalisation / C3 family claim / C4 clause binding)
- Panel A (6 seats + red team + chair, 8 agents, ~1.03M tokens): C1 SURVIVES REWORDED
  (underpowered null, MDE ~22 — the "null" wording claimed unearned precision); C2 SURVIVES
  ON CORRECTED EVIDENCE ("gain grew" struck as an arithmetic identity, t=+1.50; replaced by
  the far better-powered decoder-activity statistic, nominated/ep paired +0.13 ± 0.96 sd);
  C3 REJECTED unanimously (v9 tested one weak, need-blind dose — ~5% clean, kill-box at
  p=1.0 — not the distribution family; and the general claim contradicted v10 itself);
  C4 SURVIVES WITH DISCLOSURES (clause retired as a template: no grid named, 34.4%
  false-fire, half unmeasurable at verdict time).
- Audits that landed: the executed primary was contest_ungated, not the described composed
  decode (mode now quoted like K); the v10 header's "unfarmable" is FALSE in code (shuttle
  farm, verified by hand afterwards — clip-at-zero + free redeclaration); the hoarding
  clause closed 0/6 by a panel probe; measure_charges docstring cite stale (:1160 → :1676);
  "600 beats 1000 on everything that ships with the decode" surfaced from the panel's own
  recomputation; the methodologist's σ_run ≈ 15.6 was itself refuted by the red team + chair
  (misread file — the repeat pair is byte-identical), the second time a panel's internal
  check killed a panel claim before it reached the record.
- Pattern holds: audit seats landed ~everything; the one generative-style claim (a noise
  floor) was wrong. Chair-verification-before-synthesis caught it.

## 2026-08-28 — audit (shadow) — panel A's own v10 landing protocol
- Panel B (4 seats + chair, ~0.58M tokens, fed A's synthesis): every seat landed. The
  headline: **panel A's draft clauses reproduced the exact defect class panel A had just
  retired** — the +5.4 threshold was an 18/24-cell SE mix (→ +6.7); the "<5" charge guard
  sat inside the bimodal 2.82–17.33 comparator band (≈91% false-excuse rate); the
  decorative screen's 6/6-boxes reading was miscalibrated under any sd assumption; the
  farm screen's army-mean income conjunct was provably blind to a one-unit farm (39% of
  cap); the internalisation conjunction routed a +10–13 vp term to family death with
  P=0.991 (→ funding follows A; B∧C label only).
- Constructive survivors: conjunct-B threshold 6.6 vindicated by a measured positive
  control (script 5.11); the −52 charge floor reached independently by two seats; the
  decode-admissibility line (Level-N verdicts admit only sub-N decode content) which
  re-scored §27's "Level 2 — MET" to PARTIAL.
- Meta-lesson written into §31 itself: pre-registration fails at the moment of WRITING —
  apply grid-by-name / false-fire-rate / instrument-existence checks to the prescription,
  not just the experiment.

## 2026-08-29 — audit — the v12 plan (home-objective diagnosis + hold term + value/threat columns)
- Panel A (6 seats + red team + chair, ~1.25M tokens): C1 mechanism CONFIRMED, causal framing
  REFUTED (home neglect predates the term; total income already favours garrisoning and is
  declined; home declared at chance, not below); C2 UNADJUDICATED (the spread screen was the
  wrong estimator on a saturated platform — struck, not underpowered); C3 FATAL as specified
  (mission-pricing ill-posed under the count-cap — verified at vp_calculator.py:111; the term
  double-pays a declined state; free redeclaration = launderable annuity), one stripped
  descendant survives conditionally (constant-pot declared_objective_hold); C4 DIES both
  columns (value = zero bits under the single mission; threat = mis-signed at objective cells
  and false-safe vs charges); C5 FATAL, triple-confirmed by concordant probes (apply(_init_weights)
  re-draws every Gaussian on any width change; "rows" was the wrong axis too) — repair harness
  specified, mandatory for any future column.
- Audits that landed: the spread-screen dichotomy in the brief was FALSE (wrong estimator, not
  blind-vs-real); the split instrument cannot name an objective (a CLASS blind spot across
  three instruments); two seats' census kill-gates would have fired against an already-dead
  claim (divergence caught by the probe); v11 inherits the zero-at-adjacency bias and its
  census must measure declared-target distances.
- Survivor: the turn-0 garrison pricer (script prices doctrine, n=90 × 3 seed bases) as C2's
  gate; v12-lite only if it clears. Author's plan mostly demolished before any code existed —
  the cheapest demolition on record.
