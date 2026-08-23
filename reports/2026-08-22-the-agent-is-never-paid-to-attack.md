# The travel reward never points at ground the opponent holds — criteria, written before any training

## What was measured first (no GPU)

`just measure-shaping-gates` on the v3.0 lineage (s1-newmaps, `25v25_maps_two_mode`,
held-out nine, n=20, K=3):

    objectives the travel reward can point at     35.1%  (1.99 of 5.66 per step)
    units given an objective of their own         32.0%  (1.60 of 5)
    model-steps paid toward an assigned target    36.9%
    model-steps paid toward NEAREST (fallback)    63.1%

    why an objective was not a candidate:
      we already hold it                            56.6%
      they hold it by 2+ (one arrival cannot flip)  43.4%

## The mechanism, located exactly

`_is_positive_transition` asks whether **one more model** changes the control
label. With `_state_label` being `player` at `p >= o+1`, `contested` at `p == o`
and `opponent` at `o >= p+1`, the only qualifying moves are
`neutral -> player`, `opponent -> contested` (needs `o == p+1` exactly) and
`contested -> player`.

So **an objective the opponent holds by two or more is never a candidate**. The
travel reward cannot point at it, the unit nearest it is left unassigned, and
`fallback_to_nearest` then pays that unit to close on its *nearest* objective —
which, for a stacked army, is usually the one it is already standing on, at
distance 0 and therefore for zero reward.

That is the offence deficit as a line of code: **the agent is never paid to
attack.** It matches every symptom on file — offence negative on 3/3 seeds in
every scenario, 54.4% of model-steps on objectives against the scripts' 75.5%,
and 53.7% of income arriving as global terms rather than per-model ones.

## The arm

`closest_objective_v2` gains **`contest_deficit: int = 1`** — an objective is a
candidate when the opponent's lead over us is at most `contest_deficit`, instead
of the hardcoded 1. At `contest_deficit: 5` a whole five-model unit arriving can
flip the point, which matches the fact that units move as units.

- **Default is 1 = today's behaviour exactly**, so every existing config, the
  control lineage and `tests/test_reward_golden.py` are untouched.
- **No tensor-shape change**, so the paired estimator holds and checkpoints stay
  loadable.
- **Observability desk check PASSES**: the term keys on per-objective player and
  opponent alive counts, and `observe_objective_control: true` supplies both —
  verified against the scoring definition on 2026-08-22.
- **It ADDS income rather than moving it.** The failed anti-concentration levers
  all lowered total objective income; this one points existing travel reward at
  ground that currently pays nothing. Nothing is taken away from held points.

## What the gate did and did not forbid

⚠ It never forbade the coordinated attack. Nothing in the rules stops the agent
sending three models at a point two defenders hold, and `objective_hold` and
`vp_gain` pay for it normally once it succeeds. What was missing is the
*shaping*: `closest_objective_v2` is the only term that pays a model to move
BETWEEN objectives, and its target could never be that point. So the agent had
to discover a coordinated multi-model assault from sparse downstream signal,
while `fallback_to_nearest` actively paid it to close on the point it already
stood on.

**The granularity is the unit, not the model.** `_compute_group_assignment`
gives a candidate objective to exactly one *group*, so `contest_deficit: 5`
reads as "could a squad take this?" rather than "could three models take this?".
5 is the squad size, and the coherency work established that models here already
move as units (0.938-0.955 coherent), so the shaped behaviour is "send that
squad at that point".

## The remaining bottleneck, measured and NOT addressed here

Widening the gate raises candidate objectives, but units assigned rises only
**32.0% -> 48.1%** (1.60 -> 2.41 of 5) while *steps where one unit owns 2+
objectives* rises **29.0% -> 70.7%**.

That is `_compute_group_assignment`: it walks each candidate objective and gives
it to the **closest group**, so one well-placed unit can take several while other
units get none. The candidate gate was the binding constraint; with it widened,
greedy per-objective assignment becomes the next one.

⚠ **This is deliberately NOT in this arm.** Fixing it means a matching (each
group at most one objective) rather than an argmin, which is a second behaviour
change; stacking it would make the result uninterpretable. Recorded here so the
next step is chosen from a measurement rather than rediscovered.

⚠ Note also that the golden config's own comment for this parameter reads "5
groups over 3 objectives leaves two groups unassigned" -- that is the THREE
objective scenario, and the generated tables carry five or six. The comment is
stale; the measurement above is what holds.

## Pre-registered criteria

Three seeds, 300 epochs, `ent_coef` 0.003, recording on, **paired against the
existing s1/s2/s3 `-newmaps` controls** (same seeds, same config, same epoch
budget — no control retraining). Scored refereed at K=3, held-out nine, n=30.

- **ACCEPT** if the paired difference is positive with t > 2 **and** offence
  (agent VP minus best-script VP) improves. Offence is the target; a vp gain
  that is entirely defence again is NOT this lever working.
- **REJECT** if the paired difference is negative, or if `alive` rises while
  `held` does not — that is the hoarding getting worse.
- **INCONCLUSIVE** at |t| < 2: report as "run it longer", per the 300-epoch
  screening rule, not as refuted.

## What would make me wrong, stated in advance

The overstack result three days ago is the live caution: a term that looks bad
in the income ledger cost **−12.2 vp** to remove, because what it *prevented* was
invisible. The symmetric risk here is that the one-model gate is load-bearing —
it may be what stops the army walking into ground it cannot take and dying there.
`alive` and `held` are the readouts that would show that, and both are in the
reject rule above.

⚠ This is also the **third** consecutive attempt to fix offence with a reward
change. The two before it (`24v24` spare squads, mixed roles) left offence
unmoved at −50.5 and −42. If this one also leaves offence flat, the honest
conclusion is that offence is not reward-shapeable here and the next move is
architectural, not another term.
# Amendment to the contest_deficit pre-registration

**Written 2026-08-22 at epoch 290 of 300, BEFORE any score existed.** Recorded
here first, timestamped, precisely so it cannot be mistaken for a post-hoc
rationalisation. Prompted by an adversarial review, not by a result.

## 1. The REJECT rule was armed against the wrong failure

Original: *"REJECT if the paired difference is negative, or if `alive` rises
while `held` does not — that is the hoarding getting worse."*

That catches hoarding. **The predicted failure of THIS lever is the opposite**:
squads paid to walk at defended ruins get shot crossing open ground, so `alive`
FALLS. Under the original wording, "the agent lost 20 vp because its army died
attacking" would not have tripped the alive/held clause at all.

**Added, symmetric clause:** REJECT also if `alive` FALLS without `held` rising.
Report model-steps-in-the-open alongside, since `measure-hold-hazard` establishes
that the exposed models are the ones walking between points, not the ones
standing on them.

## 2. Prior evidence AGAINST this lever that the report failed to cite

The **teleport audit** (2026-08-11, recounted in
`reports/2026-08-16-the-cap-makes-it-a-denial-game.md`) force-moved a squad onto
contested ground, n=39 **paired**:

| | control -> teleported |
|---|---|
| the moved squad's own income | 83.03 -> **53.62**  (−29.41 ± 8.27) |
| the moved squad's survivors (of 5) | 3.03 -> **1.33** (−1.69 ± 0.28) |
| episode reward (scalar) | 20.94 -> 24.65 (+3.71 ± 1.30) |

The squad dies on a point holding **4.91 opponents**. PPO trains on the
per-model reward vector, so the gradient that squad generates gets *worse* by
35% while the team scalar improves. **`contest_deficit: 5` pays for exactly that
move, at exactly that defender count.**

The golden config header also records forcing redistribution onto *undefended*
ground at **−3.6** and **−3.2 on 180 paired episodes**. Defended ground is
strictly harder, and the defenders are in cover because objectives are ruins.

⚠ I knew the −3.2/−3.6 result — I cited it earlier in this same session — and
still wrote a report that did not engage with it. That is the error to learn
from, not the lever.

## 3. What is genuinely different this time, stated fairly

The previous nulls (`deny_high`, the redistribution arms) ran at `ent_coef`
0.03, where the policy is measured as too blunt to represent a committed squad
advance. This arm runs at **0.003**. So it is not a strict repeat.

## 4. The arrival payout was NOT changed, and that may be the whole problem

`contest_deficit` widens *candidacy* for the TRAVEL term only. `objective_hold`
still pays 1.0 / 0.5 / 0.25 for ours / contested / theirs, and raising the last
two to parity was measured a clean null (188.5 v 188.5). So the agent is now
paid to WALK toward enemy ground while still paid 4x less for ARRIVING there —
and arriving is what kills it. If this arm fails, that asymmetry is the first
thing to look at, not the gate.

## 5. Cross-branch interaction — do not confound these

⚠ `feature/movement-batch`'s engagement rule made **87% of opponent-held
objectives physically impossible to enter**. If that had landed alongside this,
`contest_deficit` would steer squads at ground they cannot occupy, and neither
result would mean anything. The engagement half has been reverted for this
reason among others. **Do not merge a movement-blocking change and this one and
then measure either.**

---

# RESULT — rejected, through the door the reviewers opened

**Measured 2026-08-22.** Three seeds, 300 epochs, `ent_coef` 0.003, scored
refereed at K=3 on `configs/evaluation/25v25_maps_take_opponent_refereed.yaml`,
held-out nine, n=30, seeds 700000+. **Paired on seed against the `-newmaps`
controls** — same seed, same config but for one parameter, same flags, same
epoch budget, so the per-seed difference is the strong estimator.

| seed | arm | control | difference |
|---|---|---|---|
| s1 | +23.2 | +26.6 | **−3.4** |
| s2 | +20.9 | +14.9 | +6.0 |
| s3 | +18.0 | +28.6 | **−10.6** |

**−2.7 ± 4.8, t = −0.55 (df=2), 1 of 3 seeds positive.**
Across tables: **−2.7 ± 3.7, t = −0.72 (df=8), ahead on 2 of 9.** The two
estimators agree in sign and magnitude — and note they are *not* independent
evidence, being one dataset sliced two ways.

Scripts on the same config, re-measured refereed at K=1: `squad_march_deny`
**−1.1**, `squad_march_take` **−2.4**. Coherency: arm 0.942–0.965 against the
scripts' 0.908–0.910.

## It failed on the ACCEPT criterion, not just on vp

The pre-registered ACCEPT rule required **offence** to improve — a vp gain that
was all defence again would not count. Offence (own VP minus the best script's)
went the **wrong way**:

| | offence | defence |
|---|---|---|
| arm | **−71.5** | +93.2 |
| control | **−61.2** | +85.5 |
| difference | **−10.3** | +7.7 |

Per seed, offence moved −8.0 / +5.3 / −28.2. So the lever that existed to fix
offence made it worse on 2 of 3 seeds, and the small defensive gain did not
cover it.

## ⚠ It failed through the clause the reviewers added, which the original rule would have MISSED

`alive` **fell on 3 of 3 seeds** (−0.075 / −0.032 / −0.037, mean **−0.048**)
while `held` did **not** rise (mean −0.007).

The original REJECT rule read *"`alive` rises while `held` does not"* — armed
against hoarding. **This failure is the opposite**, and would not have tripped
that clause at all. The symmetric clause was added at epoch 290, before any
score existed, on adversarial review. It fired.

## The mechanism is the teleport audit, reproduced by training instead of by force

The 2026-08-11 audit force-moved a squad onto contested ground and measured it
losing **1.69 of 5 models** and **29.41 of its own income** against 4.91
defenders. `contest_deficit: 5` pays for that move by gradient rather than by
teleport, and got the same answer: the army finishes ~4.8 percentage points
smaller and holds no more ground.

**Paying a policy to walk at defended ruins gets it shot crossing open ground.**
The one-model gate was not a bug; it was load-bearing, in exactly the way the
overstack penalty was.

## What this settles, and what it does not

⚠ **At t = −0.55 this is formally INCONCLUSIVE**, and this project's own rule is
that a marginal 300-epoch result means "run it longer", not "refuted". The
reason it is recorded as REJECTED rather than unresolved is that the
pre-registered accept condition was *offence improving*, and offence moved
backwards. Running longer would settle the vp difference; it would not rescue
the criterion the arm was built to satisfy.

**This is the THIRD consecutive reward-shaping attempt to leave offence flat or
worse: −50.5, −42, now −71.5.** The pre-registration said in advance that if
this happened, the honest conclusion is that offence is not reward-shapeable
here and the next move is architectural. That conclusion now stands.

**Where the evidence points instead.** `vp_gain` is *net* — denial is already
paid — but it is a **global** term, broadcast identically to every alive model.
No model can therefore prefer "take theirs" to "stand on ours", because both
change the same shared scalar by the same amount. That is a **difference-reward**
problem, not a pricing one, and no widening of a candidacy gate can reach it.
The travel term is per-model but only prices *distance closed*; the term that
prices *outcome* is global. Closing that gap means a per-model credit for the
opponent VP a model's own presence denied — a structural change to how credit is
assigned, which is what "architectural" means here.

## Ledger of what was NOT the problem

- **Not the candidacy gate.** Widening it did what it claimed mechanically — the
  "they hold it by 2+" exclusion fell 43.4% → 3.9%, units with their own
  objective rose 32.0% → 48.1% — and the policy got no better at offence.
- **Not observability.** The desk check passed; per-objective alive counts for
  both sides are in the observation and were verified against the scoring
  definition the same week.
- **Not the scenario.** `24v24_maps_spare_squads` was built specifically to pose
  the allocation question, and offence did not move there either.
