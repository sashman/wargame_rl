# Teach the agent to charge

**Goal.** A trained agent that uses the charge phase competently on a melee config —
declaring when a charge can land, aiming so it lands, and beating the scripted bar on the
same config while doing it.

Melee is a **core rule**, not an experiment. It is staying. So this document asks *how best
to teach it*, never *whether it is worth having*, and no measurement here compares a game
with melee against a game without one.

> ⚠ **This replaces the accept/reject framing of
> [the pre-registration](../reports/2026-08-25-melee-preregistration.md).** That document
> asked whether the mechanic pays and built an arm-versus-control pair to answer it. Both
> the question and the pair are retired: the pair was confounded (its control disarms the
> *opponent*, so it estimated a mirror match spanning −29.4 to +2 and containing no PASS),
> and the question was the one this project already got wrong for Advance — *"ADVANCE IS A
> CORE RULE, NOT AN ARM. Wrong question: it is staying."* Same error, one rule over. The
> **audit findings in that document all stand** and most of them matter more under this
> framing, because they are the instruments for measuring teaching.

---

## 1. What the agent has to learn

A charge is a **two-stage, all-or-nothing, unit-level** action, and every one of those
words is part of the difficulty.

1. In the **command phase**, a unit's leader declares a charge. The declaration **binds the
   whole unit**. ⚠ **It costs NO shooting** — measured, 5 of 5 models keep a legal shot either
   way. An earlier version of this document said it "spends the whole unit's shooting
   immediately"; that is the **advance's** cost, carried across by mistake. The phase order is
   command → movement → shooting → charge, so a charge cannot forfeit a shot already fired.
   What a *standing* charge costs is every FUTURE shooting phase until the unit disengages.
2. In the **charge phase**, each model of that unit picks a move, capped at
   `min(Move, 2D6)`.
3. The referee then judges the unit **as a whole**: it must end in coherency and engaged
   with **exactly one** enemy unit. Miss by any margin and **every model is put back where
   it started**, for exactly zero reward.

So a near-miss and a wild miss are indistinguishable to the learner, and the credit for a
declaration made in one phase arrives two phases later.

## 2. The obstacle, and it is the whole problem

**Stumbling into a legal charge by chance happens on 0.27% of attempts. The scripted policy
manages 66%.** PPO cannot bootstrap a policy from a 0.27% success rate.

Everything else in this document is detail. **The goal reduces to: get the agent from 0.27%
to a rate it can learn from.**

## 3. Success criteria, written before anything is trained

> ⚠ **PROVISIONAL, and the weakest part of this document.** Every target below was derived
> from `squad_march_take_charge` — which is simultaneously the teacher, the comparator and the
> target. Nothing has established that it plays *well*. This project's own record says six
> independently hand-rolled charging scripts spanned **14x** in value for nominally the same
> measurement, so "the script's rate" is not evidence of what competent play is. Copying a bar
> and calling it a target is the *"measure the bar, not the agent"* error, restated.
>
> **Four things must be measured before these numbers are trusted.** All are CPU-only:
>
> 1. **How often does the one-enemy-unit clause make a charge impossible?** `_enforce_charge`
>    reverts a unit that clips a second enemy unit, which `11-charge-phase.md` names as what
>    makes a charge fail *even on a long roll*. If two enemy units standing near each other
>    veto a charge often, then target availability is the constraint and aiming is not.
> 2. **Does a standing charge freeze both units for the rest of the game?** The fight phase
>    carries no agent action, pile-in and overrun are absent, the blade is ~0.44 expected
>    wounds over nineteen rounds, and both engagement gates are unit-level so one model in
>    contact shields five. If a charge is a permanent mutual lock, it is a **denial** tool —
>    and this agent's diagnosed pathology is that it already hoards and holds too little
>    ground. "More charges" would then be the wrong thing to want.
> 3. **Is `squad_march_take_charge` any good?** It should have to earn at least one of its
>    three roles.
> 4. **`DEFERRED: charge.beyond_move_ladder`** — see §10.



Scored on the melee config against the scripted bar on that **same** config — the standard
apparatus for every other scenario here. **No control arm**, because there is no
counterfactual game to compare against.

| readout | floor (untrained, K=1) | target | why this one |
|---|---|---|---|
| **charges standing per episode** | 0.00–1.67 | **> 3.0** | ⚠ **the primary readout.** Numerator only, hard floor at zero, monotone in competence |
| charges declared per episode | bimodal 0.00–81.7 | inside the script's band | ⚠ **bounded on BOTH sides.** An untrained net declares constantly and lands nothing |
| standing fraction | 0.003–0.013 | **> 0.35**, → the script's 0.750 | ⚠ **secondary only** — its denominator is policy-controlled, so it rises when a policy declares *less* |
| `vp_margin` v the bar | — | ahead, with a t **and** a sign count | the decision |
| `coherent` | — | ≥ the bar's | a charge is a joint move; formation is how it fails |

⚠ **Every readout is quoted at `decode_topk` = 1**, which is what training decodes at. At
K=3 the joint decoder picks legal combinations *for* the network, and the mechanism counts
measure the decoder rather than the policy.

⚠ **Floor every gate on a random-init network through the arm's own selector path before
launching**, and check the readout is **monotone in competence**. Two of the four gates in
the previous pre-registration were passed by a network that had learned nothing — one of
them by declaring *nothing at all*.

## 4. Prerequisites — three confirmed defects, ALL THREE NOW FIXED

Nothing in §5 could be measured until these were fixed. All three were bugs, not judgement
calls, each verified in the source and each re-measured after the fix. The seeded episode
digest is **9 of 9 identical to `main`**, so melee-off is still an exact no-op.

1. ✅ **FIXED — the scripted bar was exempt from a rule the agent must obey.** "A declared
   unit may not stand still" existed only in mask construction; `ActionHandler.apply` takes
   no mask, so the bar could declare a charge, submit STAY, and have the env accept it.
   ⚠ **1.8%, not the 14.7% two panels reported** — 2 of 109 declared rows, re-measured
   here; the defect was real and its published magnitude overstated ~8×. Golden configs
   were clean at **0 of 2,660**, so this was melee-only. Now 0 of 106, with
   `tests/test_scripts_obey_the_action_mask.py` pinning the class.
2. ✅ **FIXED — `declared_charge` was not observable.** It appears in `observation_builder.py` only
   inside the mask code. Any charge-gated reward term keys on state the network cannot
   perceive, which fails the cheapest standing check in `CLAUDE.md` — the one that has
   already burned ~10 GPU-hours twice.
3. ✅ **FIXED — `charge_progress` never implemented its declaration gate.** It gates on
   `charge_roll <= 0.0`, and 2D6 is never ≤ 0. Measured: the **non-charging** script earns
   **5.713/episode with 0.0% of it going to a declared unit**, against the charging
   script's 4.196 — the term pays the wrong behaviour **36% more**. It is also wired into
   no config.

## 5. The three candidate methods

### (a) Warm-start from a clone of the charging script — strongest candidate

Clone `squad_march_take_charge`, then run PPO from those weights. It starts the policy near
the script's 66% rather than at 0.27%.

⚠ **The objection on file does not apply.** `CLAUDE.md` records *"PPO cannot improve a
behaviour-cloned policy here — with a cold critic it destroys a 115.8 clone at every
`ent_coef`."* **The critic is not cold**: `behaviour_clone.py` fits one to explained
variance 0.76–0.78 and writes all **222** tensors, exactly the full network, which is what
`_apply_warm_start_weights` loads. **That caveat has never been tested with a warm critic.**

⚠ **One measured fact argues against it**, and it is the thing to settle first: across a 4×
sweep of demonstration volume, the clone's held-out aiming accuracy stayed **flat at 0.30**.
If more examples do not improve aiming, warm-starting will not either.

### (b) Shaping — `charge_progress`, once §4.2 and §4.3 are fixed

A per-model potential paid for closing on the enemy **during a declared charge**, maximal
when engaged. It converts the referee's all-or-nothing verdict into a gradient, which is
exactly what a 0.27% discovery rate needs.

⚠ **Correctly gated it is inert at its default** — roughly **0.05/episode** against
`objective_hold`'s +16.0. It needs scaling, and scaled far enough it becomes a
"walk at the enemy" term, which is the family this project has now rejected four times
(`closest_objective_v2`) and which the 2026-08-11 teleport audit priced at −29.4 of the
committing squad's own income. **The declaration gate is the entire difference**, so it has
to work before the weight is raised.

### (c) A curriculum phase

An early reward phase that pays charging, annealed out later. The machinery exists and
`reward_phases` is already per-config. Cheapest to express; least evidence behind it.

## 6. The order of work

1. **The three fixes in §4.** Engineering, no GPU. Required under every method.
2. **Re-run the clone control at house fidelity** — 1200 demonstrations, 60 epochs, three
   demonstration seed bases, ~15 GPU-minutes. ⚠ The previous run used **200 and 8**, which
   is underfit, and a REJECT was read off it; refitting the same cache to 48 epochs gives
   train 0.811 against held-out 0.173. This one measurement discriminates between (a) and
   (b)/(c): **if held-out aiming rises above 0.30, take (a); if it stays flat, imitation is
   not the route and the answer is shaping or curriculum.**
3. **Train the chosen method**, three seeds minimum, screened at ~300 epochs and quoted at
   1000+, scored at K=1 against the bar on the same config.

## 7. What would make me abandon this

- **Held-out aiming stays flat at 0.30 at full clone fidelity AND shaping leaves
  charges-standing at the untrained floor.** That would say the architecture cannot express
  a coordinated unit-level action, which is a claim about the network, not about melee —
  and it generalises to every move type the rules add next (fall back, and beyond).
- **The agent learns to charge and loses.** Then the lesson is about the scenario, not the
  teaching, and it belongs in `play-doctrine.md` rather than here.

## 8. Cautions carried forward

- ⚠ **Melee is not in any golden config, and putting it there voids every baseline and
  every agent score on them.** That is the eventual cost of "melee is core" and it should
  be paid deliberately, on one config at a time.
- ⚠ **No learned policy of any kind has ever been evaluated on melee.** Published
  checkpoints fail to load on a melee config at the observation width, and again at the
  policy head. The melee lineage starts from scratch.
- ⚠ **Six independently hand-rolled charging scripts produced +6.5 to +88.8 vp for
  nominally the same measurement — a 14× spread.** `squad_march_take_charge` is one
  heuristic and its numbers are its own, not "the value of melee".
- ⚠ **Certify any paired control on BEHAVIOUR, not on a config field.** The test that
  certified the retired pair passes, and proves the confound.
- **The charge is `partial` against the rules**: `beyond_move_ladder` discards 59.1% of
  rolls, and the target declaration is absent, so charge-A-land-on-B is legal. Closing
  either changes the action space and voids the checkpoints behaviourally.


---

## 9. The training run, pre-registered — comparator fixed before any agent exists

⚠ **Written before a single agent number exists**, because *"fix the comparator BY NAME before
measuring, and select it on the statistic you will report"* — a "best script" chosen by argmax
on the same data changes identity between cells and turns a magnitude into an artefact. It did
exactly that in the five-round report, where winner-selection bias measured **+1.4 to +2.9**.

**THE COMPARATOR IS `squad_march_take_charge`.** Named now, not chosen later. It is the only
scripted policy that uses the mechanic under test, so it is the bar for a goal about using the
mechanic. Its numbers on `configs/evaluation/25v25_maps_melee_refereed.yaml` — the held-out
nine, refereed, n=9, seeds 700000+, K=1 — measured **before any agent existed**:

| policy | decl/ep | tried/ep | **stood/ep** | frac | coherent | vp |
|---|---|---|---|---|---|---|
| **`squad_march_take_charge`** (the comparator) | 4.44 | 4.22 | **4.00** | 0.947 | 0.878 | **+13.3** |
| `squad_march_take` (never charges) | 0.00 | 0.00 | 0.00 | — | 0.904 | −11.1 |
| `squad_march_deny` | 0.00 | 0.00 | 0.00 | — | 0.921 | −36.7 |
| `squad_march_shoot` | 0.00 | 0.00 | 0.00 | — | 0.836 | −82.8 |

**Taken against the COMPLETED melee implementation** (2026-08-26): the while-moving rule,
resolution-time preconditions, the restored decline, the 2D6 charge ladder, pile-in,
alternating activation, passing, overrun, consolidate-Ongoing, engaged units barred from
advancing, and `engagement_range` at the rules' 2".

⚠ **AND THAT IS WHY IT HAD TO BE COMPLETED FIRST. Same-row, charging is worth +24.4 vp**
(+13.3 against −11.1, same opponent). On the **incomplete** implementation the same
comparison read **−1.7** — about zero. Six rules changes moved a mechanic from "does not pay"
to "pays two and a half times the per-seed noise", and any decision taken on the earlier
number would have been taken on a game this project does not play.

⚠ **The standing fraction is 0.947 against 0.604–0.804 before.** Alternating activation also
halved the seat asymmetry: the same policy on both seats read **−25.0** under whole-side
resolution and **+13.3** now, because the active player no longer inflicts every casualty
before any opposing unit swings back.

Superseded rows, in order: 5.11/**3.22**/0.744/+9.4 → 5.44/**3.11**/0.667/+11.1 →
5.44/**3.11**/0.800/+11.1 → 6.00/**3.22**/0.604/+7.2 → 4.78/**2.56**/0.575/+4.4 →
6.22/**4.56**/0.804/−25.0.

### The trichotomy, written first

- **PASS** — the agent clears `stood/ep` **> 3.0** at K=1 *and* beats `squad_march_take_charge`
  on `vp_margin` with a t **and** a sign count, on ≥ 3 seeds, with `coherent` no lower.
- **FAIL** — it does not clear the untrained floor on `stood/ep`, or it charges competently and
  still loses on vp. ⚠ These are **different** failures and must be reported as such: the first
  is about teaching, the second about the scenario, and only the first is this goal's.
- **UNDERPOWERED** — the confidence interval contains both zero and the effect that would
  matter. This must stay a reportable outcome. Per-seed paired sd is 11.3 vp and the per-episode
  sd on the map pool is 80.9–83.1, so it is the likely outcome at n=3 and must not be narrated
  into a PASS.

### Standing instruments

`just measure-charges <policy|ckpt> <config> [n] [decode_topk]` is the primary readout, and
`just measure-checkpoint` / `just measure-baselines` carry the vp comparison with `held`,
`alive`, `coherent` and `adrift`. Every row is quoted with its **K** and its **seeds**.


---

## 10. The charge cannot travel further than a walk, and the only reason on file is now void

`docs/rules/11-charge-phase.md` caps a charge at the **2D6 alone**, up to 12". Here the charge
reuses the **movement** slice, whose longest rung is the model's Move (6"), so **a charge can
never travel further than a walk however high the dice land.**

Measured (gap map row 106, 10 episodes, 203 eligible declarations — ⚠ **taken before the
command-phase declaration landed and `max_turns` went 60 → 80, so it needs re-measuring**):

- the roll exceeds Move on **59.1%** of declarations, and every inch above 6" is discarded
- **12.3%** of eligible declarations are blocked by this cap rather than by the dice
- reachability is **44.3%** where a true 2D6 ladder gives **56.7%**

⚠ **This removes the upside of the gamble.** The reason a charge is worth declaring in the
rules is that 2D6 can carry a unit *further than it could walk* — that is what it buys for the
shooting it forfeits. Implemented this way it forfeits the shooting and buys ordinary movement.

**The gap map states one reason, and only one, for deferring it:**

> *"a dedicated rung ladder still changes the output head's shape, which is what makes an
> action-space arm unpairable against its control … but it survives for the pairing reason
> **ALONE**"*

⚠ **There is no control arm any more.** Melee is a core rule, so there is no counterfactual
game to pair against, and the agent is scored against the scripted bar on the same config
(§9). **The sole objection on file evaporates under this document's own framing**, and what is
left is 12.3pp of reachability and the mechanic's whole point.

What it costs: a charge rung ladder is new actions, and this project measured *carrying* an
unused action slice at **−2.9 ± 0.67 vp** for the advance. ⚠ That is not the same situation —
charge rungs are legal only for a declared unit, so they are masked off almost always, where
the advance's were always legal. Do not carry the −2.9 across without measuring.


---

## 11. ⚠ The demonstration cache keys on the CONFIG, not the CODE

Found 2026-08-26, mid-run. `scripts/behaviour_clone.py:464` fingerprints
`config.model_dump_json()`. That catches a config change — it was added after the cache keyed
on the config *filename* and would have fitted 102-action demonstrations to a 104-action
network — but **it does not catch a code change**.

The clone control launched, then four rules fixes landed while it was collecting: the
*while-moving* condition, the resolution-time precondition re-check, the restored decline, and
the removal of `_commit_declared_units`. The last of those changes what the teacher *does* —
attempts fall from 4.67 to 3.89 per episode, i.e. **~14% of its charge decisions differ** —
so the cached demonstrations came from a teacher playing by superseded rules while the clone
would be scored under the new ones.

The whole point of the clone control is to ask whether the architecture can express **the
teacher's** behaviour, so the teacher has to be the current one. The run was killed and both
the cache and the checkpoints deleted rather than caveated.

**Standing rule: a demonstration cache must be invalidated by a change to the RULES, not only
by a change to the config.** Until the fingerprint covers the code, delete
`checkpoints/clone_data/` by hand after any change to `actions.py`, `domain/`, or a baseline
policy. ⚠ And do not launch a clone run while rules work is still in flight — this one was
launched three hours before the audit that changed the rules it was learning.
