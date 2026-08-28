# Teach the agent to fight

**Goal.** **Melee implemented as the rules define it**, and a trained agent that uses the
**charge and fight phases** competently on it — declaring when a charge can land, aiming so
it lands, and beating the scripted bar on the same config while doing it.

⚠ **IMPLEMENTATION IS PART OF THE GOAL, NOT A PREREQUISITE TO IT.** Set 2026-08-26. The
reason is measured, not stylistic: completing six rules moved charging from **−1.7 vp to
+24.4 vp same-row**, so an agent trained against a partial melee is trained against a
different game, and a bar taken on one has been voided five times in a single day. The
standing rule is this project's own — *measure the configuration that SHIPS*.

### What "implemented as the rules define it" still requires

⚠ **Checked against the code before each item is started, never trusted.** This inventory
has been wrong in BOTH directions in one day: two rules had no row at all, and one closed
item was recorded as open.

**Open — rules-correctness bugs.** The implementation permits or forbids something the
rules do not:

| item | the divergence |
|---|---|
| **charge declaration is unmasked** | `11-charge-phase.md` §Eligibility makes a unit ineligible to declare if it is *not within 12" of any enemy*, is engaged, or advanced/fell back. `get_action_mask` is **purely phase-based** and `wargame.py` never masks the `move_type` slice on either seat, so any alive model may declare in any command phase. Eligibility is enforced only later, on the MOVE, where an ineligible unit finds no legal rung. Measured consequence: **71.4% of the agent's declared model-steps hold zero legal rungs** and declarations land on eligible units *below chance* (31.4%, z = −2.54) against the teacher's 40/40. ⚠ The gap map rates this row **implemented** (`implementation-status.md:103`) — it is implemented for the move, not the declaration, and the row is wrong. |
| **3.2 consolidate Engaging** | The remaining consolidate mode; needs 2.3, which is done. |

**Open — required by the fight clause of the goal.** The fight phase resolves but the agent
decides none of it. Every one of these is a choice `12-fight-phase.md` gives a player and
the engine currently makes:

- which eligible unit activates next (the alternating-activation ordering)
- pile-in target selection, and the pile-in move itself
- the fight **type** — normal versus overrun — where a unit is eligible for both
- whether to **pass**
- the consolidate move

Promoting these to agent actions is a **scope increase over the charge-only goal** and is
what makes the fight clause achievable at all. ⚠ Until it lands, every fight-phase number
measures the ENGINE's fight policy, not agent skill.

**Design-uncertain — decide explicitly, do not slip in.** Target declaration → multi-target
charge (§6). It adds actions, measured opportunity is 7.5% of eligible unit-turns with
**0% realised**, and an expert review ranked it last on those grounds.

**Out of scope, and named rather than forgotten** — each needs a subsystem this game does
not have: sequential per-unit decisions (architecture), distance around obstacles
(pathfinding), reckless break (suppression + backfire), vertical engagement (the board is
2D), `MONSTER`/`VEHICLE` (no unit-type system), melee weapon selection and multiple profiles
(a choice with no second case), and fighting after death — **a granted ABILITY no model
has**, where implementing it as a default would make the game *incorrect*.

⚠ **"Melee" here means BOTH phases.** Charge and fight are one mechanic split across two
phases by the rules, and a goal naming only the charge measures half of it.

⚠ **THE FIGHT PHASE IS NOT SKIPPED MECHANICALLY, AND IS SKIPPED AS A DECISION.** Both are
true and the distinction is the whole of what this clause adds:

- **It resolves.** `fight` is in `skip_phases`, but `on_before_advance` fires on skipped
  phases and calls `_resolve_fight_phase` (`wargame.py:1138`), guarded on
  `state.phase is BATTLE_PHASE_ORDER[-1]`. Pile-in, Strikes First, alternating activation,
  passing, overrun, the attacks themselves and consolidate all execute. Independent
  evidence: charging is worth **+24.4 vp same-row** on the bar, which it could not be if
  no blow landed.
- **The agent decides none of it.** `get_action_mask(BattlePhase.fight)` offers exactly one
  legal action per model — STAY — because the `movement` slice is valid only in
  `{movement, charge}`. Every choice `12-fight-phase.md` gives a player is made by the
  engine: which eligible unit activates next, pile-in target selection and direction, the
  fight **type** (normal versus overrun), whether to **pass**, and the consolidate move.

**So this goal is not currently achievable as written, and that is the point of writing it
this way.** "Uses the fight phase competently" cannot be true of an agent that has no fight
action. Closing it needs fight-phase decisions promoted to agent actions — which is a
scope increase over the charge-only goal, and is now the standing scope. ⚠ **Do not read a
fight-phase result as agent skill until that lands**; until then every fight number
measures the ENGINE's fight policy, exactly as the coherency referee measured itself.

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


---

## 12. THE CLONE CONTROL, on the completed implementation — REJECT on 3 of 3

Measured 2026-08-26 at `5da54ed`. House fidelity (1200 demonstrations, 60 epochs), three seeds,
held-out nine, n=9, **K=1** (what training decodes at).

| | decl/ep | stood/ep | standing fraction | vp |
|---|---|---|---|---|
| teacher `squad_march_take_charge` | 4.22 | **3.89** | **0.921** | +13.3 |
| clone s0 / s1 / s2 | 1.33 / 2.89 / 2.67 | **0.11 / 0.67 / 0.56** | 0.083 / 0.231 / 0.208 | −96.1 / −106.1 / −30.6 |

**Verdict against the criterion committed before the numbers existed** (accept ≥ 0.45, reject
≤ 0.25): **REJECT on 3 of 3.** The gate `stood/ep > 3.0` fails by a factor of six.

### The clone is bad at EVERYTHING, and that is the finding

⚠ The within-policy control, which had to be run because the clone scores vp **−30 to −106**
against its teacher's +13.3 — *"it cannot charge"* and *"it cannot play"* predict the same
charge number:

| seed | movement agreement | charge agreement | gap |
|---|---|---|---|
| s0 | 0.529 | 0.448 | +0.081 |
| s1 | 0.496 | **0.548** | **−0.051** |
| s2 | 0.512 | 0.432 | +0.080 |

**The charge is NOT specially hard.** The gap averages +0.04 and **flips sign** — on s1 the
clone aims charges *better* than it walks. Imitation of this teacher lands at ~0.5 per-model
agreement in both phases, and that is the ceiling the charge inherits.

### `p^k` is one real term and not the dominant one

Collapsing `k` to 1 — the clone's OWN preferred heading applied rigidly to its unit, so nothing
about its judgement changes:

| seed | as played | `k` = 1 |
|---|---|---|
| s0 | 0.083 | 0.273 |
| s1 | 0.231 | 0.231 |
| s2 | 0.208 | 0.417 |

Roughly a doubling on average, **nothing at all on s1**, and it tops out at 0.42 against the
teacher's 0.92. ⚠ An earlier version of this section named `p^k` as the mechanism on the
strength of `0.448^3.47 = 0.062` matching a measured 0.083; that agreement was a coincidence
and the falsifier written to test it refuted it. ⚠ **That probe's first run also reported the
teacher at 0.079 against its own measured 0.947**, by counting the referee's REVERT as a stand
— caught by the teacher row, which is in the table precisely as a known-answer check.

### What this decides

**Warm-start is not the route**, and neither is better aiming. The pre-registered rule
*"aiming above 0.30 → warm-start"* is satisfied in the letter (0.448 / 0.548 / 0.461, against
0.30 for the old underfit clones) and void in substance: a start that stands 8–23% of its
charges and scores −30 to −106 vp is not a start worth taking, and the deficit is general
rather than charge-specific.

**So the first arm trains from scratch, with no shaping term.** `charge_progress` stays
unwired: an unshaped arm is the control any shaping decision must be measured against, and the
agent already has a reward path to charging — melee kills reach `model_kills` per model, and a
standing charge shields its unit into surviving to hold objectives.

⚠ **Read the seed spread before any of this is quoted.** Standing fraction runs 0.083–0.231 and
vp −30.6 to −106.1 on three seeds of one recipe. Means over that spread are weak evidence.


---

## 13. The untrained floor, on the COMPLETED implementation

⚠ **Measured mid-arm, because I had launched without it.** The pre-registration's own rule is
*"any behavioural readout that gates an arm must be floored on a random-init network through
the arm's OWN selector path, **before the arm launches**"* — and every floor on file was taken
before the 2D6 ladder, pile-in, alternating activation and `engagement_range` 2.0. None of them
describes the game this arm trains on, so none of them could have interpreted its result.

Three randomly-initialised `PPO_Transformer`s, saved with the Lightning prefix so they load
through the same selector path as an agent checkpoint. Held-out nine, n=9, K=1:

| random-init seed | decl/ep | stood/ep | standing fraction | coherent | vp |
|---|---|---|---|---|---|
| s0 | 0.00 | 0.00 | — | 0.574 | −227.8 |
| s1 | 0.00 | 0.00 | — | 0.841 | −198.9 |
| s2 | **93.11** | 0.89 | 0.065 | 0.816 | −236.1 |

⚠ **`declared/ep` is USELESS as a gate on this config** — the floor is bimodal from 0 to 93,
because the declaration is one argmax over two actions with arbitrary logits. Read `stood/ep`
(floor **0.00–0.89**) and the standing fraction (floor **0.065**), and read vp against a floor
of **−199 to −236**.

### What it says about the arm in flight

The trajectory sample at epoch 433 — 27.44 declares, 1.44 stood, fraction **0.052**, vp −39.4:

- **vp −39.4 against a floor of −199 to −236**: the agent has emphatically learned to PLAY.
- **standing fraction 0.052 against a floor of 0.065**: its charge competence is at the
  UNTRAINED level.

**It is learning the game and not learning the charge**, and neither half of that is visible
without this control. ⚠ Provisional: one seed, mid-training, n=9.

---

## 14. ⚠ THE PRIMARY READOUT WAS BROKEN. Every mechanism figure above is superseded.

Found 2026-08-26 by an adversarial panel's red team under a dual mandate to audit the
*instruments*, not the reasoning. Verified independently in the code before acting, and every
claim below was reproduced by hand.

### The bug

`scripts/measure_charges.py` scored a charge as **stood** if any member of a moving unit was
not where it started **after `env.step` returned**. That window does not contain only the
charge. After `_enforce_charge` has already reverted a failed charge, the same `env.step` runs
pile-in for both forces, the fight step, consolidate for both, and **the entire opponent turn**
(`turn_execution.py:55-60`), whose own pile-in moves the player's models again.

So a unit whose charge the referee **reverted**, and which the opponent then charged, was
displaced by pile-in and **scored as having stood**.

The false positives are drawn from the *failed-attempt* pool, so **the error grows with
incompetence**:

| policy | stood/ep (broken) | stood/ep (referee) | inflation |
|---|---|---|---|
| `squad_march_take_charge` | 4.00 | **3.89** | +2.9% |
| untrained `melee_floor_s2` | 0.89 | **0.22** | **+300%** |
| arm s2 @ epoch ~520 | 2.67 (ep 475) | **0.33** | **~8x** |

`stood/ep` was therefore **ANTI-monotone in competence** — the exact disease its own docstring
attributed to the standing *fraction* while asserting the numerator was immune to it
(*"the numerator alone, with a hard floor at zero"*).

**The correct reading already existed.** `charged_this_turn` is set at `actions.py:1160` only
when `_charge_preconditions_hold` **and** `_charge_is_legal` both pass — the reverting branch is
its `else` — and it is cleared inside `_resolve_fight_phase`. The old docstring warned never to
read it *after* the charge step and never tried reading it *inside* the step, which is the one
window that works. Fixed there; the known-answer row reproduces the teacher's `vp` and
`coherent` to the digit, so it is the same instrument with only the counting rule changed.

### The corrected table — held-out nine, refereed, 9 episodes, K=1, seeds 700000+

| policy | decl/ep | tried/ep | **stood/ep** | frac | coherent | vp |
|---|---|---|---|---|---|---|
| `squad_march_take_charge` | 4.44 | 4.22 | **3.89** | **0.921** | 0.878 | +13.3 |
| `squad_march_take` | 0.00 | 0.00 | 0.00 | — | 0.904 | −11.1 |
| floor s0 / s1 / s2 | 0.00 / 0.00 / 93.11 | — | **0.00 / 0.00 / 0.22** | 0.016 | — | −227.8 / −206.1 / −236.1 |
| arm s1 / s2 / s3 (ep 521/520/383) | 21.9 / 21.1 / 44.1 | 13.2 / 5.8 / 15.2 | **0.67 / 0.33 / 0.78** | **0.050 / 0.058 / 0.051** | 0.805 / 0.722 / 0.764 | −19.4 / −65.6 / −17.8 |

**The verdict is unchanged in direction and the numbers that argued it were all wrong.** The
agent lands **~6% of the teacher's rate** while declaring 21–44 charges an episode. It is
modestly above the floor (0.050–0.058 against 0.016), not at it.

⚠ **The clone control's REJECT stands a fortiori** — its clone rows were measured on the broken
instrument, so correction moves them *down*, further below their pre-registered bound. The
`stood/ep` magnitudes in §12 are superseded; the verdict is not.

### The mechanism this exposes — and it is not "the charge is hard to learn"

Declaring is **free** (the charge phase runs after shooting, so it forfeits nothing) and
**unmasked on eligibility**; a failed charge is restored to its exact start state, so it is a
state no-op; and with no shaping term the reward delta is zero. **Free action + no-op outcome +
no reward delta = no gradient.** 21–44 declarations an episode against 0.3–0.8 landings is
precisely what a policy does when nothing pushes back against a spurious declaration.

### Three more defects found in the same audit, each verified here

1. ⚠ **`charge_progress` could not fire on a charge.** It gated on
   `view.game_clock_state.phase`, and reward is calculated *after* `run_after_player_action`
   advanced the clock — so on the charge step the live phase is already the next one. The term
   paid **zero on every charge** and fired on the *shooting* step at pre-charge positions.
   **This is the lever the next spend would have wired.** Fixed by adding
   `StepContext.action_phase` (set from the value `_apply_player_action` already captures
   before advancing) and gating on it. Reward and observation goldens are **bit-identical** —
   the term is unwired, so nothing that ships changed.
   ⚠ **Four unit tests covered this gate and none could see it**: they drove the live clock into
   the charge phase and called the calculator directly, a state no real step ever produces.
   Verbatim the defect this project has already paid for twice.
2. ⚠ **"Held-out nine, n=9" visits FIVE tables.** `MapPool.draw` is uniform **with
   replacement** (its docstring defends that for *training rollouts*). Seeds 700000–700008 draw
   `table_25, table_25, table_40, table_05, table_35, table_30, table_05, table_05, table_40`
   — `table_05` three times, and `table_10/15/20/45` never. **No per-table sign count is
   obtainable**, so CLAUDE.md's own map-pool rule is inexpressible on this config. Every melee
   row above is labelled with a coverage it does not have. `measure-maps`, which iterates
   tables explicitly, is unaffected.
3. ⚠ **The `coherent` clause of the PASS gate never sees the charge phase.**
   `_record_coherency` fires only under `if phase == BattlePhase.movement`
   (`wargame.py:1414`), while §9's stated rationale for the clause is *"a charge is a joint
   move; formation is how it fails"*. The clause measures something else entirely.
4. ⚠ **The comparator under-declares on a stale cap.** `scripted_squad_march.py:305` computes
   the declaration's `reach = min(move, charge_roll)` while `select_charge` at `:416` uses the
   full roll under a comment stating the `min(Move, roll)` reading **was the bug and is now
   closed**. The file's own docstring promises both gates ask the same question. The bar sets
   every target in this goal, including `stood/ep > 3.0`.

### The lessons, as rules

- ⚠ **A DOCSTRING THAT LISTS ITS AUTHOR'S PAST MISTAKES READS AS AUDITED AND IS NOT.** Six
  panellists took `stood/ep` at its documentation's word and every one of them reached the
  wrong verdict from it. **Discount a panel's agreement to the number of seats that
  independently validated the instrument.**
- ⚠ **A metric read AFTER `env.step` is a metric read after the opponent's entire turn.** Any
  per-phase quantity has to be captured inside the step.
- ⚠ **An error that lives in the FAILURE pool is anti-monotone in skill.** Ask of every
  behavioural statistic: does its error rate depend on the competence it is measuring?
- ⚠ **Check what a seeded episode count actually covers.** Sampling with replacement is right
  for training rollouts and wrong for an eval label.

---

## 15. The arm of record — the COMPLETE implementation, pre-registered

Launched 2026-08-26 at `1cfebfa`. Three seeds, 600 epochs, `ent_coef 0.003`,
**unshaped** (`charge_progress` stays unwired, so this arm is the control any shaping
decision has to be measured against). `configs/experiments/25v25_maps_melee.yaml`,
`max_turns` **140**.

⚠ **EVERY EARLIER MELEE FIGURE IS VOID AGAINST THIS ONE**, and not merely stale. Six
changes landed under it: the declaration mask, consolidate Engaging, the drag-in clause,
activation priority, pile-in and consolidate as phases, and both as agent decisions. The
bar moved from **+13.3 → +23.9 → −17.2** across them.

### The bar, and the floor, both taken BEFORE any agent number exists

Held-out nine, refereed, n=9, seeds 700000+, K=1:

| policy | decl/ep | tried/ep | **stood/ep** | frac | coherent | vp |
|---|---|---|---|---|---|---|
| **`squad_march_take_charge`** (the comparator) | 8.00 | 7.00 | **3.67** | **0.524** | 0.850 | **−17.2** |
| `squad_march_take` (never charges) | 0.00 | 0.00 | 0.00 | — | 0.895 | −57.8 |
| floor s0 / s1 / s2 (random init) | 0.00 / 8.67 / 21.67 | — | **0.00 / 0.00 / 0.11** | 0.000–0.005 | 0.561–0.845 | −194.4 / −215.0 / −237.8 |

Charging is worth **+40.6 same-row**, up from +26.7 — the mechanic got *more* valuable as
the rules got more complete.

⚠ **The absolute level fell ~40 vp when pile-in and consolidate became agent decisions,
and that is the ENCODING, not the policy.** The scripted bar calls `domain.pile_in` for
the move the engine itself would have made and encodes that displacement; a continuous 3"
move quantised onto 16 angles × 6 rungs lands models slightly off, breaks unit coherency,
and the referee reverts the whole unit. Recorded as a measured argument for reverting that
one commit — the phases and the activation priority cost nothing and are kept either way.

⚠ **`declared/ep` is USELESS as a gate**, again: the floor spans 0.00 to 21.67. Read
`stood/ep` (floor **0.00–0.11**), the standing fraction (floor **0.000–0.005**) and vp
(floor **−194 to −238**).

### The trichotomy, unchanged from §9 and restated against these numbers

- **PASS** — `stood/ep` **> 3.0** at K=1 *and* beats `squad_march_take_charge` on
  `vp_margin` with a t and a sign count, on ≥ 3 seeds, with `coherent` no lower.
- **FAIL** — does not clear the floor on `stood/ep`, **or** charges competently and still
  loses on vp. ⚠ Different failures, reported as such: the first is about teaching, the
  second about the scenario, and only the first is this goal's.
- **UNDERPOWERED** — the interval contains both zero and the effect that would matter.
  The likely outcome at n=3, and it must not be narrated into a PASS.

⚠ **Scored at n=45 with REPEATS, not the n=9 of §9.** Two defects make n=9 unusable:
`MapPool.draw` samples **with replacement**, so seeds 700000-700008 visit **five** of the
held-out nine; and an agent row does not reproduce run to run while a scripted row does,
so the noise floor is **asymmetric**. Both are recorded in `scripts/measure_charges.py`.

⚠ **The §9 trichotomy has a gap, named before the numbers arrive.** An agent that clears
the floor but comes nowhere near the bar is neither PASS nor FAIL as written. That state
will be reported as what it is rather than forced into a bin.

---

## 16. The arm of record's VERDICT — the named gap, and the diagnosis that follows it

Scored 2026-08-26 at `b133709`, all three seeds at epoch 599 (`last.ckpt`, written by
`on_train_end` — clean completion verified per seed in `wandb-summary.json`). n=45, K=1,
seeds 700000+, refereed. Two independent repeats of the s2 row agree **to every printed
digit**, so the `_rolled_for` fix closed the irreproducibility and these rows are quotable.

| policy | decl/ep | tried/ep | **stood/ep** | frac | coherent | vp |
|---|---|---|---|---|---|---|
| `squad_march_take_charge` (comparator, **fixed bar** — see below) | 8.67 | 7.89 | **5.56** | **0.704** | 0.870 | −14.0 |
| `squad_march_take` (never charges) | 0.00 | 0.00 | 0.00 | — | 0.881 | −34.8 |
| agent s1 | 9.58 | 8.98 | 0.71 | 0.079 | 0.714 | −81.3 |
| agent s2 | 11.91 | 11.16 | **1.09** | 0.098 | 0.746 | −61.1 |
| agent s3 | 6.67 | 6.09 | 0.47 | 0.077 | 0.749 | −81.1 |

**VERDICT: the NAMED GAP.** Clears the untrained floor on `stood/ep` (0.47–1.09 against
0.00–0.11) on 3 of 3 seeds; nowhere near the bar (12–20% of it). Neither PASS nor FAIL as
§15 defines them, and §15 named this state in advance, so it is reported as what it is.

⚠ **The bar in this table is NOT §15's bar.** `stood/ep` 3.67 → **5.56** because the
comparator's declaration gate carried the retired `min(Move, roll)` cap and compared
inches against board units (`68d5204`) — it under-declared by ~22% and every §15 target
descends from it. The trichotomy's 3.0 was 82% of the throttled bar and is 54% of the
real one. Same-row, charging is worth **+20.8** here (−14.0 v −34.8).

### The diagnosis, by referee clause — and two hypotheses died on the way

Instrumented `_enforce_charge` on s2, 20 episodes, per clause, **seat-separated** (⚠ the
first run of this census read both seats and inflated the agent's conversion 12% → 43% —
the opponent is the scripted bar and its charges land in the same instrument):

| clause | agent (159 attempts) | bar (294 attempts) |
|---|---|---|
| **stood** | **11.9%** | 73.1% |
| **reached NOBODY** | **76.7%** | 15.3% |
| clipped a second unit | 1.9% | 4.4% |
| a mover ended not-closer | 5.0% | 3.1% |
| incoherent | 4.4% | 4.1% |

- ⚠ **"The coherency regression causes the conversion failure" is REFUTED** — the clause
  kills 4.4% of the agent's charges against the bar's 4.1%. (The 0.714–0.749 `coherent`
  figures are a movement-phase statistic and a separate finding.)
- Splitting `no_contact` by winnability: only **13.8%** of attempts were doomed at
  declaration (gap > roll); **62.9% were REACHABLE AND MISSED.**
- The geometry of those misses: the best-placed model's heading has **median angle error
  92.2°** against the enemy it declared on (75% of movers > 45° off), and **102.9°**
  against the nearest objective — so it is not walking to objectives either. Only 15.8%
  aimed right and fell short. **The charge-phase action is simply unlearned**: headings
  uncorrelated with anything, exactly what a 0.27%-discovery, all-or-nothing, unshaped
  signal predicts.

### What was done about it, same day (all verified to fail without their fix)

1. **Doomed declarations are masked** (`2c1fb42`): a unit whose roll cannot cover the gap
   to any enemy may not declare, mirroring the advance's own reachability mask and the
   scripted comparator's own gate. Removes the 13.8% zero-gradient trap.
2. **Coherency is recorded in every displacing phase** (`98df112`): the §15 `coherent`
   clause measured the movement phase only — 0 of the 3 melee phases it was written for.
   ⚠ Melee `coherent` is not comparable across this date.
3. **Pile-in/consolidate actually move** (`2c8d402`): eligible was identical to pinned
   (one predicate used twice in opposition) AND the endpoint was refused the engagement
   exemption. **Neither fix works alone — the exemption alone is a REGRESSION** (94.9%
   zero-delivery v 86.5%). Together: 0.103" → 0.525" mean, 86.5% → 28.3% zero. ⚠ Post-fix
   the residual zeros are **21.0% of moving pile-in units reverted by the referee**
   (quantisation misses the "end closer" clause) — so `1cfebfa`'s mechanism is real at the
   ~20% level even though its 40 vp magnitude was the n=9 block draw. The revert decision
   stays open; held to its own pre-registered criterion (mean < 1.0" → revert), 0.525"
   **fails**, and the n≥100 paired score should decide it.
4. **A seeded episode no longer depends on the reset before it** (`67e6838`).

### The next arm, launched 2026-08-26 ~21:30 — `melee-shaping-v4`

The diagnosis says the missing thing is a **gradient toward contact inside the declared
charge**, which is `charge_progress` verbatim (§5b). Wired at `value: 0.25, weight: 1.0`
(`configs/experiments/25v25_maps_melee_shaped.yaml`) — measured live at ~2–4/ep for the
scripted charger against `objective_hold`'s ~16–31, confined by the declaration gate.
**Paired control retrained on the same code** (the four fixes above void the arm of
record as a control): 3 seeds × {control, shaped}, 600 epochs, `ent_coef` 0.003, same
flags. Same action space (108), same seeds → identical inits → **paired estimator**.

**Pre-registered readouts, written before any number exists:**

- **Primary: `stood/ep` and conversion fraction at K=1, n=45**, shaped v paired control.
  The bet is that shaping moves *conversion* (the 92° aiming error), not declarations.
- **Reject** (the calculator's own rule): `vp_margin` falls against the paired control on
  2 of 3 seeds, **or** declarations rise while `held` falls — buying charges with ground.
- The §15 trichotomy still judges the shaped arm against the **fixed** bar (5.56), with
  the same named-gap escape.
- ⚠ Melee `coherent` now includes the melee phases for BOTH arms (fix 2), so neither is
  comparable to any row above this section.

---

## 17. melee-shaping-v4 VERDICT — the gradient works and the annuity is farmed: REJECT, iterate to the delta

Scored 2026-08-27 ~04:30 at `e215a5a` + the calculator's delta patch (uninvolved in these
runs). All six seeds verified at epoch 599 in their own `wandb-summary.json`. n=45, K=1,
seeds 700000+, refereed; the repeat row again matches **to every digit**.

| | decl/ep | stood/ep | frac | coherent | vp | held |
|---|---|---|---|---|---|---|
| bar `squad_march_take_charge` | 8.67 | 5.56 | 0.704 | 0.870 | −14.0 | — |
| ctl s1 / s2 / s3 | 5.67 / 8.16 / 7.73 | 0.53 / 0.40 / 0.62 | 0.053–0.097 | 0.709–0.816 | −44.1 / −79.1 / −85.6 | 1.78 / 1.40 / 1.49 |
| shp s1 / s2 / s3 | 14.60 / 14.02 / 15.29 | **1.76 / 1.31 / 1.69** | 0.100–0.128 | 0.715–0.743 | −66.6 / −83.9 / −66.6 | 1.60 / 1.58 / 1.49 |

**Paired (shp − ctl), the estimator the identical inits buy:**

- **`stood/ep` +1.23 / +0.91 / +1.07 — +1.07 ± 0.16, 3 of 3.** Conversion +0.031 / +0.047
  / +0.032, 3 of 3. **The gradient works**: the shaped arms land 3.1× the control's
  standing charges, and the pairing is tight enough that three seeds settle it.
- vp **−22.5 / −4.8 / +19.0** — mean −2.8, signs flipping.
- `held` −0.18 / +0.18 / 0.00 — flat.

**Verdict against the pre-registered rules: REJECT, on clause 1's letter.** vp fell against
the paired control on 2 of 3 seeds. Clause 2 (decl↑ while `held`↓) does **not** fire.
⚠ Clause 1 is **underpowered by this project's own recorded standard** (−2.8 ± ~12 with a
sign flip; the 2026-08-24 lesson says power-check a per-seed bound before writing it down,
and this one was not). The reject is honoured as written — and the next rule gets
power-checked.

**The mechanism behind the reject was named from the mid-run trajectory before the
endpoint existed**: the term pays the progress **LEVEL** every charge step, so a unit that
declares and hovers near contact collects an annuity without landing — decl 14–15/ep
against the bar's 8.67, conversion flat across epochs 90 → 294 → 430 while the control's
doubled. Its docstring called it "a potential"; a potential-based term pays the
**difference**, and a level on a repeatable state is farmable.

### The iteration — `pay_delta` (v5, in flight)

`charge_progress` gains `pay_delta: true`: pay the distance **closed by the charge move**
(previous step's end positions are the charge move's start, because shooting displaces
nobody), clipped at zero so the referee's revert is not a fine on the attempt. Closing
pays once; hovering pays nothing. Tests drive both forms and fail without the feature.

`configs/experiments/25v25_maps_melee_shaped_delta.yaml`, 3 seeds, 600 epochs, launched
2026-08-27 ~04:45 (`melee-shaping-v5`). ⚠ **The v4 control is REUSED, not retrained** —
training is deterministic given seed + config + code, and the only code change since it
trained is inside a calculator its config never constructs. Same action space, same seeds,
identical inits: paired.

**Pre-registered for v5**: primary is `stood/ep` and conversion paired against the v4
control; the reject rule is v4's verbatim; and the delta form's own bet, written before
any number exists — **declarations land near the bar's 8.67** (the annuity was the
inflation) **while `stood/ep` holds at or above the level form's 1.31–1.76.** If instead
declarations stay at 14+ with the annuity gone, the inflation was never the annuity and
the diagnosis is wrong — say so.

---

## 18. melee-shaping-v5 VERDICT — the delta form is a weak null, and the shaping family is exhausted

Scored 2026-08-27 ~09:30, all three seeds verified at epoch 599. n=45, K=1, seeds
700000+, refereed, paired against the reused v4 control:

| seed | ctl stood | dlt stood | Δstood | Δvp | Δheld |
|---|---|---|---|---|---|
| s1 | 0.53 | 0.58 | +0.05 | −28.8 | −0.36 |
| s2 | 0.40 | 0.71 | +0.31 | +4.5 | −0.02 |
| s3 | 0.62 | 1.49 | +0.87 | +32.0 | +0.02 |

- **The §17 bet FAILS on its stood half**: 0.58–1.49 sits below the level form's
  1.31–1.76 on 2 of 3 seeds. Declarations (5.1–15.0, mean 9.2) did land near the bar's
  8.67 — so the v4 inflation *was* partly the annuity — but the mechanism went with it.
- Reject clause 1 does **not** fire (vp fell on 1 of 3). Clause 2 is marginal by means
  (decl +2.0, held −0.12) and driven by different seeds on each half; not treated as a
  reject on its own.
- **The mechanism was predicted before the endpoint**: a reverted charge restores start
  positions, so the delta pays **zero on every near-miss** — it is a success-only bonus,
  the exact discontinuity the calculator's own docstring warns against. The mid-run
  read (stood 1.89 at epoch ~215 on 2 seeds) reversed by 600 — ⚠ the third mid-run
  reversal on file; screens are screens.

**The shaping family is exhausted: the LEVEL form has a gradient and is farmable; the
DELTA form is unfarmable and gradient-free.** Anything between (intended-position
deltas) pays the referee's revert as if it stood, which is a different farm.

### The action-space turn (v6) — mask the charge move to the referee's own clause

The pooled-logit screen also ran (2026-08-27, frozen v4-shp weights, n=20 × 3 seeds):
rigid-unit decoding lifts stood/ep **1.19–1.31×** against a pre-committed kill line of
**2×** — **the rigid charge is dead by its own rule**. Coordination is worth ~25%;
the failure is per-model DIRECTION, and averaging five wrong directions is a wrong
direction.

So v6 changes what a charge move can BE: `charge_legality` additionally masks, per
model, every movement action whose endpoint does not end **closer to the unit's charge
target** — the same "each model must end its move closer" clause the referee already
enforces, applied at action time. It removes only moves that would void the charge
anyway, adds no actions, keeps every checkpoint loadable, and turns the 77%
reached-nobody failure into a distance-and-coherency problem, which is where the bar's
0.704 conversion lives. ⚠ This is NOT the joint "if it can" mask that collapsed the bar
5.67 → 1.67 — that one forced ENGAGEMENT, a joint property no per-model mask can see;
"closer" is the rules' own per-model test.

---

## 19. melee-approach-v6 VERDICT — the mask is the first honest mechanism gain, a hair short of its own bet

Scored 2026-08-27 ~14:45, three seeds verified at epoch 599, n=45, K=1, seeds 700000+,
scored on the arm's OWN masked game (`25v25_maps_melee_approach_refereed.yaml` — ⚠ the
scenario-override path SILENTLY DROPS nested keys, so the eval config exists rather than
an override). Repeat row identical to every digit. Paired against the reused v4 control:

| seed | ctl stood | apr stood | ctl→apr conv | Δvp | Δheld | apr decl |
|---|---|---|---|---|---|---|
| s1 | 0.53 | 0.58 | 0.097→0.165 | −23.8 | −0.34 | 3.64 |
| s2 | 0.40 | **2.09** | 0.053→**0.264** | +16.9 | +0.49 | 8.58 |
| s3 | 0.62 | **2.51** | 0.085→0.192 | +20.4 | +0.13 | 13.84 |

- **The pre-registered bet fails by its letter, narrowly, on both halves**: stood ≥ 2.15
  on 1 of 3 (s2 misses by 0.06), conversion > 0.20 on 1 of 3 (s3 misses by 0.008).
- **The power-checked reject rule does NOT fire**: vp falls > 1 SE on 1 of 3 only, and no
  seed reaches the 14–15/ep spam line.
- **The two healthy seeds are the best honest melee rows ever trained here** — stood
  2.09/2.51 (v4-shp's 1.31–1.76 was farmed; this is unshaped), conversion the highest
  measured on any trained arm, vp and held both UP against the paired control.
- ⚠ **s1 collapsed to low-declaration for the third time** (v3, v5, v6 — same seed
  number, same init each time). The never-charge basin appears to be an INIT property.
  Weigh seed counts, and consider excluding-and-reporting rather than averaging over it.
- ⚠ **s1 also regressed between epoch 481 (stood 1.67, conv 0.203, decl 9.11 at n=9) and
  599 (0.58, 0.165, 3.64 at n=45)** — late-training declaration decay, the v4-s1 pattern.
  Lever-usage-as-convergence-signal applies: s1 was not settled at 599.

**Arc across the arms (mean stood/ep, all at 600 epochs, n=45, K=1):** v3 0.76 →
v4-level 1.59 (farmed, REJECTED) → v5-delta 0.93 (null) → **v6-mask 1.73, unshaped,
nothing to farm**. Against the trichotomy the verdict is still the NAMED GAP (bar 5.56 /
0.704), but the conversion gap closed from 7× to 2.7× on the best seed.

**Next (v7): the `engaged` observation column, zero-initialised, on top of the mask.**
The agent can now aim a charge; it still cannot SEE the shooting shield that makes one
worth standing (probe AUC 0.75 on trained latents against a 0.95 kill line — funded).
Zero-init makes the logits bit-identical at step 0, so **v6 is v7's paired control**.

---

## 20. v6 at SIX seeds — the approach mask replicates

Seeds 4–6 trained 2026-08-27 (⚠ relaunched once: the first attempt's recording
subprocesses crashed on a mid-run disk edit — the `observe_engaged` column landing under
live runs — killed by PID at epoch ~10 and restarted on code digest-verified bit-identical
for this config; partial checkpoints moved aside). All verified at epoch 599. n=45, K=1,
masked eval:

| seed | decl/ep | stood/ep | conv | coherent | vp | held |
|---|---|---|---|---|---|---|
| s1–s3 | (see §19) | 0.58 / 2.09 / 2.51 | 0.165 / 0.264 / 0.192 | | −67.9 / −62.2 / −65.2 | |
| s4 | 4.51 | 0.93 | 0.215 | 0.730 | −89.3 | 1.24 |
| s5 | 10.71 | 2.18 | 0.216 | 0.805 | −40.7 | 1.67 |
| s6 | 10.04 | 2.13 | 0.226 | 0.663 | −54.7 | 1.60 |

- **stood/ep 1.74 ± 0.80 against the control's 0.52 ± 0.11, unpaired t ≈ 3.7.**
- **Conversion 0.165–0.264 on 6 of 6 seeds against the control's 0.053–0.097 — the bands
  do not overlap.** The mask's effect on aim is no longer a screen; it is a replicated
  result.
- vp +6.3 against the control, not significant; no seed pays for the mechanism.
- ⚠ The "never-charge basin" is a **continuum, not a binary**: declaration rates span
  3.6–13.8/ep and s4 landed mid-range (0.93) rather than at zero. Do not model it as a
  discrete collapse mode.
- Against the bar (5.56 / 0.704): still the NAMED GAP, at 31% of the bar's standing rate
  and 30% of its conversion on the six-seed means.

**v8 launches on this baseline**: `25v25_maps_melee_engaged.yaml`, six seeds, unpaired
against these six, pre-registration in its config header.

---

## 21. v8 REJECTED — and v6 at the PLAY decode converts at the bar's rate

Scored 2026-08-28 ~03:45, six v8 seeds verified at 599, n=45, seeds 700000+.

**v8 (`observe_engaged` on top of the mask): REJECT by both of its own clauses.** Six-seed
means, unpaired against v6's six: stood 1.13 ± 0.52 v 1.74 ± 0.80 (the bet demanded an
increase and measured a decrease), vp −78.7 v −63.3 (−15.4, beyond the ~10.5 unpaired SE
— the reject clause as written). The AUC-0.75 funding argument did not cash:
⚠ **a feature the network does not represent is not thereby a feature it needs** —
visibility is not value, and the probe gate ("AUC > 0.95 kills it") only ever licensed
the spend, never promised the return. `observe_engaged` stays in the codebase, default
off; the v8 arm is retired.

**The v6 residual, by referee clause** (s3, 20 episodes, seat-separated): `not_closer`
is ELIMINATED (0.4%), and the failures moved DOWN the ladder — `no_contact` 76.7% → 53.9%
(now pure distance shortfall, since every move approaches) and `incoherent` 4.4% →
**19.9%** (five models converging on one point stretch the 2" chain — a failure the
random-heading policy never survived long enough to reach). Both residuals are JOINT
properties of the unit's combination.

**And the play-time joint decoder resolves exactly those.** v6's six seeds at K=3 — the
decode every published agent score uses:

| seed | stood/ep | conversion | vp |
|---|---|---|---|
| s1 | 1.02 | 0.719 | −44.2 |
| s2 | 4.24 | 0.872 | −64.8 |
| s3 | **5.53** | 0.773 | −35.8 |
| s4 | 1.67 | 0.676 | −31.6 |
| s5 | 3.87 | 0.744 | −11.0 |
| s6 | **5.67** | 0.787 | −36.1 |

- **Conversion 0.676–0.872 on 6 of 6 seeds — at and above the bar's 0.704.** The best two
  seeds stand 5.53/5.67 charges against the bar's 5.56.
- ⚠ The §15 trichotomy's mechanism gate stays quoted at K=1 BY DESIGN (K=3 measures the
  decoder as well as the policy), and at K=1 the verdict remains the NAMED GAP. Both
  statements are true, and the honest claim is conditional: **at the decode that ships,
  the agent executes the charge phase at the bar's conversion rate.** The mask is what
  made this true — the same six checkpoints before it converted 0.05–0.10 raw and the
  decoder cannot verify its way out of an empty candidate set.
- vp at K=3: mean −37.3 against the bar's −14.0 — the agent still loses the GAME by ~23,
  which is Level 2/3 territory (play quality), not phase-execution.

**Where the goal stands.** Level 1 — the agent plays through every phase, both seats play
every phase, and charge execution at the play decode is bar-adjacent: **substantially
met**, with the K=1 raw-policy gap as the standing caveat. Level 2/3 (the ladder, the
bar on vp) are open, and the next levers are play-quality ones, not phase-mechanics ones.

---

## 22. Level 2 measured — positive against the ladder, and one Level-3 signal

Scored 2026-08-28 ~04:45. v6 seeds 5 and 3 at K=3 (the play decode) against the
non-charging ladder on the masked eval configs; the charging bar at K=1 for the same-row
comparison. n=45, seeds 700000+, refereed.

| opponent | agent s5 (stood/conv/coherent/vp) | agent s3 | bar `take_charge` |
|---|---|---|---|
| `squad_march_take` | 3.38 / 0.772 / 0.953 / **+4.7** | 4.62 / 0.715 / 0.956 / −5.8 | 6.71 / 0.851 / 0.882 / +16.6 |
| `squad_march_deny` | 3.40 / 0.797 / 0.950 / **+24.7** | 4.62 / 0.717 / 0.947 / +4.1 | 6.07 / 0.794 / 0.882 / **+10.4** |
| `squad_march_shoot` | 3.16 / 0.768 / 0.938 / **+32.4** | 4.11 / 0.743 / 0.954 / +17.4 | 5.96 / 0.709 / 0.887 / +53.3 |

- **Positive vp in 5 of 6 agent cells.** Charge competence transfers across the ladder
  (conversion 0.71–0.80 against every opponent), and the agent's historic coherency edge
  is intact (0.938–0.956 v the bar's 0.882–0.887).
- ⚠ **One Level-3 signal, not a claim**: s5 beats the bar same-row against `deny`
  (+24.7 v +10.4) — one seed, one opponent, n=45, UNPAIRED, no t, no sign count. It also
  echoes this project's oldest pattern (the agent has always been strongest where denial
  pays), so treat it as the familiar trait resurfacing with a charge attached, and
  demand the full apparatus before quoting it: all six seeds, per-table sign counts via
  `measure-maps`, and the bar re-measured per opponent.
- Agent rows at K=3, bar at K=1 — the house convention (an agent ships with its decoder;
  a script is its own decode), stated because a table that does not name its decode gets
  compared to the wrong one.

**Level 2 — "reasonably perform against the opponent ladder" — is met on this evidence.**
Level 3 needs the six-seed, all-opponents grid and the vp gap against the charging bar
itself (−37.3 v −14.0 at K=3) closed.

---

## 23. ⚠ CORRECTION — §22's "Level 2 met" was WINNER-SELECTION BIAS. The six-seed grid.

Scored 2026-08-28 ~05:30, all six v6 seeds × four opponents, K=3, n=45. §22's two agent
rows were **s5 and s3, chosen as the best seeds by their K=3 vp** — the comparator error
CLAUDE.md § How to measure names ("a best chosen by argmax on the same data changes
identity between cells"), committed at seed level, by me, hours after quoting the rule.
The original stands above, uncorrected, per house practice.

| seed | vs `take` | vs `deny` | vs `shoot` | vs `take_charge` |
|---|---|---|---|---|
| s1 | −7.7 | −7.8 | +16.0 | −44.2 |
| s2 | −24.8 | −31.9 | +2.3 | −64.8 |
| s3 | −5.8 | +4.1 | +17.4 | −35.8 |
| s4 | −6.9 | +1.1 | +13.0 | −31.6 |
| s5 | +4.7 | +24.7 | +32.4 | −11.0 |
| s6 | −8.2 | −5.3 | +15.0 | −36.1 |
| **mean** | **−8.1** | **−2.5** | **+16.0** | **−37.3** |
| bar (K=1) | +16.6 | +10.4 | +53.3 | −14.0 |

- **What survives**: positive against `squad_march_shoot` on **6 of 6 seeds** (mean
  +16.0); conversion 0.68–0.87 against every opponent; coherency 0.94–0.96 everywhere.
  The charge SKILL is real and transfers.
- **What dies**: "positive in 5 of 6 cells" (it is 9 of 24); the `deny` Level-3 signal
  (s5's +24.7 is the seed tail — the six-seed mean is −2.5, 3 of 6 positive).
- **Level 2 verdict, corrected: PARTIAL.** Competitive with the ladder (means −8 to +16),
  clearly ahead of one opponent, behind the bar's same-row on all three.
- **Level 3: open.** Behind the charging bar's mirror row on 5 of 6 seeds (s5's −11.0 v
  −14.0 is one seed, 3 vp, n=45 — nothing).
- ⚠ The standing pattern held where the §22 framing missed it: the agent's best opponent
  is the one that shoots most (`shoot` — where its formation discipline and the shield
  pay), not the one where denial pays. Re-derive doctrine from the six-seed grid only.

---

## 24. The 1000-epoch extension — "train longer" is REFUTED as the route to Level 3

All six v6 seeds resumed 599 → 999 (resume verified advancing past 599 — the silent-death
signature checked explicitly), scored 2026-08-28 on the full grid, K=3, n=45:

| vp mean | vs `take` | vs `deny` | vs `shoot` | vs `take_charge` |
|---|---|---|---|---|
| @600 | −8.1 | −2.5 | +16.0 | −37.3 |
| **@1000** | −10.3 | −5.5 | +9.8 | −47.2 |

- **Every column flat or worse; 15 of 24 per-seed deltas negative.** The vp gap is NOT
  undertraining — the same shape as the advance arm, where 700 extra epochs also hurt.
- **What the extension DID buy: consolidation of the mechanics.** No collapse seeds
  remain (min stood 2.29 v 1.02 at 600), conversion 0.693–0.845, and coherency
  **0.916–0.948** against 0.685–0.805 at 600 — the agent's historic formation discipline
  restored on the melee game, with the charge skill intact.
- **The diagnosis this closes**: at 1000 epochs the agent is a disciplined, bar-adjacent
  charger that loses on PLAY QUALITY. The remaining Level-2/3 gap is the project's
  standing pre-melee problem — the allocation/search failure (§ The critic already
  knows) — not a melee mechanic. No melee-side iteration is likely to move it, and the
  record already shows three reward arms and four movement-side fixes failing against
  that wall.

**Where the goal's levels stand, on six-seed 1000-epoch evidence:**
- **Level 1 — met at the play decode** (all phases, both seats, rules-true, bar-adjacent
  charge execution, formation discipline restored).
- **Level 2 — partial** (ahead of `shoot`, break-even `deny`, behind `take`).
- **Level 3 — open, and now demonstrably NOT blocked on melee**: the binding constraint
  is the allocation/search problem, which predates melee and needs its own programme.

---

## 25. The reallocation decode SURVIVES its kill — the first allocation lever ever to do so

Scored 2026-08-28. `scripts/measure_reallocation_decode.py` (recipe `measure-realloc`):
per movement phase, `choose_branch` (the critic probe's own instrument, unchanged)
nominates one surplus squad on the biggest stack and the cheapest empty objective; the
critic prices a virtual full-move translation and approves on SIGN alone (corr(dV,dVP)≈0
forbids ranking); approved members are redirected onto the one shared movement-grid cell
nearest the target — rigid, referee still judges. Play-time only.

⚠ The first 24-cell screen scored every episode **exactly 0.0** — `info.get("vp_margin",
0.0)` on an info dict that carries no such key. Caught because a column of identical
zeros is not a result; fixed to read the env's own counters (`d896b3a`); the sanity row
went 0.0 → +3.9 with bit-identical mechanism counts. **A default on a `dict.get` is a
silent instrument.**

Six 1000-epoch seeds × four opponents, n=45, K=3, realloc minus the same checkpoints'
no-realloc grid rows:

| Δvp | vs `take` | vs `deny` | vs `shoot` | vs `take_charge` | overall |
|---|---|---|---|---|---|
| mean | −3.1 | +2.8 | **+9.6** | +2.2 | **+2.86 ± 2.12** |

- **Pre-registered kill (mean < +1.0 OR ≥3 of 6 seeds negative): does NOT fire** — the
  first allocation-side intervention on this project's record to survive its own screen,
  after three reward arms, four movement fixes and `assignment_optimal` all died.
- Sober: t ≈ 1.4 on the overall mean — clears the line, not significant on its own. The
  clean result is **`shoot`: positive on 6 of 6 seeds** (+3.4 to +18.9) — redistribution
  reliably pays against the opponent that punishes stacking hardest.
- Usage: ~6.6 nominations/ep, ~75% critic-approved, ~23 model-moves redirected/ep.
- **What it does NOT do**: close Level 3. The charge-bar column moves −47.2 → −45.0.
  The decode redistributes what the policy already holds; it does not make the policy
  attack. Level 2 with the decode active: `shoot` +19.4 (clearly ahead), `deny` ≈ 0,
  `take` behind.

---

## 26. CONTEST MODE — the first offence-side intervention ever to survive, at 4× spread

Scored 2026-08-28. Same decode, same kill, one change: the target is the opponent's
WEAKEST-HELD objective instead of an empty one — the attack the policy never makes.
Six 1000-epoch seeds × four opponents, n=45, K=3, paired per cell against the same
checkpoints' no-realloc rows (same seeds, same layouts — only the decode differs):

| Δvp | vs `take` | vs `deny` | vs `shoot` | vs `take_charge` | overall |
|---|---|---|---|---|---|
| contest | +8.3 | +9.8 | +14.8 | +12.2 | **+11.30 ± 3.27** |
| (spread) | (−3.1) | (+2.8) | (+9.6) | (+2.2) | (+2.86 ± 2.12) |

- **Positive on 6 of 6 seeds and 22 of 24 cells, t ≈ 3.5.** The kill does not fire, by a
  wide margin, and the effect is 4× spread mode's.
- **The bet resolved exactly as posed.** The 2026-08-11 teleport audit priced FORCED
  contests at −29.4 of the committing squad's income; the critic's sign gate refuses
  ~33% of nominations, and what it approves pays. The audit measured the move without a
  gatekeeper; the gatekeeper is what three reward arms could never express — a reward
  term pays a *behaviour class*, the critic prices *this board*.
- **This is the project's standing diagnosis cashed**: reward and critic both valued
  attacking correctly, the policy could not act on it, and a decode that lets the critic
  act at play recovers +11.3 vp without a single gradient step.
- Absolute standing with contest active: `take` ≈ −2.0, `deny` ≈ +4.3, `shoot` ≈ +24.6,
  `charge` ≈ −35.0 against the bar's +16.6 / +10.4 / +53.3 / −14.0. **The same-row gaps
  are roughly halved; Level 3 is open but no longer static.**
- Next: compose the modes (contest when a target exists, else spread), screen the
  composition identically, and graduate the winner into the selector path.

---

## 27. The composed decode — contest-else-spread graduates, and the goal's final table

Scored 2026-08-28. `mode=both`: contest when the opponent under-holds a point, spread as
the fallback. 24 cells, same kill, paired per cell:

| Δvp | vs `take` | vs `deny` | vs `shoot` | vs `take_charge` | overall |
|---|---|---|---|---|---|
| **both** | +9.2 | +10.0 | +14.7 | +12.4 | **+11.55 ± 3.35, 0/6 negative** |
| contest | +8.3 | +9.8 | +14.8 | +12.2 | +11.30 ± 3.27 |
| spread | −3.1 | +2.8 | +9.6 | +2.2 | +2.86 ± 2.12 |

No interference — contest dominates and spread fills the boards with nothing to contest.
**The composed decode is the graduating form.**

### The goal's standing, at the close of the 2026-08-26→28 programme

Absolute vp, six-seed means, n=45, K=3, composed decode active, against the bar same-row:

| | agent | bar | gap |
|---|---|---|---|
| vs `take` | −1.2 | +16.6 | −17.8 |
| vs `deny` | +4.4 | +10.4 | −6.0 |
| vs `shoot` | +24.5 | +53.3 | −28.8 |
| vs `take_charge` | −34.9 | −14.0 | −20.9 |

- **Level 1 — MET.** Every phase playable on both seats, rules-true (gap map current,
  pile-in pair fixed, comparator honest), charge conversion 0.68–0.87 at the play decode
  against the bar's 0.704, formation discipline 0.92–0.95.
- **Level 2 — MET on this evidence**: positive vp against `deny` and `shoot`,
  −1.2 against `take` — the agent performs reasonably against the whole ladder, with the
  composed decode as part of what ships (decode is already how every agent number here is
  quoted; this adds two critic-gated redirect rules to it).
- **Level 3 — OPEN, gaps halved**: −6 to −29 same-row against the bar, from −14 to −47
  before the decode. The remaining distance is play quality under gradient — the next
  programme, with the decode's survival as its first funded hypothesis (what the critic
  can execute at play, training should be able to internalise; how, without the −51.8
  training-decode trap, is the open design question).
