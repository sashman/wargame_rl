# Melee: what is still missing

> **Read §0 first.** Four of these gaps make the charge *harder* than the
> rules specify, and the melee configs have no trained lineage to protect — so fixing them
> is nearly free today and gets expensive the moment a melee agent exists.

Every melee rule this environment does not yet implement, with its `DEFERRED` tag, what it
blocks, and what has been measured about it. The authority is
[docs/rules/](rules/README.md) and the per-rule ratings live in
[implementation-status.md](rules/implementation-status.md); this page is the **work list**
derived from them, ordered by what depends on what.

Melee is a **core rule**, not an experiment — see
[melee-teaching-goal.md](melee-teaching-goal.md). So everything here is *outstanding work*,
not a menu of optional features.

⚠ **Two of these rules have no gap-map row at all** and were found by a rules-lawyer audit on
2026-08-25, not by the register. Both are marked ★ below. `tests/test_implementation_status.py`
cannot catch a missing row — it only checks that a `DEFERRED` symbol does not exist — so the
register's coverage is itself unverified.

---

## 0. Is our charge HARDER than the rules? Yes — in four ways, and one is a one-line fix

The reason this matters: **an agent should not be made to learn a mechanic that is stricter
than the game specifies.** Difficulty the rules do not ask for is difficulty that buys nothing
and costs sample efficiency. The ledger runs both ways, so here is all of it.

### HARDER than the rules

| what | ours | the rules | measured cost |
|---|---|---|---|
| **Engagement range** | **1"** | **2"** | ⚠ **The single biggest dial.** A charge must END inside it, so halving it roughly halves the target. Measured on the bar, held-out nine, n=9, K=1: standing fraction **0.667 → 0.773**, `stood/ep` **3.11 → 3.78**. |
| ~~**Charge reach**~~ | ✅ **FIXED** — spans 2D6 | the 2D6 alone, up to 12" | was 13.8–15.6% of eligible declarations blocked by the cap; now zero (§1.1) |
| **Multi-target charge** | refused — "exactly one enemy unit" | *"one or more"* | **7.5%** of charge-eligible unit-turns have 2+ reachable enemy units (§1.3) |
| **Pile in** | absent | 3" close-up in the fight phase | a unit landing at the edge of contact cannot close, and one that loses its nearest model cannot re-engage (§2.1) |
| **The declaration is made two phases early** | in the **command** phase, before the unit moves or shoots | in the **charge** phase, when positions are final | the agent must predict where it will stand after moving, and whether a charge will reach from there. **No gap-map row.** |

✅ **FIXED 2026-08-26 — a declared unit may now DECLINE.** The rules grant it explicitly
(`11-charge-phase.md` step 3: *"and the controlling player still wants to make it … Otherwise
the unit does not move"*) and the implementation had removed it, striking STAY from every
declared model holding a legal rung. It bound **30 of 31** declared units, and compounded with
the row above: a unit committed in the command phase could not back out of a charge the
movement phase had made hopeless. The scripted bar's `stood/ep` is **unchanged at 3.11**, so
the constraint only ever cost a policy that wanted to decline — i.e. the agent.

### EASIER than the rules

| what | ours | the rules |
|---|---|---|
| The charge roll | visible before the declaration | declare blind, *then* roll |
| Distance | straight-line | measured **around obstacles** — *"a target 7" away as the crow flies may be unreachable on a 7" roll"* |
| The two *"if it can"* while-moving clauses | absent | each model must end within 1" of a target, and engaged with one, **if it can** |
| The fight itself | auto-resolved, no agent action | pile-in, alternating activation, weapon and target choice, passing, overrun |

### ⚠ The melee family has NO trained lineage, so most of this is free to fix now

Changing `engagement_range` normally voids every baseline and every agent score on a config —
which is why it still defaults to 1. **But no learned policy has ever been evaluated on a
melee config at all**: published checkpoints fail to load at the observation width, and again
at the policy head. There is nothing on the melee configs to void.

So the ordinary objection to adopting the rules' value does not apply here, and the same is
true of §1.1 and §1.3. **The cost of fixing these rises the moment a melee agent exists.**

⚠ Two cautions before adopting 2" wholesale. It also changes **which shots are legal** — the
shooting engagement gate reads the same scalar — so it is a scenario change on both seats, not
a charge tweak. And `vp_margin` moved **+11.1 → −5.0** in the measurement above, because the
*opponent* charges better too; that is a two-seat effect and not a reason against, but it means
the bar must be re-measured rather than carried across. `engagement_range` is now a scenario
override (`just measure-charges ... engagement_range=2`), so this is one command, not a config
fork.

---

## 1. Charge phase

| # | Missing | Tag | Measured |
|---|---|---|---|
| 1.1 | ✅ **DONE 2026-08-26** — the charge now spans 2D6, not Move | ~~`charge.beyond_move_ladder`~~ | A dedicated charge ladder (2"/4"/6"/8"/10"/12") in the charge phase, **zero new actions**, Move-independent per the rules. Bar: `stood/ep` 3.11 → **3.22**. |
| 1.2 | **Charge target declaration** | `charge.target_declaration` | The target is DERIVED from where the unit ends. ⚠ The reason on file is **stale** — it argues a declaration would cost the declarer its move, which the command-phase `move_type` slice disproves. |
| 1.3 | **Multi-target charge** ★ | *no row* | The rules select *"one or more"* enemy units; `_enforce_charge` requires **exactly one**. Opportunity measured at **7.5%** of charge-eligible unit-turns (12 of 174 at two reachable units, 1 at three). Currently **0 of 36** moved charges clip a second unit, so the refusal costs nothing observable *yet*. **Blocked by 1.2.** |
| 1.4 | **The two "if it can" while-moving clauses** ★ | `charge.while_moving_best_effort` | *"Must end within 1" of a target if it can; must end engaged with one if it can."* The third clause — *end closer* — **was implemented 2026-08-25**; these two need a per-model reachability test. |
| 1.5 | **Declare blind, then roll, then choose targets** | `charge.blind_declaration` | The 2D6 is rolled at the start of the side's turn and is visible when the charge is declared. Deliberate: legality is gated on the roll, so a blind declaration would have no legal distance. |
| 1.6 | **Charges resolved one unit at a time, by player choice** | *partially* | `_charge_batches` does resolve unit by unit with the referee between them — but all units' actions come from **one observation**, so a player cannot condition unit 2's charge on unit 1's outcome. |
| 1.7 | **Distance measured around obstacles** | (terrain) | *"A target 7" away as the crow flies may be unreachable on a 7" roll."* Movement ignores terrain entirely, so all charge reach is straight-line. Documented under **terrain**, never mentioned in the charge rows — so 1.1's reachability figures overstate the rules' reach on both sides. |

## 2. Fight phase

| # | Missing | Tag | Measured / note |
|---|---|---|---|
| 2.1 | ✅ **DONE 2026-08-26** — the 3" close-up | ~~`fight.pile_in`~~ | `domain/pile_in.py`, both seats, active player first, models in base contact pinned, all-or-nothing at the unit. Only the CHOICE of which units pile in remains (`fight.pile_in_choice`). **Unblocks 2.2 and 3.1.** |
| 2.2 | **Passing** | `fight.passing` | Needs 2.1 — with no pile-in there is nothing to wait for. |
| 2.3 | **Alternating activation** | `fight.alternating_activation` | Order is fixed: active player's units then the opponent's, chargers first within each. The rules alternate and return to the Strikes First sub-step whenever a new such unit becomes eligible. **Blocks 2.4 and 3.2.** |
| 2.4 | **Overrun fight** | `fight.overrun` | A unit that destroys its target cannot reach a new one this phase. |
| 2.5 | **Select a melee weapon** | `fight.select_weapon` | A no-op while every model carries at most one profile. |
| 2.6 | **Select a melee TARGET unit** ★ | *no row* | A fighting model always strikes the lowest-indexed engaged defender's unit. Fires only where a model is engaged with two enemy units, which 1.3's refusal currently makes impossible. **Unblocked by 1.3.** |
| 2.7 | **Fighting after death** | `fight.fighting_after_death` | ⚠ **A granted ABILITY, not a default** — *"some rules let models attack after being destroyed"*. No model has it, so it is genuinely inert; closing it needs an ability system. The reason on file was wrong and would have made the game incorrect. |
| 2.8 | **Strikes First as a grantable ability** | (partial) | Implemented as "chargers fight first", which is the whole effect while a charge is its only source. The sub-step structure needs 2.3. |
| 2.9 | **The fight phase carries no agent action** | — | Its mask offers exactly one legal action per model, so it stays in `skip_phases` and auto-resolves on the boundary. Every item above is resolved *for* the player. |

## 3. Consolidate

| # | Missing | Tag | Note |
|---|---|---|---|
| 3.1 | **Ongoing mode** | `consolidate.ongoing` | An engaged unit does not consolidate. Needs a pile-in style move with base-contact models pinned — i.e. **2.1**. |
| 3.2 | **Engaging mode** | `consolidate.engaging` | Drags fresh enemy units into the fight and grants each a swing, which needs **2.3**. |
| 3.3 | **The player selects which objective** | `consolidate.select_objective` | The env takes the nearest in range. Arises only for a unit that already has an objective within 3". |

⚠ Objective mode **is** implemented, and fires roughly **once per five episodes** — correctly, because the modes are ordered and compulsory, so only a unit that killed everything near it reaches it.

## 4. Fall back

| # | Missing | Tag | Note |
|---|---|---|---|
| 4.1 | **Declared as a move type** | `fallback.declared_move_type` | v1 *infers* it (began engaged, then moved). A declaration would cost an action; inferring costs none. The `move_type` slice now exists, so this is cheap. |
| 4.2 | **Reckless break** | `fallback.reckless_break` | Needs a per-model backfire roll and a suppression roll. **Neither suppression nor backfire exists anywhere**, so this is the deepest item on the page. |

⚠ **Falling back currently costs an engaged unit nothing it has.** It already cannot shoot (unit-level gate) and cannot declare a charge (eligibility requires unengaged), so the only price of breaking a lock is losing the shield. Lock duration is mean **4.90** battle rounds, median 3, with **16.7%** lasting ten or more — which may be an artefact of scripted policies that never withdraw rather than a property of the game.

## 5. Supporting rules that melee depends on

| # | Missing | Note |
|---|---|---|
| 5.1 | **Engagement range is 1", the rules say 2"** (5" for the fight step) | `engagement_range` defaults to 1 because every baseline and trained result was measured at 1. Adopting the rules value changes which shots are legal — a **scenario change to be measured, not a correction**. |
| 5.2 | **No vertical component to engagement** | Board is 2D throughout. |
| 5.3 | **One melee profile per model** | `MeleeWeaponProfile` carries attacks / melee skill / strength / AP / damage; no multiple profiles. Blocks **2.5**. |
| 5.4 | ✅ **ALREADY DONE** — melee kills DO reach `model_kills` | `wargame.py:1563-1566` folds `_last_player_fight_results` into `p_kills_by_model`, covered by `tests/test_fight_phase.py:123`. ⚠ **This inventory listed it as missing and was wrong.** Verified in code before acting, which is the only reason it was not 'fixed' twice. |
| 5.5 | **`MONSTER` / `VEHICLE` distinction** | Absent, so "engaged large models can be shot at" cannot be expressed. |

---

## Dependency order

Nothing above is blocked by more than one layer, and the graph is shallow:

```
1.2 target declaration ──► 1.3 multi-target charge ──► 2.6 select a melee target
2.1 pile in ──► 2.2 passing
           └──► 3.1 consolidate Ongoing
2.3 alternating activation ──► 2.4 overrun
                           └──► 2.8 Strikes First as an ability
                           └──► 3.2 consolidate Engaging
5.3 multiple profiles ──► 2.5 select a melee weapon
(suppression + backfire, neither built) ──► 4.2 reckless break
```

**Three items unblock most of the rest**: `charge.target_declaration` (1.2), `fight.pile_in`
(2.1) and `fight.alternating_activation` (2.3).

## What to weigh when picking

- ⚠ **The whole charge mechanic is worth about +14.22 ± 8.64 vp** to the script that uses it
  best, against an n=6 vp MDE of **13.54**. Nothing on this page should be justified by an
  expected vp gain — none of them can be resolved that way. Justify on rules-faithfulness and
  on whether the agent's *option set* is wrong.
- ⚠ **Anything that adds actions breaks pairing** and needs a `dark_action_slices` control of
  identical shape. 1.2 adds `max_groups` actions; 4.1 adds one value to an existing slice;
  1.1 can be done with **zero** new actions by re-decoding the charge phase's speed rungs.
- ⚠ **Every item here voids melee baselines on the config it lands on.** Re-measure the bar
  with `just measure-charges` rather than carrying a figure across.
- **The two ★ items are the register's own blind spot.** Before implementing anything here,
  re-read the rules chapter and check a row exists for every clause in it — the audit found
  two clauses with no row, and one row (`2D6 caps the charge distance`) rated `implemented`
  when it was mask-only.


---

## 6. What "completed" means, and what it cannot mean

Set 2026-08-26. **Measurements of record are taken against the COMPLETED implementation**, not
against an intermediate one — this project's own most expensive rule is *"measure the
configuration that SHIPS"*, and a melee bar has already been voided five times in one day by
rules work landing under it.

### Completable, and the order

⚠ **This list is checked against the code before each item is started, not trusted.**
5.4 was listed as missing and turned out to be already implemented and already tested.
The inventory has now been wrong in BOTH directions in one day — two rules with no row at
all, and one closed item recorded as open.

| # | item | why it is next |
|---|---|---|
| 2.3 | ✅ **DONE** — alternating activation | `resolve_fight_step`, with the Strikes First sub-step and passing |
| 3.1 | ✅ **DONE** — consolidate Ongoing | reuses `pile_in`; the spec gives Ongoing pile-in's conditions exactly |
| 3.2 | **consolidate Engaging** | needs 2.3 |
| 2.4 | ✅ **DONE** — overrun | keyed on step-START eligibility, since a live read cannot tell a unit that LOST its target from one that never had one |
| 2.2 | ✅ **DONE** — passing | landed with 2.3; a player who cannot select hands over rather than passing, which is the rules' other branch |
| 4.1 | ⛔ **DECIDED AGAINST** — keep inferring | It costs an action and adds NO expressiveness: an engaged unit already chooses by moving or not, and its only legal move is a fall back. Implementing it would make the action space larger for nothing, against the standing instruction not to hamper the agent. ✅ What WAS a real gap and is now fixed: an engaged unit could **advance**, withdrawing `M + roll` where the rules cap a fall back at `M`. |
| 1.4 | ⛔ **TRIED AND REVERTED** — stays deferred | Implemented as a per-model mask and it **collapsed the mechanic**: the bar's attempts fell 5.67 → 1.67 and standing charges 4.56 → **1.11**. *"If it can"* is a **JOINT** property — a model can end engaged only if a legal UNIT move exists in which it does — and a per-model mask cannot see that, so it forced all five members onto contact, scattered the squad and had the referee revert the charge. Expressing it needs the joint candidate set, which lives only in the play-time decoder, and folding decoding into training measured **−51.8 vp**. |

### Design-uncertain, and NOT to be slipped in

| # | item | why it needs deciding, not implementing |
|---|---|---|
| 1.2 → 1.3 | **target declaration → multi-target charge** | Adds actions. Measured opportunity is **7.5%** of eligible unit-turns with **0%** realised, and an expert review ranked it last on exactly those grounds. There is a real encoding worth trying — each unit member names a target in the command phase, **union = the target set**, which turns the factored policy from a liability into an advantage — but that is a design to measure, not a gap to close. |

### Out of scope, and honestly so

Each of these needs a subsystem this game does not have, so "complete" cannot include them:

- **1.6 sequential per-unit decisions** — all units act from one observation; per-unit
  conditioning is an architecture change.
- **1.7 distance measured around obstacles** — movement ignores terrain entirely; this is
  pathfinding.
- **4.2 reckless break** — needs suppression **and** backfire, neither of which exists.
- **5.2 vertical engagement** — the board is 2D throughout.
- **5.5 `MONSTER`/`VEHICLE`** — no unit-type system.
- **2.5 select a melee weapon** and **5.3 multiple profiles** — feasible, but a choice with no
  second case while every model carries one profile.
- **2.7 fighting after death** — a granted ABILITY no model has; closing it means building the
  ability system, and implementing it as a default would make the game *incorrect*.

**So "completed melee implementation" means the eight completable items above, with the
design-uncertain one decided explicitly and the out-of-scope six named rather than forgotten.**
Anything measured before that is provisional and must say so.
