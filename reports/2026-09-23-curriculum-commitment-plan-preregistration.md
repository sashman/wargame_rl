# Pre-registration: can the commitment head make a good plan? — the plan-only readout, the desk check, and the plan-only rung (#384, after Stage 1)

Written 2026-09-23 11:49, **before any training round on the plan-only arms**,
on branch `feature/commitment-plan` (stacked on `feature/commitment-revision`
at `f0ebeb8`, PR #396). Parent question #384. Set as a goal by Sash
2026-09-23: check whether the commitment head can make a good plan, in four
steps — the plan-only readout, the desk check of the shipped Stage 1, the
plan-only rung with two arms, and the passing head with learned members.

## The question

Stage 1's CM3 (`reports/2026-09-22-curriculum-commitment-stage1-preregistration.md`,
amendment 1) read NULL on every seed: the head commits and keeps, the members
follow and complete, and the plan stacks 1.6–2.6 squads per claimed objective
with one always empty — 0.03 / 0.00 / 0.00 against A3's own 0.70 / 0.80 /
0.77 and the environment-assigned 0.82 / 0.85 / 0.93. Success there mixes the
head's plan with the members' execution. **Step 1 separates them**; **Step 2**
says what the head is paid and what it reads; **Step 3** trains the head alone,
with execution held at the bar's, under the credit it has and under a per-unit
credit; **Step 4** puts the passing head back with learned members.

## Step 1 — the plan-only readout (built; read 2026-09-23 on CM3's finals)

`just measure-plan <head_config> <n> <seed_base> <ckpt...>`
(`scripts/measure_plan.py`): a checkpoint on a `head` config scored as
trained and PLAN-ONLY — the network draws only the commitment decisions, the
scripted `squad_march_committed` (which reads the state; its own emit off;
re-planned before each act; a per-squad fallback where the head has not yet
written) takes every other decision — beside the bar, with the commitment
readouts and a new `distinct` column (distinct objectives claimed per turn
over the objectives on the board).

| CM3 seed | as trained | PLAN-ONLY success · turns · held | persist · claim · distinct · complete | follow · leave |
|---|---|---|---|---|
| bar `squad_march_take` | 1.000 · 5.28 · 4.00 | — | 0.99 · 1.00 · 1.00 · 1.00 | 1.00 · 0.00 |
| s1 | 0.030 · 8.00 · 2.17 | **0.550 · 7.17 · 3.51** | 0.34 · 1.19 · 0.76 · 0.87 | 0.99 · 0.00 |
| s2 | 0.000 · 8.00 · 2.64 | **0.000 · 8.00 · 1.19** | 0.89 · 2.68 · 0.33 · 0.70 | 0.96 · 0.01 |
| s3 | 0.000 · 8.00 · 1.72 | **0.170 · 7.79 · 2.58** | 0.95 · 1.68 · 0.47 · 0.90 | 0.97 · 0.01 |

(n=100, seeds 700000+, greedy head.) With execution held at the bar's, one
head's plan is half a pass and two are stacks. The plan is the wall on s2 and
s3; on s1 both the plan and the execution are short — s1's head made a
half-decent plan that its own members could not execute (0.03 as trained
against 0.55 with the bar's members), the cleanest separation of the two
halves the ladder has had.

## Step 2 — the desk check (done 2026-09-23)

- **What the head is paid.** Under `streams`, `PerStepReward._pay_close` pays
  the planning stream the delta globals, the state globals and the terminal
  bonuses — and the collector (`_land_planning`) broadcasts that one scalar
  to every unit's open commitment span. **No taken pot, no death pot, no
  claimant dilution exist** (D7's pots were never built); B6 was a proposal.
  A squad stacking on a covered objective and the squad taking the empty one
  are paid alike. (Read by the parallel session and confirmed in the code.)
- **What the head reads.** `scripts/measure_commitment_ablation.py` gained a
  `no claimants` mode (the claimant column on every objective token zeroed at
  play). On CM3's finals at n=100: 0.03 / 0.00 / 0.00 trained → 0.03 / 0.00 /
  0.00, held within 0.08. **The head does not read who has already claimed
  what**, as the members do not read the marked target (blank / misdirect /
  nearest flat, Stage 1 amendment 1). So the shipped head has no per-unit
  signal to allocate with: the credit is shared and the one observation that
  distinguishes a claimed objective from an empty one is unused.

## The build (this branch; default off; goldens byte-identical)

- `PlanningCredit.counterfactual` (`reward_timing.py`, B6): at the close, each
  living unit is credited the state globals and terminal bonuses with the unit
  IN minus the same terms with its models masked out of the distance cache;
  `StepPayment.planning_credits`, landed per unit by `_land_planning_credits`.
  The delta globals stay broadcast. `--planning-credit` on the trainer.
- **The plan-only trainer** (`--members <baseline>`): a non-emitting scripted
  seat takes every decision but the commitment draw in rollouts; those steps
  carry no policy, so the update sees commitment rows only; the in-run
  evaluation reads the run the same way (`plan_only_chooser`).
- `ScriptedSeat(emits=)`, `declaration_for`; `squad_march_committed`'s
  per-squad fallback; `measure-plan`; the `distinct` column; the
  `no claimants` ablation mode. Tests in `tests/test_per_model_plan_only.py`.

## Step 3 — the plan-only rung: two arms on A3's shape

| arm | config | flags (the one change each) | tag |
|---|---|---|---|
| **PL1** the head as shipped, plan-only | `configs/experiments/curriculum/a3_head.yaml` | `--members squad_march_committed` (planning credit `broadcast`) | `pl1` |
| **PL2** the head with the per-unit counterfactual, plan-only | the same | `--members squad_march_committed --planning-credit counterfactual` | `pl2` |

Three seeds each (1 / 2 / 3), from scratch, 122,880 rounds (the A3 cap), the
recipe otherwise CM3's (128 rounds per update: `--num-rollout-envs 4
--rollout-rounds 32`; `--ent-coef 0.003`; `gamma` 0.9; `--planning-gamma` the
default; eval and checkpoint every 512 rounds; recording on), Wandb group
`curriculum-cm-plan`. **PL1 and PL2 share their initialisation seed for seed
and differ by one flag, so their difference is paired at initialisation.**
Read at 40,960 / 81,920 / 122,880 with `measure-plan` (the PLAN-ONLY row is
the read; n=100 on 700000+), `measure-commitments` on the plan-only chooser,
and the ablation (`no claimants` beside blank / misdirect / nearest); every
read gated on each seed's log carrying the round line, the final on zero
trainers, each seed's run dir resolved on its own.

Since the shipped head reads neither the claimant counts nor anything else
per unit (Step 2), **PL2 is the arm that can move it, and PL1 is
pre-registered as PL2's control** — the same head, the same scripted members,
the credit it has today.

**Comparators, fixed by name.** The bar `squad_march_take` (1.000, distinct
1.00, claim 1.00); CM3's finals on the plan-only readout (0.550 / 0.000 /
0.170, distinct 0.76 / 0.33 / 0.47) — the head trained jointly, read the same
way; PL1 against PL2 seed for seed.

**Criteria, per seed, on the PLAN-ONLY row at 122,880:**

- **PLANS** — plan-only success ≥ 0.95 AND distinct ≥ 0.95 AND claimants per
  claimed objective ≤ 1.10 (a covering plan the bar's members complete).
- **AHEAD** — plan-only success ahead of CM3's same-seed plan-only row by more
  than two binomial SE (at n=100, 0.13 or more near 0.2–0.5), not PLANS.
- **NULL** — otherwise.

Arm verdicts: **PL1 PLANS on ≥ 2 seeds** → the shipped head learns to plan
when execution does not interfere; the joint failure was interference, and
Step 4 runs PL1's head. **PL1 not, PL2 PLANS on ≥ 2 seeds** → the shared credit
was the wall, B6 is the fix, Step 4 runs PL2's head. **Neither PLANS, PL2 ahead
of PL1 on 3/3 (paired)** → the credit helps and is not enough; the diagnosis
reads the head's entropy and persistence (exploration on the head is the next
lever) before Step 4. **Neither, and PL2 not ahead** → the head cannot learn
an assignment from reward on this shape under either credit; report, no Step
4. Readouts before any verdict, on every read: the plan-only row, distinct and
claim, persist, the commitment head's entropy and clip fraction and the
planning explained variance from the run, the ablation columns.

## Step 4 — the passing head with learned members (pre-registered here, run on a PLANS read)

**CM4**: the arm whose head PLANS, trained again with learned members on the
same config (no `--members`; the same planning credit), three seeds, 122,880
rounds; read as CM3 was (greedy n=100, the commitment readouts, the flag
ablation). Comparators: CM3 (0.03 / 0.00 / 0.00), the environment-assigned
`a3_cm` (0.82 / 0.85 / 0.93), A3 from scratch (0.70 / 0.80 / 0.77). Per
seed: **PASS** success ≥ 0.95; **AHEAD** ahead of A3 from scratch by more than
two binomial SE; **NULL** otherwise; and **STEERS** as Stage 1 defined it. Then
the half-step (`a5_points` with `assignment: head`), pre-registered in an
amendment with CM4's numbers as the prior.

## What I expect (a guess, written so it can be wrong)

PL1: NULL on every seed — the broadcast credit cannot tell a stacked
commitment from a covering one, so the head's plan-only success sits at
0.1–0.5 with distinct 0.5–0.7, as CM3's heads do. PL2: PLANS on two seeds,
AHEAD on the third — a unit paid the difference its bodies make earns
nothing for stacking, so the plan spreads within 40,960 rounds and the bar's
members finish it. If PL2 also reads NULL with distinct under 0.7, the head's
draws never explore the covering assignment (persist near 1 from early on,
commitment entropy falling under 0.5 nats), and the next lever is exploration
on the head, not credit.

## Amendment 1 — written 2026-09-23 12:33: the first launch was void (the fallback walked the unplanned units); KEEP is illegal on an empty slot; Step 1 re-read; relaunch

**What happened.** PL1 and PL2 launched 11:53 (`d145e6e`). At ~21,500 rounds
every seed's in-run plan-only read held 4.0 of 4, and a readout at n=10 on
the latest checkpoints read PLAN-ONLY success **1.000 with distinct 0.39 /
0.50 and claim 1.37 / 1.47**: the head was leaving slots EMPTY (KEEP on
nothing, which the mask offered) and `squad_march_committed`'s per-squad
fallback — built so a unit acting before a later unit's commit would not be
overridden — walked every unplanned squad to the greedy plan. The read scored
the fallback, and a head trained plan-only learns exactly that: leave the
slot empty and let the members plan. Both arms were stopped at 12:30, their
run dirs carry `VOID.md`, their logs are under
`logs/curriculum-cm-plan/void-first-launch/`, their Wandb runs are void.

**The fix (`3128963`).** `PerModelEnv._with_commit_mask` offers KEEP only
where the slot holds an objective: at a unit's first open of the episode,
and whenever the env has retired its commitment, the head must name one. A
member policy that follows a plan therefore never has an unplanned unit to
rescue. Pinned by the head-writer test and
`test_a_unit_with_an_empty_slot_is_not_offered_keep`; documented in
`docs/reward-phases.md`. **Rule: a member policy that follows a plan must
never be able to rescue an unplanned unit, or the plan-only read scores the
rescue.**

**Step 1 re-read under the rule** (CM3's finals, n=100; the CM3 heads were
trained with KEEP-on-empty legal, so this forces an objective where they
would have kept nothing):

| CM3 seed | as trained | PLAN-ONLY success · turns · held | persist · claim · distinct · complete |
|---|---|---|---|
| s1 | 0.030 · 8.00 · 2.14 | **0.530 · 7.17 · 3.47** | 0.36 · 1.22 · 0.78 · 0.86 |
| s2 | 0.000 · 8.00 · 2.65 | **0.000 · 8.00 · 1.05** | 0.90 · 2.73 · 0.33 · 0.67 |
| s3 | 0.000 · 8.00 · 1.68 | **0.130 · 7.88 · 2.51** | 0.94 · 1.90 · 0.48 · 0.84 |

Within 0.04 of the first read (0.550 / 0.000 / 0.170): the jointly-trained
heads rarely kept an empty slot, so the confound barely touched Step 1. The
comparator rows for AHEAD are these: **0.530 / 0.000 / 0.130**.

**Relaunch.** PL1 and PL2 relaunched at 12:33 from `3128963` with the same
flags, seeds, budget and reads; tags `pl1` / `pl2`, group
`curriculum-cm-plan`. Criteria unchanged. One expectation revised: with the
free plan gone, PL1's head must find a covering assignment through the
broadcast credit alone, and I expect it not to (distinct 0.5–0.7, as before).

*Addition to amendment 1 (the parallel session's reading of CM3):* CM3 as
read carried 13–23% uncommitted unit-turns at 40,960, falling to 3–9% at the
cap, so its stacking result stands (the stacked plans were real commitments)
and its `empty` column is partly the KEEP-on-nothing rule this amendment
retires. Any head arm launched from #396's chain after this inherits the
mask change (`3128963`, pinned by `test_a_unit_with_an_empty_slot_is_not_offered_keep`
and the head-writer test) once #406 merges down; until then it is on
`feature/commitment-plan` only.

## Amendment 2 — written 2026-09-23 15:47: PL1 and PL2 read at the cap — the head plans on every seed under either credit; AHEAD ×6 on the letter, PLANS on none, and the PLANS clause measured the wrong thing

Six trainers launched 12:33 from `3128963`, exited 15:33–15:36 at 122,880
rounds, no errors (Wandb `curriculum-cm-plan`: PL1 `beh6ox4q` /
`e8ynk0xb` / `3p4vyiq9`, PL2 `b4650ev8` / `vajs0nj2` / `q6yut6y3`). Every
read greedy at n=100 on 700000+ through the plan-only chooser
(`squad_march_committed` walking, its emit off, KEEP illegal on an empty
slot); the ablation through the same chooser (`ablation-pl*-*-planonly.txt`).

### The plan-only row, seed for seed

| arm · read | success | turns (bar 5.28) | distinct per turn · at the end | claimants per claimed objective | persist |
|---|---|---|---|---|---|
| PL1 40,960 | 1.000 / 1.000 / 1.000 | 5.24 / 5.14 / 5.12 | 0.74 / 0.63 / 0.84 | 1.33 / 1.51 / 1.17 | 0.77 / 0.86 / 0.77 |
| PL1 81,920 | 1.000 / 1.000 / 1.000 | 5.07 / 5.33 / 5.10 | 0.85 / 0.71 / 0.83 · 0.97 / 0.94 / 0.97 | 1.17 / 1.38 / 1.20 | 0.79 / 0.82 / 0.72 |
| **PL1 122,880** | **1.000 / 1.000 / 1.000** | **5.09 / 5.14 / 5.10** | 0.85 / 0.84 / 0.90 · **0.97 / 0.94 / 0.93** | 1.16 / 1.19 / 1.09 | 0.77 / 0.79 / 0.88 |
| PL2 40,960 | 1.000 / 1.000 / 1.000 | 5.11 / 4.89 / 5.08 | 0.68 / **0.97** / 0.70 | 1.43 / **1.02** / 1.42 | 0.84 / **0.98** / 0.84 |
| PL2 81,920 | 1.000 / 0.990 / 0.990 | 5.20 / 5.13 / 5.16 | 0.81 / 0.86 / 0.81 · 0.92 / 0.97 / 0.90 | 1.21 / 1.15 / 1.21 | 0.79 / 0.71 / 0.77 |
| **PL2 122,880** | **1.000 / 1.000 / 1.000** | **5.00 / 5.22 / 5.01** | 0.87 / 0.77 / 0.88 · **0.97 / 0.93 / 0.94** | 1.14 / 1.27 / 1.12 | 0.78 / 0.87 / 0.88 |
| the bar `squad_march_take` | 1.000 | 5.28 | 1.00 · 1.00 | 1.00 | 0.99 |
| CM3's heads (Step 1, re-read) | 0.530 / 0.000 / 0.130 | 7.17 / 8.00 / 7.88 | 0.78 / 0.33 / 0.48 | 1.22 / 2.73 / 1.90 | 0.36 / 0.90 / 0.94 |

**Verdicts on the letter.** PLANS needs success ≥ 0.95 AND distinct ≥ 0.95 AND
claim ≤ 1.10 per seed: **no seed of either arm** (distinct per turn 0.77–0.90;
s3 of PL1 clears the claim clause alone). AHEAD needs success ahead of CM3's
same-seed plan-only row by more than two binomial SE: **every seed of both
arms** (+0.47 / +1.00 / +0.87, the smallest 7 SE). Paired PL2 − PL1 at the cap:
success 0 / 0 / 0, turns −0.09 / +0.08 / −0.09, distinct at the end +0.00 /
−0.01 / +0.01 — **the two arms are the same read.** So the arm-level reading
the pre-registration wrote for this case is "neither PLANS, PL2 not ahead of
PL1 → the head cannot learn an assignment from reward on this shape under
either credit; report, no Step 4."

**That reading is wrong, and the clause that produces it is the defect.**
Both heads, from scratch, with execution held at the bar's, make plans that
the bar's own members complete at **1.000 on six of six seeds, faster than
the bar walks its own plan** (5.00–5.22 turns against 5.28), and they do it
by a third of the budget. What they do not do is name a covering assignment
at the first open: they commit, watch the board, and re-commit — persist
0.77–0.88, distinct climbing from ~0.6–0.7 per turn to ≥ 0.93 by the last
turn. The PLANS clause asked for the scripted bar's *mechanism* (a fixed
assignment from turn one, distinct 1.00 every turn) and measured the plan
by its resemblance to that, not by what the rung decides on. On the rung's
own criterion — success at n=100 with the members held fixed — these are the
best plans on the ladder, and the counterfactual credit added nothing to
them: PL1 is PL2's control, and the control passed. **The head can make a
good plan; what it could not do in CM3 was make one while its own members
were learning to walk.**

### What the reads say beyond the verdict

- **The head reads the board.** The plan-only ablation at the cap keeps
  success at 1.00 in every column, and hiding the claimant counts costs
  +0.4–0.8 turns on PL1 and +0.3–0.6 on PL2, misdirecting the committed
  relation +0.3–0.5 on PL1 — the head uses both to re-plan, unlike CM3's
  head (no-claimants flat, Step 2), and can plan without them.
- **The planning panel.** PL1: commitment entropy 0.50–0.63 nats at the end,
  planning explained variance 0.28–0.36, clip 0.18–0.26, planning return
  ~2.6. PL2: entropy 0.53–0.66, planning explained variance **0.05–0.07**,
  clip 0.19–0.28, return ~2.1. The per-unit counterfactual return is barely
  fit by the planning value head (it is read at the unit's first living
  member from a board-level embedding), and the head planned anyway: on
  this shape the advantage's direction was enough and its scale did not
  matter. A B6 with a value head that conditions on the unit is the version
  to build if the counterfactual is ever needed.
- **The in-run curve** (the plan-only evaluation, 20 episodes): success
  87.6% in the first quarter of PL1 s1 and ≥ 98.9% after; the same shape on
  every run. The head learns the plan in the first ~20k rounds.
- **The confound, closed.** With KEEP illegal on an empty slot, 612 of 612
  member moves on a PL1 checkpoint were under a live commitment and KEEP was
  chosen 0 of 204 times; the reads above are the heads' plans.

### The revision this asks of #384 (D9; proposed, Sash decides)

1. **Retire the distinct/claim clauses as a pass criterion**; keep them as
   readouts (with distinct at the end beside the mean). A plan is judged by
   the rung's criterion with execution held fixed.
2. **Step 4 (CM4) is justified on the head's evidence, not on the letter**:
   run it with **PL1's recipe** (the credit as shipped, no `--members`) —
   the counterfactual adds nothing here and its critic does not fit. The
   question CM4 asks is now sharp: the head plans when the members walk the
   plan; does it still plan when the members are learning, and if not, is it
   the members' noise on the planning return, the members' failure to
   follow, or both? Pre-registered readouts: the plan-only row of the joint
   run's head beside its as-trained row at every read (if the plan-only row
   holds 1.000 while as-trained sits low, the head is fine and the members
   are the wall; if the plan-only row itself degrades, the members'
   learning degrades the head's plan), plus persist, follow-through, and
   the ablations.
3. A middle rung if CM4 reads low: **freeze the plan-only head and train the
   members under it** (the head's weights fixed, only the member heads and
   the execution value learn) — execution alone, with a known-good plan.

Files: `plan-pl{1,2}-{40960,81920,final}.txt`, `ablation-pl*-*-planonly.txt`,
`planning-panel-final.txt`, `plan-cm3-final-v2.txt` in the session drafts.
