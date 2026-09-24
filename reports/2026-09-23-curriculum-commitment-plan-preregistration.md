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

## Amendment 3 — written 2026-09-23 21:50: Sash's decision on Step 4 — CM4 launched on PL1's recipe, and a frozen-planner members rung (FH1) beside it

Sash (2026-09-23): "launch both". Both arms on `a3_head.yaml`, three seeds,
122,880 rounds, the recipe as PL1's minus `--members`, Wandb group
`curriculum-cm-plan`.

| arm | the one change | what it asks |
|---|---|---|
| **CM4** (#405) | the head AND the members learn together from scratch (no `--members`; the broadcast credit; KEEP illegal on an empty slot, which CM3 lacked) | does the head still plan when its members are learning? |
| **FH1** | `--frozen-planner <PL1 seed-s last.pt>`: PL1's finished head, seed for seed, draws every commitment (greedy, its weights fixed, its network never in the optimiser); the executor network learns the member heads and the execution value under that plan | can the members learn to walk a known-good plan? |

**Reads** at 40,960 / 81,920 / 122,880, n=100 on 700000+, gated on the logs
and the exit: for CM4 `measure-plan`'s **as-trained row beside its own
head's plan-only row** (the same checkpoint, the head with the bar's
members), the commitment readouts, the census, the walk-off probe, the
ablation; for FH1 the split row (PL1's planner + the executor), the
planner's own plan-only row as the plan's ceiling, the commitment readouts
through the split chooser.

**Criteria, per seed at 122,880, as-trained (CM4) or split (FH1):** PASS
success ≥ 0.95; AHEAD ahead of A3 from scratch (0.700 / 0.800 / 0.770) by
more than two binomial SE; NULL otherwise. Comparators by name: CM3 (0.030 /
0.000 / 0.000), the environment-assigned `a3_cm` (0.820 / 0.850 / 0.930), A3
from scratch, and for FH1 the planner's plan-only row (1.000 ×3).

**What the pair separates.** CM4's plan-only row holding ≥ 0.95 while its
as-trained row sits low → the head is fine and the members are the wall (FH1
then says whether they can learn at all under a fixed good plan). CM4's
plan-only row itself degrading → the members' learning degrades the plan
(the noise on the planning return, or the members walking away from what
the head commits). FH1 PASS with CM4 low → the join is the problem, not
either half. FH1 low → execution under a good plan is the wall, and the
arrived-keeps / leaving-step findings on the half-step apply here too.

**Expectation (a guess, written so it can be wrong).** CM4: as-trained
0.3–0.7 with the plan-only row at 0.9–1.0 on two seeds — the head plans,
the members lag it; FH1: 0.85–1.0 on 3/3 by the cap, a turn behind the bar.

## Amendment 4 — written 2026-09-23 23:12: a width arm on the join (CW1) — is the set network's size the wall?

Sash (2026-09-23, on the 1.23-million-parameter set network: 4 blocks, width
128, 8 heads; the whole-army transformer is ~12.7 million): "it does feel
that we don't have enough parameters". The record has both sides: this
network holds the escort's plan and the six-objective clone at 0.96 and plans
the four-objective assignment at 1.000 when its members are scripted (so
plan and execution are representable at this size), and the ten-times
larger whole-army transformer never learned allocation on the real tables
either; against that, the join is a noisy-return problem (CM4's planning
return is a tenth of PL1's, its commitment head at 0.85–1.13 nats has not
converged by 46k) and a wider trunk learns faster and steadier under noise.
**Representable is not learnable at this size; the test is one flag.**

| arm | the one change | tag |
|---|---|---|
| **CW1** | CM4's recipe with `--n-layers 8 --embedding-size 256` (the whole-army trunk's depth and width; ~5 million parameters) | `cw1` |

Three seeds, 122,880 rounds, `a3_head.yaml`, everything else CM4's (head
and members from scratch, the broadcast credit, KEEP illegal on an empty
slot). Unpairable at initialisation (a shape change); layouts and seeds
shared. Comparators by name: **CM4 at matched rounds** (the same recipe at
the shipped size), A3 from scratch (0.700 / 0.800 / 0.770), the
environment-assigned `a3_cm` (0.820 / 0.850 / 0.930). Reads as CM4's: the
as-trained row beside the same head's plan-only row, the commitment
readouts, census, walk-off, ablation, the planning panel.

**Criteria, per seed at 122,880 (as trained):** PASS success ≥ 0.95; AHEAD
ahead of A3 from scratch by more than two binomial SE; NULL otherwise. And
the question this arm exists for, **WIDER**: as-trained success ahead of
CM4's same-seed row by more than two binomial SE on 2/3 or more — size moves
the join. **Readouts that separate the mechanisms**: the plan-only row (does
a wider head plan under learning members where the narrow one stacked?),
commitment entropy and the planning return against CM4's (does the wider
head converge?), follow-through and leave (do wider members walk the plan?).

**Expectation (a guess).** Not WIDER: the wider network learns the walk a
little faster and plans no better under learning members, because the
planning return is small and noisy at either size; as-trained 0.1–0.4 with
the plan-only row 0.1–0.5, commitment entropy still above 0.8 nats at the
cap. If it reads WIDER with the plan-only row near 1.000, size was the wall
on the join and the ladder's per-model rows below need re-reading at width.

## Amendment 5 — written 2026-09-23 23:21: a head-depth arm on the join (CH1) — are the decision readouts too shallow?

Sash (2026-09-23): the heads' depth should be a parameter; start at 2. Every
policy head of the set network is a single linear map on the trunk's
embedding (declaration, displacement, no-target and keep on the 2 × width
input; the target and commitment pointers a bilinear match through one
query and one key map), so any nonlinearity a decision needs is computed in
the shared trunk. `SetNetworkConfig.head_layers` (`--head-layers`) now sets
the policy heads' depth: 1 is the shipped readout (the state-dict keys
unchanged, so every checkpoint on file loads), `k` stacks `k` maps of the
trunk's width with GELUs between, on every policy head including the pointer
projections. The value heads keep their shipped two layers; the trunk is
untouched. CW1 (amendment 4) asks whether the trunk is too small; **CH1 asks
whether the readouts are**, and the two bracket "size".

| arm | the one change | tag |
|---|---|---|
| **CH1** | CM4's recipe with `--head-layers 2` (the trunk at the shipped 4 × 128) | `ch1` |

Three seeds, 122,880 rounds, `a3_head.yaml`, everything else CM4's.
Unpairable at initialisation (a shape change); layouts and seeds shared.
Comparators by name: **CM4 at matched rounds**, CW1 at matched rounds, A3
from scratch (0.700 / 0.800 / 0.770), the environment-assigned `a3_cm`
(0.820 / 0.850 / 0.930). Reads as CM4's. **Scheduled** to launch when CM4's
three trainers exit (the box carries nine), so it reads on a later clock
than CM4 and CW1.

**Criteria, per seed at 122,880 (as trained):** PASS success ≥ 0.95; AHEAD
ahead of A3 from scratch by more than two binomial SE; NULL otherwise.
**DEEPER**: ahead of CM4's same-seed row by more than two binomial SE on 2/3
or more. Readouts that separate the mechanisms as CW1's: the plan-only row
(does the deeper commitment head plan under learning members?), commitment
entropy and the planning return, follow-through and leave.

**Expectation (a guess).** Not DEEPER on success; the one place a nonlinear
readout could matter is the commitment pointer, whose per-token bilinear
score cannot compare "claimed by someone else" across tokens directly, so if
anything moves it is the plan-only row and the claimant ablation, not the
as-trained success.

## Amendment 6 — written 2026-09-24 04:32: the join read — CM4 NULL ×3 with its head planning at 0.72, FH1 NULL ×3 under PL1's planner, and a frozen planner is not a fixed plan

Both arms read at 40,960 / 81,920 / 122,880, n=100 on seeds 700000+, gated
on the round line in every seed's log and on the trainers' exit for the
final. CM4's rows are its own checkpoint as trained beside the same
checkpoint's head on the plan-only row (the bar's members walking what it
commits); FH1's are the split row (PL1's frozen planner committing, the
executor walking) beside the executor alone (its own never-trained head
committing) and the planner's plan-only ceiling. Wandb group
`curriculum-cm-plan`: CM4 7er89mzx / fcbsyjp0 / hgk41vpx, FH1 wgjef4rv /
mngkjdad / 02uqvzh3. Revision `49f3957` (the FH1 build `29bfedc`).

⚠ Two instances of this conversation ran in parallel between roughly
22:30 and 01:20 after a restart; both posted the CM4 40,960 / 81,920 and
FH1 40,960 reads on #405 and #408, so each of those reads appears twice
there, identical. The later copy of each pair is the one this amendment
was written from. Nothing else was duplicated: one chain, one set of
checkpoints, one landing.

### CM4 (#405) — head and members from scratch

| read | as trained (success · held · leave) | the same head plan-only (success · held · distinct mean / end) |
|---|---|---|
| 40,960 | 0.000 / 0.160 / 0.000 · 1.91 / 1.89 / 1.35 · 0.75 / 0.65 / n/a | 0.010 / 0.170 / 0.160 · 1.54 / 2.82 / 2.79 · 0.37 / 0.57 / 0.38 (end 0.30 / 0.68 / 0.41) |
| 81,920 | 0.000 / 0.190 / 0.210 · 1.58 / 2.68 / 2.69 · 0.55 / 0.67 / 0.70 | 0.190 / 0.530 / 0.550 · 2.01 / 3.38 / 3.42 · 0.51 / 0.69 / 0.70 (end 0.47 / 0.67 / 0.62) |
| **122,880** | **0.000 / 0.370 / 0.320** · 2.01 / 2.94 / 2.96 · 0.77 / 0.73 / 0.68 | **0.720 / 0.680 / 0.720** · 3.67 / 3.57 / 3.71 · 0.68 / 0.77 / 0.69 (end **0.88 / 0.72 / 0.84**) |

**Verdict on the letter, per seed at 122,880 as trained: NULL / NULL /
NULL.** A3 from scratch is 0.700 / 0.800 / 0.770; no seed is within two
binomial SE of it, none reaches 0.95. Comparators: CM3 (the same join
without the KEEP rule) 0.030 / 0.000 / 0.000 — the one mechanic changed
between them moved two seeds from zero to a third; the environment-assigned
`a3_cm` 0.820 / 0.850 / 0.930; PL1's head plan-only 1.000 ×3 at every read
from 40,960.

- **The head plans under learning members, late and not fully.** Its
  plan-only row climbs 0.01–0.17 → 0.19–0.55 → 0.68–0.72, covering 0.72–0.88
  of the objectives on the last turn (PL1: 0.93–0.97). At 40,960 it was
  stacking (claimants 1.64–2.44 per claimed objective); at the cap 1.23–1.43.
  A bar's members walking CM4's final plan reach 0.72, not 1.000: the plan
  is roughly a quarter of the shortfall at the cap and most of it at a
  third of the budget.
- **The members lag their own head's plan by 0.35–0.72.** Given the plan
  the same checkpoint produces, the bar's members finish at 0.68–0.72 and
  CM4's own at 0.00–0.37, with the leave share 0.68–0.77 (the bar's members
  under the same plan: 0.01). The walk-off probe reads as every per-model
  row on the half-step: a body on an objective stands still on 0.00 of its
  decisions and leaves on 0.58–0.74, paid +0.006 to +0.018 for a move that
  keeps the objective and −0.001 to −0.012 for one that leaves it — the sign
  is right and the magnitude is a few thousandths.
- **The members read the mark on two seeds (STEERS on s2 and s3).**
  Blanking the commitment relation takes s2 0.37 → 0.00 and s3 0.32 → 0.00;
  misdirecting it 0.00 / 0.17; the greedy nearest assignment in place of the
  head's 0.15 / 0.59 (s3 is BETTER under the greedy plan than under its own
  head — the plan's share of that seed's shortfall); removing the claimant
  counts from the head's view 0.00 / 0.00 / 0.06 (the head reads the counts,
  as PL1's did). s1 reads nothing (0.00 in every column) and is the seed
  whose members never left the stack: its census holds 2.0 objectives with a
  maximum stack of 4.0 and the first objective empty in 80% of episodes,
  where s2 and s3 hold 3.0 with stacks of 2.5–2.9.
- **The planning panel at the cap**: planning return 0.24 / 0.47 / 0.53
  (PL1 2.6), commitment entropy 1.00 / 0.39 / 0.65 nats (PL1 0.50–0.63),
  planning EV 0.39 / 0.09 / 0.10, planning clip fraction 0.24–0.31. The
  head's return is a fifth to a tenth of PL1's because its members complete
  a fifth to a tenth as often; the head learns from what the members
  deliver, and they deliver late.
- In-run `held` by quarter: s1 0.98 → 1.73 → 1.61 → 1.70 (flat from the
  second quarter), s2 0.86 → 2.04 → 2.71 → 2.96, s3 1.24 → 1.64 → 2.48 →
  3.06 — two seeds still climbing at the cap, one parked.

### FH1 (#408) — the members under PL1's frozen planner, seed for seed

| read | split: PL1's planner + the executor (success · held · leave · distinct end) | the executor alone, its untrained head committing (success · held) |
|---|---|---|
| 40,960 | 0.160 / 0.430 / 0.040 · 2.53 / 3.27 / 1.53 · 0.44 / 0.31 / 0.87 · 0.69 / 0.64 / 0.72 | 0.200 / 0.530 / 0.090 · 2.49 / 3.40 / 1.91 |
| 81,920 | 0.260 / 0.600 / 0.600 · 3.10 / 3.52 / 3.48 · 0.30 / 0.27 / 0.32 · 0.66 / 0.64 / 0.67 | 0.310 / 0.420 / 0.630 · 3.22 / 3.14 / 3.52 |
| **122,880** | **0.370 / 0.610 / 0.630** · 3.21 / 3.59 / 3.60 · 0.48 / 0.38 / 0.20 · **0.71 / 0.59 / 0.71** | **0.400 / 0.480 / 0.720** · 3.29 / 3.40 / 3.67 |

The planner's own ceiling on every read, with the bar's members: plan-only
1.000 ×3, distinct 0.84–0.90 per turn and **0.93–0.97 on the last turn**.
The executor's panel is healthy: explained variance 0.95–0.96 in the last
quarter, displacement entropy falling 2.4–2.8 → 1.2–1.5 nats, clip fraction
0.33–0.35.

**Verdict on the letter, per seed at 122,880 on the split row: NULL /
NULL / NULL** — below A3 from scratch (0.700 / 0.800 / 0.770) on every
seed, by 3–7 binomial SE, and below the environment-assigned `a3_cm`
(0.820 / 0.850 / 0.930) by more.

- ⚠ **A FROZEN PLANNER IS NOT A FIXED PLAN.** The head is adaptive
  (commit, watch, re-commit), so its commitments are a function of what
  its members do — and under members who arrive late and leave, the SAME
  weights that cover 0.93–0.97 of the objectives on the last turn with the
  bar's members cover **0.59–0.71** with the executor's. The plan the
  members were trained under was never the 1.000 plan; it was the 1.000
  planner reacting to their own execution, and the reaction is worse. The
  coupling runs both ways even with one side's weights fixed. Only the
  WRITER fixes a plan: `a3_cm`'s greedy arrived-keeps writer gave the same
  reward and the same trainer 0.820 / 0.850 / 0.930.
- ⚠ **What the members learn is mostly not the plan.** With their own
  never-trained head committing at random (claimants 1.8–2.4 per claimed
  objective, a stack), the same members read 0.400 / 0.480 / 0.720 —
  paired against the split row −0.03 / +0.13 / −0.09, ahead on two seeds.
  On A3's shape a covering walk is reconstructible from the board (the
  nearest objective per squad IS the assignment on most layouts), so an
  executor learns the geometry and follows the mark only where the two
  differ. This is the A3 finding (the members never read the marked
  target; the lift was the reward keying) one rung up.
- The leave share under the frozen planner falls with training (0.87 →
  0.20 on s3) and sits at 0.20–0.48 at the cap — CM4's own members
  0.68–0.77 — and `held` climbs through the run (in-run by quarter 1.22 →
  2.81 → 3.13 → 3.32, 2.15 → 3.19 → 3.53 → 3.76, 1.93 → 2.86 → 3.48 → 3.40:
  s1 still climbing, s2 and s3 flat over the last third). Two seeds are on
  a plateau below the bar; an extension would read that plateau.

### Which half fails

Both, and the answer is not the sum of two halves. **The head's half is
a quarter of CM4's shortfall and improving** (plan-only 0.72 at the cap,
1.000 when its members are the bar's). **The members' half is the larger
and the slower**: under their own head's plan they finish 0.35–0.72 behind
it; under a frozen planner they reach 0.37–0.63, behind A3 with no
commitments at all; and what they learn is a geometry walk the plan barely
enters. **And the two halves cannot be separated by freezing weights**,
because an adaptive planner's plan is a function of its executor: FH1
measured the members under a worse plan than PL1's, produced by PL1's
weights. Amendment 3's expectations, scored: CM4 as-trained 0.3–0.7 with
the plan-only row 0.9–1.0 on two seeds — read 0.00–0.37 and 0.68–0.72, the
head planning worse under learning members than guessed; FH1 0.85–1.0 on
3/3 by the cap, a turn behind the bar — read 0.37–0.63, two and a half
turns behind, and the guess was wrong about the mechanism as well as the
number (it assumed the plan was fixed).

What is now measured on A3's shape, all on the per-model trainer, the same
reward: no plan 0.70 / 0.80 / 0.77; a stable greedy plan written by the
environment **0.82 / 0.85 / 0.93**; an adaptive learned plan, frozen, 0.37 /
0.61 / 0.63; a learned plan and learned members together 0.00 / 0.37 / 0.32;
a learned plan with the bar's members 1.000 ×3. **The ordering says the
members want a plan that holds still**, and that a plan they can read off
the board is a plan they do not learn to read.

### What this asks of #384 (proposed; Sash decides)

1. **Do not run the half-step under the joint head.** Neither half holds
   on A3's shape, and the half-step is where the geometry walk fails
   (0.03–0.33 from every per-model setting), so a joint arm there would
   read the members' failure and the head's at once and separate neither.
2. **Do not extend FH1.** Two seeds are flat over the last third and the
   plan they train under is not the one the extension would be credited
   to; the number an extension would buy is already bounded by `a3_cm`.
3. **Make the head's plan hold still — the commitment a body can carry
   across steps that #384 names.** The one lever the A3 ordering points at
   is the writer's stability: the greedy arrived-keeps writer beats no
   plan and the adaptive head. A middle rung, one change: **FH1 with the
   frozen planner's commitments made sticky** (the head decides only when
   a unit's slot is EMPTY — at its first open and after the environment's
   retirement rule clears it — and KEEP is forced otherwise; a small build,
   one config flag on the head writer), read split beside alone. Prediction: split above `a3_cm`'s band on 2/3
   (the head's plan is at least the greedy one, held still), and the alone
   row unchanged. If it reads there, the same rule goes into the joint
   head's writer (CM5) before anything trains jointly again.
4. **Climb to the half-step by the staged route: PL3, the plan-only head
   on A5-points, then FH2, the members under PL3's planner under the
   sticky rule.** Both are CPU arms. PL3 asks whether the head plans five
   objectives for six squads when the bar's members execute (the bar
   solves the half-step at 1.000, so the ceiling exists); FH2 asks whether
   members can walk a plan they CANNOT reconstruct from the board — the
   executor-alone row there is the legibility readout A3's shape cannot
   give (alone reads 0.03–0.33 on the half-step, so any split lift over it
   is the plan reaching the members). That is the question #384 was built
   to answer.
5. **Size is not the lever so far.** CW1 (8 × 256) reads behind CM4 at
   two thirds of the budget on both rows; CH1 (two-layer heads) reads with
   CM4 at a third. Their finals land in amendment 7 with the WIDER and
   DEEPER clauses; neither gates the route above.

## Amendment 7 — written 2026-09-24 05:36: the two size arms — CW1 NOT WIDER (the wide head unlearned its plan), CH1 NOT DEEPER (the deeper heads' plan went backwards too) — size is not the join's lever at either end

Both arms on CM4's recipe, three seeds, 122,880 rounds, read at 40,960 /
81,920 / 122,880 on the same n=100 and seeds as CM4, each checkpoint as
trained beside its own head's plan-only row. Comparator by name: **CM4 at
matched rounds** (as trained 0.000 / 0.160 / 0.000 → 0.000 / 0.190 /
0.210 → 0.000 / 0.370 / 0.320; plan-only 0.01 / 0.17 / 0.16 → 0.19 / 0.53 /
0.55 → 0.72 / 0.68 / 0.72). Wandb group `curriculum-cm-plan`: CW1
6zuta7ol / 4wdd6u88 / py7vt1jo, CH1 qrpb6pz4 / utgi1ii9 / jo8mfh29.
Revision `49f3957`.

### CW1 (#409) — the whole-army trunk's size (8 layers × 256, ~5M parameters)

| read | as trained (success · held) | plan-only (success · held · distinct mean / end) |
|---|---|---|
| 40,960 | 0.000 / 0.000 / 0.000 · 1.63 / 1.88 / 2.16 | 0.400 / 0.010 / 0.000 · 3.29 / 1.87 / 2.17 · 0.62 / 0.51 / 0.66 (end 0.59 / 0.35 / 0.74) |
| 81,920 | 0.020 / 0.000 / 0.030 · 1.88 / 1.51 / 1.84 | 0.010 / 0.310 / 0.250 · 1.82 / 2.76 / 2.71 · 0.54 / 0.67 / 0.46 (end 0.31 / 0.56 / 0.50) |
| **122,880** | **0.000 / 0.000 / 0.210** · 1.60 / 2.22 / 2.82 | **0.000 / 0.000 / 0.000** · 1.25 / 2.03 / 1.40 · 0.53 / 0.53 / 0.37 (end 0.26 / 0.47 / 0.25) |

**Verdict, per seed at the cap as trained: NULL / NULL / NULL. WIDER: no**
— behind CM4's same-seed row on two seeds (0.00 v 0.37, 0.21 v 0.32) and
level at zero on the third, and behind CM4 on the plan-only row on every
seed by 0.68–0.72.

- **The wide head UNLEARNED its plan.** Plan-only 0.40 on s1 at a third
  of the budget, 0.31 / 0.25 on s2 / s3 at two thirds, 0.000 on all three
  at the cap, with last-turn coverage falling to 0.25–0.47 and claimants
  1.64–2.31 per claimed objective (a stack). Commitment entropy 0.40 /
  0.94 / 0.68 nats in the last quarter, planning return 0.30–0.43 (PL1
  2.6), planning EV 0.00–0.27 — s3's fell to 0.000 over the last quarter.
  A larger trunk fitted the noisy planning return faster and to nothing.
- The members read as CM4's: leave 0.51–0.64, standing still on 0.00 of
  their decisions on an objective, a point empty in 70–100% of episodes on
  s1 / s2, max stack 2.9–3.9. The ablation is flat on s1 / s2 (the members
  read nothing); on s3 the greedy nearest assignment (0.29) beats the
  head's (0.21). Executor EV 0.76–0.86 (CM4's members ~0.75), displacement
  entropy 1.6 nats.

### CH1 (#410) — two-layer policy heads on the shipped trunk

| read | as trained (success · held) | plan-only (success · held · distinct mean / end) |
|---|---|---|
| 40,960 | 0.000 / 0.020 / 0.000 · 1.29 / 2.50 / 2.34 | 0.270 / 0.200 / 0.020 · 3.02 / 2.70 / 2.34 · 0.44 / 0.50 / 0.56 (end 0.43 / 0.60 / 0.56) |
| 81,920 | 0.000 / 0.020 / 0.010 · 1.84 / 2.14 / 2.14 | 0.210 / 0.370 / 0.070 · 2.52 / 3.05 / 2.00 · 0.55 / 0.59 / 0.40 (end 0.49 / 0.60 / 0.30) |
| **122,880** | **0.020 / 0.000 / 0.000** · 1.65 / 2.02 / 1.57** | **0.090 / 0.050 / 0.000** · 2.20 / 2.03 / 1.03 · 0.46 / 0.45 / 0.38 (end 0.49 / 0.28 / 0.25)** |

**Verdict, per seed at the cap as trained: NULL / NULL / NULL. DEEPER:
no** — behind CM4's same-seed row on two seeds (0.00 v 0.37, 0.00 v 0.32) and level at zero on the third, and behind CM4 on the plan-only row on every seed by 0.63–0.72.**

- **The deeper heads' plan went backwards as the wide trunk's did.** Plan-only 0.27 / 0.20 / 0.02 at a third, 0.21 / 0.37 / 0.07 at two thirds, 0.09 / 0.05 / 0.00 at the cap, last-turn coverage 0.25–0.49, claimants 1.90–2.47 per claimed objective. Commitment entropy 0.91 / 0.99 / 1.02 nats in the last quarter (never converged; CM4's s2 reached 0.39), planning return 0.25–0.33 (PL1 2.6), planning EV 0.16–0.29, planning clip fraction 0.29–0.34.
- **The members are the worst of the three sizes.** Standing still on 0.00 of their decisions on an objective, leaving on 0.59–0.76 (paid −0.009 for the leaving step and −0.004 to +0.002 for the keeping one); the ablation is flat on every seed (blank, misdirect, nearest and no-claimants all within 0.04 of trained — nothing in the plan reaches them); the census puts 4.2–6.0 of 12 bodies on 1.7–2.0 objectives with max stacks of 3.5–5.0 and the first objective empty in 90–100% of episodes. Displacement entropy 1.9–2.0 nats at the cap (CM4's members 1.7–2.1, CW1's 1.6), executor EV 0.76–0.82.

### What the pair says

Three sizes of the same join on the same recipe, and the shipped one is the best of them on both rows:

| at 122,880 | as trained | plan-only | commitment entropy, last quarter |
|---|---|---|---|
| CM4 (4 × 128, linear heads; 1.23M) | 0.00 / 0.37 / 0.32 | **0.72 / 0.68 / 0.72** | 1.00 / 0.39 / 0.65 |
| CW1 (8 × 256; ~5M) | 0.00 / 0.00 / 0.21 | 0.00 / 0.00 / 0.00 | 0.40 / 0.94 / 0.68 |
| CH1 (4 × 128, two-layer heads) | 0.02 / 0.00 / 0.00 | 0.09 / 0.05 / 0.00 | 0.91 / 0.99 / 1.02 |

- **Size is not the join's lever, at either end.** A four-times wider and
  deeper trunk and a nonlinear readout each read NULL ×3 and behind the
  shipped network on every seed on both rows. Sash's premise ("we don't
  have enough parameters") is answered on this rung: the 1.23M network
  plans at 1.000 with scripted members and holds the six-objective clone,
  and adding parameters made the join worse, not better.
- **At every size the head's plan under learning members goes BACKWARDS
  through training on some seeds** — CW1 on all three (0.40 → 0.00 on s1),
  CH1 on two, CM4 on none (its plan-only row rose monotonically, 0.01 → 0.72).
  The larger and the more nonlinear the policy, the faster it fits a
  planning return that is a fifth to a tenth of PL1's and dominated by
  execution failures, and what it fits is a stack. The noisy return is the
  mechanism amendment 4 named, and more capacity makes it bite sooner.
- **The members do not read the plan at any size** (the ablation is flat
  on CW1 s1 / s2 and on every CH1 seed) and leave the objectives they
  reach at the same 0.5–0.76 wherever the capacity went. The executor's
  wall (amendment 6) is not a capacity wall either.
- Expectations, scored: amendment 4 guessed CW1 not WIDER at as-trained
  0.1–0.4 with plan-only 0.1–0.5 and commitment entropy above 0.8 —
  right on the clause, low on the plan (0.00) and wrong on the entropy on
  two seeds (0.40 / 0.68: the wide head converged, to a stack). Amendment
  5 guessed CH1 not DEEPER with the one place it could matter the
  plan-only row and the claimant ablation — right on the clause, and the
  plan-only row moved the WRONG way.
- ⚠ **Do not run another size arm on the join**, and read the
  1.23M-parameter set network as sufficient for every rung the ladder has
  reached. If capacity is ever the question again, ask it on a rung the
  network fails with scripted members (none so far), never on a join.

## Amendment 8 — written 2026-09-24 09:29: why the join failed, investigated — the head never planned, it SEARCHED; the members never followed a mark, they walk to the nearest objective and hover; and amendment 6's "the members want a plan that holds still" is RETRACTED

Sash (2026-09-24): "Could you investigate why this did not work?" No
training. One probe (`drafts/join_probe.py`, reads in
`drafts/join-probe-reads/`) run on the checkpoints on disk, n=100 on
seeds 700000+, the same seeds as every read above. It follows each squad
from its first commitment to the end (was the committed objective its
nearest at deployment; did it arrive there; where did it end), decomposes
each member's travel pay into what the executed move earns toward the
committed objective against what the same move earns toward the nearest,
and censuses every failed episode by what the squad committed to the empty
objective was doing. A play-time wrapper (`-sticky`) forces KEEP on every
open of a unit whose slot already holds an objective, so a head's plan can
be read with its re-commits forbidden.

### 1. The plan-only 1.000 was a SEARCH executed by fast scripted members, not a plan

| head, with the bar's members | FIRST plan covers all four objectives | success, re-commits allowed | success, re-commits FORBIDDEN at play | re-committed before first arrival |
|---|---|---|---|---|
| the bar's own writer | **1.00** | 1.000 | 1.000 | 0.05 of squads |
| PL1 s1 / s2 / s3 | **0.42 / 0.09 / 0.32** | 1.000 ×3 | **0.78 / 0.52 / 0.67** | 0.50 / 0.47 / 0.32 |
| CM4 s2 / s3 | 0.41 / 0.00 | 0.68 / 0.72 | 0.55 / 0.56 | 0.61 / 0.92 |

PL1's head commits four squads to four distinct objectives at the first
opportunity in 9–42% of episodes (the bar: 100%), re-commits 12–23% of
unit-turns and half its squads before they first arrive, and reaches
1.000 by watching the counts and moving the surplus. Forbid the
re-commits and the same head reads 0.52–0.78. The planning reward
(coverage and the success bonus at the close, broadcast) priced the
outcome of the search, never the plan; and a target switch re-anchors the
members' potential, so a re-commit costs the head nothing. **The head
learned trial and error, and the bar's members made trial and error
cheap** — they arrive at a committed objective in 3.6 turns, so two or
three rounds of re-planning fit inside eight. Every read that called PL1
"a good plan" (amendments 2, 3, 6) read the search's outcome.

### 2. No learned executor on A3 follows a mark — they walk to the nearest objective and hover

Where a squad's commitment is NOT its nearest objective (half of all
squads under every writer, the bar's included):

| members | moves closing MORE on the committed than on the nearest | arrived at the committed | pay toward committed v the same move toward nearest, per step |
|---|---|---|---|
| the bar's | **0.96** | 0.95 | +0.44 v +0.24 |
| the bar's under PL1's head | 0.95–0.96 | 0.89–0.93 | +0.43 v +0.27 |
| FH1's (under PL1's frozen head) | **0.56 / 0.72 / 0.58** | 0.66 / 0.50 / 0.65 | +0.26 v +0.20 · +0.25 v +0.22 · +0.24 v +0.21 |
| FH1 s3's with a RANDOM untrained head | 0.33 | 0.18 | +0.18 v +0.24 |
| `a3_cm` s1–s3 (the greedy environment writer) | **0.57 / 0.51 / 0.48** | 0.50 / 0.47 / 0.45 | +0.27 v +0.26 · +0.30 v +0.30 · +0.31 v +0.31 |
| CM4 s1–s3 | 0.71 / 0.68 / 0.71 | 0.36 / 0.76 / 0.82 | +0.27 v +0.16 · +0.26 v +0.17 · +0.27 v +0.16 |

The reward does price the plan: for the bar's members a move earns
+0.20 per step more toward the committed objective than toward the
nearest. The learned members realise +0.03 to +0.06 of it. They read the
mark weakly (0.56–0.72 against 0.33 under random marks) and they are
slow (arrival at a committed objective in 5.8–6.1 turns against the
bar's 3.6, pay per step +0.25 against +0.43). **And once on ANY
objective they hover**: a body standing inside an objective it is not
committed to sets out for the committed one (closing more than one inch)
on 0.32–0.61 of its moves and leaves that objective on 0.35–0.54 (the
bar's members: 1.00 and 0.82–1.00), earning +0.00 to +0.04 per step
toward the committed objective where +0.48 per six-inch step is on
offer; frozen on under 5%, so it is not gridlock. The failure census of
FH1 is one line: in **81–88%** of failed episodes the squad committed to
the empty objective is standing on another objective.

⚠ **`a3_cm`'s 0.82 / 0.85 / 0.93 was never plan-following either.** Its
members close more on the committed objective on 0.48–0.57 of moves —
chance — and the greedy writer re-derives the assignment from where the
bodies are, so the plan FOLLOWS the members. What passed there is a
nearest-unheld covering walk the members learn from the objective
counts, under a writer that never contradicts it. **Amendment 6's ordering
("the members want a plan that holds still") is RETRACTED**: the FH1
executor under PL1's head with re-commits forbidden at play reads
0.44 / 0.61 / 0.67 against 0.37 / 0.61 / 0.63 — a plan that holds still
is a plan that stays wrong, and its first plan covers the board in
5–56% of episodes. The proposal to hold the head's commitment still as
the next arm is withdrawn in that form (see § 4).

### 3. Why, then: the two halves trained each other into a search and a geometry walk

- The head is paid for the outcome, re-commits for free, and its members
  on the plan-only rung arrive in three and a half turns — so it learns
  to search, and the search converges inside the episode. Nothing prices
  the first plan.
- The members are paid a potential that half the time points at their
  nearest objective anyway, and the other half points at a mark that
  changes under them before they arrive about half the time (first
  commitment kept until arrival 0.48–0.55 in FH1, 0.13–0.62 in CM4) —
  so the mark predicts pay poorly and the nearest objective predicts it
  well; they learn geometry and the counts, as `a3_cm`'s members did.
- Slow members make the head's search fail inside eight turns (targets
  change under walkers who take twice as long, and a squad on the wrong
  objective hovers), so the head's return is a fifth to a tenth of the
  plan-only rung's and noise; a churning head makes the members'
  mark worthless. Each half's failure is the other half's training
  signal. This is the coupling of amendments 3 and 6 with its mechanism
  named.
- The hover itself — standing on an objective that pays nothing while a
  committed objective nine to nineteen inches away pays +0.48 per step —
  is the same walk-off/stay defect every per-model row on the half-step
  recorded, seen from the other side: the policy learned "reach an
  objective, then stay near it" on the majority case where that is right
  (half its squads are on their committed objective) and applies it on
  the wrong objective too. Whether it persists under a mark that does
  not move is the one thing this investigation cannot say, because no
  such mark exists on file.

### 4. What this asks of #384 (proposed; Sash decides; supersedes amendment 6 § 4 items 2–3)

1. **Close the search route on the plan-only rung first: PL1s, the
   plan-only head with commitments sticky IN TRAINING** (KEEP forced
   while a unit's slot holds an objective; the head decides at the first
   open and after the environment's retirement rule clears the slot).
   The head must then learn a one-shot covering assignment or fail.
   Readouts: first-plan coverage (the bar 1.00, PL1 0.09–0.42), success
   with the bar's members, the claimant ablation (the head reads the
   counts; a one-shot cover needs them). Pass: first-plan coverage
   ≥ 0.90 and success ≥ 0.95 on 3/3. If it fails, the planning reward
   needs a term on the plan itself (coverage of the commitment set at
   commit time), which is a build.
2. **Then FH1s: the members under PL1s's planner, sticky.** The mark
   then does not move, so the executor's mark-reading and its hover are
   measured on their own for the first time: closing-more-on-committed
   (0.56–0.72 now), arrival at the committed objective when it is not
   the nearest (0.50–0.66 now), and the hover line (leaving a wrong
   objective 0.35–0.54 now). If the hover persists under a fixed mark,
   it is the execution defect the half-step recorded and the next lever
   is on the members' side, not the head's.
3. The half-step by the staged route (amendment 6 § 4 item 4) stands,
   after 1 and 2.
4. `drafts/join_probe.py` should become a recipe (`measure-plan` gains
   the first-plan coverage, the re-commits-before-arrival share, the
   mark-following share where commit ≠ nearest, and the hover line) — a
   plan-only row without them can read 1.000 on a head that does not
   plan.
