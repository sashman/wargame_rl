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

## Amendment 9 — written 2026-09-24 21:11: the two-stream reward on the join — FH2 and CM5 pre-registered, the build and the desk check

Sash (2026-09-24), after amendment 8: keep the training setup, tune the
reward under two constraints. **(1) Following must always pay more**: a
member's only income is progress on, and presence at, the objective its
unit is committed to; nothing for an uncommitted unit; receding costs.
**(2) Following must pay the same whatever the plan**: completing any
commitment pays the same total, so the members are indifferent to the plan
and the planner — paid the outcome and nothing else — can raise its
expected reward only by a better decision. The document at
https://claude.ai/artifact/UZ7Qwumzg9Hj8FJmfwsTtM states both, the
structure and this plan; this amendment is its pre-registration.

### The structure

Members' stream, per decision, keyed to the unit's committed objective:
`w_follow × (d_prev − d_now) / d_commit` (progress as the FRACTION of
the distance to the objective's EDGE the model had when its target was
set; completing any commitment pays exactly `w_follow`) plus `w_stay`
for ending the step inside the committed objective, capped at
`stay_max` per model per commitment. Planner's stream, unchanged:
coverage 0.3 at every close and the success bonus 5.0 (speed-scaled) on
the close that succeeds — the outcome and only the outcome. No fallback
to the nearest objective, no plan-shape term, no switch charge.

**The build** (`4 files`, default off, every golden byte-identical, the
full suite 5,064 passed): `normalize_to_commit_distance` on
`closest_objective_v2` (with it, a unit holding no commitment is paid
nothing when `fallback_to_nearest` is off); `cap_per_commitment` on
`objective_stay` (a fresh budget when the model's committed objective
changes); `configs/experiments/curriculum/a3_head_r.yaml` = `a3_head`
with progress scale 2.0 normalised, stay 0.5 at crowding exponent 0 and
cap 2.0 (= 1.0 weighted), fallback off. `tests/test_two_stream_reward.py`
pins the three. And `just measure-plan` gains the four columns amendment
8 asked for: `first-plan`, `pre-arrival re-commit`, `mark`,
`leave-wrong` (the bar: 1.00 / 0.05 / 0.96 / n·a).

### The desk check (no training; the bar on the new config, n=100, seeds 700000+)

- **The bridge holds.** `squad_march_take` on `a3_head_r` reads the
  same lines as on `a3_head` to the digit: success 1.000, arrival at the
  committed objective 0.96 at turn 3.55, first commit = nearest 0.51,
  re-committed before arrival 0.05, mark-following 0.96. PL1's heads with
  the bar's members read plan-only 1.000 ×3 on it, as they must (nothing
  at play reads the reward).
- **Constraint 2 holds exactly on progress.** 1,014 completed, unswitched
  commitments: progress **0.1667 with sd 0.0000** (= 2.0 / 12, the
  per-model retimer paying every per-decision term over the model count),
  correlation with the distance at commit **−0.03**; under the old term
  0.141 ± 0.022 at **+0.95**.
- ⚠ **And it holds only up to the cap on the stay term.** Progress + stay
  reads 0.222 ± 0.020, correlation with distance **−0.45**, and only
  **33%** of the commitments made with two turns to spare reach the
  maximum 0.250 — not the "every commitment reaches 3.0" the document
  promised. The cause is the rung, not the term: `terminate_on_success`
  ends the episode at the close on which the fourth objective is occupied,
  so the LAST squad to arrive collects zero or one holding step and the
  early squads collect their budget. The residual plan-dependence is
  bounded by the cap — at most 1.0 of a commitment's 3.0, a third of the
  progress pay — and it favours arriving EARLY, which the planner's own
  stream also favours, so the members and the planner are not set against
  each other by it. Turning `terminate_on_success` off would make the
  maximum exact and would change the rung's success reading; it is not
  done here and is the first thing to try if the reads show the members
  preferring near targets. **The pre-registered desk-check line is
  amended to: progress flat (correlation within ±0.05), the stay pay never
  above its cap, and the total's correlation with distance explained by
  the episode's end alone** (a squad that arrives with two turns left
  reaches the maximum). Read the members' pay per commitment on every arm
  beside the score.

### The arms

| arm | the one change | tag | comparator by name |
|---|---|---|---|
| **FH2** | FH1 with the members' stream above: PL1's finished head seed for seed draws every commitment (greedy, frozen), the members learn from scratch on `a3_head_r` | `fh2` | FH1 at matched rounds (0.16 / 0.43 / 0.04 → 0.26 / 0.60 / 0.60 → 0.37 / 0.61 / 0.63); the executor alone; the bar |
| **CM5** | CM4 with the members' stream above: head and members from scratch on `a3_head_r` | `cm5` | CM4 at matched rounds (as trained 0.000 / 0.160 / 0.000 → 0.000 / 0.190 / 0.210 → 0.000 / 0.370 / 0.320; plan-only 0.01 / 0.17 / 0.16 → 0.19 / 0.53 / 0.55 → 0.72 / 0.68 / 0.72); A3 from scratch 0.700 / 0.800 / 0.770; `a3_cm` 0.820 / 0.850 / 0.930 |

Three seeds each, 122,880 rounds, CM4's recipe unchanged (128 rounds per
update: 32 rollout rounds × 4 envs, `ent_coef` 0.003, mean credit,
broadcast planning credit, greedy scoring with the sampled row beside it),
Wandb group `curriculum-cm-plan`, both launched together and read FH2
first. Unpairable at initialisation against FH1 / CM4 (a different
reward, same init: the seeds are shared, so the per-seed difference is a
paired estimator on the init and nothing else). Reads at 40,960 / 81,920 /
122,880, n=100 on seeds 700000+, the final gated on the trainer exiting
and every seed's log carrying the round line: FH2 the split row beside
the executor-alone row and the planner's plan-only ceiling; CM5 the
as-trained row beside the same head's plan-only row; the four new columns
on every row; the walk-off probe; for CM5 the ablation and the planning
panel (planning return, commitment entropy, planning EV) against CM4's.

### Criteria, per seed at 122,880, written before any number exists

| readout | pass mark | constraint |
|---|---|---|
| mark-following (closing more on the committed than on the nearest, where they differ) | ≥ 0.90 on 2 of 3 seeds (FH1 0.56 / 0.72 / 0.58; bar 0.96; a random mark 0.33) | 1 |
| leave-wrong (setting out from an objective the unit is not committed to) | ≥ 0.80 on 2 of 3 (FH1 0.54 / 0.43 / 0.35; bar 0.82–1.00) | 1 |
| arrival at the committed objective where it is not the nearest | ≥ 0.85 on 2 of 3 (FH1 0.66 / 0.50 / 0.65; bar 0.95) | 1 |
| realised pay differential per step, toward the committed minus toward the nearest | ≥ +0.15 (FH1 +0.03 to +0.06; bar +0.20), read with the probe's geometric pay so it compares across rewards | 1 |
| execution pay per completed commitment against distance at commit | progress correlation within ±0.05; stay never above the cap | 2 |
| first-plan coverage (CM5) | readout, not a gate: ≥ 0.60 on 2 of 3 would say the head plans first time (PL1 0.09–0.42, CM4 0.00–0.41) | 2 |
| re-commits before arrival (CM5) | readout, not a gate: ≤ 0.25 (PL1 0.32–0.50, CM4 0.50–0.76) | 2 |
| **success at the cap** | **FH2 ≥ 0.85 on 2 of 3 (FH1 0.37 / 0.61 / 0.63). CM5: PASS ≥ 0.95; AHEAD of CM4's same seed by more than two binomial SE on 3 of 3; else NULL** | both |

**Decision rules.** FH2 passes and CM5 passes → the structure holds on this
shape; next the half-step on the same recipe. FH2 passes, CM5 does not →
the planner's half; the first-plan and re-commit readouts say whether it
still searches, and the switch leak (a mid-walk re-commit pays a unit up to
1.5) is the first thing to close. FH2 fails on mark-following → one sweep
of `w_stay` at 0.25 and 1.0 and `w_follow` at 4.0 before the planner is
touched. FH2 follows but hovers (mark ≥ 0.90, leave-wrong < 0.80) → the
stay term is being read on the wrong objective; the walk-off probe's pay
inside a wrong objective must read zero from it by construction. The desk
check failing → nothing launches (it did not fail; see above).

**Expectation (a guess, written so it can be wrong).** FH2: mark-following
0.85–0.95, leave-wrong 0.7–0.9, success 0.80–0.95 on two seeds, one seed
behind (the legibility rung's 2 of 3). CM5: the members follow and the
head still searches — first-plan coverage 0.3–0.6, success 0.5–0.9,
AHEAD of CM4 on 3 of 3, PASS on at most one seed.

## Amendment 10 — written 2026-09-25 01:06: FH2 and CM5 read — FAIL on every criterion; the normalised progress under a churning head is a heavy-tailed reward that broke the members' critic

Both arms read at 40,960 / 81,920 / 122,880, n=100 on seeds 700000+, the
final gated on the trainers' exit. Wandb group `curriculum-cm-plan`: FH2
wqkn5pcw / 6m84akim / rmgpi33o, CM5 is01wzx0 / jc1nvkv2 / 5g3ai03j.
Revision `3800358`. Every comparator value below is at MATCHED rounds,
from the same probe run on FH1's and CM4's own checkpoints.

### FH2 (#411) — the members under PL1's frozen head, the new stream

| read | split (FH1) | executor alone (FH1) | mark-following (FH1) | arrival at committed, commit ≠ nearest (FH1) | leave-wrong (FH1) | pay differential per step (FH1) |
|---|---|---|---|---|---|---|
| 40,960 | 0.14 / 0.55 / 0.44 (0.16 / 0.43 / 0.04) | 0.14 / 0.56 / 0.45 (0.20 / 0.53 / 0.09) | 0.38 / 0.52 / 0.61 (0.68 / 0.63 / 0.49) | 0.23 / 0.59 / 0.64 (0.52 / 0.60 / 0.31) | 0.54 / 0.40 / 0.65 (0.32 / 0.49 / 0.85) | −0.01 / +0.02 / +0.03 (+0.05 / +0.03 / +0.03) |
| 81,920 | 0.22 / 0.50 / 0.33 (0.26 / 0.60 / 0.60) | 0.17 / 0.47 / 0.37 (0.31 / 0.42 / 0.63) | 0.54 / 0.52 / 0.56 (0.75 / 0.46 / 0.44) | 0.56 / 0.45 / 0.57 (0.59 / 0.57 / 0.56) | 0.50 / 0.27 / 0.41 (0.41 / 0.31 / 0.23) | +0.01 / +0.02 / +0.02 (+0.06 / +0.01 / −0.01) |
| **122,880** | **0.39 / 0.29 / 0.40** (0.37 / 0.61 / 0.63) | 0.35 / 0.26 / 0.24 (0.40 / 0.48 / 0.72) | 0.61 / 0.57 / 0.58 (0.56 / 0.72 / 0.58) | 0.61 / 0.58 / 0.62 (0.66 / 0.50 / 0.65) | 0.50 / 0.63 / 0.44 (0.54 / 0.43 / 0.35) | +0.03 / +0.02 / +0.03 (+0.06 / +0.03 / +0.03) |

**Verdict on the letter, per seed at 122,880: FAIL / FAIL / FAIL.** Pass marks
were mark-following ≥ 0.90 on 2/3, leave-wrong ≥ 0.80 on 2/3, arrival at
the committed objective ≥ 0.85 on 2/3, pay differential ≥ +0.15, success
≥ 0.85 on 2/3. None was met on any seed: mark-following 0.57–0.61, leaving a wrong objective 0.44–0.63, arrival 0.58–0.62, the realised pay differential +0.02 to +0.03, success 0.29–0.40 — level with FH1 on one seed and behind it on two, with the executor-alone row still level with the split row (the walk is still the geometry's). The executor's explained variance ran
**0.06–0.55** through the run (FH1's 0.84–0.96), displacement entropy
2.1–2.2 nats at the cap (FH1 1.2–1.5), clip fraction 0.22–0.31.

### CM5 (#412) — head and members from scratch, the new stream

| read | as trained (CM4) | plan-only (CM4) | first-plan coverage | re-commits before arrival | mark-following (CM4) | leave-wrong (CM4) |
|---|---|---|---|---|---|---|
| 40,960 | 0.00 / 0.00 / 0.03 (0.00 / 0.16 / 0.00) | 0.00 / 0.33 / 0.16 (0.01 / 0.17 / 0.16) | 0.00 ×3 | 0.97 / 0.76 / 0.96 | 0.66 / 0.56 / 0.51 (0.52 / 0.76 / 0.48) | 0.45 / 0.72 / 0.21 (0.80 / 0.71 / 0.75) |
| 81,920 | 0.00 / 0.00 / 0.00 (0.00 / 0.19 / 0.21) | 0.03 / 0.04 / 0.13 (0.19 / 0.53 / 0.55) | 0.00 / 0.00 / 0.09 | 0.86 / 0.73 / 0.32 | 0.37 / 0.62 / 0.65 (0.73 / 0.72 / 0.69) | 0.59 / 0.51 / 0.79 (0.62 / 0.73 / 0.75) |
| **122,880** | **0.29 / 0.00 / 0.22** (0.00 / 0.37 / 0.32) | **0.16 / 0.00 / 0.27** (0.72 / 0.68 / 0.72) | 0.04 / 0.00 / 0.35 | 0.85 / 0.98 / 0.86 | 0.43 / 0.59 / 0.52 (0.71 / 0.68 / 0.71) | 0.57 / 0.44 / 0.71 (0.80 / 0.69 / 0.71) |

**Verdict on the letter, per seed at 122,880 as trained: NULL / NULL /
NULL, and not AHEAD** (ahead of CM4's same seed on one of three). The
head is far worse than CM4's: its plan-only row at the cap is 0.16 / 0.00 /
0.27 against 0.72 / 0.68 / 0.72, its first assignment covers the board in
0–35% of episodes, and in 50–79% of failed episodes no squad is committed
to the empty objective at all. The ablation is flat on every seed (blank
within 0.01 of trained): the members read no mark. Planning panel in CM4's
band (return 0.36–0.50, commitment entropy 0.88–0.97 nats, planning EV
0.22–0.33). **The members' critic collapsed: explained variance 0.07–0.37
in the last quarter against CM4's ~0.75**, displacement entropy 2.1–2.4
nats (CM4 1.7–2.1).

### The mechanism: normalised progress under a churning head is a heavy-tailed reward

Measured on the finals with the desk-check probe (n=30). Progress is the
fraction of the distance the model had when its target was set, and the
head re-commits units mid-walk (85–98% of squads before their first
arrival). A re-commit made when the unit is already near its new target
sets a SMALL anchor, and every step after it pays a large fraction:

| | spans whose anchor is under 8 in | per-step progress pay p99 | max | min |
|---|---|---|---|---|
| the bar on the new config | 3 of 387 (0.8%) | +0.07 | +0.17 | 0.00 |
| CM5 s1 as trained | **395 of 1,224 (32%)** | +0.13 | +0.17 | **−0.33** |
| CM5 s3 as trained | 126 of 1,233 (10%) | +0.10 | +0.17 | **−0.71** |

One six-inch step can then pay a whole commitment (+0.167, the per-step
maximum) or, receding, several times a commitment's worth in one move. The
members' return became a function of the head's future re-commits, which
their critic cannot predict — explained variance 0.06–0.55 on both arms
against 0.84–0.96 for the same members under the old stream — so the
advantages PPO trained on were noise, the displacement head stayed diffuse,
and the members learned less than under the old stream, not more. The
reward met constraint 2 on paper (every completed commitment pays the
same, verified to four decimals on the bar) and violated it dynamically:
a re-commit changes what a step pays by an order of magnitude, and under
the head writer re-commits are the norm. The bar's stable plan never
exposed it, which is why the desk check passed.

**Which decision rule fired.** FH2 fails on mark-following, which
amendment 9 routes to a weight sweep — and the mechanism above says the
weights are not the limiting factor: at any `w_follow` the per-step pay
under a churning head is heavy-tailed in proportion, and the critic breaks
the same way. The sweep is not run. What the two arms establish:

- **The stay term did what it was built for**, on the small scale it was
  given: a move that keeps the objective pays +0.01 to +0.05 per step
  against −0.06 to +0.02 for one that leaves it (the old stream: a few
  thousandths either way), and the members leave a wrong objective toward
  the committed one on 0.66–0.85 of their moves on two CM5 seeds where
  CM4's did on 0.56–0.79. It is not what failed.
- **The normalisation is what failed**, and it failed through its
  interaction with churn, which the plan-only rung (a stable, fast-executed
  plan) never exercised. A normalised potential is safe only under a plan
  that does not re-anchor, or with the anchor floored (a fraction of the
  ORIGINAL distance, never of the distance at a re-commit).
- **Sash's two constraints are the right ones and this build did not
  deliver the second.** "Following pays the same for every plan" has to
  hold per STEP under every plan the head can produce, not per completed
  commitment under the bar's.

### What this asks of #384 (proposed; Sash decides)

1. **Do not sweep the weights.** The rule fired on the letter, and the
   mechanism says a sweep measures the same defect at another scale.
2. **Fix the normalisation, one change**: anchor the fraction to the
   unit's distance at its FIRST commitment of the episode (or floor the
   anchor at, say, 12 inches), so a re-commit can never make a step pay
   more than a normal step; verify on CM4's own finals with the desk-check
   probe that per-step pay p99 and min stay within the bar's band under a
   churning head BEFORE launching. Then FH2b / CM5b on the same recipe.
   Predicted: the members' explained variance back above 0.8; whether
   mark-following then clears 0.90 is the open question — the shared
   eastward geometry (cos 0.5–0.7 between the two target directions) still
   pays the nearest objective's walk almost as well as the committed one,
   and the legibility rung's members followed a plan that disagreed with
   the geometry 83% of the time, not one that agreed half the time.
3. **Read a member reward's per-step distribution under a CHURNING plan
   in every desk check from here**, not only under the bar: the bar's
   stable plan is the one case a re-anchoring potential cannot expose.

## Amendment 11 — written 2026-09-25 09:39: the anchor floored — FH2b and CM5b pre-registered, with the desk check that amendment 10 asked for

Sash (2026-09-25): proceed with the recommended steps. One change against
`a3_head_r.yaml`: **`normalize_min_distance: 12.0`** on the travel term
(`configs/experiments/curriculum/a3_head_rf.yaml`). The fraction's anchor
is floored at twelve inches, so a step can never pay more than a normal
step from twelve inches out; a target set from nearer than the floor pays
a smaller fraction, the safe side. No weight sweep (the tail scaled with
the weights). Everything else FH2's / CM5's: CM4's recipe, PL1's head seed
for seed as FH2b's frozen planner, head and members from scratch on CM5b.

### The desk check, this time under a CHURNING writer

Amendment 10's rule: read a member reward's per-step pay under a plan
that re-commits, not only under the bar. The desk-check probe (n=30,
seeds 700000+) on CM4's own final checkpoints playing the new configs —
the same churning heads, the reward re-computed:

| writer | config | per-step progress pay p99 | max | min | completed commitment pays |
|---|---|---|---|---|---|
| the bar | rf (floored) | +0.069 | +0.083 | 0.000 | 0.1667 ± 0.0000 (= 2.0/12) |
| CM4 s1 | r (un-floored) | +0.125 | +0.167 | **−1.950** | 0.1667 |
| CM4 s1 | **rf** | +0.073 | **+0.083** | **−0.083** | 0.1667 |
| CM4 s2 | r | +0.066 | +0.167 | −0.418 | 0.1667 |
| CM4 s2 | **rf** | +0.061 | +0.083 | −0.062 | 0.1667 |
| CM4 s3 | r | +0.167 | **+0.856** | **−1.224** | 0.1667 |
| CM4 s3 | **rf** | +0.078 | +0.083 | −0.083 | 0.1667 |
| CM5 s1 / s3 | **rf** | +0.068 / +0.075 | +0.083 | −0.061 / −0.081 | 0.1667 |

Under the un-floored fraction CM4's own heads produced steps paying up to
five times a whole commitment (s3: +0.86, −1.22; s1: −1.95) — worse than
CM5's finals showed, because CM4's heads re-commit nearer their targets
(5–12% of spans under eight inches). With the floor every writer's
per-step pay sits inside the bar's band: max +0.083 (a six-inch step
from twelve inches, half a commitment), min −0.083, p99 0.06–0.08 against
the bar's 0.069. Completing a commitment from beyond the floor still pays
exactly 2.0/12. **The check passes**, on the writer that failed it.

### The arms

| arm | the one change against | tag | comparators by name |
|---|---|---|---|
| **FH2b** (#413) | FH2: the floor | `fh2b` | FH2 at matched rounds (split 0.14 / 0.55 / 0.44 → 0.22 / 0.50 / 0.33 → 0.39 / 0.29 / 0.40; executor EV 0.06–0.55), FH1 (0.16 / 0.43 / 0.04 → 0.26 / 0.60 / 0.60 → 0.37 / 0.61 / 0.63; EV 0.84–0.96), the executor alone, the bar |
| **CM5b** (#414) | CM5: the floor | `cm5b` | CM5 at matched rounds (as trained 0.00 / 0.00 / 0.03 → 0.00 ×3 → 0.29 / 0.00 / 0.22; plan-only 0.00 / 0.33 / 0.16 → 0.03 / 0.04 / 0.13 → 0.16 / 0.00 / 0.27; members' EV 0.07–0.37), CM4 (as trained 0.000 / 0.160 / 0.000 → 0.000 / 0.190 / 0.210 → 0.000 / 0.370 / 0.320; plan-only 0.01 / 0.17 / 0.16 → 0.19 / 0.53 / 0.55 → 0.72 / 0.68 / 0.72), A3 from scratch 0.700 / 0.800 / 0.770 |

Three seeds each, 122,880 rounds, CM4's recipe, Wandb group
`curriculum-cm-plan`, launched together, FH2b read first; reads at
40,960 / 81,920 / 122,880 as amendment 9's, the final gated on the exit,
the probes on the comparators' own checkpoints at matched rounds.

**Criteria: amendment 9's, unchanged** (mark-following ≥ 0.90 on 2/3,
leave-wrong ≥ 0.80 on 2/3, arrival at the committed objective where it is
not the nearest ≥ 0.85 on 2/3, pay differential ≥ +0.15, FH2b success
≥ 0.85 on 2/3; CM5b PASS ≥ 0.95 / AHEAD of CM4 by two binomial SE on 3/3
/ NULL), the decision rules as amendment 9's. **One new readout, not a
gate: the members' explained variance over the last quarter ≥ 0.80**
(FH2 0.06–0.55, CM5 0.07–0.37; FH1 0.84–0.96, CM4 ~0.75). It is the
number the floor exists to restore, and it is read at 20,480 rounds
already: a red panel there says the floor was not the whole of the
mechanism.

**Expectation (a guess, written so it can be wrong).** The critic
recovers (EV ≥ 0.8 on 3/3 by the second quarter). FH2b lands in FH1's
band or a little above it (split 0.4–0.7), with mark-following 0.6–0.75
— short of 0.90, because the committed and the nearest direction share
most of the walk on this board (cos 0.5–0.7) and the members that did
learn to follow (LR1) trained under a plan that disagreed with the
geometry 83% of the time. CM5b: the head still searches (first-plan
coverage under 0.5), success 0.2–0.5, AHEAD of CM4 on at most two seeds.
If FH2b's members still walk the geometry with a healthy critic, the
lever is the plan distribution the members train under, which is a change
to the setup and Sash's call.

## Amendment 12 — written 2026-09-25 13:37: FH2b and CM5b read — the floor restored the critic and the members now follow the plan; the planner is the failing half

Both arms read at 40,960 / 81,920 / 122,880, n=100 on seeds 700000+, the
final gated on the trainers' exit. Wandb group `curriculum-cm-plan`: FH2b
svyy25yo / wmf72ony / y2dm29nw, CM5b mxvxjcj2 / 2lhpmurj / 44jipi4h.
Revision `a56fb39`. Comparators at matched rounds from the same probe on
their own checkpoints.

### FH2b (#413) — the members under PL1's frozen head, the floored stream

| read | split (FH2 · FH1) | executor alone | mark-following (FH2 · FH1) | arrival at committed, commit ≠ nearest (FH2 · FH1) | leave-wrong | pay differential per step |
|---|---|---|---|---|---|---|
| 40,960 | 0.49 / 0.43 / 0.66 (0.14 / 0.55 / 0.44 · 0.16 / 0.43 / 0.04) | 0.43 / 0.46 / 0.64 | 0.62 / 0.62 / 0.56 (0.38 / 0.52 / 0.61 · 0.68 / 0.63 / 0.49) | 0.68 / 0.78 / 0.64 (0.23 / 0.59 / 0.64 · 0.52 / 0.60 / 0.31) | 0.45 / 0.59 / 0.24 | +0.04 / +0.05 / +0.03 |
| 81,920 | 0.62 / 0.81 / 0.50 (0.22 / 0.50 / 0.33 · 0.26 / 0.60 / 0.60) | 0.54 / 0.75 / 0.61 | 0.63 / 0.75 / 0.59 (0.54 / 0.52 / 0.56 · 0.75 / 0.46 / 0.44) | 0.73 / 0.78 / 0.62 (0.56 / 0.45 / 0.57 · 0.59 / 0.57 / 0.56) | 0.42 / 0.36 / 0.36 | +0.05 / +0.08 / +0.02 |
| **122,880** | **0.84 / 0.79 / 0.46** (0.39 / 0.29 / 0.40 · 0.37 / 0.61 / 0.63) | 0.71 / 0.75 / 0.52 | 0.75 / 0.72 / 0.60 (0.61 / 0.57 / 0.58 · 0.56 / 0.72 / 0.58) | 0.84 / 0.88 / 0.71 (0.61 / 0.58 / 0.62 · 0.66 / 0.50 / 0.65) | 0.41 / 0.42 / 0.65 | +0.09 / +0.08 / +0.03 |

Executor explained variance 0.94 / 0.92 / 0.92 in the last quarter, displacement entropy 1.4–1.8 nats (FH2 0.06–0.55, FH1 0.84–0.96).

**Verdict on the letter, per seed at 122,880: FAIL / FAIL / FAIL on the letter — and the nearest miss on the ladder.**
No mark was met: success 0.84 / 0.79 / 0.46 against ≥ 0.85 on 2/3 (two seeds within a hundredth and six hundredths of it, and +0.47 / +0.18 / −0.17 against FH1's finals of 0.37 / 0.61 / 0.63); arrival at the committed objective 0.84 / 0.88 / 0.71 against ≥ 0.85 on 2/3 (one seed clears, one misses by a hundredth); mark-following 0.75 / 0.72 / 0.60 against 0.90; leaving a wrong objective 0.41 / 0.42 / 0.65 against 0.80; the pay differential +0.09 / +0.08 / +0.03 against +0.15. The executor-alone row (0.71 / 0.75 / 0.52) is still close to the split row.

### CM5b (#414) — head and members from scratch, the floored stream

| read | as trained (CM5 · CM4) | plan-only (CM5 · CM4) | first-plan | re-commits before arrival | mark-following | arrival at committed | pay differential |
|---|---|---|---|---|---|---|---|
| 40,960 | 0.01 / 0.00 / 0.00 (0.00 / 0.00 / 0.03 · 0.00 / 0.16 / 0.00) | 0.16 / 0.00 / 0.00 (0.00 / 0.33 / 0.16 · 0.01 / 0.17 / 0.16) | 0.00 ×3 | 0.95 / 0.47 / 0.59 | 0.53 / 0.66 / 0.79 | 0.65 / 0.98 / 0.77 | +0.07 / +0.18 / +0.16 |
| 81,920 | 0.02 / 0.00 / 0.26 (0.00 ×3 · 0.00 / 0.19 / 0.21) | 0.39 / 0.04 / 0.36 (0.03 / 0.04 / 0.13 · 0.19 / 0.53 / 0.55) | 0.14 / 0.00 / 0.06 | 0.78 / 0.61 / 0.84 | 0.47 / 0.61 / 0.82 | 0.18 / 0.80 / 0.77 | +0.04 / +0.15 / +0.19 |
| **122,880** | **0.00 / 0.00 / 0.17** (0.29 / 0.00 / 0.22 · 0.00 / 0.37 / 0.32) | **0.00 / 0.01 / 0.82** (0.16 / 0.00 / 0.27 · 0.72 / 0.68 / 0.72) | 0.00 / 0.00 / 0.09 | 0.45 / 0.84 / 0.70 | **0.86 / 0.61 / 0.75** | **0.86 / 0.94 / 0.64** | **+0.19 / +0.13 / +0.14** |

Members' explained variance 0.67 / 0.71 / 0.85 in the last quarter (CM5
0.07–0.37, CM4 ~0.75), displacement entropy 1.7–1.8 nats (CM5 2.1–2.4).
Planning panel: return 0.45 / 0.39 / **0.84** (s3 rising 0.33 → 0.84 by
quarter), commitment entropy 0.89 / 0.91 / 0.71, planning EV 0.45 / 0.78 /
0.18. Ablation: s3 reads the mark (blank → 0.00); s1 and s2 are at zero
in every column. Failure census: on s1 and s2 the empty objective has NO
squad committed to it in 92–100% of failed episodes.

**Verdict on the letter, per seed at 122,880 as trained: NULL / NULL /
NULL, and not AHEAD.**

### What the pair says

- **The floor did what it was built for.** The members' critic is back
  (EV 0.67–0.91 against 0.06–0.55), and with it the members do what the
  reward pays for: on CM5b they arrive at the objective their unit is
  committed to on 0.64–0.94 of the squads whose commitment is not the
  nearest, walk toward it rather than the nearest on 0.61–0.86 of moves,
  set out from a wrong objective on 0.63–0.68, and realise +0.13 to +0.19
  per step of the +0.20 the mark pays (CM5: +0.01 to +0.03). **Constraint
  1 is delivered by this stream.** On FH2b, under PL1's covering plan,
  success reaches 0.84 / 0.79 / 0.46 — ahead of FH1's finals by +0.47 and +0.18 on two seeds and behind by 0.17 on the third, with the members arriving at their committed objective on 0.84 / 0.88 / 0.71 of the disagreeing squads (FH1 0.66 / 0.50 / 0.65) and walking toward it rather than the nearest on 0.75 / 0.72 / 0.60 of moves (FH1 0.56 / 0.72 / 0.58; the bar 0.96).
- ⚠ **Faithful members on a stacked plan hold one or two objectives, and
  two heads of three never left the stack.** CM5b's members follow a plan
  that commits every squad to the same one or two objectives (maximum
  stack 4.8 / 8.3, first-plan coverage 0.00). The planner's stream is the
  outcome, broadcast: every unit's commitment decision is paid the same
  coverage and the same success bonus whether it spread or stacked, and
  with members who go where it points, a stacking head has no unit-level
  reason to send one squad elsewhere. Under the bar's members (PL1) the
  head's search converged anyway because the bar arrives in three and a
  half turns; under learned followers who take five or six, a stack made
  in turn one is never repaired inside eight. s3's head found a covering
  plan in the last quarter (plan-only 0.82) and its members had not
  caught up.
- **The join's two halves have swapped roles.** Amendment 6 read the
  members as the larger failing half; with the floored stream they are
  the working half and the planner is the wall — the same reading the
  plan-only rung gave when it showed the head searches rather than plans.

**Which decision rule fired.** On the letter, FH2b fails on mark-following, which amendment 9 routes to a weight sweep (`w_stay` 0.25 and 1.0, `w_follow` 4.0). The reading argues against running it as the next arm: the members' mark-following sits at 0.60–0.75 under a plan that agrees with the nearest objective for half the squads and whose direction shares most of the walk with the nearest (cos 0.63–0.68), exactly the band amendment 11 predicted, and the one set of members that reached 0.90 (the legibility rung) trained under a plan that disagreed with the geometry 83% of the time. A weight sweep would price the same geometry at another scale. The two arms together say the members' half works well enough to expose the planner's, and the planner's is where the join now fails.

### What this asks of #384 (proposed; Sash decides)

1. **Keep the floored stream** (`a3_head_rf.yaml`) as the members' reward
   from here: it is the first per-model reward on this ladder under which
   the members demonstrably follow a plan they cannot reconstruct from the
   board.
2. **The planner's half is now the question, and its stream is the
   place.** The outcome-only broadcast stream pays a stacking head the
   same at every unit; the per-unit counterfactual credit (B6, built,
   `--planning-credit counterfactual`) was a paired null under the bar's
   members, where the search converged regardless, and has never been
   read under members who follow. One arm: CM5b's recipe with the
   counterfactual credit (CM5c), read on first-plan coverage, the
   plan-only row and success against CM5b at matched rounds. Prediction:
   the heads leave the stack (plan-only ≥ 0.5 on 2/3 by 81,920).
3. If CM5c does not move the head, a plan-shape term on the planning
   stream (coverage of the commitment set at commit time) is the next
   lever — Sash's second constraint permits it, since it pays the planner
   for a property of its own decision and nothing of the members'.
4. Read a member reward's per-step pay under a churning writer in every
   desk check (the amendment-10 rule stands), and read the planner's
   first-plan coverage beside success on every head arm.

## Amendment 13 — written 2026-09-25 23:57: two probes select the next two arms — FH3 (the members under the rotated writer on the floored stream) and CM6 (a plan-shape term on the planner's stream)

Sash (2026-09-25): "what should you investigate next?" — "go". Two
no-training probes on the finals of amendment 12, then the arms they
select.

### Probe 1 — the planner's outcome stream cannot see spreading under these members

CM5b's own games (greedy, n=30 per seed, seeds 700000+) forked at the
first turn's close: the control keeps the head's plan; the counterfactual
moves ONE stacked squad's commitment onto an objective no squad is
committed to, that squad held to its commitment in both branches. Read:
the realised planning return (0.3 × coverage at every close + the success
bonus), paired.

| CM5b seed | forked squad arrives: control → spread | planning return: control → spread | paired Δ | success: control → spread |
|---|---|---|---|---|
| s1 (stacks) | 0.77 → **0.00** | 0.490 → 0.490 | **+0.000 ± 0.015** | 0.00 → 0.00 |
| s2 (stacks) | 0.17 → 0.77 | 0.510 → 0.520 | +0.010 ± 0.014 (t 0.7) | 0.00 → 0.00 |
| s3 (learned to spread) | 0.67 → 0.67 | 0.717 → 0.956 | **+0.239 ± 0.140** (t 1.7) | 0.11 → 0.30 |

On s1 the redirected squad never goes (27 inches to an objective its
members have never been sent to; they walk to the stack); on s2 it goes
and the return barely moves (it arrives late, held 1.50 → 1.57); only
where the head already spreads does spreading pay. **The per-unit
counterfactual credit would attribute the outcome's difference, which is
nothing on the stacking seeds — CM5c is withdrawn.** A term that pays the
planner for the shape of its plan, independent of execution, is the one
that can move a stacked head.

### Probe 2 — the members still read the board first

The same checkpoints played under the ROTATED environment writer
(`a3_legible.yaml`, no squad's assignment its nearest objective), n=100:

| members | arrived at the committed objective | arrived at the nearest instead |
|---|---|---|
| LR1 (trained under the rotated writer, old reward) | **0.89 / 0.96 / 0.46** | 0.13 / 0.00 / 0.81 |
| FH1 (PL1's head, old reward) | 0.40 / 0.32 / 0.44 | 0.71 / 0.69 / 0.80 |
| FH2 (PL1's head, un-floored stream) | 0.26 / 0.29 / 0.32 | 0.64 / 0.63 / 0.71 |
| **FH2b** (PL1's head, floored stream) | **0.49 / 0.53 / 0.28** | 0.76 / 0.88 / 0.78 |
| the bar `squad_march_committed` | arrived at committed 0.96, at nearest 0.00; END on committed 0.96 / on nearest 0 |

FH2b's 0.75 mark-following under PL1's plan was mostly the walk the
geometry gives; handed a mark that disagrees with the board, its members
go to the nearest objective three times in four. The members that follow
such a mark (LR1) trained under a writer that never agreed with the
board. **The plan distribution the members train under is the lever on
their side**, and the floored stream should make that lever bite harder
than it did on LR1.

### The build (default off; every existing config and golden untouched; the full suite 5,067 passed)

- `commitment_coverage`: a global calculator, the share of objectives
  some living unit is committed to, classed a state global so under the
  head writer it lands on the PLANNING stream and never reaches a member
  (constraint 2 holds: it pays the planner for a property of its own
  decision). `configs/experiments/curriculum/a3_head_rf_pc.yaml` =
  `a3_head_rf.yaml` plus that term at 0.3.
- `configs/experiments/curriculum/a3_legible_rf.yaml` = the legibility
  rung's config (`assignment: rotated`, success
  `all_units_on_commitment`) with `a3_head_rf.yaml`'s member terms.

**Desk check.** The bar's member pay on `a3_head_rf_pc` is the floored
stream's to four decimals (progress 0.1667 ± 0.0000 per completed
commitment, per-step p99 +0.069, max +0.083, min 0); CM5b's s1 head on it
reads per-step max +0.083 / min −0.066; PL1's plan-only ceiling still
1.000 and CM5b s1's plan-only row still 0.00 (nothing at play reads the
reward). On `a3_legible_rf` the committed bar's member pay is 0.1667 ±
0.0000 per commitment, per-step p99 +0.062, min 0 (the rotated writer
never re-anchors).

### The arms

| arm | the one change | tag | comparators by name |
|---|---|---|---|
| **FH3** (#415) | the members trained under the ROTATED environment writer on the floored stream (`a3_legible_rf.yaml`); no frozen planner | `fh3` | LR1 at matched rounds (arrival 0.89 / 0.96 / 0.46, success 0.59 / 0.94 / 0.40 at the cap); FH2b's members at play on the rotated config (0.49 / 0.53 / 0.28); the bar |
| **CM6** (#416) | CM5b plus `commitment_coverage` 0.3 on the planning stream (`a3_head_rf_pc.yaml`); head and members from scratch | `cm6` | CM5b at matched rounds (as trained 0.01 / 0.00 / 0.00 → 0.02 / 0.00 / 0.26 → 0.00 / 0.00 / 0.17; plan-only 0.16 / 0.00 / 0.00 → 0.39 / 0.04 / 0.36 → 0.00 / 0.01 / 0.82; first-plan 0.00 / 0.00 / 0.09); CM4; A3 from scratch |

Three seeds each, 122,880 rounds, CM4's recipe, Wandb group
`curriculum-cm-plan`, launched together; reads at 40,960 / 81,920 /
122,880 on seeds 700000+, the final gated on the exit.

**Criteria, per seed at 122,880.** FH3: arrival at the committed objective
on the rotated config ≥ 0.85 on 2/3 and success (`all_units_on_commitment`)
≥ 0.85 on 2/3; readouts: the same members under PL1's head on
`a3_head_rf_pc` (split row against FH2b's 0.84 / 0.79 / 0.46; arrival at a
non-nearest committed objective against 0.84 / 0.88 / 0.71), executor EV
≥ 0.80. CM6: PASS success ≥ 0.95; AHEAD of CM4's same seed by two
binomial SE on 3/3; NULL otherwise; **SPREADS**: first-plan coverage
≥ 0.60 on 2/3 (CM5b 0.00 / 0.00 / 0.09) and the plan-only row ≥ 0.50 on
2/3 at 81,920; readouts: re-commits before arrival, the members' arrival
at the committed objective (not below CM5b's 0.86 / 0.94 / 0.64),
members' EV, the planning panel.

**Decision rules.** FH3 passes → the members' half is closed and FH3's
members are the warm start for the next joint arm. FH3 fails on arrival
while LR1 passed → the floored stream is worse than the old one under a
disagreeing writer, which the per-step probe on LR1's checkpoints would
then have to explain. CM6 SPREADS and passes → the planner's half is
closed on this shape. CM6 SPREADS and the members do not follow the
spread plan → join the two: CM6's head with FH3's members. CM6 does not
SPREAD → the term is too small against the outcome (the planning return
under a stack is ~0.5 per episode; the term adds up to 0.3 × 8 closes) and
the weight, not the mechanism, is the next arm.

**Expectation (a guess, written so it can be wrong).** FH3: arrival
0.85–0.95 on two seeds, success 0.7–0.9; under PL1's head at play, split
0.8–0.9 with arrival at a non-nearest objective ≥ 0.85. CM6: SPREADS on
2/3 (first-plan 0.6–0.9), success 0.4–0.8, AHEAD of CM4 on two seeds, PASS
on none — the members lag the plan as they did on CM5b's s3.

## Amendment 14 — written 2026-09-26 03:20: FH3 and CM6 read — the plan-shape term makes the head plan first time; the rotated writer makes one seed's members read the mark; CM6 NULL / AHEAD / AHEAD with the best joint reads on the ladder, FH3 FAIL on the letter with the best executor on it

Both arms read at 40,960 / 81,920 / 122,880, n=100 on seeds 700000+, the
final gated on the trainers' exit. Wandb group `curriculum-cm-plan`: FH3
5hcx6fxq / jf05mxtd / 88agvpkj, CM6 siuw5cne / t8nd9hot / nx139kf1.
Revision `2e021a2`.

### CM6 (#416) — CM5b plus `commitment_coverage` 0.3 on the planning stream

| read | as trained (CM5b · CM4) | plan-only (CM5b) | first-plan coverage (CM5b) | re-commits before arrival | members' arrival at a non-nearest committed objective (CM5b) | mark-following |
|---|---|---|---|---|---|---|
| 40,960 | 0.04 / 0.05 / 0.59 (0.01 / 0.00 / 0.00 · 0.00 / 0.16 / 0.00) | 1.00 / 0.94 / 0.52 (0.16 / 0.00 / 0.00) | **1.00 / 1.00 / 0.27** (0.00 ×3) | 0.41 / 0.85 / 0.73 | 0.31 / 0.66 / 0.76 (0.65 / 0.98 / 0.77) | 0.20 / 0.42 / 0.66 |
| 81,920 | 0.23 / 0.62 / 0.70 (0.02 / 0.00 / 0.26 · 0.00 / 0.19 / 0.21) | 0.95 / 0.95 / 0.78 (0.39 / 0.04 / 0.36) | **0.99 / 0.90 / 0.49** (0.14 / 0.00 / 0.06) | 0.36 / 0.23 / 0.31 | 0.33 / 0.55 / 0.83 (0.18 / 0.80 / 0.77) | 0.39 / 0.46 / 0.72 |
| **122,880** | **0.00 / 0.88 / 0.94** (0.00 / 0.00 / 0.17 · 0.00 / 0.37 / 0.32) | 0.86 / 0.58 / 0.94 (0.00 / 0.01 / 0.82) | 0.96 / 0.98 / 0.66 (0.00 / 0.00 / 0.09) | 0.68 / 0.46 / 0.18 | 0.34 / 0.65 / 0.93 (0.86 / 0.94 / 0.64) | 0.49 / 0.54 / 0.76 |

Panels at the cap: planning return 1.5–2.2 (CM5b 0.3–0.5; the plan-shape term dominates the stream), planning EV 0.72–0.75, commitment entropy 0.46–0.59 nats, members' EV 0.81 / 0.82 / 0.92, displacement entropy 1.5–1.7. Ablation: s3 reads the mark (blank → 0.00), s2 does not (blank 0.90 ≈ trained 0.88), s1 is at zero in every column. Failure census: s1's failures are the committed squad standing on another objective (72%) or far away (23%) — the head's plan covers the board in 96% of episodes and the members walk to the nearest objective (arrival 0.34, at the nearest 0.40).

**Verdict on the letter, per seed at 122,880 as trained: NULL / AHEAD / AHEAD** (s2 0.88 and s3 0.94 clear CM4's same seed by 10 and 12 binomial SE; s3 is one hundredth short of PASS; s1's 0.00 is a members' failure under a covering plan)**. SPREADS: MET** (first-plan coverage 0.99 / 0.90 / 0.49 at 81,920 against ≥ 0.60 on 2/3; the plan-only row 0.95 / 0.95 / 0.78 against ≥ 0.50; CM5b 0.14 / 0.00 / 0.06 and 0.39 / 0.04 / 0.36).

### FH3 (#415) — the members under the rotated writer on the floored stream

| read | arrival at the committed objective on the rotated config (never the nearest) | at the nearest instead | under PL1's head on `a3_head_rf_pc`: split (FH2b) | arrival at a non-nearest committed objective under PL1 (FH2b) |
|---|---|---|---|---|
| 40,960 | 0.83 / 0.24 / 0.40 | 0.10 / 0.69 / 0.93 | 0.82 / 0.04 / 0.61 (0.49 / 0.43 / 0.66) | 0.88 / 0.35 / 0.46 (0.68 / 0.78 / 0.64) |
| 81,920 | 0.93 / 0.51 / 0.41 | 0.04 / 0.94 / 0.91 | **0.95 / 0.91 / 0.19** (0.62 / 0.81 / 0.50) | 0.82 / 0.54 / 0.50 (0.73 / 0.78 / 0.62) |
| **122,880** | **0.94 / 0.38 / 0.50** | 0.01 / 0.95 / 0.90 | **0.98 / 0.70 / 0.50** (0.84 / 0.79 / 0.46) | 0.92 / 0.43 / 0.68 (0.84 / 0.88 / 0.71) |

Comparators: the committed bar on the rotated config 0.96; LR1's members
(the same writer, the old reward) 0.89 / 0.96 / 0.46 at their cap; FH2b's
members at play on the rotated config 0.49 / 0.53 / 0.28. Executor EV 0.76 / 0.83 / 0.83 at the cap, displacement entropy 1.3–1.7 nats. s1 under PL1's head: 5.54 turns (the bar 5.28), pay differential +0.16 per step (the bar +0.20), mark-following 0.86; the plan-only ceiling under PL1 is 1.000.

**Verdict on the letter, per seed at 122,880: FAIL / FAIL / FAIL on the letter — one seed of three clears both marks (s1: arrival 0.94, success 0.94 on the rotated config), the other two walk to the nearest objective (0.95 / 0.90)** (arrival on
the rotated config ≥ 0.85 on 2/3; success ≥ 0.85 on 2/3).

### What the pair says

- **The plan-shape term is the planner's lever.** With `commitment_coverage` on the planning stream the head writes a covering assignment at the first opportunity on every seed (first-plan coverage 0.96 / 0.98 / 0.66 against CM5b's 0.00 / 0.00 / 0.09) and keeps it (persist 0.89–0.96, re-commits before arrival 0.18–0.68). The outcome-only stream never did that on any joint arm (CM3, CM4, CM5, CM5b). The term pays the planner for a property of its own decision, so constraint 2 holds, and the planning critic fits it (EV 0.72–0.75 against 0.09–0.45 before).
- **With a planner that plans and members that follow, the join works: 0.88 and 0.94 on two seeds of three**, the first joint arm ahead of A3 from scratch (0.70 / 0.80 / 0.77), and ahead of CM4's same seed by 10–12 binomial SE. s3 is a full join: the head covers first time (0.66), its members arrive at a non-nearest committed objective on 0.93 of squads and set out from a wrong one on 0.90 of moves, blank → 0.00. s2 reaches 0.88 by a different route — members who do not read the mark (ablation flat) under a head whose covering plan agrees with the geometry.
- **The remaining failure is the members', on three of six seeds.** CM6 s1's head plans perfectly (0.96) and its members walk to the nearest objective (arrival 0.34): 0.00. FH3 s2 and s3, trained under a writer that never agrees with the board, still go to the nearest (0.95 / 0.90 at the cap). The members that learn to read the mark do so early and completely (FH3 s1 at 0.83 by a third of the budget, 0.94 at the cap; CM6 s3 similar); the ones that do not learn the geometry first and never switch. Three seeds cannot say what separates the two; LR1 read 2 of 3 on the old reward, FH3 1 of 3 on this one, and both are consistent with a coin flip per seed.
- **FH3 s1 is the best executor on the ladder.** Under PL1's head it reads 0.98 (the ceiling with the bar's members is 1.000) at 5.54 turns against the bar's 5.28, arriving at a non-nearest committed objective on 0.92 of squads with a realised pay differential of +0.16 per step against the bar's +0.20 — the first learned members whose mark-following is within reach of the script's.

**Which decision rule fired.** CM6: "SPREADS and the members do not follow the spread plan → join the two: CM6's head with FH3's members" fired on s1 (and did not need to on s2 / s3). FH3: "fails on arrival while LR1 passed" fired on the letter, but LR1's 2 of 3 and FH3's 1 of 3 are not distinguishable at three seeds and s1 is ahead of LR1's best on every column; the per-step probe on LR1's checkpoints is not run.

### What this asks of #384 (proposed; Sash decides)

1. **Keep both**: the floored stream for the members and `commitment_coverage` 0.3 on the planner's stream (`a3_head_rf_pc.yaml`) are the reward from here. Neither has a measured downside on this shape.
2. **CM7, the join of the two working halves** (start axis): CM6's recipe with the members warm-started from FH3 s1's executor (the mark-reader) and the head from scratch, three seeds (the three seeds share the one executor — read them as that executor's band, not as seed variance). Prediction: as trained ≥ 0.90 on 3/3, PASS on at least one. If the warm start is overwritten (as A5i's was under guns), the join must be trained from scratch with the rotated writer as a first stage.
3. **Then the half-step** (six squads, five objectives) on `rf_pc`'s reward, from scratch and from CM7 — the shape the commitment layer was built for, where the geometry walk reads 0.03–0.33 and no per-model policy has passed.
4. **On the members' coin flip**: the seeds that read the mark do so by a third of the budget. Read every executor at 40,960 on the rotated config and restart the seed that walks to the nearest — cheaper than three full runs — until the mechanism that separates the two is found (a difference in the first few thousand rounds' plans is the place to look).

## Amendment 15 — written 2026-09-26 10:27: CM7, the join of the two working halves; and the half-step pre-registered (CM8 from scratch, CM8w from CM7)

Sash (2026-09-26): "run CM7 and the 40,960 executor reads now, pre-register
the half-step while they train, and launch the half-step when CM7's first
read is in."

### CM7 (#417) — CM6's recipe, the members warm-started from the plan-reader

`a3_head_rf_pc.yaml`, head and members trained together, the network
warm-started from FH3 s1's executor (`--warm-start-from
per-model-a3_legible_rf-2026-09-25-23-58-19-s1fh3/last.pt`, fresh
optimiser): the member heads carry the plan-reading walk (arrival at a
never-nearest committed objective 0.94 on the rotated config, 0.98 under
PL1's head); the commitment head in that checkpoint was never trained
(FH3 ran under the environment writer), so the planner starts from its
initialisation. Three seeds, 122,880 rounds, CM4's recipe; launched
10:22, Wandb dqlxji5x / fnszyqqk / 8a1rqszk. The three seeds share one
warm start and are read as that executor's band.

**Comparators by name:** CM6 at matched rounds (as trained 0.04 / 0.05 /
0.59 → 0.23 / 0.62 / 0.70 → 0.00 / 0.88 / 0.94; first-plan coverage
1.00 / 1.00 / 0.27 → 0.99 / 0.90 / 0.49 → 0.96 / 0.98 / 0.66; arrival at a
non-nearest committed objective 0.31 / 0.66 / 0.76 → 0.33 / 0.55 / 0.83 →
0.34 / 0.65 / 0.93), FH3 s1's executor at play (0.98 under PL1's head;
0.94 on the rotated config), CM4, A3 from scratch (0.700 / 0.800 / 0.770).

**Criteria, per seed at 122,880 as trained:** PASS success ≥ 0.95; AHEAD
of CM4's same seed by two binomial SE on 3/3; NULL otherwise.
**SURVIVES**, the clause this arm exists for, read at 40,960: the
members' arrival at the committed objective on the rotated config
(`a3_legible_rf.yaml`, the executor read) ≥ 0.80 on 3/3. A read of ≤ 0.5
means the new planner overwrote the warm start, as A5i's walk was
overwritten under guns, and the join must be trained from scratch with
the rotated writer as a first stage. **The restart rule** (Sash's
"restart the seed that walks to the nearest"): a seed whose executor
reads < 0.5 at 40,960 is stopped and restarted once with a new seed
(the head's initialisation and the rollouts change; the warm start does
not), recorded as such.

**Expectation (a guess).** SURVIVES on 3/3 (0.85–0.95); as trained
≥ 0.90 on 3/3 by the cap, PASS on at least one; first-plan coverage
≥ 0.9 by 40,960.

### The half-step — `a5_points_head_rf_pc.yaml`

`a5_points_cm.yaml`'s scenario (six squads of three over five
objectives, A5's deployment band, ten rounds) with `a3_head_rf_pc.yaml`'s
reward block and the policy writing the commitments. The shape the
commitment layer was built for: every per-model setting from scratch read
0.00–0.33 there, the geometry walk leaves one objective empty, and the
whole-army control reads 0.91 / 0.81 / 0.95.

**Desk check (no training, n=100 on seeds 700000+ for the bar, 30 for the
head).** The bar `squad_march_take` reads success 1.000 in 6.71 turns
with first-plan coverage 1.00 and re-commits before arrival 0.08; its
member pay is 0.1097 ± 0.0036 per completed commitment (2.0 / 18 =
0.1111: the near column's commitments start under the twelve-inch floor
on some layouts and pay a little less, the safe side; correlation with
the distance at commit +0.41 for that reason), per-step p99 +0.055, max
+0.056, min −0.004. CM6 s3's head loaded onto the half-step (a churning
learned writer on the new shape): per-step p99 +0.050, max +0.056, min
−0.042 — inside the band. ⚠ **And that head, which never saw this board,
plans it: plan-only 1.000 in 6.50 turns with first-plan coverage 0.98
and re-commits before arrival 0.17** (the bar 6.71 / 1.00 / 0.08), while
as trained it reads 0.05 because its A3 members do not transfer (0.15 of
bodies on objectives). The set network's planner carries across the
army size; the members are what the half-step will test.

| arm | the one change | tag | comparators by name |
|---|---|---|---|
| **CM8** (#418) | the half-step from scratch on `a5_points_head_rf_pc.yaml` | `cm8` | the per-model trainer from scratch on the half-step (0.030 / 0.060 / 0.330 at 122,880); the whole-army control (0.910 / 0.810 / 0.950); the bar (1.000, 6.71); CM8w at matched rounds |
| **CM8w** (#419) | the same warm-started from CM7's final checkpoint (the seed that reads best on CM7's letter), fresh optimiser | `cm8w` | CM8 at matched rounds; the per-model warm start from A4x on the half-step (0.46 / 0.26 / 0.35 at 122,880, 0.34 / 0.17 / 0.60 at 245,760); the control; the bar |

Three seeds each, 122,880 rounds, CM4's recipe, the ladder's once-only
extension to 245,760 if an arm is still rising at the cap. CM8 launches
when CM7's 40,960 read is in; CM8w when CM7's final is.

**Criteria, per seed at 122,880 as trained:** CM8 — PASS success ≥ 0.95;
AHEAD of the per-model scratch read (0.03 / 0.06 / 0.33) by two binomial
SE on 3/3; NULL otherwise. CM8w — PASS ≥ 0.95; AHEAD of CM8's same seed
by two binomial SE on 2/3 (the transfer clause); NULL otherwise. Readouts
on both: first-plan coverage (a covering plan claims all five), re-commits
before arrival, the members' arrival at the committed objective and
mark-following, held, the census, the planning panel and the members'
EV at 20,480 (a red panel there is the answer, as the reward-shape arms
taught), and on CM8w the members' arrival at 20,480 (does the walk survive
the change of board).

**Expectation (a guess).** CM8: the head plans the half-step by 40,960
(first-plan ≥ 0.8; the A3 head already does at play), the members lag as
CM6's did, as trained 0.3–0.7 at the cap, AHEAD on 3/3, PASS on none.
CM8w: ahead of CM8 at every read, 0.6–0.9 at the cap, PASS on one seed at
most; the transfer clause met.

## Amendment 16 — written 2026-09-26 13:09: CM7 read — AHEAD / PASS / AHEAD, the first PASS of a joint arm on this ladder: the join works when the soldiers already read the plan

CM7 (#417): CM6's recipe (`a3_head_rf_pc.yaml`, head and members
trained together), the network warm-started from FH3 s1's plan-reading
executor, the planner from its initialisation. Read at 40,960 / 81,920 /
122,880, n=100 on seeds 700000+, the final gated on the exit. Wandb
dqlxji5x / fnszyqqk / 8a1rqszk. Revision `c06bcfe`. The three seeds share
one warm start and are that executor's band.

| read | as trained (CM6 at the same read) | plan-only | first-plan coverage | re-commits before arrival | arrival at a non-nearest committed objective | executor read on the rotated config (SURVIVES ≥ 0.80) |
|---|---|---|---|---|---|---|
| 40,960 | 0.91 / 0.79 / 0.92 (0.04 / 0.05 / 0.59) | 0.98 / 1.00 / 1.00 | 0.81 / 0.78 / 0.97 | 0.13 / 0.14 / 0.03 | 0.88 / 0.98 / 0.88 | **0.84 / 0.92 / 0.88 — met on 3/3** |
| 81,920 | 0.65 / 0.88 / 0.91 (0.23 / 0.62 / 0.70) | 1.00 / 1.00 / 0.96 | 0.99 / 0.89 / 0.59 | 0.33 / 0.20 / 0.69 | 0.87 / 0.94 / 0.94 | 0.62 / 0.81 / 0.90 |
| **122,880** | **0.92 / 0.99 / 0.80** (0.00 / 0.88 / 0.94) | 1.00 / 1.00 / 1.00 | 0.93 / 0.90 / 0.97 | 0.11 / 0.04 / 0.05 | **0.97 / 0.98 / 0.96** | 0.86 / 0.87 / 0.87 |

At the cap: held 3.92 / 3.99 / 3.74 in 7.55 / 6.93 / 7.12 turns (the bar
5.28); the ablation STEERS on every seed (blank → 0.00 / 0.00 / 0.00,
misdirect 0.14 / 0.00 / 0.02); s3's failures are squads that arrived and
left (81%), s1's a squad standing on another objective (88%); the walk-off
probe pays a move that keeps the objective +0.021 to +0.031 against
+0.009 to +0.020 for one that leaves it. Panels: members' EV **0.91 / 0.97
/ 0.91**, planning EV 0.59 / 0.82 / 0.54, commitment entropy 0.39 / 0.20 /
0.47 nats, planning return 2.4–2.9 (CM6 1.5–2.2).

**Verdict on the letter, per seed at 122,880 as trained: AHEAD / PASS /
AHEAD.** s2 at 0.99 is the first PASS of a joint arm on this ladder and
the first per-model policy with a learned planner to reach the bar's
success on A3's shape; s1 and s3 clear CM4's same seed (0.00 / 0.32) by
more than ten binomial SE. SURVIVES was met on 3/3 at 40,960; the restart
rule never fired.

### What it says

- **The join works when the soldiers already read the plan.** The same
  recipe from scratch (CM6) read 0.00 / 0.88 / 0.94 with one seed's
  members walking to the nearest objective under a perfect plan; started
  from members who read the mark, it reads 0.92 / 0.99 / 0.80 with the
  members arriving at a non-nearest committed objective on 0.96–0.98 of
  squads and reading the mark on every seed. The planner learned on top
  of them from its initialisation, first-plan coverage 0.78–0.97 by a
  third of the budget.
- **The plan-reading walk survives a planner trained on top of it, with
  one wobble.** s1's rotated read fell 0.84 → 0.62 at 81,920 (its success
  0.91 → 0.65) while its planner churned (re-commits before arrival 0.13
  → 0.33), and recovered to 0.86 / 0.92 by the cap as the planner settled
  (0.11). The co-adapting planner can pull the members off the mark
  mid-run; it did not keep them there.
- **The residual is speed and the walk-off, not allocation**: 6.9–7.6
  turns against the bar's 5.3, and on s3 the squads that fail are ones
  that arrived and left. The stay term's margin (+0.02 to +0.03 for
  keeping against +0.01 to +0.02 for leaving) is small; a larger stay
  weight is the obvious knob and was not run.

### What this asks of #384 (proposed; Sash decides)

1. **A3's join is closed on this recipe**: `a3_head_rf_pc.yaml`'s reward
   with the members started from a plan-reader. Record it as the ladder's
   first joint PASS and move the question to the half-step, where CM8
   (from scratch) and CM8w (from CM7 s2, the PASS seed) are running.
2. The seed lottery on the members' side stands as the open mechanism;
   CM7's cost was one extra run (FH3) to obtain a plan-reader. A
   two-stage recipe — the rotated writer first, the head second — is the
   procedure until the lottery is understood.
3. If the half-step's members chase the head's churn (CM8's first read:
   86–100% re-committed before arrival, the members arriving and leaving),
   the lever is on the planner's stability, not the members: a KEEP bias
   or a churn cost on the planning stream, pre-registered as its own arm.
