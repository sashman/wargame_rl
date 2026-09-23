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
