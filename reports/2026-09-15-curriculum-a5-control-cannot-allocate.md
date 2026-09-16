# Curriculum rung A5: the whole-army control cannot put a squad on every point — NULL (scenario) as pre-registered, and a defect in the clause

**Verdict first.** On the last movement-only rung of the ladder (#340, arm
#353) — eight squads of three, six objectives, ten rounds, success only
when every point has a body on it — the whole-army control reads success
**0.890 / 0.820 / 0.600** at its 60-epoch cap and **0.800 / 0.940 /
0.980** after the pre-registered extension to 120 epochs (245,760
rounds). Two of three seeds miss the bound at the extended budget, which
the pre-registration's letter calls **NULL (scenario)**, and that verdict
is recorded. The per-model arm was not launched under it. But the
scenario is not misconfigured: the script scores **1.000** on it, and the
control's failure has a shape — `held` 5.76 / 5.93 / 5.98 of 6 with
`on_obj` 0.93 / 0.93 / 0.96, i.e. **twenty-two of twenty-four bodies on
points and one point empty** in the failing episodes — that is the
allocation failure the record already attributes to the whole-army
trainer at this exact shape (`CLAUDE.md` § Allocation, § Holding pays —
"the agent stacks"). A NULL clause that fires when the control cannot do
what the script can is measuring the control, and the ladder was built
to measure the per-model arm against it. A second arm on this rung,
A5b, is pre-registered with the clause corrected: the control gates the
budget, the script gates the scenario, and the per-model arm is read
against both.

## Provenance

| field | value |
|---|---|
| date | 2026-09-15 (control 01:26–01:38; extension 01:39–01:50) |
| GPU / no-GPU | GPU (RTX 4090), alongside the A2 per-model arm |
| seeds | 1 / 2 / 3, extended in place with `--resume-ckpt-path` |
| n | 100 at seed base 700000 (final); 30 at 500000 (in-run) |
| config | `configs/experiments/curriculum/a5.yaml` — unrefereed by design |
| decode | K=1, undecoded |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take`: success 1.000, held 6.00, turns 6.77, coherent 0.822 |
| opponent | none |
| budget | 60 epochs of 2048 steps, extended once to 120 |
| code revision | `5845e1c` on `feature/curriculum-a5` (PR #354) |
| checkpoint | `last.ckpt` at 60 and at 120 |
| coherency | 0.58 / 0.64 / 0.58 at 60; 0.62 / 0.57 / 0.47 at 120 |
| Wandb | `curriculum-a5`: `koti5sv5` / `jemeom83` / `49n4y1l5` (to 60), `5wh7plaj` / `gmonofrt` / `jlnshwd0` (60–120) |
| pre-registration | `reports/2026-09-15-curriculum-A5-preregistration.md` at `297b252`; amendment 1 (the cap read and the extension) at `5845e1c` |

## The read

| row | success | turns | vs bar | held | on obj | coherent |
|---|---|---|---|---|---|---|
| `squad_march_take` | 1.000 | 6.77 | — | 6.00 | 0.856 | 0.822 |
| control s1 at 60 | 0.890 | 8.31 | +1.54 | 5.88 | 0.934 | 0.576 |
| control s2 at 60 | 0.820 | 8.50 | +1.73 | 5.81 | 0.922 | 0.637 |
| control s3 at 60 | 0.600 | 8.79 | +2.02 | 5.58 | 0.945 | 0.577 |
| **control s1 at 120** | **0.800** | 8.32 | +1.55 ± 0.12 | 5.76 | 0.927 | 0.619 |
| **control s2 at 120** | **0.940** | 7.96 | +1.19 ± 0.10 | 5.93 | 0.930 | 0.570 |
| **control s3 at 120** | **0.980** | 7.21 | +0.44 ± 0.09 | 5.98 | 0.964 | 0.474 |

The in-run curves and rounds-to-pass are read from Wandb once the A2
per-model arm has exited and are appended to #353. Seed 1 moved
*backwards* between 60 and 120 (0.890 → 0.800) while seeds 2 and 3 moved
up (0.820 → 0.940, 0.600 → 0.980); at n=100 the SE is 0.02–0.05, so the
first is a real move and the three seeds are not converging on one
answer.

## What it says

- **The verdict is NULL (scenario) on the letter, and the letter is
  wrong here.** The clause was written for a rung nobody can solve, and
  the script solves this one every time. What failed is the pipeline
  control, on the shape the record already says it fails on: more bodies
  than points, and a policy that stacks. A5 is the first rung where the
  control does not simply learn the task slower than the script; it
  does not learn it at all in 245k rounds, and the way it fails is the
  full game's known failure in miniature.
- **The control's failure is a finding, not a null.** Every earlier rung
  read the control as a yardstick. Here the yardstick is the thing the
  ladder was built to improve on. A per-model arm that passes A5 would be
  the first evidence for #283's premise — per-step credit against a
  shared allocation problem — and a per-model arm that fails the same
  way would say the architecture does not buy it. The letter of the
  pre-registration forbids launching it; the point of the ladder demands
  it. The resolution is a second arm with the clause corrected before it
  runs, not a reading of the first arm against a clause it did not sign.
- **Coherency collapsed with the extension** (0.58–0.64 → 0.47–0.62),
  with nothing paying for it and twenty-four bodies squeezing onto six
  radius-4 discs. It is a readout on this rung; the E rungs' referee will
  price it.
- **The clause moves.** From A5b on: the whole-army control calibrates
  the budget (6× its slowest pass, or the cap if it never passes) and is
  read beside the per-model arm as the comparison it is; NULL (scenario)
  requires the **script** to fail its own criterion; a control that fails
  while the per-model arm passes is reported as exactly that. This goes
  in #340 as an amendment to its contract, and in `CLAUDE.md`.

## What was not done

- The per-model arm did not run under this arm.
- The control was not extended past 120; the once-only extension was
  the pre-registered limit.
- No recording of which point the control leaves empty, or whether it is
  the same point across episodes.
