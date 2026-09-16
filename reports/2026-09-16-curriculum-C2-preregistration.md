# Pre-registration: curriculum rung C2 — the enemy squad on the point fires at whoever approaches

Written 2026-09-16, **before any training number on this rung exists**,
on branch `feature/curriculum-c2` (stacked on PR #366). Issue #368,
parent question #340, rung **C2** — the "guns" axis, their side first.

## The scenario

`configs/experiments/curriculum/c2.yaml` is `c1.yaml` — four unarmed
squads of three, four points in a column at x=35, radius 4, eight
rounds, one enemy unit of three standing on the point at (35, 27) —
with the enemy unit **armed and firing**: a rifle of range 12 and one
attack per model, the `hold_and_shoot` seat (holds its ground, fires at
the nearest valid unit under the env's own mask each shooting phase),
and the shooting phase **stepped** (`skip_phases: [command, charge,
fight]`; every rung below stepped movement only, and the default skips
shooting whatever the weapons say). Success is still
`all_objectives_occupied` over our bodies; the reward is C1's, so nothing
pays for a casualty avoided or taken.

**The build the rung needed, shipped with it.** `hold_deployment` shoots
at nothing on purpose; `hold_and_shoot` is that seat with
`squad_march_shoot`'s target rule, registered under its own name and
pinned by `tests/test_hold_and_shoot.py` (the blockers never move;
approaching squads lose bodies).

## Why only their guns — the design change from #340's letter

#340's C2 row reads "C1 with every unit armed, opponent shoots back".
Measured before this file was written, on that scenario: our twelve
rifles wipe the three blockers in two shooting phases, and **the phase
facade ends the battle on an opponent wipe** (#317 — against the rules,
which the per-model facade follows: a wiped army keeps ceding ground
until the battle ends). The scripted bar therefore reads **0.380** on the
phase facade and **1.000** on the per-model facade on the same seeds
(`just measure-bridge`: "BRIDGE DIVERGES"), with `alive` 0.95 on both —
the guns kill almost nobody of ours; the wipe rule ends the game before
the last squad arrives. No whole-army control could be read beside a
per-model arm on that scenario. So this rung arms **their** side only,
and our guns become C3's axis, where they are one squad's and the
blocker has to die before the unarmed squads arrive.

⚠ **Standing consequence for the ladder, to be written into CLAUDE.md
when this rung lands:** on any rung where the opponent can be wiped, the
whole-army control's episodes end early and its numbers are not
comparable to the per-model arm's until #317 is fixed. The E rungs are
scored on the phase facade's own evaluation family, so the control there
is compared against itself; the C and D rungs avoid wipes by design.

## The bar, measured first

`squad_march_take` (unarmed here, so it cannot fire), seeds 700000+ at
n=100, **both facades identical on every shared field**: success
**0.900**, `held` 2.92, `alive` **0.842**, `on_obj` 0.909, turns 9.60
phase-clock (two stepped phases per round, so about 4.8 rounds), vp −2.7
± 0.8, coherent 0.932. The script loses about two bodies of twelve on the
way and misses a point in one episode of ten. That is the pass mark's
own level: #340's C-rung mark is 0.90, and the bar sits on it.

## The two arms

| | whole-army control | per-model arm |
|---|---|---|
| trainer | `train.py` via `just train-curriculum-control` | `train_per_model.py` |
| budget | 60 epochs of 2048 steps, extended once to 120 if still rising | runs to **122,880 rounds** at 128 per update; read at the checkpoint the control's budget names (6× its slowest rounds-to-pass, floor 20,480, cap 122,880; 2× the cap, read once, when 6× exceeds it); resumed in place if the 2×-cap rule applies (#346: cadences passed explicitly) |
| start | from scratch | `--warm-start-from checkpoints/per_model/per-model-c1-2026-09-16-12-43-32-s{1,2,3}c1/last.pt` (C1 at 245,760: 0.990 / 0.930 / 0.990), seed for seed |
| seeds | 1, 2, 3 | 1, 2, 3 |
| flags | defaults (`ent_coef` 0.03) | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003` |
| in-run eval | the trainer's own, seeds 500000+ | every 512 rounds, n=30, 500000+ |
| logging | Wandb `curriculum-c2`, tag `c2-ctl` | Wandb `curriculum-c2`, tag `c2` |

Both arms launch together, beside T1's three runs and A5b's last
minutes. If the control fails its own criterion at 120 epochs, that is a
finding about the control (A5's clause) and the per-model arm is read at
245,760 beside it.

## Criteria

Read greedy, no decode, n=100 on 700000+.

- **PASS:** success ≥ **0.90** on **all three** seeds at the budget, and
  no in-run dip below 0.75 after the first rolling pass (the drift
  clause, set one notch under the mark as A2b's was).
- **FAIL:** any seed below 0.90 at the budget. The next step is the
  census: which point is short, and whether the bound squad dies on the
  way (the enemy's fire) or ends beside the disc (allocation, C1's
  residual).
- **NULL (scenario):** only if the script fails its own criterion. It
  reads 0.900, on the mark.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts, not criteria:** `alive` against the bar's 0.842 (#340's
  "alive ≥ bar − 0.1" is a readout: the reward pays nothing for a body);
  vp paired against the script; `held`; turns against the script's 9.60
  phase-clock and the control's; `on_obj`; coherency greedy and sampled;
  rounds to rolling 50 / 80 / 90%; the panel over the last quarter; the
  control beside every row.

Power: binomial SE 0.030 at p=0.90, n=100. The bar sits on the bound,
so a seed at 0.87 is one SE under it and still a fail on the letter;
this rung will be read with that in mind and the verdict will not be
softened for it.

## What I expect (a guess, written so it can be wrong)

The per-model arm passes on all three seeds inside 40k rounds at about
the bar's `alive` (0.80–0.86): the warm start carries C1's approach and
the enemy's fire costs a body or two, not the point. It may learn to
hang the bound squad back a turn and arrive later, which the success
criterion does not punish. The control passes at 60 epochs on two
seeds and needs its extension for the third, as on C1. If instead the
arm reads under 0.90 on a seed, the census says the bound squad is dead
or short on the enemy's point, not the neighbours — the opposite of C1.

## Amendment 1 — written 2026-09-16 22:50, after the control's 60-epoch read and before any per-model number

**The control at 60 epochs**, n=100 on 700000+, `last.ckpt`: success
**0.710 / 0.690 / 0.760**, `held` 2.73–2.76, turns 13.8–14.5 phase-clock
(about seven rounds of eight — it arrives with the game nearly over)
against the script's 9.60, `on_obj` 0.93–0.97, coherent 0.79–0.81, vp
−8.1 to −11.0 against the script's −2.7. In-run (n=30, rolling five):
80% at epoch never / never / 33; 90% never; the last ten evaluations
0.57–0.83 on all three. Wandb `curriculum-c2`: `pqkc7sp3` / `conri3cb`
/ `ews7lbi8`.

**The extension.** Under the mark on every seed with the curves still
moving, the control gets its once-only extension to **120 epochs**
(`--resume-ckpt-path` from `last.ckpt`, launched 22:46, suffix
`-ctl-x2`). By the symmetric-cap rule the per-model arm's budget is
**245,760 rounds**, read once at the end: the arm running to 122,880
will be resumed in place when it gets there (#346: cadences explicit),
and its 122,880 checkpoint is a readout. If the control fails its own
criterion at 120, that is a finding about the control (A5's clause) and
the arm is read at 245,760 beside it. The criteria are unchanged.
