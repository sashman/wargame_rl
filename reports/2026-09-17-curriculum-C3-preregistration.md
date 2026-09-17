# Pre-registration: curriculum rung C3 — one of our squads is armed, and the blockers have to die before the unarmed squads walk in

Written 2026-09-17 04:10, **before any training number on this rung
exists**, on branch `feature/curriculum-c3` (stacked on PR #369). Issue
#370, parent question #340, rung **C3** — the "guns" axis, our side,
mixed: the first rung that asks for a plan with an order to it.

## The scenario

`configs/experiments/curriculum/c3.yaml` is `c2.yaml` — four squads of
three, four points in a column at x=35, radius 4, one enemy unit of three
standing on the point at (35, 27) and firing — with three changes that
are one axis: **squad 0 carries a rifle that outranges the blockers**
(range 18, four attacks; squads 1–3 stay unarmed), **the blockers are
tough and lethal inside their reach** (four wounds each, eight attacks
each at range 12), and the game runs **twelve rounds** (24 phase-clock
turns) because the plan takes about six rounds and the walk after it
three. Success is still `all_objectives_occupied` over our bodies; the
reward is C2's; nothing pays for a kill or a body.

**Why those numbers.** #340's letter: "a plain `squad_march_take` must
FAIL this rung, or it is not testing sequencing." With the blockers as
C2 had them (one wound, one attack) plain `take` PASSES at 0.95–1.00
whatever our rifle: its armed squad shoots on the march and outranges
the blockers, so it clears the point just before its unarmed squads walk
in — sequencing by geometry, no plan needed. A sweep at n=20 on the
per-model facade (2026-09-17, twenty variants of wounds × attacks ×
range × rounds, `scratchpad/c3sweep/`) found the blockers have to
outlast that incidental fire: at four wounds and eight attacks `take`
reads 0.550 with 36% of its bodies dead and the blockers still alive in
two thirds of episodes, while the escort reads 1.000. That variant is
the config.

**The two builds the rung is first to need, shipped with it.**
`scripted_escort` (#330): every armed squad steers as one body at the
nearest live enemy unit and stops just inside its own reach; every
unarmed squad walks its `squad_march_take` vector but stops just outside
the enemy's reach and waits; once the last enemy is dead, or with no
armed squad at all, everyone plays `squad_march_take` unchanged
(`tests/test_scripted_escort.py`: no unarmed body inside the blockers'
reach while one lives, every blocker killed, and bit-identical to `take`
on C2). And the #317 fix: the phase facade ended the battle on an
opponent wipe unconditionally, against the rules the per-model facade
follows; it now does so only under `terminate_on_opponent_elimination`
(default `False`, the mirror of the player-side switch), pinned by
`tests/test_opponent_wipe_termination.py`. Without it no whole-army
control could be read on a rung whose point is the wipe.

## The bar, measured first

Seeds 700000+ at n=100, **both facades identical on every shared field**:

| policy | success | turns | held | alive | on obj | vp | coherent | first kill → blockers wiped → first unarmed body on their point (phase-clock turns); kill precedes arrival |
|---|---|---|---|---|---|---|---|---|
| **`scripted_escort`** (the bar) | **0.990** | 15.25 | 3.97 | **0.963** | 0.915 | +5.3 ± 1.1 | 0.855 | 6.0 → 11.0 (99%) → 12.9 (90%); **1.00** |
| `squad_march_take` (must fail) | **0.590** | 15.25 | 2.66 | 0.615 | 0.957 | +10.1 ± 3.0 | 0.937 | 6.7 → 12.5 (31%) → 8.9 (50%); 0.77 |

The escort loses one body in twenty; `take` loses four or five of
twelve, its armed squad walks into reach and dies, and its unarmed squad
reaches the blockers' point in half the episodes and is shot off it.

## The arms

| | whole-army control | per-model arm | per-model from scratch (companion readout) |
|---|---|---|---|
| trainer | `train.py` via `just train-curriculum-control` | `train_per_model.py` | `train_per_model.py` |
| budget | 60 epochs of 2048 steps, extended once to 120 if still rising | 122,880 rounds at 128 per update, read at the checkpoint the control's budget names (6× its slowest rounds-to-pass, floor 20,480, cap 122,880; 2× the cap, read once, when 6× exceeds it; resumed in place, cadences and `--n-eval-episodes` explicit, #346) | the same |
| start | from scratch | `--warm-start-from checkpoints/per_model/per-model-c2-2026-09-16-22-18-5{9,9,8}-s{1,2,3}c2/last.pt` (C2 at 245,760: 0.890 / 0.920 / 0.950), seed for seed — the C rungs' rule, which stands on S3 and C1 | from scratch — #340's consequence of T1's FAIL ("later rungs train from scratch"), run beside the warm-started arm so the rung reads both and the rule is measured on a rung with guns |
| seeds | 1, 2, 3 | 1, 2, 3 | 1, 2, 3 |
| flags | defaults (`ent_coef` 0.03) | `--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003` | the same, no warm start |
| in-run eval | the trainer's own, seeds 500000+ | every 512 rounds, n=30, 500000+ | the same |
| logging | Wandb `curriculum-c3`, tag `c3-ctl` | Wandb `curriculum-c3`, tag `c3` | Wandb `curriculum-c3`, tag `c3s` |

Nine trainers on the GPU, launched together; no CPU read runs beside
them until the first exits.

## Criteria

Read greedy, no decode, n=100 on 700000+, from `last.pt` at the budget.

- **PASS:** success ≥ **0.90** on **all three** seeds of the warm-started
  arm, **and** the first kill precedes the first unarmed arrival on the
  blockers' point in ≥ **0.90** of episodes on every seed (#340's C3
  clause; measured by the sequencing census, `scratchpad/escort_census.py`,
  at n=100), and no in-run dip below 0.75 after the first rolling pass.
- **FAIL:** any seed below 0.90 at the budget, or the ordering clause
  missed on any seed. The census says which: dead unarmed squads (walked
  in early), a live blocker (the armed squad never closed or never fired),
  or C1's spread residual.
- **NULL (scenario):** only if the script fails its own criterion. It
  reads 0.990 / ordering 1.00.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter.
- **Readouts, not criteria:** `alive` (bar 0.963; `take` 0.615); vp
  paired; `held`; turns against the bar's 15.25 and the control's;
  `on_obj`; coherency greedy and sampled; rounds to rolling 50 / 80 /
  90%; the panel; the control beside every row; **the from-scratch
  companion beside every row** (its own success at n=100 and its in-run
  curve — if it reads within one SE of the warm-started arm on every
  seed, the C-rung warm-start rule bought nothing here and is reported
  as such; if the warm-started arm is ahead 3/3, the rule stands with a
  guns rung behind it).

Power: binomial SE 0.030 at p=0.90, n=100. The ordering clause is a
binomial at n=100 too, SE 0.030 at 0.90.

## What I expect (a guess, written so it can be wrong)

The warm-started arm is at 80% within 10k rounds on the movement half
(it inherits C2's approach) and passes on two seeds by 122,880 with the
third under the mark, because the shooting decision is new and the
unarmed squads' waiting is the thing per-step credit has never had to
learn: nothing in the reward pays for standing outside a circle. The
from-scratch companion reads under 0.5 at 122,880. The control passes at
60 epochs on no seed and at 120 on one or two: it walks everyone in, as
`take` does, and its bodies die.

## Amendment 1 — written 2026-09-17 04:25, after the control's 60-epoch read and before any per-model number

**The control at 60 epochs** (eleven minutes: the scenario is small),
n=100 on 700000+, `last.ckpt`: success **0.790 / 0.560 / 0.870**, turns
19.15 / 19.34 / 18.15 phase-clock of 24 (the bar's 15.25: it arrives in
round nine or ten), `held` 3.56 / 2.66 / 3.61, `on_obj` 0.96 / 0.77 /
0.92, coherent 0.73–0.80, vp +12.9 / +9.6 / +9.4 against the bar's +5.3
(it kills more and holds less). In-run (n=30, rolling five): 80% at
epoch 35 / never / 56, 90% at 37 / never / never; the last ten
evaluations 0.57–0.90. Wandb `curriculum-c3`: `xt95et6x` / `b00ym3pi` /
`a9ubzqa7`.

**The extension.** Under the mark on every seed with the curves still
moving, the control gets its once-only extension to **120 epochs**
(`--resume-ckpt-path` from `last.ckpt`, launched 04:22, suffix
`-ctl-x2`). By the symmetric-cap rule the per-model arm's budget — and
the from-scratch companion's — is **245,760 rounds**, read once at the
end: both are resumed in place at 122,880 (#346: cadences and
`--n-eval-episodes` explicit) and the 122,880 checkpoint is a readout.
The criteria are unchanged.

## Amendment 2 — written 2026-09-17 04:36, after the control's 120-epoch read and before any per-model number

**The control at 120 epochs**, n=100 on 700000+, `last.ckpt` of the
`-ctl-x2` runs: success **0.870 / 0.830 / 0.840**, turns 18.27 / 17.31 /
17.66 of 24 (the bar's 15.25), `held` 3.61 / 3.29 / 3.48, `on_obj`
0.94–0.95, coherent 0.74–0.79, vp +6.2 / +7.0 / +6.9 against the bar's
+5.3. In-run (n=30, rolling five): 80% at epoch 64 / 96 / 64, 90% at
88 / never / never; the last ten evaluations 0.67–0.93. Wandb
`curriculum-c3`: `bpa07dn5` / `do7u35wa` / `ib9ymhgv`.

**What this is.** The whole-army control fails C3's own criterion on
all three seeds at its extended budget, on a rung the script passes at
0.990 — A5's clause, as on C2: a finding about the control, not a NULL.
It gets closer than it did on C2 (0.79 / 0.69 / 0.78 there) and it is
a round or more behind the escort; whether its bodies walk in early and
die, or the armed squad never clears the point, is the control's census,
a readout for the report. **The per-model arm and its companion are
read at 245,760 rounds beside this row, criteria unchanged.**
