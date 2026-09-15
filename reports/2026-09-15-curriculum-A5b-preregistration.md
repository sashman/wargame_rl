# Pre-registration: curriculum rung A5, second arm (A5b) — the per-model arm read beside the control, not gated by it

Written 2026-09-15, **before any per-model number on this rung exists**,
on branch `feature/curriculum-a5` (PR #354 carries the rung's arms).
Issue #355, parent question #340, rung **A5** — a follow-up to #353,
which read **NULL (scenario)** as pre-registered when the whole-army
control missed the bound on 2 of 3 seeds at its extended budget
([report](2026-09-15-curriculum-a5-control-cannot-allocate.md)).

## The one change

**The attribution clause.** The scenario, the config
(`configs/experiments/curriculum/a5.yaml`, unchanged at `297b252`), the
bar, the seeds and the per-model regime are A5's. What changes is how the
per-model read is attributed: the whole-army control calibrates the
budget and is read *beside* the per-model arm as the comparison it is;
NULL (scenario) requires the **script** to fail its own criterion. A
control that cannot do what the script can is a finding about the
control, and this rung is where the ladder first meets one.

## Comparator

`squad_march_take`, both facades, seeds 700000+ at n=100: success
**1.000**, `held` 6.00, turns 6.77 (sd 0.68), `on_obj` 0.856, coherent
0.822. Beside it, the whole-army control at 120 epochs (245,760 rounds):
success **0.800 / 0.940 / 0.980**, `held` 5.76 / 5.93 / 5.98, `on_obj`
0.93 / 0.93 / 0.96, turns 8.32 / 7.96 / 7.21, coherent 0.62 / 0.57 / 0.47.

## Criteria

Read at the END of the budget from `last.pt`, greedy, no decode, n=100
on seeds 700000+ (`just measure-rung`).

- **PASS:** success (every objective occupied at the end) ≥ 0.95 on
  **all three** seeds. Reported beside the control's 1 of 3: the
  per-model arm allocates eight squads over six points within 122,880
  rounds where the whole-army trainer does not within 245,760.
- **FAIL:** any seed below 0.95 at budget. Read beside the control's
  shape: if the per-model arm fails the same way — `held` below 6 with
  `on_obj` above 0.9, one point empty while bodies stack on another —
  neither trainer solves allocation at this shape and per-step credit did
  not buy it here; if it fails differently, the report says how.
- **NULL (scenario):** only if the script fails its own criterion. It
  does not.
- **Pass with a defect:** the bound holds but the health panel is red
  over the last quarter (the ratio line at 1.6–1.9 is this regime's
  normal).

**Readouts, not criteria:** turns paired against the script and the
control; `held` and `on_obj`; coherency greedy and sampled; per-model
decisions per round.

Power: binomial SE 0.022 at p=0.95, n=100. The control's best seed sits
0.98 at n=100 and its worst 0.80 — nine SE apart — so a 3/3 per-model
pass is a move, not noise, and so is a 0/3.

## Budget, regime, seeds

| | per-model arm |
|---|---|
| budget | **122,880 rounds** (the cap; 6× anything the control needs exceeds it), 128 rounds per update (`--rollout-rounds 32 --num-rollout-envs 4`) |
| seeds | 1, 2, 3; rollout layouts at seed×100+ |
| in-run eval | every 512 rounds, n=30, seeds 500000+, greedy, passive pair and `eval/mean_turns` logged |
| final score | n=100, seeds 700000+ |
| other flags | defaults (`gamma` 0.9, `ent_coef` 0.03, `lr` 3e-4) |
| logging | Wandb group `curriculum-a5`, one run per seed |
| start | from scratch (T1 is the warm-started pair) |
| order | launches after A4's per-model arm (#351) has been read — the ladder's order holds |

Launch: `just train-per-model-arm 122880 3 curriculum-a5 a5 "--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512 --checkpoint-every-rounds 512 --n-eval-episodes 30" configs/experiments/curriculum/a5.yaml`.

## What I expect (a guess, written so it can be wrong)

The per-model arm passes on 2 or 3 seeds and needs most of the cap to do
it. Its `held` at the end is higher than the control's on every seed —
per-step credit against a shared coverage term is the mechanism #283
argued for, and this is the first rung that tests it. Coherency, unpaid,
ends below the control's.

## Amendment 1 — written 2026-09-15 before the launch: the budget and the entropy coefficient

Two rows in the table above contradict rules the ladder had already
adopted when this was written, and both are corrected here before any
number exists.

- **Budget: 245,760 rounds, read once at the end**, not 122,880. The
  cap rule is symmetric (A3x): the control on this rung ran its once-only
  extension to 120 epochs, 245,760 rounds, so the arm under test gets the
  same rounds before the rung is called — and A4x (read this morning:
  0.980 / 1.000 / 1.000 at 245,760 where 122,880 read 0.93 / 0.93 / 0.97)
  confirmed that when 6× the control exceeds the cap the per-model budget
  is 2× the cap. Reading A5b at 122,880 would repeat the asymmetry A3's
  FAIL measured.
- **`--ent-coef 0.003`**, not the default 0.03. Every per-model arm runs
  at 0.003 from A3 on (A2b: at 0.03 the arm solves a rung and then
  unlearns it). The "other flags" row above was carried from A5's
  control-side table.

Launch, as amended: `just train-per-model-arm 245760 3 curriculum-a5 a5b
"--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512
--checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003"
configs/experiments/curriculum/a5.yaml`, on the code of PR #362's tip
(`bd15db9`: the A5 branch plus the travel term's matching flag, which
`a5.yaml` does not set), beside the nine A3 speed-screen runs already on
the GPU. The criteria are unchanged; the in-run curve is read for drift
after any first pass, as the A2b clause requires.
