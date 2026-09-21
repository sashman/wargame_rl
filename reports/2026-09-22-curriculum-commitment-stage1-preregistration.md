# Pre-registration: the commitment head on A3's shape (CM3, #398) — Stage 1 of the commitment layer

**Written 2026-09-22 01:33, before any training number exists.** Parent
#384 (D1–D9 decided 2026-09-20; Stage 1 opened on Sash's "proceed as
recommended" 2026-09-22 after the legibility rung read); build #397; arm
#398. Branch `feature/commitment-revision` (PR #396, stacked on #388);
never merged to `main`.

## The question

The legibility rung ([the revision's pre-registration, amendment 1](2026-09-21-curriculum-commitment-revision-preregistration.md))
showed two seeds of three learning to READ a commitment when reading it is
the only way to win. Stage 1 asks whether the network can CHOOSE it: at each
squad's first open of the turn the set network's commitment head draws KEEP
or an objective, the environment writes it, the members are paid the travel
potential keyed to that objective alone (the execution stream), and the
commitment decision is paid the close's outcome terms — coverage and the
rung's success bonus — on a planning stream with a semi-Markov return over
the squad's own commitment steps and its own value head (D1 with K = 1, D2
pure, D4 no switch price, D6 semi-Markov, D8 no supervision). On A3's shape
the head must learn an allocation (four squads to four distinct objectives)
and the members must follow it.

## The one change

`a3_head.yaml` = `a3.yaml` with `commitments.assignment: head`; success
A3's own `all_objectives_occupied`. Three seeds (1 / 2 / 3) from scratch,
122,880 rounds at 128 rounds per update (`--num-rollout-envs 4
--rollout-rounds 32`), `--ent-coef 0.003`, `--planning-gamma 0.99`,
eval and checkpoint every 512, a greedy episode recorded at every
checkpoint, Wandb group `curriculum-cm-s1`. Scored greedy with the passive
fingerprint, n=100 on seeds 700000+, sampled beside greedy at the end.
Every read gated on each seed's log carrying its round line, the final on
zero trainers; run directories resolved from the config stem.

## Comparators, fixed by name

| | 40,960 | 81,920 | 122,880 |
|---|---|---|---|
| A3's own runs (`a3.yaml`, no layer) | 0.400 / 0.350 / 0.340 | 0.590 / 0.670 / 0.610 | 0.700 / 0.800 / 0.770 |
| `a3_cm` (the environment's greedy assignment given) | 0.520 / 0.350 / 0.240 | 0.890 / 0.650 / 0.750 | 0.820 / 0.850 / 0.930 |

The bar `squad_march_take`, which writes its own greedy plan into the
state under `head`, measured first on the per-model facade (n=100, seeds
700000+): **success 1.000 in 5.28 turns**, held 4.00, coherent 0.927;
persist 0.99 (1712), claim 1.00 / max 1, complete 1.00, follow 1.00
(6000), leave 0.00 (263).

## Criteria — per seed, as the revision's reading asked

At 122,880, per seed:

- **PASS** — success ≥ 0.95.
- **AHEAD** — not PASS, and ahead of A3's own seed by more than two
  binomial SE (at n=100 a difference of 0.13 clears it near 0.75).
- **NULL** — otherwise.

And per seed, the head's legibility: **STEERS** if the flag ablation
(`measure-commitment-ablation`, trained / blank / misdirect / nearest)
drops success by more than 0.20 under BLANK — the members follow the head's
pointer rather than the geometry. The arm's verdict is the count, written
as PASS / AHEAD / NULL with the STEERS count beside it. Readouts before the
verdict, on every read: the commitment readouts (persist, claimants — does
the head allocate DISTINCT objectives?, complete, empty, follow, leave; the
bar's row beside), the ablation table, the census, the walk-off probe, the
member panel and the planning panel (`train/planning/*`), sampled beside
greedy at the end. A planning explained variance below 0.2 over the last
quarter is a defect named in the amendment, whatever the success reads.

## What follows each reading (D9)

- **≥ 2 seeds PASS or AHEAD with STEERS**: the head chooses and the members
  follow; Stage 1 stands, and the next arm is the D7 ablation on this arm
  (the planning stream reduced to the success bonus alone, paired) before
  the half-step is revisited with the head.
- **AHEAD/PASS without STEERS**: the head's choice is not what the members
  execute; the diagnosis reads the ablation columns and the claimants
  readout, and names whether the head learned KEEP-only (persist 1.00,
  claimants at the greedy allocation the members walk anyway).
- **NULL on ≥ 2 seeds**: the diagnosis reads the planning panel first (a
  critic that never fit → the return's construction; a critic that fit and a
  head that never moved → the surrogate's scale), then the claimants (a head
  that stacks squads on one objective is a head that has not learned to
  allocate), before any change to the design.

## What I expect (a guess, written so it can be wrong)

One PASS, one AHEAD, one NULL, with STEERS on the two that move: the head
learns KEEP early (persistence near 1.00 from the first quarter), the
allocation comes from where each squad's first open lands it, and a squad
that commits to a neighbour's objective is the failure mode on the NULL
seed. Planning explained variance 0.3–0.6: four commitment steps per turn
against a coverage signal that mostly moves at the end.
