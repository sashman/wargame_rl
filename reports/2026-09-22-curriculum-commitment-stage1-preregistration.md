# Pre-registration: the commitment head on A3's shape (CM3, #398) — Stage 1 of the commitment layer

**Written 2026-09-22 01:35, before any training number exists.** Parent
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

⚠ A first launch of the three seeds at 01:33 ran on code the hooks had
refused to commit (two type errors in a test file); it was stopped within a
minute, its run directories deleted, and nothing from it is read. The runs
below start after this file and the build (`be3fbed`) are on the branch.

## Amendment 1 — written 2026-09-22 03:24: CM3 read at the cap — NULL / NULL / NULL, STEERS 0; the head commits, keeps and stacks

Read after every seed's log carried its `rounds 122880` line and no trainer
remained (chain `cm3_reads.sh`; directories resolved from the config stem);
n=100, seeds 700000+, greedy, `last.pt` after exit. Launched 01:36, the
last trainer exited ~03:05. Wandb `curriculum-cm-s1`: g8azma3g / tbz5jz2i /
1l1qkjng. Every run recorded a greedy episode at every checkpoint.

| rounds | CM3 s1 / s2 / s3 | A3's own | `a3_cm` |
|---|---|---|---|
| 40,960 | 0.000 / 0.000 / 0.000 | 0.400 / 0.350 / 0.340 | 0.520 / 0.350 / 0.240 |
| 81,920 | 0.110 / 0.070 / 0.000 | 0.590 / 0.670 / 0.610 | 0.890 / 0.650 / 0.750 |
| 122,880 | **0.030 / 0.000 / 0.000** | 0.700 / 0.800 / 0.770 | 0.820 / 0.850 / 0.930 |

**Per seed: NULL, NULL, NULL; STEERS on none** (the flag ablation moves
success and held by hundredths on every seed). The arm is far BEHIND A3
from scratch at every read: a head that chooses badly costs more than no
head, because the travel keying carries its choice to the members.

Readouts at the cap, the bar's row beside (persist 0.99, claim 1.00 / max
1, complete 1.00, follow 1.00, leave 0.00):

| seed | held | persist | claimants per claimed objective | complete | empty | follow | leave | max stack |
|---|---|---|---|---|---|---|---|---|
| s1 | 2.17 | 0.53 | 1.64 / max 4 | 0.43 | 0.06 | 0.74 | 0.73 | 3.5 |
| s2 | 2.64 | 0.85 | **2.56** / max 4 | **0.95** | 0.09 | 0.87 | 0.19 | **5.3** |
| s3 | 1.72 | 0.91 | 2.05 / max 4 | 0.18 | 0.03 | 0.80 | 0.80 | 3.5 |

s2 is the clearest reading: the head keeps its choices (0.85), commits
every squad (empty 0.09), the members follow (0.87) and the plan is
COMPLETED (0.95) — 10.2 of twelve bodies on objectives by turn 7 — and it
holds 2.64 of four because two and a half squads claim each claimed
objective and one objective is empty in every census episode. The head
learned to commit, to keep and to be followed; it did not learn to
allocate. s1 switches (persist 0.53) and holds 2.17; s3 keeps (0.91) a
plan the members complete in 18% of unit-cases and holds 1.72. Uncommitted
unit-turns fell from 13–23% at 40,960 to 3–9%. Sampled play is 2–5 vp below
greedy, the same held.

**Panels over the last quarter.** Members: clip fraction 0.24–0.32,
explained variance 0.75–0.85, displacement entropy 1.75–1.95 nats. Planning:
explained variance **0.23 / 0.25 / 0.33** (above the 0.2 defect line, and
low), return mean 0.26–0.34, commitment entropy 0.90–0.99 nats of 1.61
(from 1.1–1.2 at 40,960), commitment clip fraction 0.27–0.32 — the head
moved throughout and settled on the wrong distribution.

### The D9 diagnosis

1. **The planning panel.** The critic fit weakly and the head moved: this
   is not a return that never reached the head. It is a return the head
   cannot use to tell its choices apart.
2. **The claimants.** 1.6–2.6 squads per claimed objective at the cap, from
   2.3–3.0 at 40,960: the head has not learned to allocate. Every squad's
   commitment step is paid the SAME close — the army's coverage and the
   rung's success bonus, broadcast — so a squad that piles onto an objective
   a neighbour already claims and a squad that takes the empty one receive
   identical credit, and the only route to telling them apart is the
   planning critic's reading of the claimant counts on the objective tokens,
   which at explained variance 0.25 it barely does. The whole-army record's
   critic-probe finding is the same fact from the other side: the value can
   know the stack is wrong while the policy gradient cannot find the
   redistribution.
3. **The members.** The flag ablation is flat on every seed, as on Stage
   0's A3: on this shape the members walk the geometry, and the head's
   choice reaches them through the travel keying (follow-through 0.74–0.87),
   not through the relation. Stacked commitments therefore stack the walk.
4. **The keying without allocation is the harm.** A3 from scratch derives a
   per-squad target every step and reads 0.70–0.80; the environment's greedy
   assignment reads 0.82–0.93; the head's assignment reads 0.00–0.03. The
   order is: a good fixed plan > a re-derived plan > a learned plan that
   stacks. The head is not a null on this rung; it is a cost, until it
   allocates.

### The revision this asks of #384 (proposed; Sash decides)

- **B6, the per-unit counterfactual on the planning stream** — the build the
  design held behind follow-through, and the reading calls for it now: pay
  each squad's commitment step the DIFFERENCE its bodies make to the
  outcome (coverage with the squad's models masked out, subtracted from
  coverage with them in; the success bonus likewise), so a squad stacking
  on a covered objective earns 0 and the squad taking the empty one earns
  the whole objective. A difference reward, not a switch price (D4 stands),
  computed at the close in the retimer from the same distance cache. One
  arm, three seeds, same shape, read against this one.
- Not proposed: a longer budget (flat from 81,920), an entropy change (the
  head is at 0.9 nats and moving), or the half-step (the head must allocate
  four before it allocates five).
