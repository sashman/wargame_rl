# Auditing the travel reward: the panel's mechanism is wrong, and the term is mostly inert

**2026-08-23. No GPU.** Held-out nine, 10 episodes per table,
`configs/experiments/24v24_maps_spare_squads.yaml` (the config that trains), agent at K=3.

An expert panel nominated `closest_objective_v2` + `fallback_to_nearest: true` as the cause
of the agent's stacking, with a specific mechanism: it "pays +0.081 per inch closed on the
**CENTRE POINT** of each model's own nearest objective", saturating only within ~0.63" of
that centre, so **STAY is strictly dominated**, every model is dragged onto a spot two bases
can occupy, and **squadmates are paid to walk apart**.

CLAUDE.md had flagged this term twice and it had never been audited. It has now been.

## Verdict on each claim

| claim | verdict |
|---|---|
| pays 0.081 per inch closed | **CONFIRMED.** `progress_scale: 6.0` over a board diagonal of 74.40" = **0.0806/inch**. |
| the pull is to the objective's **CENTRE POINT** | **REFUTED.** |
| **STAY is strictly dominated**, hence the ~0% stay rate | **REFUTED.** |
| squadmates are **paid to walk apart** | **REAL BUT SMALL** — 8.0% of squad-steps. |
| most pay is **fallback**, not a coordinated assignment | **CONFIRMED** — 73.6%. But see the trap below. |

## Why the centre-point mechanism is wrong

`_distances_to_objectives` in `env_components/distance_cache.py` says it outright: *"A
marker's is the distance to its centre; **an area's is the distance to its outline, zero
inside**."* And the training config produces objectives that are **areas**: six of them,
every one with `area is not None` and `radius_size 0.0`.

So the pull saturates at the **area boundary**, not at a point of measure zero. There is no
centre-point magnet. The panel read `norms_offset = max(to_objective - base_radius, 0)` and
took `to_objective` for a centre distance; for an area it is not.

**And the measurement confirms it: 43.5% of paid model-steps are already INSIDE their
target**, where this term pays exactly zero however the model moves. STAY is not dominated
for those models — nothing in this term distinguishes standing from shuffling once inside.

## The measured gates

| | agent (K=3) | `squad_march_take` |
|---|---|---|
| objectives the travel reward can point at | 35.8% (2.03 of 5.68) | 16.0% (0.91 of 5.67) |
| units given an objective of their own | 21.0% (1.68 of 8) | 9.8% (0.79 of 8) |
| steps where one unit owns 2+ objectives | 29.3% | 10.3% |
| model-steps paid toward an assigned target | 26.4% | 15.8% |
| **model-steps paid toward NEAREST (fallback)** | **73.6%** | **84.2%** |
| **SQUAD-steps split across 2+ targets** | **8.0%** | 4.8% |
| **paid model-steps already INSIDE their target** | **43.5%** | 67.7% |
| non-candidate: we already hold it | 55.1% | 58.6% |
| non-candidate: they hold it by 2+ | 44.9% | 41.4% |

## ⚠ The trap in reading that table, and it is the important part

**A scripted policy is not a control for what a reward term does.** The script does not
learn from reward at all, so its column describes *where its models happen to stand*, while
the agent's column describes *what the agent was paid*. The two are not the same quantity.

That matters because the script is **worse on every gate** — it points at fewer objectives
(16.0% v 35.8%), assigns fewer units (9.8% v 21.0%) and takes more fallback pay (84.2% v
73.6%) — **and it allocates better**, holding 3.28 distinct objectives to the agent's
2.08–2.30.

**So none of these gates explains the allocation gap.** Whatever separates the two policies,
it is not that the agent's travel reward points somewhere worse than the script's would.

## What the audit does establish

**The travel term is largely inert, and where it is not, it is negative.**

- 43.5% of paid model-steps earn **exactly zero** from it (already inside the target).
- 64.2% of objectives are not candidates for anybody — 55.1% "we already hold it", 44.9%
  "they hold it by 2+". That second half is the `contest_deficit` gate, already widened and
  **already REJECTED** (−2.7 ± 4.8, offence went −61.2 → −71.5).
- Net income on file is progress **+0.08** against the overstack penalty's **−0.90**, so the
  term's whole net contribution is negative — and removing that penalty was measured at
  **−12.2 ± 5.5 paired, 3/3 seeds negative**.

**The split-squad effect is real and small.** 8.0% of squad-steps have members holding two
or more different targets, against the script's 4.8%. It exists, it is 1.7x the script's
rate, and at that frequency it cannot be the mechanism behind a 33% shortfall in objectives
held.

## What this closes

⚠ **Do not fund a `fallback_to_nearest` change on the panel's rationale.** The mechanism it
named does not exist in this configuration, and the gate statistics do not separate the
agent from a policy that allocates better.

⚠ **This is now the FOURTH consecutive time the travel reward has been nominated and come
back empty** — the candidate gate (`contest_deficit`, rejected), the overstack penalty
(removal rejected), the potential-invariance defect (real, but the term nets negative
anyway), and now the fallback mechanism. **Stop nominating `closest_objective_v2`.**

## The one thing worth carrying forward

The term is **mostly inert** — 43.5% of its paid steps pay zero and its net income is
negative. A term that is inert cannot be the cause of a behaviour, but it also cannot be
carrying the travel signal the agent would need. That is consistent with everything else on
file: **there is no working travel gradient, and four attempts to build one have failed.**

## Reproduce

    just measure-shaping-gates <policy|ckpt> configs/experiments/24v24_maps_spare_squads.yaml 10 configs/evaluation/maps_heldout 3
