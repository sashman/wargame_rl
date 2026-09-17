# Pre-registration: curriculum rung D1 — the escort cloned into the set network, scored on the final-board rung

Written 2026-09-17 14:50, **before any clone of the escort exists at
house fidelity** (a 20-episode × 5-epoch smoke clone of the pipeline was
run before this file to check the plumbing; its numbers are not a
result), on branch `feature/curriculum-d1` (stacked on PR #371). Issue
#373, parent question #340, rung **D1** — the "start" axis, supervised.
Carries the per-model behaviour clone build (#331).

## The question

C3 and C3b (#370, #372) found that neither trainer learns the escort's
plan from reward at this budget: the per-model arm from C2 learned to
shoot the blockers off the point and then parked (0.20 / 0.13 / 0.23 on
C3b), the from-scratch run learned almost nothing, and the whole-army
control walked in and died. Before D2 asks whether PPO can improve a
policy that starts with the plan, D1 asks whether the set network can
**hold** one: copy the escort's decisions into the network by imitation
and see whether the copy reproduces them and plays the rung as the
teacher does. #340's row: per-head action match ≥ 0.95 held-out; the
clone's score not worse than the teacher, paired at n=100; ordering
within 5 pp.

## The build, shipped with the rung

`model/per_model/clone.py` + `scripts/behaviour_clone_per_model.py` +
`just behaviour-clone-per-model` (#331). `record_demonstrations` plays a
per-model chooser on the player seat and records every decision step as
the `Transition` the PPO update reads, with the teacher's model and the
head column that names its action; `action_to_column` is the inverse of
`SetAgent._decode`, and the recorder decodes every recorded pair back and
compares, so the map cannot drift silently. `fit_clone` minimises the
joint negative log-prob from `evaluate_transitions`; `match_report`
scores greedy agreement on held-out transitions (selector, each head,
joint); `save_clone` writes a per-model checkpoint at zero rounds that
the resolver plays and `--warm-start-from` accepts, with a `.clone.json`
of provenance beside it. Pinned by `tests/test_per_model_clone.py`. ⚠
**The critic is not fitted**: the clone's value head is at
initialisation. That is D2's question.

## The one change

The weights come from imitation, not reward. `just
behaviour-clone-per-model scripted_escort
configs/experiments/curriculum/c3b.yaml 300 40 <out> <seed>`: 300
demonstration episodes at seeds 800000+ (the last 60 held out), 40
epochs, batches of 64, Adam at 3e-4 — the house-fidelity lesson (the
whole-army clone at 120 × 8 under-fitted; at 1200 × 60 it carried its
teacher). **Two clones from identical data, fit seeds 0 and 1**: clone
twice before quoting a clone (the two whole-army clones that differed
only in initialisation measured 115.8 and 111.1). No PPO.

## Comparator, measured first

`scripted_escort` on `c3b.yaml`, seeds 700000+ at n=100, both facades
identical: success **0.990**, `held` 3.97, `alive` 0.963, `on_obj` 0.993,
vp +63.5 ± 2.0, coherent 0.858; first kill at turn 6.0, blockers wiped at
11.0 (99%), first unarmed body on their point at 12.9 (90%), the kill
before the arrival in **1.00**. Beside it, what reward produced on the
same scenario: the per-model arm from C2 0.200 / 0.130 / 0.230, from
scratch 0.02–0.06, the whole-army control 0.280 / 0.000 / 0.450.

## Criteria

Read greedy, no decode, n=100 on 700000+, from each clone's `.pt`.

- **PASS:** on **both** clones, held-out `joint` match ≥ **0.95**; success
  at n=100 ≥ **0.96** (the teacher's 0.990 less one binomial SE of 0.01,
  rounded down to the next hundredth — "not worse than the teacher" at
  this n); the kill-before-arrival share ≥ **0.95** (within 5 pp of the
  teacher's 1.00).
- **FAIL:** any clause missed on either clone. The census says which:
  a clone that matches and fails the rung is the whole-army lesson (a
  per-decision fit does not inherit a joint property — here, the
  unarmed squads' *waiting* is a joint property of twelve decisions per
  turn); a clone that fails to match is the network unable to represent
  the plan at this fidelity, and the loss curve says whether more epochs
  would.
- **NULL (scenario):** only if the teacher fails its own criterion on
  the scoring seeds. It reads 0.990.
- **Readouts:** the match by head; the loss per epoch; `held`, `alive`,
  `on_obj`, vp paired against the teacher; coherency greedy and sampled
  (`just measure-per-model-eval-mode`); decisions per episode; the
  by-turn census; C3b's rows beside every row.

Power: binomial SE 0.010 at p=0.99, n=100, on success; the ordering
share the same; the match is over ~5,000 held-out decisions, SE under
0.01 at 0.95.

## What I expect (a guess, written so it can be wrong)

Both clones match above 0.95 on the displacement head and the selector
and below it on the unit-pointer head (few shooting decisions per
episode, most of them "the only target"), for a `joint` near 0.93–0.96;
success 0.90–0.97, one clone passing and one a hair under; ordering
above 0.95 on both, since "wait" is the most common decision and the
easiest to copy. If instead both pass cleanly, D2 starts from the better
clone; if both fail on success while matching, the failure is the
whole-army lesson and the report says which joint property broke.
