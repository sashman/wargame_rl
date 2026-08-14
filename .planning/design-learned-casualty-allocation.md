# Design: casualty allocation as a learned action

Status: **proposal**, not implemented. Written 2026-08-14.

Lives in `.planning/` rather than `docs/` because `docs/` describes behaviour the
env actually has, and the docs-drift hook treats it as live. Move the parts that
survive into `docs/shooting.md` and `docs/rules/implementation-status.md` when
this ships.

---

## Summary

When a unit takes hits, the defender chooses which model dies. This environment
picks the lowest-indexed survivor. The proposal is to make that choice an
**action the policy takes**, because the right choice depends on the whole board
and trades off objectives no heuristic can rank in advance.

**There is a blocking prerequisite** — the best motivating case is currently
*unobservable*, so the action would be unlearnable for the reason that most
justifies it. See § The blocking prerequisite.

## Why this is an action and not a heuristic

Three reasons to prefer one casualty over another. They are not variations on a
theme; they **disagree about who should die**, which is the argument.

### 1. Deny the target (the motivating case)

Shooting declares a *unit*, and `compute_unit_shooting_masks` makes a unit
targetable only when **some model in it is visible** and **some model is in
range**. So if one model is standing in the open and the rest are behind a ruin,
removing that model drops the whole unit out of every later shooter's mask for
the remainder of the phase.

This is a **step change, not a gradient**. Objective control and coherency each
shift a reward slightly; this flips a boolean — shootable-by-everything to
shootable-by-nothing. And it is the case a coherency-preserving heuristic gets
*backwards*: the exposed model is often inside the coherent body, so the two
objectives point at different models.

Note it does **not** depend on models blocking line of sight. They do not, since
#182 — only terrain does. This works purely through terrain hiding the remnant
plus unit-level targetability.

### 2. Hold the objective

Control is a strict count comparison. Losing the model that is *on* the point
can flip control where losing one standing off it changes nothing. VP accrues
every round, so this is worth more than one model's firepower.

### 3. Keep the unit coherent

Removing a model from the middle of a chain splits the unit into two components
and fails the connectivity clause; removing one from the end does not. Today's
`alive[0]` is effectively a random pick with respect to position, so it splits
strung-out units for no reason.

### And they conflict

The exposed model may be the one on the objective. The model whose removal keeps
the chain whole may be the one keeping the unit targetable. Ranking these
against each other *is the decision*, and it is board-dependent. That is what a
policy is for and what a fixed rule cannot do.

## What exists today

`domain/shooting.py:_allocate_target`:

```python
wounded = [m for m in members if m.is_alive and m.has_lost_wounds]
if wounded:
    return wounded[0]
alive = [m for m in members if m.is_alive]
return alive[0] if alive else None
```

Preferring a wounded model is correct and follows the rules. The fallback —
lowest index — is arbitrary: index has no relation to position, so it models an
*incompetent* defender. `docs/rules/implementation-status.md` already rates
allocation **partial**.

At `max_wounds: 1` every allocated wound kills, so allocation *is* the choice of
who dies. With `max_groups: 5` and five models per unit, the decision is one of
at most five, made several times a phase.

## The blocking prerequisite

**The policy cannot currently see which of its models is exposed.** The
observation carries location, distances to objectives, the group one-hot, the
nearest same-group distance, wounds, combat stats, expected damage per target,
and optionally unit strength, objective-control and the two coherency columns.
It carries **nothing about visibility**. `track_exposure` computes exactly this
quantity, but it feeds the eval metrics — it is measurement, not input.

So reason 1, the strongest, keys on something invisible to the network. This
repo has paid for that mistake twice (the overstack penalty, and
`objective_hold.surplus_value`), and it is why `unit_coherency` refuses to be
configured without `observe_coherency`.

**Therefore: add a per-model `observe_exposure` column before or with this
action, not after.** One flag per model — "at least one live enemy can see and
reach me" — reusing the same LoS matrix the shooting mask already builds each
step, so the cost is a reduction and not a new trace.

One caution: `observe_threat_count` was added for the cover experiments,
measured **null**, and was removed. That is not evidence against this. That
input was there to shape a *continuous* behaviour (take cover) whose reward was
diffuse; this one gates a *discrete* choice with a step-change consequence, and
the action cannot be expressed at all without it.

## Action design

### Shape

Follow the shooting precedent. `ActionSlice` is a registry
(`env_components/actions.py`) and shooting registered a per-unit categorical
with its own mask. Allocation fits the same frame.

**Rejected — one designated casualty per unit per step.** Simplest, fixed shape,
easy log-prob. But it is too weak for reason 1: exposure changes as the
opponent's units fire *in sequence*, so the right model to remove for the second
shooter is not the right one for the first. A single designation made at the
start of the step cannot express that.

**Recommended — a priority *ordering* over each unit's models.** The policy
emits one score per own model; when a unit takes a hit, the defender removes the
live model with the lowest priority, still preferring an already-wounded model
first, per the rules. Properties:

- Fixed shape: `n_models` scores, one per model, like the value head.
- Handles multiple casualties in one phase without re-designating.
- Degrades correctly: with all scores equal it reproduces today's behaviour.
- Log-prob: sample the order as a Plackett–Luce draw over the unit's live
  models, whose log-prob is a sum of log-softmax terms. PPO needs a log-prob for
  a *sampled* action, and this gives one without a variable-length action space.

**Open question.** Ordering is chosen once per step but consumed across the
opponent's whole shooting sequence, so it still cannot react *within* the
sequence — it only removes the need to re-designate. Whether that matters is
measurable: instrument how often the optimal casualty changes between two
shooters in one phase. Do that before building the harder version.

### Timing

Allocation is **reactive** — it happens during the opponent's phase, not the
player's. Introducing a genuine decision point there means restructuring
`turn_execution` and the clock, which is a much larger change and would alter
episode length. Emitting the ordering in advance, during the player's own step,
avoids all of it. This is the main reason the design is shaped this way.

### Masking

The ordering only ranks *own* models, so the mask is the alive mask. No new
legality rules. A dead model's score is ignored rather than masked, matching how
the shooting head handles dead tokens.

## Symmetry: the opponent allocates too

If the player learns allocation and the opponent keeps `alive[0]`, the change is
a **difficulty reduction wearing a rules change's clothes** — every baseline
would move for a reason that is not the rule.

The opponent needs at least a scripted allocator. Recommended: a
coherency-preserving pick (remove a model whose removal leaves the unit
connected, preferring one off an objective), registered in the opponent policy
registry. It should be applied to *both* sides as the new default so the
scripted baselines are measured under it.

## Tests

Following `tests/CLAUDE.md`, behavioural and hand-placed rather than structural:

- The designated model dies, not `alive[0]` — with the control that under equal
  scores the behaviour is byte-identical to today. A change that cannot be
  turned off cannot be A/B'd.
- An already-wounded model is still taken first, whatever the ordering says.
  That is the rule, and the ordering is the *tiebreak*, not an override.
- Removing the last visible model makes the unit untargetable on the next
  `compute_unit_shooting_masks` call — the motivating case, asserted end to end
  through the real mask rather than a helper. (The unit-exemption bug that #182
  fixed survived precisely because its test used a helper.)
- Multiple casualties in one phase consume the ordering in order.
- The opponent's allocator is exercised too, or the symmetry above is untested.
- Golden gates must stay bit-identical with the feature off.

## Measurement

This changes what dies, so it **moves every baseline on every config** — the
third such change this week. Budget for it:

1. Re-measure floor, bar and rungs with `just measure-baselines <config> 100 ""
   700000` on the affected configs.
2. Expect the *opponent* to get tougher too. A drop in agent score is not
   evidence the feature is bad.
3. Report `eval/coherency_rate` and `models_out_of_coherency` beside vp — the
   coherency-preserving allocator alone should move them, separately from any
   learned effect.
4. Screen at 300 epochs, quote at 1000, two seeds minimum. The four-seed arm
   running now exists because two seeds could not resolve the last effect.

## Risks

- **Cost for a small decision space.** One of ≤5 models, and much of the time
  the choice is forced or irrelevant. Measure the headroom first: replay
  recorded matches and count how often an oracle allocator would have changed
  the outcome. If the answer is "rarely", stop here.
- **Credit assignment is thin.** The consequence of an allocation arrives on the
  *opponent's* next activation, mediated by their targeting. That is a long path
  for PPO through a single scalar.
- **Interaction with `unit_coherency`.** If both ship, a coherency-shaped reward
  and a coherency-relevant action are two levers on one behaviour, and
  attributing a result needs an arm for each.

## Sequencing

1. **Measure the headroom** (oracle-allocator replay). Cheap, and it can kill
   the whole proposal.
2. **Add `observe_exposure`** — useful on its own, and blocking for reason 1.
3. **Ship the scripted coherency-preserving allocator for both sides**,
   re-baseline. This alone may capture most of reason 3.
4. **Then the learned ordering**, measured against step 3 rather than against
   today, so the learned part is isolated from the heuristic part.

Steps 1–2 are hours. Step 3 is a day with re-baselining. Step 4 is the real
work and should not start until 1 has said it is worth it.
