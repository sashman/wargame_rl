# Coherency was never deployed

**2026-08-12.** The unit coherency rule has been rated *divergent* in the gap map
since the spec landed. Asked to implement it accurately, the first thing built
was not an enforcement mechanism but a **predicate and a measurement** — and the
measurement moved the whole question.

## What the rule is

`docs/rules/03-moving.md` § Coherency. A unit of more than one model is in
coherency while every model is:

- within **2"** of at least one *other* model in the unit (the **chain**), and
- within **9"** of *every other* model in the unit (the **spread**),

and the unit forms a **single connected group**.

That third clause is not implied by the first two, and the counterexample is
ordinary: two fire teams of two, 5" apart, satisfy the chain (each model has a
partner) and the spread (nothing exceeds 9") while the unit is plainly in two
pieces. Connectivity on the chain graph is what separates them. Implementing
only the stated conditions is the classic error, and it is why the predicate
here computes components rather than distances.

Distances are **base to base**, per `01-core-concepts.md` ("measure to and from
the closest part of the model"), matching how this environment already measures
engagement. At the default 32mm base that widens the chain to 3.26"
centre-to-centre. This matters more than it looks: read literally against
centre distances, the legal band between "bases touching" (1.26") and the chain
limit is 0.74" — narrower than the smallest move the action space can make
(1.0"). Base to base it is 2.0", twice the minimum move. The rules' own
measuring convention is the difference between a controllable constraint and an
uncontrollable one.

## What was measured

`just measure-coherency <policy|ckpt> <config> [n]` — new. Both forces, every
step, at the rules' 2"/9" and at whatever the config calls `group_max_distance`,
so the cost of adopting the rules' *numbers* is separable from the cost of
enforcing the *concept*.

On `configs/golden/25v25_shooting_opponent.yaml`, 20 episodes, seeds 700000+:

| policy | units coherent | steps coherent | deployment coherent | models out |
|---|---|---|---|---|
| `hold_deployment` (STAY) | 0.006 | 0.000 | 0.000 | 20.14 |
| `random` | 0.094 | 0.018 | 0.000 | 18.09 |
| `squad_march` | 0.285 | 0.088 | 0.000 | 7.22 |
| `squad_march_shoot` (the bar) | 0.248 | 0.033 | 0.000 | 8.13 |
| trained agent (real-maps s1) | 0.306 | 0.022 | 0.000 | 15.01 |

**Nothing in this repo has ever been in coherency.** The bar — a policy written
to hold formation — satisfies the rule on 3.3% of steps. The trained agent
manages 2.2%.

## The cause is deployment, and `hold_deployment` proves it

Every policy deploys incoherently in **every** episode, because placement does
not depend on the policy. `wargame_model_placement` anchors each model within
`group_max_distance` of one *random* already-placed squadmate: that bounds the
nearest neighbour and leaves the unit's overall span unbounded, so a 5-model
squad can legally deploy strung across 40" and in several pieces.

The clean isolation is `hold_deployment`, the STAY baseline: it never moves, and
it still scores **0.006**. Whatever is wrong is wrong before the first action.

## The fix, and what it recovers

`coherency.enforce_at_deployment` places each model within the chain distance of
an already-placed squadmate *and* within the spread cap of all of them —
connected by construction, since each new model attaches to a body that is
already one piece. Same measurement, same seeds:

| policy | units coherent | steps coherent | deployment | models out |
|---|---|---|---|---|
| `hold_deployment` (STAY) | 0.006 → **0.992** | 0.000 → **0.965** | **1.000** | 20.14 → **0.08** |
| `random` | 0.094 → 0.105 | 0.018 → 0.024 | **1.000** | 18.09 → 17.75 |
| `squad_march` | 0.285 → **0.799** | 0.088 → **0.462** | **1.000** | 7.22 → **1.16** |
| `squad_march_shoot` | 0.248 → **0.804** | 0.033 → **0.428** | **1.000** | 8.13 → **1.29** |

Three things worth reading off this:

- **`hold_deployment` → 0.992** is the proof the fix is complete at the point it
  acts. The residual 0.008 is its own force being shot apart, not placement.
- **The marching baselines go 0.25 → 0.80 with no enforcement during play.**
  Roughly three quarters of the breach was inherited from the set-up. No reward
  term and no constraint mechanism was involved.
- **`random` barely moves, 0.094 → 0.105.** That is the control: random
  movement destroys coherency wherever it starts. A metric that improved here
  too would be measuring something other than the policy.

## What deploying legally costs — it pays

Re-measured at n=100, seeds 700000-700099, both configs on identical layouts:

| policy | golden (parent) | coherent deploy | delta | win |
|---|---|---|---|---|
| `random` | −128.4 | −122.3 | +6.1 | 0.01 → 0.03 |
| `greedy_nearest` | −57.6 | −52.2 | +5.4 | 0.13 → 0.13 |
| `split_evenly` | −49.2 | −41.3 | +7.9 | 0.26 → 0.26 |
| `squad_march` | −26.4 | −3.6 | **+22.8** | 0.33 → 0.48 |
| **`squad_march_shoot` (the bar)** | **+38.0** | **+58.9** | **+20.9** | 0.75 → 0.82 |
| `contest_and_spread` | +18.7 | +39.3 | +20.6 | 0.64 → 0.79 |

The parent's bar reproduces the known real-geometry figure of +38.0 exactly,
which is the check that the measurement is sound.

**Legality is not a tax here — it is worth ~21 vp to every policy that
manoeuvres as squads**, and ~6 to the ones that do not. `random` gaining least
is the control: a policy that scatters on the first step cannot keep what the
set-up gave it. The mechanism is visible in `alive` — 0.450 → 0.592 for the bar,
0.363 → 0.465 for `squad_march`. A squad that starts concentrated concentrates
its fire, kills faster, and takes fewer casualties doing it.

Two consequences:

- **The bar on this config is +58.9.** Every trained number on the parent is
  void here, including the agent's +6.08 ± 3.09 over the bar, which was measured
  against +38.0.
- **The ordering does not flip**, unlike the one-distance precedent, where
  `contest_and_spread` edged past the bar. The bar keeps its lead and slightly
  widens it, 19.3 → 19.6.

## What this changes about the plan

The five expert reviews fanned out for this task split on the enforcement
mechanism — action masking (unsound: coherency is a joint constraint and a
per-model mask is a product set), move-revert, projection, attrition. That
argument turns out to be **downstream of a much cheaper fact**. Three quarters
of the problem was a placement bug, and the mechanism debate is only about the
remaining quarter.

The rules also settle the priority, against the intuition that attrition is the
enforcement: the spec's *primary* consequence is the end-of-move revert
(`03-moving.md` § Making a move), and End-of-Turn attrition is a **backstop**
for breaks caused by something other than the unit's own move. On the table
nobody loses models to coherency in a normal game, because the illegal move is
never made. **If attrition fires often, the end-of-move check is wrong** — which
is a design constraint, and a test, not a preference.

## Enforcing the move: a tax, and a measurable cliff

Deployment fixed the set-up; the residual ~20% is real manoeuvre breakage. The
rules' primary consequence is a revert, so both readings of it were built and
measured — n=100, seeds 700000+, identical layouts:

| policy | deploy only | `revert_unit` | `revert_model` |
|---|---|---|---|
| `random` | −122.3 | −119.0 | −122.3 |
| `greedy_nearest` | −52.2 | −27.9 | −35.2 |
| **`split_evenly`** | **−41.3** | **−97.6** | **−47.7** |
| `squad_march` | −3.6 | +0.9 | −1.0 |
| **`squad_march_shoot` (bar)** | **+58.9** | **+44.2** | **+41.5** |
| `contest_and_spread` | +39.3 | +41.7 | +32.1 |

**1. Enforcing the move is a tax, where deploying legally was a bonus.** The bar
gives back ~15–17 of the +20.9 that coherent deployment bought. At n=100
*unpaired* this config resolves ~10–18 vp, so the direction is clear and the
magnitude is not — pairing across configs would be needed to quote a number.

**2. The two modes are not distinguishable on the bar.** 44.2 against 41.5 is
2.7 apart, far inside noise. Reading that as a winner is reading noise.

**3. The undo cliff is real, and it lands exactly where it was predicted to.**
`split_evenly` sends model *i* to objective *i mod n*, shattering every squad by
construction. Under `revert_unit` it collapses **−41.3 → −97.6**, far outside
any noise, and the diagnostics say why: `on_obj` 0.831 → **0.030**, `alive`
0.101 → **0.962**. Almost nobody arrives and almost nobody dies — **the army is
frozen**, every move cancelled, standing still. `random` shows the same
signature (`alive` 0.982, `on_obj` 0.013). Standing still is this project's
worst measured failure mode (−40.4). `revert_model` halves the damage by letting
the coherent body keep moving: a 50 vp gap between the modes on one policy.

The modes tie on competent policies and separate sharply on incompetent ones —
and a *learning* policy starts incompetent. `revert_model` is the safer default
for training even though `revert_unit` is the faithful rule, because the spec's
version would spend early training teaching the agent that movement does not
happen.

**Why they tie where they tie:** the spread condition is collective. Once one
model exceeds the cap from the rest, no model is within the cap of every other,
so all are in breach and `revert_model` reverts everyone too. The modes separate
only while a break is *local*.

**A first version of this was wrong, and the numbers above are the corrected
ones.** The naive revert put a model back onto ground another had legally taken
— models resolve sequentially against live positions, so one may move into a
square a lower-indexed model vacated. Two overlapping bases is *another* illegal
state, and `03-moving.md` checks it in the same breath as coherency ("no model
is left on top of another model", under the same "if any check fails, the move
cannot be made"). So the enforcement was laundering one illegal state into
another. The revert now **cascades**: a displaced model's move has failed too,
and it goes back as well, to a fixpoint. It converges because each pass reverts
at least one more model and the worst case is the whole force at its start,
which is legal by construction. Every qualitative conclusion survived the fix;
the tax grew from ~10 to ~15 vp.

**And the residual confirms the spec's own priority.** With the move rule on,
the player still sits at 0.908 of units coherent rather than 1.000 — units
broken by **casualties**, where there is no move to undo. That is precisely what
`03-moving.md` reserves End-of-Turn attrition for, and it is the test the rules
lawyer named: if attrition ever fires often, the revert is wrong. Here it would
fire rarely, and only on the cases it exists for.

## Status

- `domain/coherency.py` — the predicate. Both conditions plus connectivity, base
  to base, alive models only. Decides nothing; the same definition serves the
  metric, and will serve any enforcement or reward without three copies drifting.
- `scripts/measure_coherency.py` / `just measure-coherency`.
- `coherency.enforce_at_deployment` — the § Setting up rule, **off by default**.
- `configs/experiments/25v25_coherent_deploy.yaml` — the arm, carrying both
  tables above in its header.

Everything is off by default and verified byte-identical when off: both
bit-identical goldens pass unmodified, and 1169 tests are green.

**Not done, deliberately.** Attrition — the backstop for the ~9% of unit-steps
broken by casualties rather than by a move. Three separate points are worth
recording before that work starts:

1. **The observability desk check fails today.** The only cohesion input is a
   nearest-neighbour *scalar* normalised by the board diagonal — 2" is 2.7% of
   one column's range — with no distance to the furthest squadmate (the spread
   condition has no tensor at all) and nothing exposing connectivity. This
   repo's own rule is to add the input before training a term that keys on it.
2. **`_same_group_closest_distance` counts corpses.** It takes every model's
   location with no alive filter, and a destroyed model keeps its position
   forever, while the `group_cohesion` *reward* masks the dead. The agent's
   cohesion input and its cohesion reward disagree about who is in the unit, and
   they diverge as casualties mount. A live bug, independent of coherency.
3. **Turning deployment on is a scenario change and voids the baselines on that
   config.** Precedent: the milder one-distance version moved the bar
   +58.9 → +63.3 and flipped the top two baselines
   (`configs/experiments/25v25_coherent_spawn.yaml`). Re-measure floor and bar
   on identical layouts before quoting anything.
