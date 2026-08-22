# The travel reward never points at ground the opponent holds — criteria, written before any training

## What was measured first (no GPU)

`just measure-shaping-gates` on the v3.0 lineage (s1-newmaps, `25v25_maps_two_mode`,
held-out nine, n=20, K=3):

    objectives the travel reward can point at     35.1%  (1.99 of 5.66 per step)
    units given an objective of their own         32.0%  (1.60 of 5)
    model-steps paid toward an assigned target    36.9%
    model-steps paid toward NEAREST (fallback)    63.1%

    why an objective was not a candidate:
      we already hold it                            56.6%
      they hold it by 2+ (one arrival cannot flip)  43.4%

## The mechanism, located exactly

`_is_positive_transition` asks whether **one more model** changes the control
label. With `_state_label` being `player` at `p >= o+1`, `contested` at `p == o`
and `opponent` at `o >= p+1`, the only qualifying moves are
`neutral -> player`, `opponent -> contested` (needs `o == p+1` exactly) and
`contested -> player`.

So **an objective the opponent holds by two or more is never a candidate**. The
travel reward cannot point at it, the unit nearest it is left unassigned, and
`fallback_to_nearest` then pays that unit to close on its *nearest* objective —
which, for a stacked army, is usually the one it is already standing on, at
distance 0 and therefore for zero reward.

That is the offence deficit as a line of code: **the agent is never paid to
attack.** It matches every symptom on file — offence negative on 3/3 seeds in
every scenario, 54.4% of model-steps on objectives against the scripts' 75.5%,
and 53.7% of income arriving as global terms rather than per-model ones.

## The arm

`closest_objective_v2` gains **`contest_deficit: int = 1`** — an objective is a
candidate when the opponent's lead over us is at most `contest_deficit`, instead
of the hardcoded 1. At `contest_deficit: 5` a whole five-model unit arriving can
flip the point, which matches the fact that units move as units.

- **Default is 1 = today's behaviour exactly**, so every existing config, the
  control lineage and `tests/test_reward_golden.py` are untouched.
- **No tensor-shape change**, so the paired estimator holds and checkpoints stay
  loadable.
- **Observability desk check PASSES**: the term keys on per-objective player and
  opponent alive counts, and `observe_objective_control: true` supplies both —
  verified against the scoring definition on 2026-08-22.
- **It ADDS income rather than moving it.** The failed anti-concentration levers
  all lowered total objective income; this one points existing travel reward at
  ground that currently pays nothing. Nothing is taken away from held points.

## What the gate did and did not forbid

⚠ It never forbade the coordinated attack. Nothing in the rules stops the agent
sending three models at a point two defenders hold, and `objective_hold` and
`vp_gain` pay for it normally once it succeeds. What was missing is the
*shaping*: `closest_objective_v2` is the only term that pays a model to move
BETWEEN objectives, and its target could never be that point. So the agent had
to discover a coordinated multi-model assault from sparse downstream signal,
while `fallback_to_nearest` actively paid it to close on the point it already
stood on.

**The granularity is the unit, not the model.** `_compute_group_assignment`
gives a candidate objective to exactly one *group*, so `contest_deficit: 5`
reads as "could a squad take this?" rather than "could three models take this?".
5 is the squad size, and the coherency work established that models here already
move as units (0.938-0.955 coherent), so the shaped behaviour is "send that
squad at that point".

## The remaining bottleneck, measured and NOT addressed here

Widening the gate raises candidate objectives, but units assigned rises only
**32.0% -> 48.1%** (1.60 -> 2.41 of 5) while *steps where one unit owns 2+
objectives* rises **29.0% -> 70.7%**.

That is `_compute_group_assignment`: it walks each candidate objective and gives
it to the **closest group**, so one well-placed unit can take several while other
units get none. The candidate gate was the binding constraint; with it widened,
greedy per-objective assignment becomes the next one.

⚠ **This is deliberately NOT in this arm.** Fixing it means a matching (each
group at most one objective) rather than an argmin, which is a second behaviour
change; stacking it would make the result uninterpretable. Recorded here so the
next step is chosen from a measurement rather than rediscovered.

⚠ Note also that the golden config's own comment for this parameter reads "5
groups over 3 objectives leaves two groups unassigned" -- that is the THREE
objective scenario, and the generated tables carry five or six. The comment is
stale; the measurement above is what holds.

## Pre-registered criteria

Three seeds, 300 epochs, `ent_coef` 0.003, recording on, **paired against the
existing s1/s2/s3 `-newmaps` controls** (same seeds, same config, same epoch
budget — no control retraining). Scored refereed at K=3, held-out nine, n=30.

- **ACCEPT** if the paired difference is positive with t > 2 **and** offence
  (agent VP minus best-script VP) improves. Offence is the target; a vp gain
  that is entirely defence again is NOT this lever working.
- **REJECT** if the paired difference is negative, or if `alive` rises while
  `held` does not — that is the hoarding getting worse.
- **INCONCLUSIVE** at |t| < 2: report as "run it longer", per the 300-epoch
  screening rule, not as refuted.

## What would make me wrong, stated in advance

The overstack result three days ago is the live caution: a term that looks bad
in the income ledger cost **−12.2 vp** to remove, because what it *prevented* was
invisible. The symmetric risk here is that the one-model gate is load-bearing —
it may be what stops the army walking into ground it cannot take and dying there.
`alive` and `held` are the readouts that would show that, and both are in the
reject rule above.

⚠ This is also the **third** consecutive attempt to fix offence with a reward
change. The two before it (`24v24` spare squads, mixed roles) left offence
unmoved at −50.5 and −42. If this one also leaves offence flat, the honest
conclusion is that offence is not reward-shapeable here and the next move is
architectural, not another term.
