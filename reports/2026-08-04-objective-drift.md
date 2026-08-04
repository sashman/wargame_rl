# Objective drift: the agent reaches objectives, then abandons them

**Date:** 2026-08-04
**Method:** per-step `at_objective` traces reconstructed from recorded match event logs
**Data:** 5 recorded episodes, 4v4 config, `recordings/ppo721_ep{0..4}.jsonl`

## Question

Aggregate metrics showed `objective_coverage` and `closest_objective/progress` improving
while the fraction-based success criteria stayed flat. What is the agent actually doing
inside an episode?

## Method

Event logs were decoded with `JsonMatchCodec` and replayed through `ReplayController`,
which reconstructs a full `GameStateSnapshot` per step from delta-encoded events. For each
step the number of player models with any `at_objective` flag set was counted.

```python
log = JsonMatchCodec().decode(open(path, "rb").read())
snaps = ReplayController(log).iter_snapshots()
counts = [sum(1 for m in s.player_models if any(m.at_objective)) for s in snaps]
```

## Result

Models on objectives, per step, out of 4:

```
ep0  00000111111221133333333333333112211111111   peak 3 -> final 1
ep1  (peak 3 -> final 2)
ep2  00011001122333311000000003322110011222211   peak 3 -> final 1
ep3  00011111122112222002222112222224            peak 4 -> final 4  (terminated on success)
ep4  (peak 4 -> final 4)                          (terminated on success)
```

Per-episode summary, with each model's final distance to its nearest objective
(objective radius = 3):

| Episode | Steps | Peak on-obj | Final on-obj | Final distances |
|---|---|---|---|---|
| ep0 | 40 | 3/4 | 1/4 | 2.8, 5.1, 5.4, 6.7 |
| ep1 | 40 | 3/4 | 2/4 | 2.2, 3.0, 5.0, 6.3 |
| ep2 | 40 | 3/4 | 1/4 | 2.0, 5.0, 5.4, 5.8 |
| ep3 | 31 | 4/4 | 4/4 | 2.2, 2.2, 2.2, 3.0 |
| ep4 | 31 | 4/4 | 4/4 | 1.0, 1.0, 1.0, 2.2 |

**ep2 is the clearest case:** 3 models on objectives at step ~15, then **zero for eight
consecutive steps**, then back to 3, then down to 1 at the end. That is active abandonment
and re-approach, not positional noise.

Episodes that ran the full 40 steps ended with 1–2 of 4 on objectives, at distances of
5–6 against a radius of 3 — outside, having previously been inside. The two episodes
ending at 4/4 (ep3, ep4) are the two that **terminated early on success**: the episode
stopped at the good moment, so the final state is favourable by construction.

## Interpretation

`closest_objective` with `progress_scale` computes `10 x distance_closed`. This is a
**potential function**: over any trajectory that leaves an objective and returns, the
contributions cancel to exactly zero. The policy is therefore mathematically indifferent
between holding a point and wandering off and back.

`objective_coverage` pays only when out-counting the opponent at an objective and
saturates at roughly two models a point, so it does not oppose the remaining models
drifting.

At the time these episodes were recorded, **no reward term paid for occupancy per step**,
so no gradient held models in place. This is the same gap recorded as
[D3](2026-08-04-mechanism-defects.md#d3--the-gated-quantity-had-no-dense-reward-behind-it).

**Status of this interpretation:** the drift is *measured*; the potential-function
explanation is *inferred* from the calculator's definition. It has not been tested by
ablating `progress_scale`.

## Consequences

**Success is judged at the final step**, so drift makes the criteria under-report
capability by roughly the peak-to-final ratio:

| Scale | Peak on objectives | Final on objectives |
|---|---|---|
| 4v4 (this data) | 3/4 | 1/4 |
| 25v25 (earlier measurement) | 6–8 / 25 | 2.5–4 / 25 |

Both give a ratio near 3x. This directly explains why gates calibrated on final-step
measurements sat far below observable behaviour, and why an early sizing attempt that used
*peak* occupancy produced thresholds that measured 0% in practice.

## Secondary findings

**Terrain and shooting were inert in this config.** All five episodes ended 4–4 alive with
`weapons: []` on every model. An empty weapon list means a model cannot shoot; terrain
blocks only line-of-sight; line-of-sight only matters for shooting. Nothing about terrain
was exercised.

**The analyser's "high idle rate" warning is a false positive.** It fired on all five
episodes at 48–52%. `_analyze_movement` (`analysis.py:291-295`) counts every `Stay` as
idle regardless of battle phase. With `skip_phases: [command, charge, fight]`, exactly half
of all steps are the shooting phase, where `Stay` is the only legal action. The ~50%
figure is structural, not behavioural. Already noted in `CLAUDE.md`.

## Actions taken

- `models_at_objectives` weight raised to 1.0 (`approach_objectives`) and 1.5
  (`mass_on_objectives`), above the saturating `objective_coverage`. This is the only term
  that makes holding strictly better than oscillating.
- `terminate_on_success` fixed to consult the configured criteria — but **not enabled**,
  since it would inflate `success_rate` ~3x by construction. See
  [mechanism defects](2026-08-04-mechanism-defects.md#latent-issue-fixed-but-deliberately-not-enabled).

## Limitations

**These are 4v4 recordings, not 25v25.** No 25v25 snapshots existed: `--record-events` was
passed to five 25v25 runs and wrote nothing, because the log was serialised only after
`trainer.fit()` returned and every run was stopped mid-flight. This is now fixed — the log
is written at each epoch start — so future runs produce this data as a side effect.

The 25v25 peak-vs-final figures quoted above come from earlier aggregate measurements, not
from step traces, and corroborate the ratio without replacing a direct trace.

**Five episodes, one checkpoint, one config.** Enough to establish that drift occurs and to
identify a plausible mechanism; not enough to quantify its frequency across policies.
