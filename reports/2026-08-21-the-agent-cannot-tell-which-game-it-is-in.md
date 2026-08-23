# 2026-08-21 — The agent cannot tell which game it is in

**Question.** Is the `contest_and_spread` loss real, and does the agent need
retraining on `25v25_maps_coherency`?

**Answer.** The loss is real, unchanged at six seeds and still not
statistically settled — and it is not the interesting result. Adding a fifth
opponent found a **−75.9 vp deficit on 0 of 9 tables**, the largest and most
significant ever measured here, and it has the same cause. The agent's defence
is excellent and its offence is capped, so its lead is worth whatever denial is
worth against that particular opponent. `25v25_maps_coherency` does not need
retraining: it is the same scenario against the weakest opponent, which is
exactly the case the agent is worst at.

## What was measured

Three more seeds of the documented recipe (`configs/golden/25v25_maps_two_mode.yaml`,
`ent_coef` 0.003, 300 epochs), taking the lineage from three seeds to six. Launch
arguments were diffed against a seed-1–3 run before pooling: identical but for
`--seed`. All six scored from `last.ckpt`, written at `on_train_end`, so every
seed is compared at the same epoch.

Scored on the **refereed** eval configs, nine held-out tables, n=30 at seeds
700000+, verified top-3 joint decode. Scripts re-measured against each opponent,
because swapping the opponent voids every baseline on a config.

A fifth eval config was added — `configs/evaluation/25v25_maps_vs_advance_and_shoot.yaml`
— differing from the other four in exactly one parsed field, which
`tests/test_map_config_pairs.py` now pins.

| opponent | agent | best script | gap | t | sign |
|---|---|---|---|---|---|
| `squad_march_deny` | **+26.4** | −8.9 (`take`) | **+35.4** | 4.49 | **9/9** |
| `squad_march_take` | **+25.1** | −1.1 (`deny`) | **+26.1** | 3.32 | 8/9 |
| `squad_march_shoot` | **+39.2** | +23.0 (`take`) | +16.2 | 1.64 | 7/9 |
| `contest_and_spread` | +20.8 | **+30.2** (`take`) | **−9.5** | −1.18 | 4/9 |
| `advance_and_shoot` | +61.4 | **+137.2** (`deny`) | **−75.9** | **−7.12** | **0/9** |

Coherency 0.938–0.955 against a scripted 0.867–0.908, in every row including
both losses.

## Six seeds replicated the table rather than changing it

Every pre-existing row moved by under 2 vp. `deny` firmed from 8/9 to **9/9**,
`shoot` remains unsettled at t=1.64, and `contest_and_spread` went −8.4 → −9.5.

What six seeds *did* settle is that the loss is not a one-seed artefact. The
previous note recorded "one seed carries it (+30.9 / +8.1 / +26.4)". At six the
band is +8.1 / +11.2 / +22.9 / +25.1 / +26.4 / +30.9 — **five of the six behind
the script's +30.2**, and the sixth ahead by 0.7. Seed sd is 9.1 against the
three-seed 12.1 — still large next to a 9.5 vp effect, which is why t is
-1.18 rather than the sign being in doubt. What changed is that no single seed
carries the result any more.

## One trait explains all five rows

Split each gap into **offence** (what the agent scores minus what the script
scores) and **defence** (what the script concedes minus what the agent
concedes):

| opponent | script concedes | offence | defence | gap |
|---|---|---|---|---|
| `squad_march_deny` | 223.5 | −60.8 | **+96.1** | +35.3 |
| `squad_march_take` | 219.6 | −56.3 | **+82.3** | +26.1 |
| `squad_march_shoot` | 197.1 | −42.0 | +58.2 | +16.2 |
| `contest_and_spread` | 184.2 | −48.0 | +38.5 | −9.5 |
| `advance_and_shoot` | 128.0 | −71.3 | **−4.5** | −75.8 |

**Offence is flat** at −42 to −71 across every opponent — the agent always
scores 40 to 70 fewer points than the best script. **Defence collapses** from
+96 to zero. The gap tracks what the best script concedes at **r = +0.991**
(n=5).

The mechanism is visible in `held`: the agent holds **1.89 to 2.11** objectives
against every opponent alike, while the scripts reach **2.93 to 3.86** against
the weak ones. Against `advance_and_shoot` both sides concede ~130 — the
opponent is too weak to score whoever it plays — so the agent's discipline earns
nothing and only the deficit remains.

This is the same failure in every row, priced differently. It is not a
`contest_and_spread` problem.

## The abandonment behind it is opponent-invariant

`just measure-objective-split` at K=3 (the flag this work added — it previously
decoded argmax, describing a player that does not ship) on three seeds against
two opponents:

- The agent puts **exactly 0.00 models on the bottom three objectives** in every
  seed and every matchup.
- Abandonment is **0.645 against `contest_and_spread` and 0.653 against
  `squad_march_take`** — invariant to the opponent.
- The redistribution ceiling is **+1.53 to +2.50 objectives** everywhere: moving
  models surplus to `opponent_count + 1` onto the cheapest lost objective would
  roughly double `held`.

⚠ Those runs use the config's own map pool — the 36 **training** tables — so
read them as a within-agent contrast across opponents, not as a held-out result.

`objective_hold.crowding_exponent`, the lever built to price exactly this, is
**already at 1.0 in this config**, and the config header already records that it
does not fix it here.

## Why `contest_and_spread` is the mild version

`contest_and_spread` allocates `count + 1` models to the cheapest objectives and
**skips any it cannot win**. Against a script that masses 4.73 and 3.00 on its
top two points, those are conceded outright. The agent masses 3.60 and 2.23 —
concentrated enough to abandon most of the board, not concentrated enough to
deter a counter-allocator — so more of its points read as winnable and get
contested. `advance_and_shoot` does not allocate at all, so it concedes to
anyone who turns up, which is why the same trait costs eight times as much
there.

## `25v25_maps_coherency` is retired as a training config

Comments stripped, it is byte-identical to `25v25_maps_two_mode` except for
`config_name` and the opponent. It is the same scenario against
`advance_and_shoot` — the matchup the agent is *worst* at relative to a script.
Training three seeds there would have bought an agent number against the weaker
of two opponents. Its agent column now comes from scoring the `two_mode` lineage
on the refereed eval config instead, at no GPU cost.

`tests/test_map_config_pairs.py` pins both this and the single-variable property
of the eval family, and was checked for sensitivity before being trusted: the
same helper reports `{config_name, coherency}` for the `repair` variant and
`{config_name, observe_unit_centroid}` for the centroid one.

## What this does and does not support

- **Supported.** The agent is a better *defensive* player than any script, by a
  wide and replicated margin, and holds formation better than any of them in
  every matchup. Against the three opponents that score heavily it wins by 16 to
  35 points.
- **Supported.** Its offensive deficit is real, large, constant across
  opponents, and now the binding constraint. It is not caused by the opponent.
- **Not supported.** That the `contest_and_spread` matchup is a distinct
  weakness. t=−1.18 on 4/9 is not a settled result on its own; it is only
  interpretable as the mild end of the `advance_and_shoot` finding.
- **Not established.** That re-allocation would actually work. The
  redistribution ceiling is deliberately optimistic — no travel time, no return
  fire — so a large ceiling does not rule re-allocation *in*.

## Method notes

- **Absolute score measures the opponent.** The agent scores its *highest*
  (+61.4) in the matchup it loses worst. Only the same-row comparison means
  anything.
- **The analysis parser was validated before use** — run against the old
  three-seed log it reproduced every published figure exactly (+23.7/t=3.14,
  +17.2/t=1.78, +33.4/t=4.00, −8.4/t=−1.11), so anything new it printed was a
  result rather than a parsing difference.
- **The scoring was chained to training completion in a detached process.** Two
  in-session monitors were torn down at session boundaries; the detached chain
  was unaffected.
