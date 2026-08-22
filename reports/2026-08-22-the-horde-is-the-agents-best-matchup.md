# The horde is the agent's best matchup ever measured — because denial finally has a price

**Date:** 2026-08-22
**Config:** `configs/experiments/30v15_fast_horde_vs_elite.yaml` (trained),
scored on `..._refereed.yaml`
**Measurement:** held-out nine, n=30, seeds 700000+, K=3 verified decode, refereed
**Seeds:** 3 × 300 epochs, `ent_coef` 0.003, recording on

## The question

The scripted screen ([preregistration](2026-08-22-asymmetric-armies-preregistration.md))
accepted this scenario as HEALTHY because the script ranking is **side-dependent**:
`contest_and_spread` is 2nd of 4 commanding the elite and **last of 4** commanding
the horde. The question a trained agent answers is whether that side-dependence is
something a policy can exploit, or just a quirk of how the scripts are written.

## Result

The agent beats the best script by **+48.5**, on **9 of 9 tables**, at the largest
effect size recorded in this project.

| | `vp_margin` | win | plr VP | opp VP | `held` | `alive` | `coherent` | `adrift` |
|---|---|---|---|---|---|---|---|---|
| agent s1 | **+29.5** ± 16.3 | 0.74 | 143.7 | 114.1 | 1.72 | 0.352 | 0.933 | 0.59 |
| agent s2 | **+8.2** ± 9.3 | 0.62 | 111.2 | 103.0 | 1.38 | 0.374 | 0.930 | 0.59 |
| agent s3 | **+11.0** ± 9.3 | 0.66 | 124.0 | 113.0 | 1.43 | 0.287 | 0.960 | 0.30 |
| `squad_march_deny` | −32.3 ± 11.5 | 0.38 | 155.1 | 187.4 | 1.00 | 0.125 | 0.922 | 0.45 |
| `squad_march_take` | −46.4 ± 14.2 | 0.34 | 153.9 | 200.3 | 0.78 | 0.086 | 0.927 | 0.40 |
| `squad_march_shoot` | −66.3 ± 15.6 | 0.27 | 144.5 | 210.8 | 0.79 | 0.095 | 0.894 | 0.54 |
| `contest_and_spread` | −90.7 ± 10.9 | 0.20 | 125.3 | 216.0 | 0.60 | 0.077 | 0.891 | 0.62 |

Agent mean **+16.2**, best script **−32.3**, gap **+48.5**.

Two independent estimators, both significant and in agreement:

- **across seeds:** +48.5 ± 6.7, t = 7.26 (df=2, p≈0.018)
- **across tables:** +48.6 ± 9.2, t = 5.30 (df=8), **agent ahead on 9 of 9**
  (per seed: 9/9, 8/9, 8/9)

⚠ **UNPAIRED.** Model counts (30 v 25) and `max_groups` (6 v 5) differ from every
other lineage here, so no control shares an initialisation. The per-table estimator
above is the pairing that *is* available.

## What it does NOT show

⚠ **This is not evidence the agent got better.** Per the standing rule, absolute
score measures the opponent. The agent has never been retrained, retuned or given a
new lever — it is the documented recipe on a new scenario. What moved is the *price
of the thing it was already good at*.

## The mechanism: the same trait, at a different price

Decomposed against `squad_march_deny` (offence = agent's own VP minus the script's;
defence = what the script concedes minus what the agent concedes):

| seed | offence | defence | gap |
|---|---|---|---|
| s1 | −11.4 | **+73.3** | +61.9 |
| s2 | −43.9 | **+84.4** | +40.5 |
| s3 | −31.1 | **+74.4** | +43.3 |

**Offence is negative on all three seeds. Defence carries the entire gap.** That is
the identical decomposition recorded on the mirror scenario, where offence ran −42 to
−71 and defence ran +96 down to zero.

What changed is what denial is worth. The mirror's finding was that *the gap tracks
what the best script concedes*, at r = +0.991. Here the elite opponent concedes
**187.4** to the best script and **103–114** to the agent — so there is ~80 vp of
denial available, and the agent collects it. On `advance_and_shoot`, where both sides
conceded ~130, the same policy trait was worth nothing and the agent finished −75.9
behind on 0 of 9 tables.

**So this row is the +0.991 correlation continuing to hold, not a new capability.**

## The one genuinely new observation: `held` inverted

On every previous scenario the agent held **less** ground than the scripts (1.9–2.1
against 2.9–3.9) while keeping more of its army alive — the hoarding finding. Here it
holds **more** (1.38–1.72 against 0.60–1.00) *and* keeps far more alive (0.287–0.374
against 0.077–0.125).

The hoarding did not go away; it stopped costing ground. Control is a **headcount**,
so a 30-model horde that survives outnumbers 15 elites on every point it reaches.
The same behaviour that surrenders objectives in a 25v25 mirror wins them when the
army is twice the size and the models are individually expendable.

⚠ This makes the horde side a **poor scenario for studying the offence deficit** —
the deficit is present (offence negative on 3/3) but its usual symptom is masked.
Do not read a healthy `held` here as the allocation problem being solved.

## The referee tax on this scenario is enormous — every screen number is void as a bar

| policy | unrefereed (screen) | refereed | tax |
|---|---|---|---|
| `squad_march_take` | −6.9 | −46.4 | **−39.5** |
| `squad_march_deny` | −15.4 | −32.3 | −16.9 |
| `squad_march_shoot` | −33.1 | −66.3 | −33.2 |
| `contest_and_spread` | −47.8 | −90.7 | −42.9 |

Thirty models in six squads of five, moving 12", break formation constantly, and
`revert_unit` + `attrition` prices it. It also **reorders the bar**: unrefereed
`take` leads at −6.9; refereed `deny` leads at −32.3. Any comparison drawn against
the screen's numbers would have been drawn against the wrong policy.

This is the "measure the configuration that ships" rule applying to a *scenario* and
not just a config field. The screen was the right tool for a go/no-go and the wrong
tool for a bar.

## What this does and does not license

- **Does:** the asymmetric scenario is real, non-degenerate, and separates policies
  by more than any scenario measured here. It is a viable place to run arms.
- **Does not:** it says nothing about whether the agent allocates. Offence is still
  negative on 3/3 seeds. The allocation question stays where
  [the spare-squads report](2026-08-22-spare-squads-pose-the-question-the-agent-still-cannot-answer.md)
  left it.
- **Untested:** the **elite side**. Only the horde was trained. Whether an agent
  commanding 15 elites also beats its scripts — and whether it beats them by less,
  which is what the denial-price account predicts — is the obvious next arm and
  costs one training run.

## Reproduce

```
just train configs/experiments/30v15_fast_horde_vs_elite.yaml 300   # x3 seeds
just measure-maps <ckpt>/last.ckpt \
    configs/experiments/30v15_fast_horde_vs_elite_refereed.yaml 30 \
    configs/evaluation/maps_heldout 3
just measure-maps squad_march_deny \
    configs/experiments/30v15_fast_horde_vs_elite_refereed.yaml 30 \
    configs/evaluation/maps_heldout 1
```
