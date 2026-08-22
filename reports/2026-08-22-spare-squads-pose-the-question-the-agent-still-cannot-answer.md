# Spare squads pose the allocation question. The agent still cannot answer it.

**2026-08-22.** Every score below: held-out nine, n=100, seeds 700000+, and the
decode named in the row. Scripts are K=1, agents K=3 (`verify_moves` on).

## The question

Six `v3.0` seeds beat the best script by a wide margin on `25v25_maps_two_mode`,
and **all of it is denial**: offence about −56, defence about +82. `held` sits
near 2.05 against the scripts' 2.63. Three explanations had already been refuted
without training -- the twenty-round clock, weapon range, and the travel-reward
gate.

The live one was structural: with **five squads against five or six
objectives there is never a spare squad**, so the scenario may never ask the
question at all. `squad_march_take` and `squad_march_deny` differ *only* in what
a squad with nowhere assigned does, which makes their difference the direct
instrument.

## The instrument says the scenario was NOT asking

`take` − `deny`, paired per episode, n=100 per cell, K=1, three independent
layout sets. Sign convention: positive means `take` ahead.

| seed base | control, 5 squads of 5 | **8 squads of 3** |
|---|---|---|
| 700000 | +5.7 | +20.2 |
| 800000 | **−9.2** | +11.6 |
| 900000 | +9.8 | +16.2 |
| **mean** | **+2.1** (sd 10.2) | **+16.0** (sd 4.3) |
| episodes identical | 25 / 25 / 31 | **2 / 5 / 4** |

**On five squads the difference changes sign across layout sets** -- it is
indistinguishable from zero, and a single seed set reading +5.7 would have said
otherwise. On eight squads it is +16.0 and positive in all three. Between
configs that is +13.9, t ≈ 2.2.

The identical-episode count is the mechanical half of the same fact and is not a
statistic at all: **two to five games in a hundred play out identically with
spare squads, against twenty-five to thirty-one without.**

### It is the squad COUNT. Roles are a measured null.

Two arms were built first and both were wrong, in an instructive way.
`25v25_maps_mixed_roles` gave five squads five different guns; the 40-model
version gave eight squads the same. Both showed `take` and `deny` tying at
t ≈ −0.03 and −0.01, which looked like confirmation that nothing poses the
question -- and was an artefact. Those configs fire **45 shots a round against
the control's 25**, at higher Strength, with 15 of 25 models outranging it two to
three times. `alive` collapses 0.432 → 0.203 → **0.135**. An army of five
survivors cannot spread over six objectives, so the tie was attrition.

`25v25_maps_mixed_roles_matched` repeats the roles at **exactly 25 shots**, AP 1
throughout, each role a trade rather than a gain. Its paired difference is
**+5.7 ± 7.9** -- the control's number to one decimal. **Roles change nothing.**
This was pre-registered as the confound before any of it was measured.

## Trained on the config that does ask, the agent still does not answer

`configs/experiments/24v24_maps_spare_squads.yaml` is the golden config with
**only** the squad structure changed -- same rifle, same maps, same reward,
verified to differ in `models`, `opponent_models`, `max_groups` and the two
counts and nothing else. Three seeds, 300 epochs, `ent_coef` 0.003. Scored on the
refereed twin at epoch 299 for all three (`last.ckpt`; the highest `ppo-NNN` is
epoch **145** for s1 against **292** for s3, so scoring those would have compared
different epochs).

| | vp_margin | plr VP | opp VP | held | alive | coherent | adrift |
|---|---|---|---|---|---|---|---|
| `squad_march_take` (K=1) | **+6.0** ± 3.0 | 213.9 | 207.9 | **2.80** | 0.309 | 0.945 | 0.36 |
| `squad_march_deny` (K=1) | −6.9 ± 3.2 | 205.6 | 212.4 | 2.41 | 0.274 | 0.941 | 0.39 |
| `squad_march_shoot` (K=1) | −76.2 ± 5.6 | 179.8 | 256.0 | 1.84 | 0.392 | 0.877 | 0.78 |
| agent s1 (K=3) | +4.5 ± 7.6 | 153.5 | 149.0 | 2.02 | 0.519 | 0.964 | 0.31 |
| agent s2 (K=3) | +17.5 ± 8.8 | 171.5 | 154.0 | 2.32 | 0.536 | 0.967 | 0.29 |
| agent s3 (K=3) | +23.4 ± 8.3 | 165.2 | 141.8 | 2.16 | 0.532 | 0.964 | 0.33 |
| **agent mean** | **+15.1** ± 5.6 | 163.4 | 148.3 | **2.17** | **0.529** | 0.965 | 0.31 |

**Gap +9.1, t ≈ 1.44, three seeds, UNPAIRED** -- `max_groups` 5 → 8 is a
tensor-shape change, so the paired estimator is unavailable, which is exactly the
class the project already flags as least measurable. Per-seed spread is 4.5 to
23.4, sd 9.7, consistent with the ~11 vp documented for this family.

Decomposed: **offence −50.5, defence +59.6.** Against roughly −56 / +82 on the
previous scenario. **The offence term did not move** -- the change is inside one
seed's worth of noise -- and the win is still entirely denial.

`held` is 2.17 against 2.80, a shortfall of **0.63**. Before this arm it was 2.05
against 2.63, a shortfall of **0.58**. Fixing the scenario left it where it was.

## What the agent is actually doing, in one number

**It finishes with 52.9% of its army alive against the scripts' 27.4–30.9%, and
holds fewer objectives.** Nearly twice the survivors, less ground. It is not
losing a fight for the points; it is declining to have one.

`just measure-vp-cap` shows the same thing per step, on the arm config:

| | decode | at 3+ objectives | above the cap (pays 0) | VP/step |
|---|---|---|---|---|
| `squad_march_take` | K=1 | **49.1%** | 21.4% | 11.38 |
| `squad_march_deny` | K=1 | 45.3% | 11.8% | 11.26 |
| agent s1 / s2 / s3 | K=3 | 28.3 / 32.5 / 26.7% | 4.9 / 5.4 / 2.3% | 9.75 / 10.32 / 9.95 |
| **agent mean** | K=3 | **29.2%** | 4.2% | 10.01 |

It did improve -- the same measurement on the previous scenario put the `v3.0`
agent at **22.3%**. But the script on this config is at 49.1%, so most of the
gap survives.

## The cap taxes the SCRIPTS, not the agent

The mission pays `min(15, controlled * 5)`, so the **fourth** objective is worth
nothing while the tables carry five or six. Measured on
`25v25_maps_two_mode`, n=20:

    squad_march_take  (K=1)  55.6% of steps at 3+, 23.9% above -- 10.1% discarded
    squad_march_deny  (K=1)  46.3%                ,  6.6%      --  2.9% discarded
    the v3.0 agent    (K=3)  22.3%                ,  2.0%      --  1.1% discarded

So the cap penalises **`take` hardest**, and `take`'s surplus-grabbing is
precisely what distinguishes it from `deny` -- the cap compresses the very
difference used as the instrument, and `take` wins anyway. The agent never
reaches the cap, so **its offence shortfall is real and almost entirely
payable**: scaling its VP rate against `take`'s puts it near 57 vp, which
independently reproduces the −56.3 offence term measured a different way.

## The finding

**The scenario genuinely was not posing an allocation problem, spare squads fix
that, and the agent still cannot allocate.** This is the alternative the goal
named in advance, and it is what landed. The offence framing survives; the
scenario explanation does not.

Do not read this as "the arm failed". It answered its question, and it
disqualified two others on the way.

## Do not re-run

- **Mixed weapon profiles as an allocation lever.** Firepower-matched, they
  reproduce the control's paired difference to one decimal. The unmatched
  versions only measured their own lethality.
- **A single layout set for a script-vs-script difference.** The control's
  `take` − `deny` reads +5.7, −9.2 and +9.8 on three; one of them alone supports
  any conclusion you like.

## Open, and what it points at

The agent hoards. It keeps 53% of its army alive and holds 0.63 fewer objectives
than a script that keeps 31%. Nothing measured so far explains *why* survival is
being preferred to ground when `vp_gain` is net and the cap is nowhere near
binding for it. That -- not the scenario, and not the profiles -- is where the
offence deficit lives.
