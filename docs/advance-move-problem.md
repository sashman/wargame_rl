# The advance move made the agent worse. Why?

A handoff brief. Everything here is measured; the open question is at the end.

---

## 1. The game, in one paragraph

Two armies of 25 models on a 60×44 inch board, five to twenty rounds. Each round
has a **movement phase** and a **shooting phase**. Objectives are scored by
**headcount** — whoever has more models within ~3" of a marker controls it and
banks victory points that round. Models are physical discs (0.63" radius): a
model may walk *through* a friendly model but may not *stop* on one, and may not
walk through an enemy at all. Models belong to five-model **units** that must
stay in formation (every model within 2" of a squadmate, 9" across the unit).

## 2. The advance move

The rule being implemented (`docs/rules/09-movement-phase.md`):

> **Advance move.** Maximum distance: an advance roll (**one D6**) **plus** the
> unit's Move. After moving, until the end of the turn, only `[RUN AND GUN]`
> weapons may be fired.

No weapon in this project has that ability, so in practice:

**An advance trades the unit's entire shooting phase for up to +6" of reach.**

It is chosen **per unit**, and the roll happens **before** the move is chosen, so
the player knows what reach is on offer when deciding.

### How it was implemented

- A new **action slice appended after the existing ones**: `n_movement_angles ×
  n_advance_speed_bins` = 16 × 3 = 48 new actions. Action space 102 → **150**.
  Appended rather than widening the existing speed bins, because the movement
  slice is decoded angle-major/speed-minor and widening it would renumber every
  existing action.
- Distance is `fraction × (Move + roll)`, so bins span the whole allowance and a
  unit **can** advance a short distance — `Move + D6` is a maximum, not a fixed
  step.
- If **any** model in a unit picks an advance action, the whole unit advances
  and the whole unit loses its shooting. (Otherwise: advance one model onto the
  objective, keep four shooting — an exploit, not a divergence.)
- Both the roll and the "already advanced" flag are in the observation.
- `n_advance_speed_bins: 0` is the default and is a byte-for-byte no-op.

All of this is verified by tests, including that the opponent is blocked from
shooting after advancing, at unit level.

---

## 3. The result

Three seeds, 300 epochs, `ent_coef` 0.003, scored refereed with top-3 joint
decoding on nine held-out tables, n=30, seeds 700000+.

| | vp_margin | `held` | `alive` |
|---|---|---|---|
| **advance arm** s1 / s2 / s3 | **+10.8 / −12.4 / −8.4** (mean **−3.3**) | 1.68 | 0.446 |
| **control** (identical config, no advance) | +26.6 / +14.9 / +28.6 (mean **+23.4**) | 1.94 | 0.535 |
| best script (`squad_march_deny`) | −1.1 | 2.42 | 0.433 |

**Unpaired difference −26.7 ± 8.3, t = −3.20.** The control beat the best script
by +24.5; the arm is 2.2 *behind* it. Every seed is below every control seed.

⚠ The comparison is **unpaired** and cannot be otherwise: adding actions changes
the policy's output layer, so no initialisation is shared. The layouts and seeds
*are* identical, and the two configs are verified to be the same game for a
policy that never advances (scripts score to the same decimal on both).

---

## 4. What has been ruled out

### It is not freezing

The agent piles models onto the same objective and they block each other. That
looked like the answer:

| | frozen orders | distance delivered |
|---|---|---|
| scripted policies | ~11% | **91.8%** |
| advance arm | 18–28% | **70–77%** |

**But the control agent freezes 26.3% and delivers 76.4%** — indistinguishable
from the arm. Trained agents freeze at that rate *because they stack*, with or
without the advance move. The scripts were never the right comparison. **This
explanation was proposed and then refuted; do not re-run it.**

### It is not that the agent ignores the feature

Advance actions are 8.1% / 23.1% / 11.8% of orders for s1 / s2 / s3.

### It is not that the weights are broken — and this SPLITS the loss

Forbidding advance actions **at play time, on the same trained weights, on the
same layouts and seeds** (n=10 per table, held-out nine, K=3):

| seed | as trained | advance forbidden | delta |
|---|---|---|---|
| s1 | +9.1 | +19.9 | **+10.9** |
| s2 | +1.8 | +5.7 | **+3.9** |
| s3 | −3.1 | +7.8 | **+10.8** |
| **mean** | **+2.6** | **+11.1** | **+8.5** |

Positive on 3 of 3. **Using the option costs ~8.5 vp.** The network is not
broken — the same weights play materially better when simply denied the choice.

**But it recovers only about a third of the gap.** The control is **+23.4**; the
arm with advance forbidden is **+11.1**. So roughly:

    ~8.5 vp   the agent choosing a bad option at play time   (recoverable)
    ~12 vp    the policy it learned is simply worse           (not recoverable)

⚠ The n differs between these numbers — the forbid probe is n=10 per table, the
headline scores are n=30 — so the split is approximate. The *direction* is not
in doubt; the sizes are ±a few vp.

**Both candidate explanations below are therefore true, and roughly evenly.**

### The usage/damage correlation does NOT survive

| seed | advance usage | vp as trained | gain from forbidding |
|---|---|---|---|
| s1 | 8.1% | +9.1 | +10.9 |
| s3 | 11.8% | −3.1 | +10.8 |
| s2 | **23.1%** | +1.8 | **+3.9** |

An earlier note in this project read the usage ordering as monotone in the
damage. With the full probe it is not: **s2 advances nearly three times as often
as s1 and loses the least by giving it up.** Whatever makes advancing costly, it
is not simply "more advancing, more harm". Treat that reading as retracted.

## 5. The two candidate explanations

### (a) The trade is genuinely bad in this scenario

Objectives are scored by headcount every round, and the agent's standing
weakness is documented: it is a strong *defensive* player whose offence never
improves. Giving up a unit's entire shooting phase for reach it cannot convert
into held ground may simply be a losing trade here — the extra 3.5" of expected
reach does not get a unit onto a *new* objective, it just gets it deeper into
the crowd it is already in.

If so, the agent is **correctly** learning to use a bad option and being punished
for the exploration — and the mechanic is fine, the scenario just does not reward
it.

### (b) It is an exploration / capacity cost, not a mechanic cost

48 of 150 actions — **32% of the action space** — are new. At a fixed 300 epochs
the policy must cover a third more action space, and every advance it samples
while learning costs a unit's shooting. The arm may simply be *under-trained*
relative to its control rather than worse.

The project's own rule is that a marginal 300-epoch result means "run it
longer", not "rejected".

**Both are true.** The forbid probe splits the loss: ~8.5 vp is (a), the agent
choosing a bad option, and the remaining ~12 vp is (b), a worse learned policy.
What is *not* settled is whether (b) is fixable by training longer or is a
permanent cost of the larger action space.

---

## 6. Why this is hard to settle

- **No pairing.** Adding actions changes the output head, so the two arms cannot
  share an initialisation. The usual estimator (worth roughly an order of
  magnitude here) is unavailable.
- **Unpaired seed spread on this config is ~11 vp**, so three seeds resolve only
  large effects. −26.7 clears that bar; anything subtler would not.
- **1000 epochs costs real GPU time** and would only answer (b).

## 7. Suggested next steps, cheapest first

1. ~~Finish the forbid probe~~ **DONE** — see above. It recovers ~8.5 of ~26.7,
   so the loss is roughly one third "bad choice at play" and two thirds "worse
   policy".
2. **Anneal advance during early training** — if the loss is
   exploration cost, starting with advance masked and unmasking later should
   recover most of it, cheaply.
3. **Check whether an advance ever reaches a NEW objective.** The premise of the
   feature is reach. Measure the distance from each advancing unit to its nearest
   *unheld* objective before and after. If advances never close that gap, (a) is
   established and the mechanic is fine but the scenario cannot use it.
4. **Only then** consider 1000 epochs.

## 8. Context a newcomer will need

- **Three reward-shaping attempts to fix the agent's offence have all failed**
  (−50.5, −42, −71.5); the last made offence actively worse. That is why the
  project moved to building game mechanics instead.
- **The agent hoards**: it finishes with far more of its army alive than the
  scripts while holding fewer objectives. Everything above happens against that
  background.
- **Scores are only comparable within a row.** The opponent, the referee
  settings and the decode all change the number; a `vp_margin` without them is
  meaningless.
- Read `CLAUDE.md` § *How to measure here* before running anything.
