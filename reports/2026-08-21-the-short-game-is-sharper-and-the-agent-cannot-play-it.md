# 2026-08-21 — The short game is sharper, and the agent's edge does not survive it

**Question.** Does shortening the game from 20 battle rounds to the tabletop-legal 5
make it harder to tell whether a change helped — fewer rounds to average the dice
over, and a board nobody can cross? And does longer weapon range open the game up?

**Answer.** No to the first, on both counts: **measurement gets sharper, not
blunter**, and the scripts hold the same ground at either length. No to the second:
**doubling weapon range changes nothing** but the casualty rate. What the work
found instead is that the agent's entire advantage is **time-dependent** — at 20
rounds it is +27.8 clear of the best script, at 5 rounds it is **−5.7 behind** —
and that its offensive deficit is *not* an artefact of the long game, which was the
hypothesis this was meant to test.

All of it is scripts and existing checkpoints. **No training was run.**

## Method

Nine held-out tables, refereed eval config (`revert_unit` + `attrition`) against
`squad_march_take`, seeds 700000+. Scripts at **n=100**; the six `v3.0` checkpoints
at **n=30 with verified top-3 decode**, matching the published table so the
20-round column is a replication check. Scenario scalars set with the
`key=value` overrides added in #227, so both horizons are the *same config* with one
number changed — not two hand-copied files.

The decision criteria below were written **before any number was seen**.

## 1. Five rounds is a viable scenario

| | 20 rounds | 5 rounds |
|---|---|---|
| `hold_deployment` (the floor) | −203.2 ± 5.5 | −31.1 ± 0.7 |
| `random` | −204.8 ± 5.7 | −30.9 ± 0.7 |
| `contest_and_spread` | −46.1 ± 5.6 | −13.9 ± 0.9 |
| `squad_march_shoot` | −39.5 ± 3.8 | −8.9 ± 0.8 |
| `squad_march_deny` | −2.6 ± 3.2 | −1.1 ± 0.8 |
| `squad_march_take` | **−2.7 ± 2.5** | **−0.9 ± 0.7** |

Error bars are across maps, which is the unit this evaluation generalises over.

- **Not degenerate.** The floor sits **30 error bars** below the best script at five
  rounds, against 33 at twenty. Standing still is no closer to winning.
- **Resolution improves.** Best-to-worst script spread divided by its own error bar
  goes **~11 → ~16**. Scores shrink ~3×, error bars shrink ~5×.
- **Ranking is identical**, including the `take`/`deny` tie at the top.
- **Paired sensitivity improves.** `take` vs `deny`, paired on layout: t = −0.56 at
  twenty rounds, **t = −1.89** at five.

### The noise composition inverts

`measure-noise-floor`, `turn_order` pinned so the first-player draw is not booked
under "the scenario":

| | dice (within layout) | scenario (between layouts) |
|---|---|---|
| 20 rounds | **75.3** | 43.5 |
| 5 rounds | 6.6 | **15.2** |

The dice term falls **11×** and the layout term only **3×**. Long games let one
unlucky early exchange compound for fifteen more rounds; short ones do not. The
practical consequence is favourable: at five rounds the dominant noise term is the
one every measurement here already controls, since `measure-maps` and
`measure-paired` hold the layout fixed.

## 2. The agent's advantage is time-dependent, and collapses

Six seeds, n=30, K=3, same nine tables, same opponent:

| | 20 rounds | 5 rounds |
|---|---|---|
| agent `vp_margin` | **+25.1** (sd 8.8) | **−6.6** (sd 1.1) |
| best script | −2.7 | −0.9 |
| **gap** | **+27.8** | **−5.7** |
| agent `held` | 2.05 | 1.83 |
| best script `held` | 2.63 | 2.63 |
| agent win rate | 0.68 | 0.25 |

The 20-round column replicates the published +25.1 / −1.1 / +26.1 within noise, so
the pipeline is measuring what it measured before.

Splitting the gap the way [the offence–defence report](2026-08-21-the-agent-cannot-tell-which-game-it-is-in.md)
does — offence is what the agent scores minus what the script scores, defence is
what the script concedes minus what the agent concedes:

| | offence | defence | sum | actual gap |
|---|---|---|---|---|
| 20 rounds | −59.8 | **+87.4** | +27.7 | +27.8 |
| 5 rounds | −8.2 | **+2.5** | −5.8 | −5.7 |

The decomposition closes to 0.1 vp at both horizons. And the two halves scale
completely differently: the VP scale itself shrinks ~5.3× (222 → 42), **offence
shrinks 7.3× — roughly with the scale — while defence shrinks 35×.** Denial is an
accumulating quantity that needs rounds to bank; taking ground is a fixed fraction
of whatever is on offer.

So the agent does not have a small edge that shrinks. It has an edge made almost
entirely of a term that only exists in a long game.

⚠ **This is a transfer result and must not be read as "the agent would be bad at a
five-round game".** All six seeds were *trained* at 20 rounds; the scripts do not
plan and are horizon-agnostic, so their columns are clean and the agent's is not.
The asymmetry was stated before the measurement, and it biases against the
hypothesis being tested — which makes the refutation below safe, and any claim
about trained five-round performance unsupported.

## 3. The offence deficit is NOT an artefact of the long game — refuted

The hypothesis was that the scripts reach more objectives by walking across the
table, which only twenty rounds affords, so a short game would narrow the gap. Two
predictions, both wrong:

- **"The scripts will lose objectives at five rounds."** They do not:
  `squad_march_take` holds **2.63 at both horizons**, `squad_march_deny` 2.44 →
  2.59, `squad_march_shoot` 2.25 → 2.18.
- **"The agent will lose fewer, so the gap narrows."** The reverse: the agent's
  `held` falls 2.05 → 1.83 while the scripts' does not move.

The premise was also wrong. The plan asserted the scripts "reach `held` ≈ 4.0 by
walking the table over twenty rounds". On this refereed matchup against the
strongest opponent they reach **2.1–2.6**. The 2.9–3.9 figure in `CLAUDE.md` is
against *weaker* opponents, and carrying it to this config was the error.

**The deficit is a genuine allocation failure and survives the scenario change.**

## 4. Uniform weapon range does nothing

Both sides raised together, so the match stays a mirror. Five rounds, n=100:

| range | floor | `shoot` | `take` | `contest` | best−worst | `alive` (take) |
|---|---|---|---|---|---|---|
| 12″ | −31.1 | −8.9 | −0.9 | −13.9 | 13.0 | 0.708 |
| 18″ | −31.5 | −8.6 | −0.6 | −13.4 | 12.8 | 0.619 |
| 24″ | −32.3 | −8.3 | −0.1 | −12.5 | **12.4** | 0.562 |

Every policy moves by under 1.5 vp across a doubling of range, the ordering is
unchanged, and the best-to-worst spread **narrows slightly**. Paired `take` vs
`shoot` is flat: −8.8 / −7.7 / −9.7, t = −5.6 / −4.8 / −5.7. The only thing that
moves is the casualty rate, `alive` 0.708 → 0.562.

This also re-runs, for scripts and on the real tables, the arm that once collapsed
a trained agent to 6.8% win at range 24
([2026-08-05](2026-08-05-stochastic-terrain-and-cover.md)). It did nothing here —
which supports that report's own correction: the collapse was a property of a
policy that managed distance and nothing else, not of the game at longer range.

**Raising a number on both sides at once leaves the mirror intact.** If the game
feels compressed, range is not what is compressing it.

## 5. Two of this work's own criteria were badly designed

Recorded because the pre-registration is worth nothing if only the confirming half
is written up.

- **"More than 20% tied episodes means degenerate" is not a degeneracy test.** It
  fails the *known-good* 20-round scenario too (24 of 100 for `take` vs `deny`).
  VP arrives in steps of 5 under a per-round cap, so ties are common by arithmetic,
  and worse, the count tracks how *similar the two policies are*, not the scenario:
  `take` vs `deny` ties 47 times at five rounds, while `take` vs `shoot` on the same
  config ties 10. Criterion discarded.
- **The travel-distance premise was measured on the wrong matchup**, as above.

The criteria that did work were the floor-to-best separation in error bars, and the
spread-to-error-bar ratio. Both are policy-agnostic and both said *healthy*.

## What this does and does not support

- **Supported.** Five rounds is a measurable scenario — better resolution, same
  ranking, floor 30 error bars down. It is not the reason to avoid the tabletop
  length.
- **Supported.** The agent's advantage is overwhelmingly denial, and denial needs
  rounds. Defence scales 35× with the horizon where offence scales 7×.
- **Supported.** Uniform weapon range is a null on this board, for scripts.
- **Refuted.** That the offensive deficit is an artefact of the 20-round clock.
- **Not established.** Anything about how a policy *trained* at five rounds would
  play. Every agent number here is a transfer result.
- **Not established.** That asymmetric or heterogeneous profiles would open the game
  up. That is the untested version of the range question — the mirror was never
  broken here, only scaled.

## Reproducing

```bash
just measure-maps  <policy|ckpt> configs/evaluation/25v25_maps_take_opponent_refereed.yaml \
                   100 configs/evaluation/maps_heldout 1 "" rounds=5
just measure-paired squad_march_take squad_march_deny <config> 100 700000 rounds=5
just measure-noise-floor <config> 10 10 squad_march_take rounds=5 turn_order=player
```
