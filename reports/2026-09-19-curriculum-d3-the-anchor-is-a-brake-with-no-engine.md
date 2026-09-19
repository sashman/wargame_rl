# Curriculum rung D3: anchored PPO holds a clone with two thirds of the rung still to gain, and takes no step toward it — the D-route is imitation only

**Verdict first.** Two arms of anchored PPO from a clone of plain
`squad_march_take` on C3b (#340; the rung where the clone reads 0.361
and the escort 0.978), three seeds each, 122,880 rounds, read greedy at
n=180 on 700000+:

| arm | anchor | verdict | success at the end | paired v the clone | ordering (kill before arrival / blockers wiped) |
|---|---|---|---|---|---|
| **D3a** | coef 10, target 0.03 (D2c's) | **HOLDS; IMPROVES not shown** | 0.339 / 0.356 / 0.333 | −0.022 ± 0.026 / −0.006 ± 0.026 / −0.028 ± 0.024 | 0.79–0.83 / 0.64–0.67 (the clone 0.82 / 0.64) |
| **D3b** | coef 1, target 0.10 (A5g's) | **HOLDS; IMPROVES not shown** | 0.328 / 0.344 / 0.356 | −0.033 ± 0.024 / −0.017 ± 0.023 / −0.006 ± 0.026 | 0.78–0.84 / 0.63–0.64 |

Twenty-four reads over 20k–122,880 rounds, every one within a binomial
SE of the clone's 0.361 and none above it; the by-phase census the
clone's on every column; sampled play the same policy as greedy. The
question this rung was built to separate — is the anchor a brake with
no engine, or was there nothing left to gain on D2 and A5 — is
answered: **with 0.62 of success to gain and the teacher's plan as the
reward's own signal, anchored PPO on this trainer moved nothing at
either coefficient.** The D-route on the per-model facade is
imitation: what the anchor holds is what the clone was, and what the
clone was is what the rung reads. The goal set on 2026-09-19 is
answered in the negative on its first item, and its third (decompose
the conjunction on A5-points) is not run, by the goal's own ordering.

## Provenance

| field | value |
|---|---|
| date | 2026-09-19: clone fitted 01:13–02:19, read 02:42; arms launched 02:42, finished 07:25–07:35; final reads 07:35–07:55 |
| GPU / no-GPU | GPU (RTX 4090), six D3 trainers beside E1's six per-model runs |
| seeds | 1 / 2 / 3 per arm, every seed from the same clone (fit seed 0) — three seeds off one warm start are that clone's band, not three samples |
| n | 180 at seed base 700000 (every read, the ordering census); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 1,200 games at 800000+ for the clone, the last 240 held out |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the clone on identical seeds (success, turns); per seed across arms on the same layouts |
| comparator | the clone `checkpoints/per_model/clones/take-c3b-1200-s0.pt` at n=180: success 0.361, held 2.51, alive 0.531, vp 60.8 ± 3.3, coherent 0.870, first kill 7.0, kill before arrival 0.82, blockers wiped 0.64 (its teacher `squad_march_take` 0.339 / 2.46 / 0.525 / 0.79 / 0.62); `scripted_escort` 0.978 / 3.97 / 0.965 / 62.6 ± 1.6 / 0.839 / 1.00 / 0.99 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, four wounds, eight attacks at range 12 |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, the default `mean` credit, from the clone's cold critic; no extension |
| code revision | pre-registered at `10eeb1d`; runs on `d8a1059`; the branch `feature/per-model-actor-credit`, PR #383 |
| checkpoint | `last.pt` at 122,880 and every 512 rounds under `checkpoints/per_model/per-model-c3b-2026-09-19-02-42-*-s{1,2,3}d3{a,b}` |
| coherency | greedy at 700000+: D3a 0.895 / 0.877 / 0.884, D3b 0.859 / 0.880 / 0.879 (the clone 0.870, the escort 0.839, the teacher 0.934); sampled at 900000+ 0.81–0.85 |
| Wandb | `curriculum-d3`: D3a `vjgk8f5w` / `i6pri90x` / `kjppfvfs`; D3b `vv1w0k0f` / `sfk1u00s` / `k3u5dz2m` |
| pre-registration | `reports/2026-09-19-curriculum-D3-preregistration.md` (+3 amendments) |

## The read

### The clone is its teacher, with the escort's plan out of reach

`squad_march_take` cloned into the set network from 1,200 games on
C3b (D1b's recipe) reads **0.361** against its teacher's 0.339 on the
same 180 seeds; held 2.51 against 2.46, alive 0.531 against 0.525;
held-out per-head match declaration 0.956, displacement 0.807,
unit-pointer 1.000, joint 0.569 (the opening order, at chance as D1
recorded). The census says what the gap to the escort is: the teacher's
armed squad does fire first in most episodes (kill before arrival 0.79)
and does not finish the job (blockers wiped 0.62), and its unarmed
squads walk in while the blockers live (alive 0.53). The escort's
order is "wipe, then walk"; the teacher's is "walk, shooting on the
way". That order is the whole of the 0.62 of success between them, and
it is the order the arm from C2 half-learned from reward on C3b (wipes
the blockers in 75–79% of episodes before stopping).

### Both anchors hold the clone, at every read

| rounds | D3a s1 / s2 / s3 | D3b s1 / s2 / s3 |
|---|---|---|
| the clone | 0.361 | 0.361 |
| 20,480 | 0.372 / 0.339 / 0.350 | 0.322 / 0.383 / 0.406 |
| 40,960 | 0.367 / 0.361 / 0.367 | 0.361 / 0.367 / 0.372 |
| 61,440 | 0.378 / 0.367 / 0.389 | 0.328 / 0.350 / 0.344 |
| **122,880** | **0.339 / 0.356 / 0.333** | **0.328 / 0.344 / 0.356** |

Twenty-four reads in 0.32–0.41, the clone's binomial band (0.361 ±
0.036 at n=180), no trend on either arm, and at the end no seed above
the start. The census at the end:

| row | success | held of 4 | alive | first kill (turn) | first unarmed arrival | kill before arrival | blockers wiped |
|---|---|---|---|---|---|---|---|
| the clone | 0.361 | 2.51 | 0.531 | 7.0 (94%) | 8.9 (62%) | 0.82 | 0.64 |
| D3a s1 / s2 / s3 | 0.339 / 0.356 / 0.333 | 2.44 / 2.56 / 2.49 | 0.519 / 0.534 / 0.524 | 7.0 / 7.0 / 6.8 | 8.6 / 8.7 / 8.9 | 0.79 / 0.81 / 0.83 | 0.64 / 0.65 / 0.67 |
| D3b s1 / s2 / s3 | 0.328 / 0.344 / 0.356 | 2.47 / 2.54 / 2.51 | 0.522 / 0.535 / 0.519 | 7.1 / 6.9 / 7.1 | 8.8 / 8.6 / 8.8 | 0.84 / 0.78 / 0.84 | 0.64 / 0.64 / 0.63 |

Not a column moved: the first kill at the same turn, the unarmed
arrival at the same turn, the blockers wiped in the same share of
episodes, the same third of the army dead. Sampled play (n=100 on
900000+, paired against greedy) is the same policy: −3.4 to +1.1 ± 2.5
vp, held within 0.06, coherency 0.81–0.85 against greedy's 0.86–0.90 —
no do-nothing fingerprint (stationary 0.45–0.48, the clone's 0.44).

**The letter of the clauses**, the paired per-episode success difference
against the clone on the same 180 seeds:

| seed | D3a: diff, SE, t, episodes differing | D3b: diff, SE, t, episodes differing |
|---|---|---|
| s1 | −0.022 ± 0.026, t −0.85, 22 of 180 | −0.033 ± 0.024, t −1.42, 18 of 180 |
| s2 | −0.006 ± 0.026, t −0.22, 21 | −0.017 ± 0.023, t −0.73, 17 |
| s3 | −0.028 ± 0.024, t −1.15, 19 | −0.006 ± 0.026, t −0.22, 21 |

Every seed within two paired SE of the clone (HOLDS), none above it
(IMPROVES not shown), none below by two SE (DESTROYS not triggered).
Only 17–22 episodes in 180 end differently from the clone at all: after
122,880 rounds of reward the policy is the clone on nine tenths of the
episodes and a coin-flip on the rest.

### What the two coefficients say

D2c's setting (10 / 0.03) and A5g's (1 / 0.10) differ by an order of
magnitude in the coefficient and threefold in the target, and they read
the same. The looser anchor was the one the pre-registration guessed
might "find the order partially or slide toward D2 plain's collapse";
it did neither. A5h (0.1 / 0.3) lost the A5 clone slowly and D2 plain
(no anchor) lost the escort clone in 2,560 rounds, so the coefficient
has a floor below which the clone is destroyed and, above it, a plateau
on which nothing is gained — there is no setting on file at which
reward both keeps the plan and adds to it.

## What it says

- **The D-route on the per-model facade is imitation only, and the
  ladder's D and start-axis rows should be read as such.** D2c held the
  escort clone (0.95 → 0.94–0.96), A5f and A5i held the bar's clones
  (0.93 → 0.93; 0.96 → 0.96–0.98), and each could be read as "nothing
  left to gain". D3 removes that reading: from 0.361 with 0.62 to gain
  and the gap a plan the same reward has taught from scratch, the
  anchor held the clone at every read and every coefficient that holds.
  A rung passed from a clone is passed by the clone.
- **The anchor's coefficient is not the IMPROVES lever, and no more of
  them should be run.** Four settings across three rungs: 10 and 1
  hold, 0.1 loses slowly, 0 destroys. Between "held" and "lost" there
  is no measured "lifted". Whatever lifts a clone here is not the
  strength of the tether to it.
- **What is left for reward on this trainer is the from-scratch route,
  and the record says what it can and cannot learn there**: a walk to
  the bar's speed (A0–A4x), a mask-free engagement rule (C1), the first
  half of an ordered plan (C3b's 0.20), and not a conjunction over five
  or six points (A5-points 0.03–0.33) nor the second half of the plan.
  The goal's third item — decompose the conjunction on A5-points — asks
  about reward from scratch, which this rung did not test; it is
  recorded as not run by the goal's ordering, not as answered.
- **Three seeds off one clone read as one policy.** Six runs, two
  coefficients, twenty-four reads, all within one SE of one number.
  The next test of "can reward improve a start" needs a different
  kind of start (a partially trained scratch policy, or the C2 arm
  that half-learned the plan) rather than another clone.

## What was not done

- One fit seed for the clone; D1's two fit seeds agreed to a thousandth
  and the read here is the teacher's to two hundredths, so a second fit
  was not spent.
- No critic pre-fit (D2b) and no unanchored arm (D2): both destroyed
  the escort clone within 2,560 rounds on this scenario and were not
  re-run on a weaker one.
- No whole-army control: the rung asks about the per-model trainer's
  anchor, which has no whole-army counterpart.
- Goal 3 (per-point terminal bonus or unique-coverage pay on A5-points)
  was not built or run.

## Addendum — written 2026-09-19 11:30: why training does not improve the clone, measured

Three causes, each read off the runs' own logs (`wandb/run-*/run-*.wandb`,
by quarter of the run) and one measurement of the reward.

**1. The anchor owns the gradient.** The adaptive coefficient
(Schulman's rule, doubling above 1.5× the target, capped at 10,000)
saturated at the cap inside the first quarter on D3a because the
measured drift sat at 0.075–0.083 nats per decision and never reached
the 0.03 target. The pre-clip gradient norm read 14,000–17,000 with
`max_grad_norm` 0.5 and the clip binding on 100% of steps, so the
update direction is the anchor's; the policy loss (0.05) is six
millionths of the loss. D3b's coefficient hovered at 5–9 with the
drift at its 0.10 target, gradient norm 8–17, still clipped every step;
its loss is 96% anchor. The same saturation is in every anchored run
on file — D2c (coef 10,000, gradient 24,000–29,000) and A5i (10,000,
20,000–23,000): **"the anchor holds" has meant "the anchor is the
whole update", at every coefficient that holds.**

**2. There is nothing to explore from.** The clone's heads are
near-deterministic: entropy 0.03 nats on the declaration head, 0.24 on
displacement, 3×10⁻⁵ on the shooting head, ~0 on the unit pointer; a
from-scratch policy at the same stage (A5-points) reads 0.11 and
1.6–2.3. Sampled rollouts are the greedy policy (the paired
sampled − greedy difference was −3 to +1 vp), so the alternative the
rung asks for — the unarmed squads wait while the rifles work — is
never rolled out and its advantage is never measured. The anchor
forbids the cure: entropy added to a near-deterministic reference is
KL to it, so `ent_coef` 0.003 pushes against a coefficient of 10,000.

**3. The reward pays the wrong order first.** The re-timed C3b reward
stream, n=40 on 700000+, gamma 0.9 per round as PPO discounts it:

| policy | success | episode reward | discounted from the start | rounds 1–6 | rounds 7–12 |
|---|---|---|---|---|---|
| `scripted_escort` | 1.00 | 9.79 | 3.96 | 1.25 | 8.54 |
| `squad_march_take` | 0.30 | 6.23 | 3.24 | **2.55** | 3.69 |
| the clone | 0.30 | 6.26 | 3.22 | 2.47 | 3.79 |

Walking in at once earns twice the escort's reward in the first six
rounds (the travel term pays per inch closed, now); the escort's
payoff — coverage of the cleared point and the terminal bonus — lands
in rounds 7–12 and at the end. Undiscounted the plan is worth +3.6 per
episode; discounted to the rounds where the decision is made it is
worth **+0.7**, spread over ~120 decisions, against a per-decision
return sd of 1.7 and a critic at explained variance 0.45–0.58. The
advantage of waiting is a fraction of the noise on any one decision,
and negative on the decisions that matter most (the first moves).

Any one of the three would slow improvement; together they forbid it.
The escort clone (D2c) and the bar's clones (A5f, A5i) sat under the
same three, which is why HOLDS was the ceiling there too. **What the
record now says about the D-route:** the anchor as built (an adaptive
coefficient capped at 10,000 on a clipped gradient) is a freeze, not a
tether; a clone of a deterministic teacher has no entropy to explore
with; and a plan whose payoff is late loses to a travel term that pays
now under gamma 0.9. A tether that could lift a clone would need a
bounded coefficient (or a trust region on the policy step rather than
a penalty), an entropy floor the anchor does not tax, and a reward or
horizon under which the plan's payoff reaches the decisions that
choose it — three builds, and the third is the same wall the
conjunction rungs hit.
