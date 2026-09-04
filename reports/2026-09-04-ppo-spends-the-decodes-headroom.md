# PPO did not fail — it spent the decode's headroom

**2026-09-04**, code `d5ec7d4`, no GPU training. Six paired seeds, n=45, seeds
700000+, `configs/evaluation/25v25_maps_melee_approach_refereed.yaml`, scored
through the §46/§47 harness so all three tables are directly comparable.

`docs/melee-teaching-goal.md` §48 records one open question and calls it "the
whole question": **why gradient descent cannot hold a basin it was placed in,
with a fitted critic.** §47 had ruled out the record's own explanation ("with a
cold critic") by measuring a warm one, and left the cause unknown. Nothing in
the line had probed it.

This is the probe. It cost no training.

## The answer

PPO is not failing. It is succeeding at a different objective from the one it is
scored on, and the two are in direct opposition.

| regime | clone | after 300 epochs of PPO | paired Δ | t | signs |
|---|---|---|---|---|---|
| `K=1 cd=0` — **the regime PPO trains in** | −85.53 | −65.82 | **+19.72** | 8.22 | **6/6** |
| `K=1 cd=1` | −69.05 | −50.83 | +18.22 | 3.50 | 6/6 |
| `K=3 cd=0` | −32.18 | −28.72 | +3.47 | 0.52 | 2/6 |
| `K=3 cd=1` — **the regime it is scored in** | **−10.67** | **−25.58** | **−14.92** | −2.60 | 1/6 |

Rollouts are collected at `decode_topk=1` with no charge decode, because
decoding is deliberately kept out of training. Scoring uses `K=3` plus the
charge decode. **In its own regime PPO improved the policy by +19.7 vp on 6 of 6
seeds at t=8.2.** It then lost 14.9 in the regime that counts.

**Decode headroom — the same checkpoint at `K=3 cd=1` minus itself at `K=1 cd=0`:**

| | headroom |
|---|---|
| the §46 clone | **+74.87 vp** |
| after 300 epochs of PPO | **+40.23 vp** |

**PPO bought 19.7 vp of unaided skill and spent 34.6 vp of headroom doing it.**

The mechanism is visible in the behaviour, not just the score. Unaided coherency
rises 0.754 → 0.818 and unaided standing charges 0.95 → 2.04/ep: the policy
really did learn, on its own, the formation and charge execution it previously
could not manage. **Both policies reach 0.96 coherency once decoded.** So what
PPO learned is exactly what the decode already supplies for free, and the
capacity spent acquiring it came out of the property the decode needs.

## The harness reproduces the two published numbers to the decimal

Not a claim about my arithmetic — the check that makes the table comparable at
all. `barclone-s{1..6}` at `K=3 cd=1` gives **−10.67** with per-seed values
+2.2 / −16.2 / −13.2 / −11.9 / −17.8 / −7.1, identical to §46's printed list;
the six §47 checkpoints give **−25.58**. The scripted bar re-measured the same
day gives **−5.3 / +20.2 / +11.8 / +56.6**, identical to §38 on all four cells.

## ⚠ A prediction registered before the cell existed, and FALSIFIED

I registered that the **charge decode** was the misaligned piece, on the
reasoning that a declaration which fails unaided wastes a turn while a decoded
one lands — so PPO would learn to under-declare. Bound: `Δ(K=1, cd=1) ≤ +5.0`,
falsified at `≥ +12.0`.

**Measured +18.22.** Supplying charge execution alone leaves the misalignment
essentially untouched. The declaration count does fall as predicted (12.98 →
10.71/ep) but that is not where the vp are.

The misalignment is the **joint coherent decode**: Δ collapses from +19.7 to
+3.5 the moment formation is decoded. And that one **cannot** be moved into
training — constrained sampling was measured at −51.8 vp from scratch and −43.7
warm-started, scored decoded both times
([report](2026-08-20-decoding-does-not-belong-in-training.md)). So the training
regime cannot be made equal to the play regime by any route currently on file.

## What is left: bound the drift

The §47 runs' intermediate checkpoints say the damage is **progressive**.
Scored at `K=3 cd=1`: epochs ≤100 mean **−7.3** (5 checkpoints), epochs >100
mean **−24.7** (13). ⚠ Those epochs are `ppo-NNN` — top-k by *training reward* —
so they are seed-specific and not a clean curve, and **"the early window beats
the clone" is NOT established**: at n=45 a single point carries SE ≈ 9, and 5
points at −7.3 against the clone's −10.67 is well inside noise. It is enough to
motivate a trust-region intervention and no more.

The drift itself is enormous. `KL(ppo ‖ clone)` on real states, per model:
**mean 2.06 / 2.65 / 2.34 nats** across three seeds (median 1.1–1.8, p90
5.5–8.2). That is comparable to the policy's entire entropy — the per-model
policy is effectively unrecognisable.

A KL anchor to the warm-start weights is shipped as `--kl-ref-coef` /
`--kl-ref-target`. Its arm is pre-registered and in flight; nothing here is a
result about whether it works.

## ⚠ Training is NOT bit-reproducible, and the record says it is

CLAUDE.md: *"Training is deterministic given seed + config + code (within one
setting of `--tf32`)"*, and on that basis *"never retrain a control that already
exists at the same epoch budget."* Two independent pairs of 2-epoch runs —
identical code, seed, config and flags — produced four different weight digests:

| config | tensors identical | max abs diff | mean rel diff |
|---|---|---|---|
| `configs/golden/25v25_shooting_opponent.yaml` | **110 / 222** | 1.19e−07 | 0.0000 |
| `configs/experiments/25v25_maps_melee_approach.yaml` | **0 / 222** | 6.36e−03 | **0.0064** |

The rollout envs *are* seeded deterministically (`ROLLOUT_SEED_BASE + env_idx`,
and note that base does **not** depend on `--seed`), so this is float
nondeterminism on the GPU amplified by chaotic dynamics: a last-bit difference
flips a sampled action, and the episode diverges. On the golden config two
epochs barely amplify it; on the melee map-pool config they amplify it by four
orders of magnitude.

**What this changes.** Pairing two arms by seed still holds *at initialisation*
— that is where its measured value came from (+7.5 and +7.2, 0.3 apart) — but
the trajectories are not shared, so a paired difference carries a rerun-noise
term nobody here has measured. ⚠ **This also made the no-op check I had
pre-registered for the KL anchor impossible**: a bit-identical digest at
`--kl-ref-coef 0` cannot be produced, because the same build cannot reproduce
*itself*. The no-op rests on static reading instead — at coefficient 0 no
reference network is constructed, nothing is copied, no stream is drawn, and the
loss is assembled by the identical expression — which is the same standard the
self-play flag is held to. **The magnitude of rerun noise on a 300-epoch score
is unmeasured and is a real gap in every paired figure in this record.**

## What this does NOT establish

- **Nothing about whether the anchor works.** That arm is in flight.
- **Nothing about the goal.** No cell moved; the best policy on file is still
  §46's clone at −10.67 against the bar's −5.3.
- **Not that PPO is the wrong optimiser.** It is that the objective it is given
  during rollout is not the objective the policy is scored on, and the gap
  between them is worth ~35 vp on this config.
- The `K=1 cd=0` column is a **regime**, not a policy anybody plays. Its
  absolute values (−85.5, −65.8) are not comparable to anything published here.

## The rule this earns

**Measure a policy in the regime it is TRAINED in, not only the one it is scored
in, whenever a play-time decode stands between the two.** The whole of §47's
mystery is one column that nobody had ever printed. It cost eighteen inference
runs.

⚠ And its corollary: **a play-time decode makes the corresponding training-time
skill worthless.** Any capacity the learner spends acquiring what the decode
supplies is spent twice, and here it is spent *against* itself. That applies to
all three members of the decode family on file — formation, surplus
reallocation, and the charge.
