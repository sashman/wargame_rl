# Pre-registration: curriculum rung D2 — PPO from the escort clone

Written 2026-09-17 16:25, **before any PPO round from the clone exists**,
on branch `feature/curriculum-d1` (PR #374). Issue #376, parent
question #340, rung **D2** — the "start" axis, reward from a policy that
holds the plan.

## The question

D1b's clone (#375, fit seed 0, 1,200 games) holds the escort's plan on
C3b: success 0.960, kill before arrival 1.00, alive 0.953, held 3.93,
vp +59.6, against the escort's 0.990 / +63.5. On the same scenario,
PPO from a start without the plan reached 0.20 (C3b's arm from C2) and
0.02–0.06 from scratch. Does per-model PPO **improve** a policy that
starts with the plan, **hold** it, or **destroy** it — the whole-army
record's outcome, where PPO with a cold critic destroyed a clone at
every entropy setting (`CLAUDE.md` § Settled)?

## The one change

The starting weights. `train_per_model.py --env-config-path
configs/experiments/curriculum/c3b.yaml --rounds 122880 --seed {1,2,3}
--warm-start-from checkpoints/per_model/clones/escort-c3b-1200-s0.pt
--num-rollout-envs 4 --rollout-rounds 32 --eval-every-rounds 512
--checkpoint-every-rounds 512 --n-eval-episodes 30 --ent-coef 0.003`,
Wandb group `curriculum-d2`, tag `d2`. The value head starts at its
random initialisation: the clone fits no critic, by design, so this arm
measures PPO from a clone as the whole-army arm did. No KL anchor.

## Comparator, measured first

The clone itself, paired per episode on identical seeds: success 0.960,
held 3.93, alive 0.953, on_obj 0.965, vp +59.6 ± 2.3, coherent 0.763,
kill before arrival 1.00 (n=100 on 700000+; the n=180 read of the clone
is taken beside the arm's). `scripted_escort`: 0.990, +63.5 ± 2.0.

## Criteria

Read greedy, no decode, n=180 on 700000+, from `last.pt` at 122,880;
the in-run curve at n=30 on 500000+ every 512 rounds, its first point
being the clone's own score.

- **IMPROVES:** vp margin paired against the clone > 0 with t ≥ 2 on
  every seed, success ≥ 0.96 and kill before arrival ≥ 0.95 on every
  seed at the end.
- **HOLDS:** success ≥ 0.94 (one SE under the clone at n=180) with
  ordering ≥ 0.95 on every seed, and vp paired against the clone not
  significant on any seed.
- **DESTROYS:** success at the end < 0.90 on any seed. **Drift
  prediction, written first:** in-run success falls below 0.90 within
  20,480 rounds on every seed and does not recover by 122,880. If it
  holds, #332 (a per-model KL anchor to the clone) is the next build; if
  PPO holds or improves the clone without one, #332 stays on hold.
- **NULL (scenario):** only if the escort fails its own criterion.
- **Readouts:** the in-run curve; `alive`, `held`, `on_obj`, coherency
  greedy and sampled; the health panel (explained variance from a cold
  critic first); the census by phase at the end; the drift between the
  clone and the final policy — the final policy's held-out match against
  the escort's demonstrations, beside the clone's 0.72 on the
  displacement head.

Power: paired vp SE against the clone about 2 at n=180 (both near +60,
per-episode sd ~25 on this scenario); binomial SE 0.015 at p=0.96.

## What I expect (a guess, written so it can be wrong)

DESTROYS, by the drift prediction: the cold critic's first updates are
noise, the clone's success falls under 0.90 inside 10k rounds on every
seed, and the policy settles near C3b's arm (0.2–0.3) with the order
partly kept (0.7–0.9). If instead it HOLDS, the per-model trainer has
done what the whole-army one could not, and D2's second arm is the
anchor as a control rather than a rescue.
