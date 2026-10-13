# One episode per update — the first per-model runs are void for regime, not for the architecture

Read 2026-09-14 off the run directories, **no GPU** and no new training.
This is the first pipeline-health finding of the per-model curriculum
(issue #288's successor question), and it lands the diagnostic that the
2026-09-11 investigation made and did not write down.

## Verdict

**Every per-model training run to date is VOID for REGIME.** The stage-1
calibration sweep (six cells, `checkpoints/per_model/calibration_stage1/`)
and the three observe runs that followed all trained on **one rollout env**,
updating every 8, 16 or 32 rounds against the phase facade's **1024 rounds per
update** — one episode or less of experience per gradient update, re-read
four times. Their flat evaluation curves say nothing about whether the set
network can learn this game. They are also void for **code**: their
checkpoints predate the facade tag and the audited network and cannot be
loaded on `main`, so the rescoring this report set out to do is impossible
and is not done.

Nothing in the architecture is exonerated by this either. The curriculum
that follows (E1 onward) is where that question gets asked, one rung at a
time, with the instruments this report shipped.

## Provenance

| | |
|---|---|
| runs read | `calibration_stage1/per_model_25v25_maps_two_mode_cal_g90_l{90,95}_r{8,16,32}_s1_20260907_105201_*` (revision `3e54619`, seed 1); `per_model_25v25_maps_two_mode_observe_r32_s0_20260907_141837_*` (`fccc89b`, seed 0); `..._observe_r32_s1_s1_20260907_151244_*` (`49388a5`, seed 1); `..._observe_s0_20260907_115119_*` (`4907b88`, seed 0, no checkpoint) |
| config | `configs/golden/25v25_maps_two_mode.yaml`, unrefereed (the training config), opponent `squad_march_take` |
| in-run eval | n=20, seeds 500000+, **greedy**, no decode (the per-model facade runs none) |
| comparator | the phase facade's `random` on the same config: **−222.5** (all 45 tables, n=30, seeds 700000+, `CLAUDE.md` § The board) — a different seed band and facade, quoted as the floor's order of magnitude only |
| old facade's regime | `train.py`: 2048 steps per epoch at two steps per round = 1024 rounds per update, 4 epochs of minibatches |
| code revision of this reading | the Stage 0 branch, `feature/per-model-pipeline-health` |
| paired | no — nothing here is a comparison between arms |

## The regime table

The driver at `3e54619` ran one env (its `PerModelPPOConfig` has no
`num_rollout_envs`), so rounds per update is `rollout_rounds` itself:

| run | rollout rounds | envs | **rounds / update** | vs 1024 | budget / killed at |
|---|---|---|---|---|---|
| cal `g90_l90_r8`, `g90_l95_r8` | 8 | 1 | **8** | ÷128 | 51,200 / 69,120 · 61,440 |
| cal `g90_l90_r16`, `g90_l95_r16` | 16 | 1 | **16** | ÷64 | 51,200 / 53,760 |
| cal `g90_l90_r32`, `g90_l95_r32` | 32 | 1 | **32** | ÷32 | 51,200 / 53,760 |
| observe `r32_s0` (`fccc89b`) | 32 | 1 | **32** | ÷32 | 25,600 / 37,856 |
| observe `r32_s1` (`49388a5`) | 32 | 1 | **32** | ÷32 | 25,600 / 30,208 |
| observe `s0` (`4907b88`) | 8 | 1 | **8** | ÷128 | 307,200 / 17,920 |

A 25v25 game on this config is 20 rounds, so a 16-round rollout is less than
one episode. With `n_update_epochs` 4 and `minibatch_size` 64 over roughly
300–1200 decision steps, each update took 20–75 gradient steps on that one
part-episode — and at the logged per-update `approx_kl` (median 0.006–0.007
on the observe runs) the policy drifted **32–128× further per round** than
the same per-update figure means at 1024 rounds per update.

## What the curves show

Eval `vp_margin` (greedy, n=20) at the last four evaluations of each cell,
with the eval-side stationary share beside it:

| cell | rounds | vp_margin | stationary share |
|---|---|---|---|
| `g90_l90_r8` | 46,080 · 53,760 · 61,440 · 69,120 | −187.5 · −190.0 · −147.3 · −178.3 | 0.00 · **0.68** · 0.21 · 0.00 |
| `g90_l95_r8` | 38,400 · 46,080 · 53,760 · 61,440 | −193.5 · −241.5 · **−216.8** · −145.2 | 0.19 · 0.48 · **0.94** · 0.00 |
| `g90_l90_r16` | 30,720 · 38,400 · 46,080 · 53,760 | −189.7 · −222.2 · −176.0 · −221.5 | 0.09 · 0.53 · 0.00 · 0.00 |
| `g90_l95_r16` | 30,720 · 38,400 · 46,080 · 53,760 | −223.8 · −173.7 · −237.2 · −194.5 | **0.93** · 0.01 · 0.63 · 0.00 |
| `g90_l90_r32` | 30,720 · 38,400 · 46,080 · 53,760 | −205.0 · −176.7 · −215.5 · −199.8 | 0.23 · 0.00 · 0.11 · 0.16 |
| `g90_l95_r32` | 30,720 · 38,400 · 46,080 · 53,760 | −188.8 · −205.7 · −181.2 · −219.7 | 0.12 · 0.07 · 0.00 · 0.22 |
| observe `r32_s0` | 7,680 · 15,360 · 23,040 · 30,720 | −226.2 · −117.0 · −175.3 · −239.2 | **1.00** · 0.00 · 0.00 · 0.00 |
| observe `r32_s1` | 7,680 · 15,360 · 23,040 | −191.3 · −163.3 · −138.8 | 0.42 · 0.00 · 0.00 |

Three things, none of them a result about the network:

1. **Every cell sits within the eval's own noise of the random floor.** At
   n=20 on this config the eval `vp_margin` carries an SE of roughly 18, and
   the curves move ±50 between consecutive evaluations. Nothing separates
   the six cells, and nothing separates any of them from −222.5.
2. **The passive fingerprint swings 0 ↔ 1 between evaluations 7,680 rounds
   apart.** The greedy policy's stationary share reads 0.00 at one
   evaluation, 0.94 at the next and 0.00 again. The training-side share
   (sampled) on the observe runs spans 0.00–1.00 too. That is not a policy
   that learned to do nothing; it is a declaration head whose argmax flips
   under a diffuse distribution, and the greedy read amplifies the flip
   into a whole-army do-nothing turn. **The greedy score and the sampled
   policy were different objects** — which is why Stage 0 ships sampled
   play as a diagnostic beside the greedy score.
3. **The critic never settled.** `train/explained_variance` on the observe
   runs spans −3.6 to 0.99 (`r32_s0`) and −6.6 to 0.98 (`r32_s1`) across
   consecutive updates, which is what fitting a value function to one
   part-episode at a time looks like: it explains the batch it just saw and
   nothing about the next.

## What this report was going to measure and cannot

The plan called for rescoring the killed cells' checkpoints **greedy and
sampled** at n=30 on the tuning band (900000+), to price the fingerprint
directly. The checkpoints carry no facade tag and a `state_dict` from the
pre-audit set network (`feed_forward.net.*` against `main`'s
`feed_forward.fc` / `proj`, and different shapes on `self_relation_bias`,
`cross_relation_bias`, `pointer_relation_bias`, `phase_embedding`), so
`load_checkpoint` refuses them on `main` and there is no honest way to
score them. The instrument exists (`just measure-per-model-eval-mode`) and
was verified on a Stage 0 smoke checkpoint; **no number from it is quoted
here**, because a fresh two-update network says nothing about these runs.

## What ships with this report (Stage 0 of the curriculum)

- **The regime, explicit.** `train/rounds_per_update`,
  `train/gradient_steps_per_round` and `train/num_rollout_envs_resolved` on
  every update row; a warning naming the visible CPU count when the env
  count auto-clamps below the device ceiling; `--eval-seed-base` in
  provenance; `just train-per-model-arm` refuses a flag string that does not
  name `--num-rollout-envs`, and launches detached with `setsid`.
- **The passive fingerprint on the shared result.** `EvalResult.stationary_share`
  / `hold_fire_share`, counted off the `open` steps the per-model evaluator
  steps (`MoveDeclaration.stationary`, `ShootDeclaration.hold_fire`), logged
  as `eval/stationary_share` / `eval/hold_fire_share`, printed as `stat` /
  `hold` by `measure-checkpoint`, `measure-maps` and `measure-paired`, `-`
  on the phase facade.
- **Sampled play as a diagnostic.** `build_per_model_chooser(..., greedy=False)`
  on a seeded generator; `evaluate_spec(..., greedy=False)`, refused by name
  on the phase side; `just measure-per-model-eval-mode`.
- **Resume and warm start.** `--resume-from` (in place, optimizer and
  generator restored, knobs refused if changed, `metrics.jsonl` appended,
  provenance `resumed_from`) and `--warm-start-from` (weights only, any
  scenario with the same displacement head, provenance `warm_started_from`).
- **The health panel** (`docs/metrics.md` § The per-model health panel):
  advantage moments before normalisation, return and value moments, per-head
  entropy, the ratio's tails, cumulative KL per 1k rounds, the eval SE — and
  the test that gives it meaning, thirty-six gradient steps on one fixed
  rollout driving the policy loss down monotonically.

## Standing rules (into `CLAUDE.md` § How to measure here)

- **State the rounds per update on every per-model row, and never let the
  env count come from CPU affinity.** A per-update statistic (`approx_kl`,
  `clip_fraction`, a loss) at 32 rounds per update is not the same
  quantity as at 1024.
- **Score a per-model checkpoint greedy WITH the passive fingerprint, and
  read the sampled score beside it before calling a flat curve "nothing
  learned".** A greedy do-nothing fingerprint over a sampled policy that
  scores is a diffuse declaration head, not an empty policy.
- **A void run is void for a named reason.** These are void for regime and
  for code; they are not evidence against the set network, and the next
  claim about the architecture has to come from a rung that passes its
  health panel.
