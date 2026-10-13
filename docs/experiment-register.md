# Experiment register

The runs that matter, one entry each. A run is here because it is a reference
point: the best policy on a task, a result that changed a decision, or the
control other runs are compared to. Everything else lives in `reports/`.

## Rules for editing this file (for Claude and any other AI)

1. **Never add, remove or rewrite an entry without Sash's explicit yes in the
   conversation.** Propose it first: the run, and one line on why it matters.
2. **Keep it small.** One entry per run, using the template below and nothing
   more. No narrative, no adjectives, no "notably"/"importantly", no
   speculation. If a detail is not in the template, it belongs in the report.
3. **Plain language.** Someone who has not followed the project should be able
   to read an entry. Spell out what a name means the first time it appears in
   that entry; no unexplained jargon or flag names in the "What it tested" line.
4. **Numbers only from a read on file** (a report, an amendment, an issue
   comment), always with what they are: success per seed, which checkpoint
   (end of training or best), n, and the comparator by name.
5. **Every entry must say how to reproduce it**: config path, the non-default
   flags, the starting checkpoint (or "from scratch"), and the code revision.
6. **Link, do not copy**: the report or amendment, the GitHub issue(s), the
   Wandb runs. If a link does not exist, write "none".
7. **When a later run changes what an entry means** (retraction, a better run
   on the same task), do not edit the old result. Add one line under it
   starting with `Superseded:` or `Retracted:` and a link — and only with
   Sash's yes (rule 1).
8. **Newest first** within each section. Sections are by task (the board or
   rung the run was on), created only when an entry needs one.

## Entry template

```markdown
#### <Name> — <one-line verdict>

- **Date / revision:** YYYY-MM-DD · `<commit>`
- **What it tested:** <one sentence, plain language>
- **Changed from:** <comparator by name> — <the one thing that differs>
- **Result:** <success s1 / s2 / s3> at <end of training | best checkpoint>, n=<n>; comparator <its numbers>
- **Reproduce:** `<config path>` · `<non-default flags>` · start: <checkpoint path | from scratch>
- **Links:** report <link> · issues <#N, #M> · Wandb <run links>
```

## Runs

Checkpoints for these runs are in `checkpoints/per_model/` (the run directory
ends in the seed and tag, e.g. `-s1pc`). The register does not protect them.

### Six squads, five objectives (A5-points)

Six squads of three soldiers, five objectives, ten rounds, no opponent.
Success: every objective occupied. The scripted player (`squad_march_take`)
succeeds 1.00 in 6.71 rounds.

#### PC — a planner learns the scripted player's plan from scratch

- **Date / revision:** 2026-09-28 · `670f8f7`
- **What it tested:** Can the planner (the part that picks each squad's objective) learn from scratch, if scripted soldiers carry out whatever plan it writes? Planner paid for covering the objectives, for its plan covering them, and a bonus for finishing early.
- **Changed from:** nothing — it is the control for PV.
- **Result:** 1.00 / 1.00 / 1.00 at end of training, n=100, in 6.57–6.68 rounds; best checkpoint 1.00 ×3, reached by 20,480 rounds on two seeds.
- **Reproduce:** `configs/experiments/curriculum/a5_points_head_pf_pcm.yaml` · `--members squad_march_committed --num-rollout-envs 4 --rollout-rounds 32 --ent-coef 0.003 --rounds 122880 --seed <1-3>` · start: from scratch
- **Links:** report [amendment 29](../reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-29) · issues #440, #384 · Wandb [s1](https://wandb.ai/wargame_rl/wargame_rl/runs/vack5vct) [s2](https://wandb.ai/wargame_rl/wargame_rl/runs/zi5wlefq) [s3](https://wandb.ai/wargame_rl/wargame_rl/runs/c6ho4pe6)

#### PV — the same planner paid only in victory points

- **Date / revision:** 2026-09-28 · `670f8f7`
- **What it tested:** Does the game's own score (victory points, per-round cap off) teach a planner from scratch, with no reward for the shape of the plan?
- **Changed from:** PC — the planner's reward only.
- **Result:** 0.92 / 1.00 / 0.99 at end of training, n=100, in 6.75–7.24 rounds; best checkpoint 0.98 / 1.00 / 1.00. PC 1.00 ×3. Passes on two seeds; its first plan rarely covers the board, it repairs by redirecting squads.
- **Reproduce:** `configs/experiments/curriculum/a5_points_head_pf_vp.yaml` · `--members squad_march_committed --num-rollout-envs 4 --rollout-rounds 32 --ent-coef 0.003 --rounds 122880 --seed <1-3>` · start: from scratch
- **Links:** report [amendment 29](../reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-29) · issues #441, #384 · Wandb [s1](https://wandb.ai/wargame_rl/wargame_rl/runs/h113g6dr) [s2](https://wandb.ai/wargame_rl/wargame_rl/runs/xoa9yzbz) [s3](https://wandb.ai/wargame_rl/wargame_rl/runs/v3gqtfs5)

#### CM15 — best policy with planner and soldiers trained together

- **Date / revision:** 2026-09-27 · `aef0db1`
- **What it tested:** Planner and soldiers trained together, with the soldiers paid for following the plan (and, at a success, exactly the pay the early finish cuts off) and the planner paid in victory points (per-round cap off).
- **Changed from:** CM14 — the planner's reward (CM14 kept the older coverage reward).
- **Result:** 0.15 / 1.00 / 0.98 at end of training, n=100 (seed 1's planner collapsed late); best checkpoint 1.00 / 1.00 / 0.98 in 6.9–7.2 rounds. CM14 0.00 / 0.98 / 0.97. Not learned from scratch: the same setup from scratch (CM15s) read 0.00 ×3.
- **Reproduce:** `configs/experiments/curriculum/a5_points_head_pf_vp.yaml` · `--num-rollout-envs 4 --rollout-rounds 32 --ent-coef 0.003 --rounds 122880 --seed <1-3>` · start: CM9c of the same seed at 81,920 rounds (`per-model-a5_points_head_rf_pc-2026-09-26-19-10-27-s<N>cm9c/pm-00081920.pt`)
- **Links:** report [amendment 27](../reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-27) · issues #436, #384 · Wandb [s1](https://wandb.ai/wargame_rl/wargame_rl/runs/rw8d257w) [s2](https://wandb.ai/wargame_rl/wargame_rl/runs/6ahb00ot) [s3](https://wandb.ai/wargame_rl/wargame_rl/runs/8vswalfh)

### Four squads, four objectives, plan never the nearest (A3 rotated)

Four squads of three, four objectives. The environment writes each squad's
objective and never gives a squad its nearest one; success only when every
squad is on its own objective. Soldiers who walk to the nearest objective fail.

#### FH3 seed 1 — soldiers that follow the plan

- **Date / revision:** 2026-09-25 · `2e021a2`
- **What it tested:** Can soldiers trained from scratch learn to go where the plan says rather than to the nearest objective?
- **Changed from:** LR1 — the soldiers' reward (progress toward the committed objective, measured from at least 12", plus a capped stay term).
- **Result:** seed 1 arrives at its committed objective on 0.94 of squads (at the nearest 0.01); seeds 2 and 3 walk to the nearest (0.38 / 0.50). Under a learned planner (PL1) seed 1 succeeds 0.98 in 5.54 rounds (scripted 5.28). n=100, end of training. LR1 seed 1 0.89. Its weights are the start of CM7, the first joint run to pass a rung.
- **Reproduce:** `configs/experiments/curriculum/a3_legible_rf.yaml` · `--num-rollout-envs 4 --rollout-rounds 32 --ent-coef 0.003 --rounds 122880 --seed 1` · start: from scratch · weights: `per-model-a3_legible_rf-2026-09-25-23-58-19-s1fh3/last.pt`
- **Links:** report [amendment 14](../reports/2026-09-23-curriculum-commitment-plan-preregistration.md#amendment-14) · issues #416, #384 · Wandb [s1](https://wandb.ai/wargame_rl/wargame_rl/runs/5hcx6fxq)
