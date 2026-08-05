# Experiment reports

Findings from training experiments, kept for retrospection. Each report records what
was asked, what was measured, and what the measurement does and does not support.

## Index

| Report | Question | Outcome |
|---|---|---|
| [2026-08-05 — stochastic terrain and cover](2026-08-05-stochastic-terrain-and-cover.md) | Does the agent learn to use terrain for cover when the opponent shoots back? | **No — it learns range management.** Deleting every ruin moves exposure 0.116 → 0.120; return fire alone moves it 4.7x. At weapon range 24, where distance stops working, the policy collapses to 6.8% win. Reward *and* opponent curricula both unnecessary. Objective-placement defect found and fixed: 25% of episodes had overlapping objectives |
| **[2026-08-04 — correction: what was actually broken](2026-08-04-correction-what-was-actually-broken.md)** | **Read this first.** Is the policy learning at all? | **No.** The movement head never left initialisation; a 12-line heuristic beats 945 epochs of training 80% to 17%. Four further defects found; most conclusions below retracted |
| [2026-08-04 — reward-phase curriculum](2026-08-04-reward-phase-curriculum.md) | Why does the 25v25 curriculum never leave phase 0, and can it reach `win_at_the_end`? | ⚠️ Substantially retracted — training reward never left phase 0, so no cross-run conclusion holds |
| [2026-08-04 — mechanism defects](2026-08-04-mechanism-defects.md) | Detailed evidence for each defect found | D1/D2/D4 survive (arithmetic); D3/D5's numbers measured a near-random policy |
| [2026-08-04 — objective drift](2026-08-04-objective-drift.md) | What does the agent actually do during an episode? | Drift is real; ⚠️ its explanation is superseded — a uniform policy produces the same trace |
| [2026-08-04 — reward calculation changes](2026-08-04-reward-calculation-changes.md) | What changed in the reward, and what does the evidence support? | Reward became per-model; two calculators added, three dropped, one arithmetic bug fixed. Win 0.17 → 0.93–0.97 vs a 0.67 movement bar — but **confounded** with the PPO fixes that shipped alongside |

## Conventions

**Distinguish measurement from inference.** Every quantitative claim states how it was
obtained. Claims that are inferred rather than measured say so.

**Report negative results.** A refuted hypothesis costs a run to establish and costs
another run every time it is retried by someone who does not know it failed.

**State confounds.** Most runs here changed more than one variable, were stopped early,
and used a single seed. That is adequate for finding defects and inadequate for ranking
hyperparameters. Reports say which of the two they support.

**Prefer the live metric.** `success_rate` from a best checkpoint is not the same
quantity as the per-epoch `success_rate` a gate reads. See
[mechanism defects, D4](2026-08-04-mechanism-defects.md#d4--rungs-calibrated-against-the-wrong-distribution).

**Quote every number against a floor and a bar.** A `success_rate` alone says nothing —
17% read as progress until a scripted baseline scored 80% on the same board. Run
`just measure-baselines <env_config>` before interpreting any result; training logs
`eval/baseline_*` automatically. Note every number in these reports was measured against
`scripted_advance_to_objective`, an opponent that never fires — the `squad_march_shoot`
bar of 1.00 is partly an artefact of that, and falls to 0.60 against
`scripted_advance_and_shoot`.

**Verify the thing you are tuning is the thing being trained.** Seven runs tuned reward
phases whose rewards never reached the gradient.

## Reproducing

```bash
# Scripted baseline scores -- the floor and the bar. Run this before reading any result.
# Pass a seed_base to score on the same layouts as measure-checkpoint.
just measure-baselines <env_config> [n_episodes] [record] [seed_base]

# Score a checkpoint through the baselines' own code path, so the two are comparable
just measure-checkpoint <checkpoint> <env_config> [n_episodes] [record]

# Rolling-mean metrics for a run (point readings are n-sample binomials -- do not trust them)
just run-summary <run_id> [bucket_size]

# Per-phase criteria rates and the whole min_fraction curve for a checkpoint
just measure-phase-gates <checkpoint> <env_config> 40

# Behavioural analysis of a recorded match
just replay-summary <file>
just analyze <file>
just analyze-compare <files...>
```

Recordings live in `recordings/` (gitignored). Runs are in the `wargame_rl/wargame_rl`
Weights & Biases project; run IDs are given in each report.
