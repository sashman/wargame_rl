# Experiment reports

Findings from training experiments, kept for retrospection. Each report records what
was asked, what was measured, and what the measurement does and does not support.

## Index

| Report | Question | Outcome |
|---|---|---|
| [2026-08-04 — reward-phase curriculum](2026-08-04-reward-phase-curriculum.md) | Why does the 25v25 curriculum never leave phase 0, and can it reach `win_at_the_end`? | Reached `win_at_the_end`; cause was four mechanism defects, not policy quality |
| [2026-08-04 — mechanism defects](2026-08-04-mechanism-defects.md) | Detailed evidence for each defect found | Four confirmed, all fixed |
| [2026-08-04 — objective drift](2026-08-04-objective-drift.md) | What does the agent actually do during an episode? | Reaches objectives, then abandons them; peak occupancy ~3x final |

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

## Reproducing

```bash
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
