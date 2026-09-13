"""Score per-model checkpoints two ways on one seed set: greedy, and sampled.

A per-model checkpoint is SCORED greedy -- the argmax of every head, the
policy the checkpoint would play -- and TRAINED sampled, the policy its
rollouts actually drew. The two can disagree badly: the 2026-09-07
calibration cells sat at the random floor with the do-nothing fingerprint
(stationary share 1.0) under the greedy read while their sampled
training-side margin drifted up. Which of those is "the policy" decides
whether a flat eval curve means nothing learned or means the argmax of a
still-diffuse policy is a corner nobody sampled.

One row per checkpoint per mode, on identical seeds, with the paired
greedy-minus-sampled difference; the sampled row draws from a generator
seeded from the first seed, so it reproduces. The passive pair on every row
is the fingerprint the question is about.

Usage: just measure-per-model-eval-mode <env_config> <n_episodes> <seed_base> <ckpt.pt>...
"""

from __future__ import annotations

import sys

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.evaluation import (
    EvalResult,
    format_optional_metric,
    paired_difference,
)
from wargame_rl.wargame.scoring import evaluate_spec
from wargame_rl.wargame.selectors import is_per_model_checkpoint, label_for

# The tuning band (`CLAUDE.md` § How to measure here): this is a diagnostic
# of a checkpoint, not a score, so it stays off the evaluation band.
TUNING_SEED_BASE = 900_000


def format_row(label: str, mode: str, result: EvalResult) -> str:
    """One aligned row of the two-mode table."""
    return (
        f"{label:<30}{mode:<8}{result.vp_margin:>10.1f}"
        f"{format_optional_metric(result.vp_margin_se, 1):>8}"
        f"{100.0 * result.win_rate:>7.0f}{result.objectives_held:>7.2f}"
        f"{result.final_fraction_alive:>8.3f}"
        f"{format_optional_metric(result.coherency_rate):>10}"
        f"{format_optional_metric(result.stationary_share, 2):>6}"
        f"{format_optional_metric(result.hold_fire_share, 2):>6}"
        f"{format_optional_metric(result.mean_decisions, 1):>10}"
    )


def main() -> None:
    """Print the greedy and sampled rows of every checkpoint, then the pairing."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 5:
        print(__doc__)
        raise SystemExit(1)
    config_path = argv[1]
    n_episodes = int(argv[2])
    seed_base = int(argv[3]) if argv[3] else TUNING_SEED_BASE
    checkpoints = argv[4:]
    rejected = [c for c in checkpoints if not is_per_model_checkpoint(c)]
    if rejected:
        raise SystemExit(
            f"greedy against sampled is a per-model checkpoint diagnostic; "
            f"{', '.join(rejected)} is not a .pt"
        )

    env_config = load_env_config(config_path, **overrides)
    env_config.render_mode = None
    seeds = [seed_base + i for i in range(n_episodes)]

    print(
        f"\n{config_path}{describe(overrides)}  ({n_episodes} episodes, "
        f"seeds {seeds[0]}-{seeds[-1]}; sampled rows draw from seed {seeds[0]})\n"
    )
    header = (
        f"{'checkpoint':<30}{'mode':<8}{'VP margin':>10}{'±SE':>8}{'win%':>7}"
        f"{'held':>7}{'alive':>8}{'coherent':>10}{'stat':>6}{'hold':>6}"
        f"{'decisions':>10}"
    )
    print(header)
    print("-" * len(header))
    for checkpoint in checkpoints:
        label = f"{label_for(checkpoint)}/{checkpoint.rsplit('/', 1)[-1]}"[:28]
        greedy = evaluate_spec(checkpoint, env_config, seeds, label, greedy=True)
        sampled = evaluate_spec(checkpoint, env_config, seeds, label, greedy=False)
        print(format_row(label, "greedy", greedy))
        print(format_row(label, "sampled", sampled))
        mean, error = paired_difference(greedy, sampled)
        print(
            f"{'':<30}{'greedy - sampled':<18}{mean:>+8.1f} "
            f"± {format_optional_metric(error, 1)} paired\n"
        )


if __name__ == "__main__":
    main()
