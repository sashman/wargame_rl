"""The bridge check for a curriculum config: a script scores identically
through both facades, and its row IS the rung's bar.

Every curriculum rung is trained on both facades (the whole-army trainer is
the pipeline control), so before an arm opens the scenario has to be the
same game under both steps. This scores one scripted policy through
`evaluate_selector` (the phase facade) and `evaluate_per_model_chooser` (the
per-model facade) on identical seeds and prints both rows, the success rate
and rounds-to-success the per-model side measures, and whether every shared
field agreed. A disagreement is a facade divergence on this scenario
(`tests/test_per_model_bridge.py` names the orderings that legitimately
diverge) and the rung is redesigned, not trained.

Usage: just measure-bridge <env_config> [n_episodes] [policy] [seed_base]
"""

from __future__ import annotations

import sys

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.baseline.evaluate import evaluate_selector, selector_for
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.evaluation import (
    EVAL_WAVE_SIZE,
    EvalResult,
    format_optional_metric,
)
from wargame_rl.wargame.envs.per_model import (
    PerModelEnv,
    evaluate_per_model_chooser,
    scripted_chooser,
)
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.wargame import WargameEnv

HELDOUT_SEED_BASE = 700_000
# The fields both facades measure; a difference in any is a divergence.
SHARED_FIELDS = (
    "final_fraction_at_objectives",
    "win_rate",
    "player_vp",
    "opponent_vp",
    "worst_cohesion_gap",
    "final_fraction_alive",
    "objectives_held",
    "coherency_rate",
    "models_out_of_coherency",
    "vp_margin_per_episode",
    "objectives_held_per_episode",
    "win_per_episode",
    "turns_per_episode",
)


def format_row(label: str, result: EvalResult) -> str:
    """One aligned row: the fields that decide a rung."""
    return (
        f"{label:<12}{result.vp_margin:>10.1f}"
        f"{format_optional_metric(result.vp_margin_se, 1):>8}"
        f"{result.final_fraction_at_objectives:>8.3f}{result.objectives_held:>7.2f}"
        f"{result.final_fraction_alive:>8.3f}"
        f"{format_optional_metric(result.coherency_rate):>10}"
        f"{format_optional_metric(result.success_rate, 3):>9}"
        f"{format_optional_metric(result.mean_turns, 2):>8}"
        f"{format_optional_metric(result.stationary_share, 2):>6}"
        f"{format_optional_metric(result.hold_fire_share, 2):>6}"
    )


def main() -> None:
    """Score the policy through both facades and print the comparison."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    config_path = argv[1]
    n_episodes = int(argv[2]) if len(argv) > 2 and argv[2] else 100
    policy_name = argv[3] if len(argv) > 3 and argv[3] else "squad_march_take"
    seed_base = int(argv[4]) if len(argv) > 4 and argv[4] else HELDOUT_SEED_BASE

    env_config = load_env_config(config_path, **overrides)
    env_config.render_mode = None
    seeds = [seed_base + i for i in range(n_episodes)]

    phase_env = WargameEnv(env_config)
    phase = evaluate_selector(
        selector_for(build_baseline_policy(policy_name)), phase_env, seeds, policy_name
    )
    phase_env.close()
    envs = [PerModelEnv(env_config) for _ in range(min(EVAL_WAVE_SIZE, n_episodes))]
    per_model = evaluate_per_model_chooser(
        scripted_chooser(build_baseline_policy(policy_name), envs),
        envs,
        seeds,
        policy_name,
        retimers=[PerStepReward(env) for env in envs],
    )
    differing = [
        field
        for field in SHARED_FIELDS
        if getattr(phase, field) != getattr(per_model, field)
    ]

    print(
        f"\n{config_path}{describe(overrides)}  `{policy_name}`  "
        f"({n_episodes} episodes, seeds {seeds[0]}-{seeds[-1]})\n"
    )
    header = (
        f"{'facade':<12}{'VP margin':>10}{'±SE':>8}{'on obj':>8}{'held':>7}"
        f"{'alive':>8}{'coherent':>10}{'success':>9}{'turns':>8}{'stat':>6}{'hold':>6}"
    )
    print(header)
    print("-" * len(header))
    print(format_row("phase", phase))
    print(format_row("per-model", per_model))
    if differing:
        print(f"\nBRIDGE DIVERGES on: {', '.join(differing)}")
        raise SystemExit(2)
    print("\nbridge identical on every shared field")


if __name__ == "__main__":
    main()
