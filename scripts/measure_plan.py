"""The plan-only readout (#384): is the commitment head's plan a good one?

A checkpoint on a `head` config is scored twice on identical seeds -- as
trained (the network plays every decision) and PLAN-ONLY (the network draws
only the commitment decisions; the scripted `members` policy, by default
`squad_march_committed`, which reads the state, takes every other decision,
re-planning before each act) -- beside the scripted bar. The plan-only row
holds execution at the bar's, so its success, turns and held read the plan
alone; the commitment readouts (persist, claimants, distinct, complete) say
what the plan looks like.

Usage: just measure-plan <head_config> <n_episodes> <seed_base> <ckpt...> [members] [planner=<ckpt>]

With `planner=<ckpt>` (the frozen-planner rung, FH1) each checkpoint is an
EXECUTOR: its as-trained row, then a SPLIT row where the planner's head
commits and the checkpoint walks; the planner's own plan-only row prints
first as the ceiling.
"""

from __future__ import annotations

import sys

from scripts.measure_commitments import run_chooser
from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.evaluation import format_optional_metric
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.evaluate import evaluate_per_model_chooser
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.scoring import evaluate_spec
from wargame_rl.wargame.selectors import (
    build_plan_only_chooser,
    build_split_chooser,
    is_per_model_checkpoint,
    label_for,
)

HELDOUT_SEED_BASE = 700_000
DEFAULT_MEMBERS = "squad_march_committed"
BAR = "squad_march_take"


def main() -> None:
    """Print the bar, then per checkpoint the as-trained and plan-only rows."""
    argv = sys.argv
    if len(argv) < 5:
        print(__doc__)
        raise SystemExit(1)
    config_path, n_episodes = argv[1], int(argv[2])
    seed_base = int(argv[3]) if argv[3] else HELDOUT_SEED_BASE
    rest = argv[4:]
    members = DEFAULT_MEMBERS
    planner: str | None = None
    kept: list[str] = []
    for token in rest:
        if token.startswith("planner="):
            planner = token[len("planner=") :]
        elif not is_per_model_checkpoint(token):
            members = token
        else:
            kept.append(token)
    checkpoints = kept
    config = load_env_config(config_path)
    config.render_mode = None
    seeds = [seed_base + i for i in range(n_episodes)]
    print(
        f"\nplan-only readout on {config_path} ({n_episodes} episodes, seeds "
        f"{seeds[0]}-{seeds[-1]}; members {members} on the plan-only rows)\n"
    )
    print(
        "| policy | mode | success | turns | held | on obj | persist | claim | distinct (mean / end) | complete | follow | leave |"
    )
    bar = evaluate_spec(BAR, config, seeds, BAR)
    bar_tally = run_chooser_for(BAR, None, config, seeds, members)
    print(_row(BAR, "script", bar, bar_tally))
    if planner is not None:
        # The frozen planner's own plan, executed by the bar's members: the
        # ceiling of every split row below it.
        p_envs = [PerModelEnv(config) for _ in range(min(8, n_episodes))]
        p_chooser = build_plan_only_chooser(
            planner, members, p_envs, seed=int(seeds[0])
        )
        p_result = evaluate_per_model_chooser(
            p_chooser.choose,
            p_envs,
            seeds,
            p_chooser.label,
            retimers=[PerStepReward(env) for env in p_envs],
        )
        p_env = PerModelEnv(config)
        p_tally = run_chooser(
            build_plan_only_chooser(
                planner, members, [p_env], seed=int(seeds[0])
            ).choose,
            p_env,
            config,
            seeds,
        )
        print(_row(f"planner {label_for(planner)}", "PLAN-ONLY", p_result, p_tally))
    for checkpoint in checkpoints:
        label = f"{label_for(checkpoint)}/{checkpoint.rsplit('/', 1)[-1]}"
        trained = evaluate_spec(checkpoint, config, seeds, label)
        trained_tally = run_chooser_for(checkpoint, None, config, seeds, members)
        print(_row(label, "as trained", trained, trained_tally))
        if planner is not None:
            s_envs = [PerModelEnv(config) for _ in range(min(8, n_episodes))]
            s_chooser = build_split_chooser(
                planner, checkpoint, s_envs, seed=int(seeds[0])
            )
            s_result = evaluate_per_model_chooser(
                s_chooser.choose,
                s_envs,
                seeds,
                s_chooser.label,
                retimers=[PerStepReward(env) for env in s_envs],
            )
            s_env = PerModelEnv(config)
            s_tally = run_chooser(
                build_split_chooser(
                    planner, checkpoint, [s_env], seed=int(seeds[0])
                ).choose,
                s_env,
                config,
                seeds,
            )
            print(_row(label, "SPLIT (planner>this)", s_result, s_tally))
            continue
        envs = [PerModelEnv(config) for _ in range(min(8, n_episodes))]
        chooser = build_plan_only_chooser(checkpoint, members, envs, seed=int(seeds[0]))
        plan_only = evaluate_per_model_chooser(
            chooser.choose,
            envs,
            seeds,
            chooser.label,
            retimers=[PerStepReward(env) for env in envs],
        )
        plan_env = PerModelEnv(config)
        plan_chooser = build_plan_only_chooser(
            checkpoint, members, [plan_env], seed=int(seeds[0])
        )
        plan_tally = run_chooser(plan_chooser.choose, plan_env, config, seeds)
        print(_row(label, "PLAN-ONLY", plan_only, plan_tally))
    print()


def run_chooser_for(spec, _unused, config, seeds, members):  # type: ignore[no-untyped-def]
    """The commitment tally of `spec` playing every decision itself."""
    from scripts.measure_commitments import run

    return run(spec, config, seeds)


def _row(label, mode, result, tally):  # type: ignore[no-untyped-def]
    return (
        f"| {label} | {mode} | {format_optional_metric(result.success_rate, 3)} | "
        f"{format_optional_metric(result.mean_turns, 2)} | {result.objectives_held:.2f} | "
        f"{result.final_fraction_at_objectives:.3f} | {tally.row_fields()} |"
    )


if __name__ == "__main__":
    main()
