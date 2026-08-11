"""Which reward calculator actually pays the agent, and how much of it is global.

A reward config is a budget, but nothing in the repo reports how that budget is
actually *spent* by a trained policy. Weights are not shares: a term with a large
weight that rarely fires is cheap, and a small global term that pays every model
on every step can quietly dominate. Three rounds of arms here tuned what
objectives pay while the largest single share sat in a term nobody adjusted.

Reports each calculator's total contribution per episode, its share of gross
income, and — the number that motivated this script — **how much of the ledger is
global**. A global term is broadcast whole to every model regardless of what that
model did, so it is a floor: income a model receives for standing still. When the
floor is large, a shaping term that pays for *moving* is competing against free
money, and the policy will look unresponsive to weight changes that never touch
the floor.

Two readings this is for:

- Before training an arm, check the term you are about to tune is actually a
  meaningful share. `model_kills` looked like the driver of a range-managing
  policy and turned out to be 4.5% of income.

  **Read that inference narrowly.** A share of *mean* income rules a term out as
  the largest income stream. It does **not** rule it out as the driver of a
  behaviour: what moves a policy gradient is a term's *variation across the
  actions being compared*, not its share of the mean. `objective_hold` is 46% of
  income and nearly constant across the choices of a model already standing on a
  point, while `model_kills` is 4.5% but arrives as a lumpy 2.0 in one model's
  own row. Ruling a term out as a behavioural driver needs a differential, which
  this script does not produce.
- When a shaping term is not producing the behaviour it prices, compare its
  share against the global floor before raising its weight.

Note `closest_objective_v2`'s breakdown keys are sub-parts of its own total and
are reported separately: summing them beside the calculators double counts, and
`target_obj_idx` is an objective index rather than reward at all.

Accepts either a scripted baseline name or a checkpoint path, so a policy and a
baseline are measured by one code path on identical layouts.

Usage: just measure-income-share <policy|ckpt> <env_config> [n_episodes]
"""

from __future__ import annotations

import sys
from collections import defaultdict

from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_checkpoint import HELDOUT_SEED_BASE, build_selector
from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector, selector_for
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
)
from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


def _selector_for_policy(policy_name: str, env: WargameEnv) -> ActionSelector:
    """A baseline by registry name, or a policy loaded from a checkpoint path."""
    if policy_name in get_registry():
        return selector_for(build_baseline_policy(policy_name))
    select, _net = build_selector(policy_name, env)
    return select


def _calculator_kinds(env: WargameEnv) -> dict[str, str]:
    """Map each calculator name in the active phase to "global" or "per-model".

    Read off the live phase rather than the config, so a term that is registered
    as global cannot be mislabelled here by a stale reading of the YAML.
    """
    phase = env.phase_manager.current_phase
    kinds = {name: "per-model" for name, _calc in phase.per_model_calculators}
    for name, calculator in phase.global_calculators:
        kinds[name] = "global" if isinstance(calculator, GlobalRewardCalculator) else ""
    return kinds


def main() -> None:
    """Print the per-calculator income ledger for one policy."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy_name = sys.argv[1]
    config_path = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    env = create_environment(env_config=env_config)
    select = _selector_for_policy(policy_name, env)
    kinds = _calculator_kinds(env)

    totals: dict[str, float] = defaultdict(float)
    steps = 0

    for episode in range(n_episodes):
        observation, _info = env.reset(seed=HELDOUT_SEED_BASE + episode)
        done = False
        while not done:
            action = select(observation, env)
            observation, _reward, terminated, truncated, _step_info = env.step(action)
            done = terminated or truncated
            steps += 1
            for key, value in env.last_reward_breakdown.items():
                totals[key] += float(value)

    named = {name: totals.get(name, 0.0) for name in kinds if name in totals}
    gross = sum(abs(value) for value in named.values()) or 1.0

    print(f"\n{policy_name}   {config_path}")
    print(f"{n_episodes} episodes, {steps} steps\n")
    print(f"{'calculator':<24}{'kind':>11}{'per episode':>14}{'share':>9}")
    print("-" * 58)
    for name, value in sorted(named.items(), key=lambda item: -abs(item[1])):
        print(
            f"{name:<24}{kinds[name]:>11}{value / n_episodes:>14.2f}"
            f"{abs(value) / gross:>9.3f}"
        )
    print("-" * 58)

    global_share = sum(
        abs(value) for name, value in named.items() if kinds[name] == "global"
    )
    print(f"\nglobal share of gross income   {global_share / gross:>8.3f}")
    print(
        "  a global term pays every model whatever it did, so this is the floor\n"
        "  a model earns for standing still -- what a movement term competes with\n"
    )

    components = {
        name: value
        for name, value in totals.items()
        if name not in kinds and abs(value) / n_episodes > 0.01
    }
    if components:
        print(f"{'breakdown component':<44}{'per episode':>14}")
        print("-" * 58)
        for name, value in sorted(components.items(), key=lambda item: -abs(item[1])):
            print(f"{name:<44}{value / n_episodes:>14.2f}")
        print()


if __name__ == "__main__":
    main()
