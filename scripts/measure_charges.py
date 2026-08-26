"""Does a policy use the charge phase, and does it use it competently?

The primary readouts of [the melee teaching goal](../docs/melee-teaching-goal.md),
which exists because melee is a **core rule**: the question is how well a policy
charges, never whether charging is worth having.

    just measure-charges <policy|ckpt> <config.yaml> [n_episodes] [decode_topk]

⚠ **Read `stood` per episode, not the standing fraction.** The fraction's
denominator is the policy's own declaration count, so it *rises* when a policy
declares less -- it is not monotone in competence, and a gate written on it
rejects policies that land more charges. `stood/ep` is the numerator alone, with
a hard floor at zero.

⚠ **Quote the K.** At `decode_topk` 3 the joint decoder picks legal combinations
FOR the network, so these counts measure the decoder: a randomly-initialised
network stands 1.17-3.67 charges an episode at K=3 and 0.00-1.67 at K=1. Training
decodes at K=1, so **K=1 is the column that decides**.

⚠ **A charge STOOD iff the referee did not put its models back where they
started.** Never read `charged_this_turn` after the charge step: with `fight` in
`skip_phases` the fight resolves on the boundary inside the same step and clears
the flag, so it always reads False. That has cost two measurements.
"""

from __future__ import annotations

import sys

import numpy as np

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

STAY_ACTION = 0


def measure(
    selector_spec: str,
    config_path: str,
    n_episodes: int,
    decode_topk: int,
    overrides: dict[str, str],
) -> dict[str, float]:
    """Per-episode charge counts for one policy, judged by the referee's verdict."""
    env = create_environment(load_env_config(config_path, **overrides))
    declared = attempted = stood = 0
    vp = 0.0
    coherent_steps = 0.0
    coherent_total = 0
    try:
        selector = build_action_selector(selector_spec, env, decode_topk=decode_topk)
        for episode in range(n_episodes):
            observation, _ = env.reset(seed=700000 + episode)
            done = False
            while not done:
                phase = env.game_clock_state.phase
                action = selector.select(observation, env)
                units: dict[int, list[int]] = {}
                if phase is BattlePhase.charge:
                    for index, model in enumerate(env.wargame_models):
                        if model.is_alive and getattr(model, "declared_charge", False):
                            units.setdefault(int(model.group_id), []).append(index)
                    declared += len(units)
                before = {
                    index: np.array(env.wargame_models[index].location, copy=True)
                    for members in units.values()
                    for index in members
                }
                moving = {
                    group
                    for group, members in units.items()
                    if any(action.actions[i] != STAY_ACTION for i in members)
                }
                attempted += len(moving)

                observation, _r, terminated, truncated, info = env.step(action)

                # The referee reverts the WHOLE unit, so a unit that attempted a
                # charge and still sits on its start positions was reverted.
                for group in moving:
                    if any(
                        not np.array_equal(before[i], env.wargame_models[i].location)
                        for i in units[group]
                    ):
                        stood += 1
                done = terminated or truncated
            # ⚠ The POLICY'S OWN figure, not the realised one. This config
            # referees with `enforce_move: revert_unit`, under which the
            # realised rate is 1.000 whatever the policy does -- a metric
            # sampled after a corrective wrapper measures the wrapper, and
            # reading it that way once published a policy intending 0.630 as
            # 1.000. `evaluate.py` prefers the same field for the same reason.
            intended = env.intended_coherency_rate
            if intended is not None:
                coherent_steps += float(intended)
                coherent_total += 1
            vp += float(env.player_vp - env.opponent_vp)
    finally:
        env.close()
    return {
        "declared": declared / n_episodes,
        "attempted": attempted / n_episodes,
        "stood": stood / n_episodes,
        "fraction": stood / attempted if attempted else float("nan"),
        "vp": vp / n_episodes,
        "coherent": coherent_steps / coherent_total if coherent_total else float("nan"),
    }


def main() -> None:
    """Print one row of charge counts for the named policy or checkpoint."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    selector_spec = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 else 20
    decode_topk = int(argv[4]) if len(argv) > 4 else 1

    print(
        f"{config_path}{describe(overrides)}  ({n_episodes} episodes, "
        f"seeds 700000+, decode_topk={decode_topk})\n"
    )
    result = measure(selector_spec, config_path, n_episodes, decode_topk, overrides)
    print(
        f"  {'policy':38s} {'decl/ep':>8s} {'tried/ep':>9s} {'stood/ep':>9s} "
        f"{'frac':>7s} {'coherent':>9s} {'vp':>8s}"
    )
    print(
        f"  {selector_spec[:38]:38s} {result['declared']:8.2f} "
        f"{result['attempted']:9.2f} {result['stood']:9.2f} {result['fraction']:7.3f} "
        f"{result['coherent']:9.3f} {result['vp']:+8.1f}"
    )


if __name__ == "__main__":
    main()
