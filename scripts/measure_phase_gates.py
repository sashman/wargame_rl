"""Measure every reward phase's success criteria against one checkpoint.

Answers "is the ladder ahead passable?" before a run reaches those phases, using
the same evaluation path training uses: run the episode to its end, then check
each phase's criteria against `env.last_step_context`.

Usage: just measure-phase-gates <checkpoint> <env_config> [n_episodes]
"""

from __future__ import annotations

import sys

import torch
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.criteria.registry import build_criteria
from wargame_rl.wargame.envs.types.config import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.net import TransformerNetwork


def main() -> None:
    checkpoint_path = sys.argv[1]
    config_path = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())

    env = create_environment(env_config=env_config)
    agent = Agent(env)
    net = TransformerNetwork.from_checkpoint(env, checkpoint_path)
    net.eval()

    criteria = [
        (
            phase.name,
            build_criteria(phase.success_criteria.type, phase.success_criteria.params),
        )
        for phase in env_config.reward_phases
    ]
    hits = {name: 0 for name, _ in criteria}
    final_fractions: list[float] = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.get_action(net, obs, epsilon=0.0)
                obs, _, done, truncated, _ = env.step(action)
                done = done or truncated
            ctx = env.last_step_context
            if ctx is None:
                continue
            for name, criterion in criteria:
                if criterion.is_successful(env, ctx):
                    hits[name] += 1
            final_fractions.append(
                ctx.distance_cache.fraction_at_objectives(
                    alive_mask=alive_mask_for(env.player_models)
                )
            )

    print(f"checkpoint : {checkpoint_path.split('/')[-1]}")
    print(f"episodes   : {n_episodes}\n")

    # The whole min_fraction curve from one sweep, so ladder rungs can be sized
    # from data instead of re-running an evaluation per candidate value.
    final_fractions.sort(reverse=True)
    print("min_fraction -> success rate")
    for candidate in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        rate = sum(1 for f in final_fractions if f >= candidate) / len(final_fractions)
        models = candidate * len(env.wargame_models)
        print(f"  {candidate:>4.2f} ({models:>4.1f} models) -> {rate:.2f}")
    median = final_fractions[len(final_fractions) // 2]
    print(
        f"  final fraction: median={median:.2f} "
        f"best={final_fractions[0]:.2f} worst={final_fractions[-1]:.2f}\n"
    )
    print(f"{'phase':<22} {'criteria':<24} {'threshold':>9} {'measured':>9}  verdict")
    for phase, (name, _) in zip(env_config.reward_phases, criteria):
        rate = hits[name] / n_episodes
        verdict = "PASSABLE" if rate >= phase.success_threshold else "BLOCKED"
        print(
            f"{name:<22} {phase.success_criteria.type:<24} "
            f"{phase.success_threshold:>9.2f} {rate:>9.2f}  {verdict}"
        )


if __name__ == "__main__":
    main()
