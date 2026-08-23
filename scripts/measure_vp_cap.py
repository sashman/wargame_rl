"""How often the mission's per-turn VP cap binds, and what it discards.

`DefaultVPCalculator` pays `min(cap_per_turn, controlled * vp_per_objective)` --
15 and 5 by default, so the **fourth** objective a side controls is worth
exactly nothing while the generated tables carry five or six. That makes "send
the spare squad to take one more point" a decision the rules can price at zero,
and no amount of training lifts a rules ceiling.

Reading `held` cannot see this: it is an end-state snapshot of how many points a
side finished on, with no notion of which of them were paid. This reports the
whole per-step distribution of controlled objectives for both sides, the share
of steps at and above the cap, and the fraction of VP the cap throws away.

What it established, on `configs/golden/25v25_maps_two_mode.yaml`, held-out
nine, n=20, seeds 700000+:

    squad_march_take  (K=1)  55.6% of steps at 3+, 23.9% above -- 10.1% discarded
    squad_march_deny  (K=1)  46.3%              ,  6.6%        --  2.9% discarded
    the v3.0 agent    (K=3)  22.3%              ,  2.0%        --  1.1% discarded

So the cap taxes the SCRIPTS, and `take` hardest -- it penalises precisely the
surplus-grabbing that distinguishes `take` from `deny`. The agent never reaches
it: sitting on two objectives 44% of the time, its offence shortfall is real and
almost entirely payable.

⚠ Score the arm config, not its refereed twin, when comparing against these
numbers -- attrition deletes models in breach and therefore moves control.

Usage: just measure-vp-cap <policy|ckpt> <env_config> [n_episodes] [decode_topk]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.measure_maps import config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.mission.vp_calculator import DefaultVPCalculator
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

HELDOUT_SEED_BASE = 700_000
DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")


def cap_and_rate(config: WargameEnvConfig) -> tuple[int, int]:
    """The mission's `(cap_per_turn, vp_per_objective)`, from the config itself.

    Read rather than hardcoded: a mission tuned to a different cap would make
    every percentage below wrong while still printing.
    """
    params = dict(config.mission.params)
    reference = DefaultVPCalculator()
    return (
        int(params.get("cap_per_turn", reference.cap_per_turn)),
        int(params.get("vp_per_objective", reference.vp_per_objective)),
    )


def collect(
    policy: str,
    config: WargameEnvConfig,
    seeds: list[int],
    decode_topk: int,
) -> tuple[Counter[int], Counter[int], Counter[int]]:
    """Per-step controlled-objective counts for both sides, over every map."""
    player: Counter[int] = Counter()
    opponent: Counter[int] = Counter()
    objectives_per_table: Counter[int] = Counter()

    for terrain_map in load_maps(DEFAULT_MAPS_DIR):
        per_map = config_for_map(config, terrain_map)
        env = create_environment(env_config=per_map)
        select = build_action_selector(policy, env, decode_topk).select
        for seed in seeds:
            observation, _info = env.reset(seed=seed)
            done = False
            while not done:
                observation, _r, done, _t, _i = env.step(select(observation, env))
                player_cache = compute_distances(
                    env.player_models,
                    env.objectives,
                    alive_mask=alive_mask_for(env.player_models),
                )
                opponent_cache = compute_distances(
                    env.opponent_models,
                    env.objectives,
                    alive_mask=alive_mask_for(env.opponent_models),
                )
                held_by_player, held_by_opponent = (
                    objective_ownership_from_norms_offset(
                        player_cache.model_obj_norms_offset,
                        opponent_cache.model_obj_norms_offset,
                        player_cache.obj_radii,
                    )
                )
                player[int(np.sum(held_by_player))] += 1
                opponent[int(np.sum(held_by_opponent))] += 1
                objectives_per_table[len(env.objectives)] += 1
        env.close()
    return player, opponent, objectives_per_table


def report(name: str, histogram: Counter[int], cap: int, per_objective: int) -> None:
    """Print the control distribution and what the cap discarded."""
    total = sum(histogram.values())
    if not total:
        return
    at_cap = cap // per_objective
    earned = sum(min(cap, c * per_objective) * n for c, n in histogram.items())
    uncapped = sum(c * per_objective * n for c, n in histogram.items())
    binding = sum(n for c, n in histogram.items() if c >= at_cap)
    surplus = sum(n for c, n in histogram.items() if c > at_cap)
    discarded = 100.0 * (uncapped - earned) / uncapped if uncapped else 0.0
    print(f"  {name}")
    print(
        "    control distribution: "
        + "  ".join(
            f"{c}:{100.0 * histogram[c] / total:.0f}%" for c in sorted(histogram)
        )
    )
    print(
        f"    steps at the cap (>={at_cap} controlled): {100.0 * binding / total:.1f}%"
    )
    print(
        f"    steps ABOVE the cap (>{at_cap}, surplus pays 0): "
        f"{100.0 * surplus / total:.1f}%"
    )
    print(
        f"    VP earned {earned / total:.2f}/step vs {uncapped / total:.2f} uncapped"
        f"  -> the cap discards {discarded:.1f}%"
    )


def main() -> None:
    """Print how often the per-turn VP cap bound, and what it threw away."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 20
    decode_topk = int(argv[4]) if len(argv) > 4 and argv[4] else 1

    config = load_env_config(config_path, **overrides)
    cap, per_objective = cap_and_rate(config)
    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    player, opponent, per_table = collect(policy, config, seeds, decode_topk)

    total_tables = sum(per_table.values())
    print(f"\n{policy} on {config_path}{describe(overrides)}")
    print(
        f"  ({n_episodes} episodes per map, seeds {seeds[0]}-{seeds[-1]}, "
        f"decode_topk={decode_topk}, cap {cap} at {per_objective} per objective)"
    )
    print(
        "  objectives per table: "
        + ", ".join(
            f"{k} ({100.0 * v / total_tables:.0f}%)"
            for k, v in sorted(per_table.items())
        )
    )
    report("PLAYER", player, cap, per_objective)
    report("OPPONENT", opponent, cap, per_objective)
    print()


if __name__ == "__main__":
    main()
