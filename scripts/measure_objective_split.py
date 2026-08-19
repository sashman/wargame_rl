"""Break `objectives_held` down into *why* an objective was not held.

`held` says a policy controls 1.42 of 3 objectives; it does not say whether the
missing ones were abandoned, contested and narrowly lost, or lost by a mile. Those
three call for different fixes, and the aggregate cannot tell them apart.

Reports, per objective at episode end, the (player, opponent) counts sorted by
player count, the outcome class, and the **redistribution counterfactual**: how
many objectives the same surviving models would hold if every model surplus to
`opponent_count + 1` on an already-held objective were moved to the closest
objective the policy lost. That number bounds what any pure re-allocation lever
can buy — if it is ~0, the deficit is not misallocation and no reward reshaping
will close it.

Accepts either a scripted baseline name or a checkpoint path, so the agent and
the bar are measured by one code path on identical layouts.

Usage: just measure-objective-split <policy|ckpt> <env_config> [n_episodes]
"""

from __future__ import annotations

import sys

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_checkpoint import HELDOUT_SEED_BASE
from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector


def _counts_at_end(env: WargameEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-objective (player count, opponent count, radius) at the current state.

    Same rule VP scores on: a model counts when its centre is inside the disc,
    and control needs strictly more models than the other side.
    """
    alive = alive_mask_for(env.wargame_models)
    cache = compute_distances(env.wargame_models, env.objectives, alive_mask=alive)
    player_counts = (cache.model_obj_norms_offset[alive] <= cache.obj_radii).sum(axis=0)

    opponent_alive = alive_mask_for(env.opponent_models)
    if env.opponent_models:
        opponent_norms = compute_distances(
            env.opponent_models, env.objectives, alive_mask=opponent_alive
        ).model_obj_norms_offset
        opponent_counts = (opponent_norms <= cache.obj_radii).sum(axis=0)
    else:
        opponent_counts = np.zeros(len(env.objectives), dtype=int)

    return (
        np.atleast_1d(player_counts).astype(int),
        np.atleast_1d(opponent_counts).astype(int),
        np.atleast_1d(cache.obj_radii),
    )


def _redistribution_gain(player: np.ndarray, opponent: np.ndarray) -> int:
    """Extra objectives held if surplus models moved to the cheapest losses.

    Surplus is everything past `opponent_count + 1` on a held objective — those
    models change nothing about who scores. Spending them on the *cheapest*
    unheld objectives first is the best case for any re-allocation, so this is an
    upper bound rather than an estimate.

    Ignores travel time and return fire, both of which only reduce the gain. A
    zero here therefore rules re-allocation out; a large one does not rule it in.
    """
    held = player > opponent
    surplus = int(np.clip(player - opponent - 1, 0, None)[held].sum())

    # Cost to flip an unheld objective: one more model than the opponent has.
    costs = sorted(int(o - p + 1) for p, o in zip(player[~held], opponent[~held]))
    gained = 0
    for cost in costs:
        if cost <= surplus:
            surplus -= cost
            gained += 1
    return gained


def _classify(player: int, opponent: int) -> str:
    """Name why this objective did or did not score."""
    if player > opponent:
        return "held"
    if player == 0:
        return "abandoned"
    if opponent - player <= 2:
        return "lost_close"
    return "lost_far"


def _pad_to(rows: list[np.ndarray], width: int) -> np.ndarray:
    """Stack ragged per-episode rows into one matrix, padding short ones with NaN.

    Objectives vary per table on a `map_pool` config, so the rows are ragged.
    NaN rather than 0 because a table with five objectives has no sixth, and a
    zero there would read as a sixth objective that nobody is standing on.
    """
    padded = np.full((len(rows), width), np.nan, dtype=float)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = row
    return padded


def main() -> None:
    """Print the objective-by-objective breakdown for one policy."""
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    policy_name = sys.argv[1]
    config_path = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    with open(config_path) as handle:
        env_config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    env_config.render_mode = None
    env = create_environment(env_config=env_config)

    resolved = build_action_selector(policy_name, env)
    select: ActionSelector = resolved.select
    label = resolved.label

    seeds = [HELDOUT_SEED_BASE + i for i in range(n_episodes)]

    classes: dict[str, int] = {
        "held": 0,
        "lost_close": 0,
        "lost_far": 0,
        "abandoned": 0,
    }
    # Ranked within each episode so "the objective it committed to" is comparable
    # across episodes -- objective index is arbitrary, occupancy rank is not.
    ranked_player: list[np.ndarray] = []
    ranked_opponent: list[np.ndarray] = []
    surpluses: list[int] = []
    gains: list[int] = []
    alive_totals: list[int] = []

    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = select(observation, env)
            observation, _r, terminated, truncated, _i = env.step(action)

        player, opponent, _radii = _counts_at_end(env)
        order = np.argsort(-player)
        ranked_player.append(player[order])
        ranked_opponent.append(opponent[order])
        for p, o in zip(player, opponent):
            classes[_classify(int(p), int(o))] += 1
        held_mask = player > opponent
        surpluses.append(int(np.clip(player - opponent - 1, 0, None)[held_mask].sum()))
        gains.append(_redistribution_gain(player, opponent))
        alive_totals.append(int(alive_mask_for(env.wargame_models).sum()))

    # A `map_pool` config draws a different table each episode and the real
    # tables carry 5 OR 6 objectives, so these rows are ragged and `np.stack`
    # raises. Pad the short rows with NaN rather than 0: a table that simply has
    # no sixth objective must not read as a sixth objective standing empty,
    # which is what a zero would say to every rate below.
    n_objectives = max(len(row) for row in ranked_player)
    player_matrix = _pad_to(ranked_player, n_objectives)
    opponent_matrix = _pad_to(ranked_opponent, n_objectives)
    total = sum(len(row) for row in ranked_player)

    print(f"\n{label}   {config_path}")
    print(f"{n_episodes} episodes, seeds {seeds[0]}-{seeds[-1]}\n")

    print("per-objective end state, ranked by player occupancy within each episode")
    print(f"{'rank':<8}{'player':>9}{'opponent':>10}{'held rate':>12}{'n':>8}")
    print("-" * 47)
    for rank in range(n_objectives):
        present = ~np.isnan(player_matrix[:, rank])
        if not present.any():
            continue
        held_rate = float(
            (player_matrix[present, rank] > opponent_matrix[present, rank]).mean(),
        )
        # `n` is printed because a rank past the fifth exists on only some of
        # the tables, and a mean over a third of the episodes should not be
        # read like the others.
        print(
            f"{rank + 1:<8}{np.nanmean(player_matrix[:, rank]):>9.2f}"
            f"{np.nanmean(opponent_matrix[:, rank]):>10.2f}{held_rate:>12.2f}"
            f"{int(present.sum()):>8}"
        )

    print(f"\n{'outcome':<14}{'share of objectives':>21}")
    print("-" * 35)
    for name, count in classes.items():
        print(f"{name:<14}{count / total:>21.3f}")

    # NaN compares false, so a padded slot cannot count as held.
    with np.errstate(invalid="ignore"):
        mean_held = float((player_matrix > opponent_matrix).sum(axis=1).mean())
    print(f"\nmodels alive at end        {np.mean(alive_totals):>8.1f}")
    print(f"objectives held            {mean_held:>8.2f}")
    print(f"surplus models on held     {np.mean(surpluses):>8.2f}")
    print(f"redistribution ceiling     {mean_held + np.mean(gains):>8.2f}", end="")
    print(f"   (+{np.mean(gains):.2f})\n")


if __name__ == "__main__":
    main()
