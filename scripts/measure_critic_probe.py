"""Does the critic believe the stack is right -- and is it?

The agent finishes with far more of its army alive than the scripts while
holding fewer objectives. Two accounts of that have been on file, and they
prescribe opposite work:

  REWARD  -- the return contains a survival premium the game does not, so the
             critic has learned to value a surplus model standing on a held
             point above the same model taking an empty one. Fix the reward.
  SEARCH  -- the critic already prefers the spread state and the policy simply
             never finds it. Fix exploration or the representation; no reward
             change will help.

There is also a third possibility nobody costed: the stack is CORRECT. A
`measure-objective-split` redistribution ceiling of +2.20 objectives says
surplus models could hold more ground, but that ceiling is deliberately
optimistic -- it charges no travel time and no return fire -- and forced
redistribution has measured *negative*. If spreading genuinely loses, the
"offence deficit" is a mis-description and every shaping proposal aimed at it
is aimed at nothing.

This separates all three, on frozen weights, with no training. At a chosen
battle round it takes a live game, finds an over-stacked objective and an empty
one, and rigidly translates one surplus squad from the first to the second --
rigid because a translation preserves the unit's internal geometry and so
cannot break the 2" chain, which would otherwise confound the value read with a
coherency penalty. It then reports two numbers per branch:

  dV      the critic's per-model values summed over the army, counterfactual
          minus factual -- what the agent BELIEVES redistribution is worth
  dVP     the realised `vp_margin`, counterfactual minus factual, from rolling
          both branches to the end of the episode -- what it IS worth

Reading the two together is the whole point:

  dV < 0, dVP < 0   the stack is correct and the critic knows it. The offence
                    framing is wrong; stop shaping against it.
  dV < 0, dVP > 0   the critic is miscalibrated in the direction the survival
                    premium predicts. Reward attribution is the story.
  dV > 0, dVP > 0   the critic already wants the spread. The failure is SEARCH;
                    do not spend a run on reward shaping.
  dV > 0, dVP < 0   the critic is optimistic about ground it cannot hold.

`reverse` runs the same probe backwards -- it takes a squad already holding a
point of its own and stacks it onto the army's biggest pile. This is the control
for off-distribution optimism. Both counterfactuals are states the policy never
produces, so a critic that simply extrapolates upward would score BOTH above the
factual. If `dV` is positive when spreading and negative when stacking, the
preference is directional and real; if it is positive both ways, the forward
result is an artefact and means nothing.

Both branches continue from a `deepcopy` taken at the branch point, so they
share the dice stream up to the first divergence -- common random numbers, not
a perfect pairing, because the two states consume the RNG at different rates
once they differ. It still removes most of the layout and deployment variance,
which is what dominates here.

Usage: just measure-critic-probe <ckpt> <env_config> [n_episodes] [rounds] [decode_topk]
"""

from __future__ import annotations

import copy
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.measure_checkpoint import HELDOUT_SEED_BASE, build_selector
from scripts.measure_maps import config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_counts_from_norms_offset,
)
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.model.net import TransformerNetwork

DEFAULT_MAPS_DIR = Path("configs/evaluation/maps_heldout")
VALUE_PREFIX = "ppo_model.value_network."
# Squadmates are planted this far apart: inside the 2" chain with margin, and
# wide enough that two bases never overlap at the radii in use here.
SQUADMATE_SPACING = 1.2


@dataclass
class Branch:
    """One factual/counterfactual pair taken at a single branch point."""

    table: str
    seed: int
    battle_round: int
    donor_group: int
    moved: int
    stack_count: int
    value_before: float
    delta_value: float
    delta_vp_margin: float


def load_value_network(checkpoint: str, env: WargameEnv) -> TransformerNetwork:
    """Rebuild the critic from a PPO checkpoint.

    `TransformerNetwork.from_checkpoint` resolves the *policy* prefix, so the
    value network needs its own load: it is a second network of the same shape
    under `ppo_model.value_network.`, and only it carries `value_head`.
    """
    state_dict = torch.load(checkpoint, weights_only=False, map_location="cpu")[
        "state_dict"
    ]
    value_state = {
        key[len(VALUE_PREFIX) :]: tensor
        for key, tensor in state_dict.items()
        if key.startswith(VALUE_PREFIX)
    }
    if not value_state:
        raise SystemExit(
            f"No value network in {checkpoint} (looked for {VALUE_PREFIX})"
        )
    network = TransformerNetwork.from_state_dict(env, value_state, is_policy=False)
    network.eval()
    return network


def army_value(network: TransformerNetwork, env: WargameEnv) -> float:
    """The critic's per-model values for the live board, summed over the army.

    Summed rather than averaged because the mean is a per-survivor figure and
    would *rise* when a model dies, which is the confound this whole script
    exists to price.
    """
    with torch.no_grad():
        values = network(observation_to_tensor(env._get_obs(), network.device))
    alive = torch.as_tensor(
        alive_mask_for(env.player_models), dtype=torch.bool, device=values.device
    )
    return float(values.squeeze(0)[alive].sum())


def in_range_counts(models: list[WargameModel], env: WargameEnv) -> np.ndarray:
    """Per-objective count of alive models in range, the scoring definition."""
    cache = compute_distances(models, env.objectives, alive_mask=alive_mask_for(models))
    return objective_counts_from_norms_offset(
        cache.model_obj_norms_offset, cache.obj_radii
    )


def choose_branch(
    env: WargameEnv, min_stack: int, reverse: bool = False
) -> tuple[int, int] | None:
    """Pick a donor squad on an over-stacked objective and an empty target.

    With `reverse`, the roles swap: the donor holds a point of its own and the
    target is the army's biggest pile. That is the control described above.

    Returns `(group_id, objective_index)`, or None when this board offers no
    such move -- which is itself informative and is counted in the report.

    The donor must leave the stacked objective still controlled: the question
    is what a *surplus* model is worth, not what abandoning a point is worth.
    """
    player_counts = in_range_counts(env.player_models, env)
    opponent_counts = in_range_counts(env.opponent_models, env)

    stacked = [
        index
        for index in range(len(env.objectives))
        if player_counts[index] >= min_stack
        and player_counts[index] > opponent_counts[index]
    ]
    empty = [
        index
        for index in range(len(env.objectives))
        if player_counts[index] == 0 and opponent_counts[index] == 0
    ]
    if not stacked or not empty:
        return None

    biggest = max(stacked, key=lambda index: player_counts[index])
    if reverse:
        # Donor is a squad on some OTHER held point; target is the biggest pile.
        held_elsewhere = [
            index
            for index in range(len(env.objectives))
            if index != biggest
            and 0 < player_counts[index] <= min_stack
            and player_counts[index] > opponent_counts[index]
        ]
        if not held_elsewhere:
            return None
        source = max(held_elsewhere, key=lambda index: player_counts[index])
    else:
        source = biggest
    cache = compute_distances(
        env.player_models,
        env.objectives,
        alive_mask=alive_mask_for(env.player_models),
    )
    on_source = cache.model_obj_norms_offset[:, source] <= cache.obj_radii[source]
    alive = alive_mask_for(env.player_models)

    by_group: dict[int, int] = {}
    for index, model in enumerate(env.player_models):
        if alive[index] and on_source[index]:
            by_group[int(model.group_id)] = by_group.get(int(model.group_id), 0) + 1
    if not by_group:
        return None

    donor = max(by_group, key=lambda group: by_group[group])
    if reverse:
        # No surplus requirement: abandoning the point is the whole point.
        return donor, biggest
    # Keep the point: the surplus has to be genuinely surplus.
    if player_counts[source] - by_group[donor] <= opponent_counts[source]:
        return None

    target = min(
        empty,
        key=lambda index: float(
            np.min(cache.model_obj_norms_offset[:, index][alive & on_source])
        ),
    )
    return donor, target


def relocate_squad(env: WargameEnv, group_id: int, objective_index: int) -> int:
    """Rigidly translate a squad onto an objective; return how many moved.

    Rigid because the unit's internal geometry is what the coherency rule
    constrains: translating every model by the same vector cannot break a
    chain that was intact. Where the objective is too small to hold the squad's
    original spread, models are re-planted on a small ring instead, still
    inside the 2" chain.
    """
    objective = env.objectives[objective_index]
    centre = objective.location
    members = [
        index
        for index, model in enumerate(env.player_models)
        if int(model.group_id) == group_id and model.stats.get("current_wounds", 1) > 0
    ]
    if not members:
        return 0

    occupied = [
        (float(model.location[0]), float(model.location[1]))
        for index, model in enumerate(env.player_models)
        if index not in members and model.stats.get("current_wounds", 1) > 0
    ] + [
        (float(model.location[0]), float(model.location[1]))
        for model in env.opponent_models
        if model.stats.get("current_wounds", 1) > 0
    ]

    for rank, index in enumerate(members):
        model = env.player_models[index]
        angle = 2.0 * np.pi * rank / max(len(members), 1)
        radius = 0.0 if rank == 0 else SQUADMATE_SPACING
        x = float(centre[0]) + radius * float(np.cos(angle))
        y = float(centre[1]) + radius * float(np.sin(angle))
        clearance = 2.0 * model.base_radius + 0.05
        for _ in range(12):
            if all(
                (x - ox) ** 2 + (y - oy) ** 2 >= clearance**2 for ox, oy in occupied
            ):
                break
            x += clearance
        model.location = position(x, y)
        model.previous_location = None
        occupied.append((x, y))
    return len(members)


def play_out(env: WargameEnv, select: ActionSelector) -> float:
    """Run to the end of the episode and return the realised `vp_margin`."""
    observation = env._get_obs()
    done = False
    while not done:
        observation, _reward, done, _truncated, _info = env.step(
            select(observation, env)
        )
    return float(env.player_vp - env.opponent_vp)


def probe_episode(
    env: WargameEnv,
    select: ActionSelector,
    network: TransformerNetwork,
    branch_round: int,
    min_stack: int,
    table: str,
    seed: int,
    reverse: bool = False,
) -> Branch | None:
    """Play to `branch_round`, fork, relocate one squad, and score both sides."""
    observation = env._get_obs()
    done = False
    while not done:
        state = env.game_clock_state
        if state.battle_round is not None and state.battle_round >= branch_round:
            break
        observation, _reward, done, _truncated, _info = env.step(
            select(observation, env)
        )
    if done:
        return None

    chosen = choose_branch(env, min_stack, reverse)
    if chosen is None:
        return None
    donor_group, target = chosen

    factual = copy.deepcopy(env)
    counterfactual = copy.deepcopy(env)
    stack_count = int(np.max(in_range_counts(env.player_models, env)))

    value_before = army_value(network, factual)
    moved = relocate_squad(counterfactual, donor_group, target)
    if moved == 0:
        return None
    value_after = army_value(network, counterfactual)

    return Branch(
        table=table,
        seed=seed,
        battle_round=int(env.game_clock_state.battle_round or 0),
        donor_group=donor_group,
        moved=moved,
        stack_count=stack_count,
        value_before=value_before,
        delta_value=value_after - value_before,
        delta_vp_margin=play_out(counterfactual, select) - play_out(factual, select),
    )


def summarise(label: str, branches: list[Branch], attempted: int) -> None:
    """Print both deltas with standard errors, sign counts and their correlation.

    The correlation is the load-bearing statistic. `dV` is in reward units and
    `dVP` in victory points, so their *magnitudes* are not comparable and a
    mean `dV` near zero could always be argued to be a scale artefact. Whether
    they move together cannot: if the critic had any grip on what redistributing
    is worth, the branches it valued most would be the ones that paid most.
    """
    if not branches:
        print(f"  {label}: no branch point found in {attempted} episodes")
        return
    values = [branch.delta_value for branch in branches]
    margins = [branch.delta_vp_margin for branch in branches]

    def stat(sample: list[float]) -> str:
        if len(sample) < 2:
            return f"{sample[0]:+7.2f}       -"
        error = statistics.stdev(sample) / np.sqrt(len(sample))
        return f"{statistics.mean(sample):+7.2f} +/-{error:5.2f}"

    correlation = (
        float(np.corrcoef(values, margins)[0, 1])
        if len(values) > 2 and statistics.stdev(values) > 0
        else float("nan")
    )
    print(
        f"  {label}: n={len(branches):3d}/{attempted:3d}"
        f"  dV {stat(values)} ({sum(1 for v in values if v > 0)}/{len(values)}+)"
        f"  dVP {stat(margins)} ({sum(1 for m in margins if m > 0)}/{len(margins)}+)"
        f"  corr {correlation:+.2f}"
        f"  |  V {statistics.mean([b.value_before for b in branches]):6.1f}"
        f"  stack {statistics.mean([b.stack_count for b in branches]):.1f}"
        f"  moved {statistics.mean([b.moved for b in branches]):.1f}"
    )


def main() -> None:
    """Fork live games, relocate a surplus squad, and price it two ways."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    checkpoint = argv[1]
    config_path = argv[2]
    n_episodes = int(argv[3]) if len(argv) > 3 and argv[3] else 10
    rounds = (
        [int(part) for part in argv[4].split(",")]
        if len(argv) > 4 and argv[4]
        else [3, 6, 10]
    )
    decode_topk = int(argv[5]) if len(argv) > 5 and argv[5] else 3
    reverse = len(argv) > 6 and argv[6].lower() in {"reverse", "true", "1"}
    min_stack = 3

    config = load_env_config(config_path, **overrides)
    maps = load_maps(DEFAULT_MAPS_DIR)
    seeds = [HELDOUT_SEED_BASE + index for index in range(n_episodes)]

    print(f"\ncritic probe: {checkpoint}")
    print(f"  on {config_path}{describe(overrides)}")
    print(
        f"  {len(maps)} held-out tables x {n_episodes} episodes, seeds "
        f"{seeds[0]}-{seeds[-1]}, decode_topk={decode_topk}, "
        f"branch rounds {rounds}, surplus stack >= {min_stack}"
        + ("  [REVERSE: stacking, the control]" if reverse else "")
    )
    print(
        "\n  dV  = critic's summed army value, counterfactual - factual "
        "(what the agent believes)"
    )
    print(
        "  dVP = realised vp_margin, counterfactual - factual "
        "(what it is actually worth)\n"
    )

    for branch_round in rounds:
        collected: list[Branch] = []
        attempted = 0
        for terrain_map in maps:
            per_map = config_for_map(config, terrain_map)
            probe_env = create_environment(env_config=per_map)
            select, _net = build_selector(checkpoint, probe_env, decode_topk, False)
            network = load_value_network(checkpoint, probe_env)
            for seed in seeds:
                probe_env.reset(seed=seed)
                attempted += 1
                branch = probe_episode(
                    probe_env,
                    select,
                    network,
                    branch_round,
                    min_stack,
                    terrain_map.name,
                    seed,
                    reverse,
                )
                if branch is not None:
                    collected.append(branch)
            probe_env.close()
        summarise(f"round {branch_round:>2}", collected, attempted)
    print()


if __name__ == "__main__":
    main()
