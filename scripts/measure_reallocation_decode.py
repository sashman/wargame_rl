"""Does a critic-DIRECTED reallocation decode buy vp at play?  The kill screen.

The panel slate's R5, funded by the critic-probe result (`just
measure-critic-probe`): the critic values spreading a surplus squad correctly
(+2.63 ± 0.32, realising +3.85 ± 1.81 over 634 forked games) and the policy
does not do it. `corr(dV, dVP)` is ~0, so the critic can supply a DIRECTION and
never a ranking — this decode therefore builds exactly ONE candidate per
movement step and asks only for the sign.

    just measure-realloc <ckpt> <config.yaml> [n_episodes] [decode_topk] [min_stack]

Per movement phase, after the normal decode:

1. `choose_branch` (imported from the probe, unchanged — same instrument, same
   definitions) nominates a surplus donor squad on the army's biggest stack and
   the cheapest empty objective, or nothing.
2. The donor is VIRTUALLY translated one full move toward the target, the
   critic prices both boards, and the move is approved iff dV > 0.
3. Approved, the donor's members are redirected onto the ONE shared
   (angle, rung) of the movement grid that brings the squad centroid closest to
   the target — rigid, so a chain that was intact stays intact, and the env's
   own referee still judges the executed move.

⚠ PLAY-TIME ONLY. Folding any decode into training measured −51.8 vp.

⚠ PRE-REGISTERED KILL (before any number existed): mean vp gain against the
same checkpoints' no-realloc rows < +1.0, or negative on 3 of 6 seeds — dead,
per the slate's own rule. The no-realloc control rows are the existing grid
scores at the same n, seeds and K.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from scripts.measure_critic_probe import army_value, choose_branch, load_value_network
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector


def measure(
    checkpoint: str,
    config_path: str,
    n_episodes: int,
    decode_topk: int,
    min_stack: int,
    overrides: dict[str, str],
) -> None:
    """Score one checkpoint with the reallocation decode active."""
    env = create_environment(load_env_config(config_path, **overrides))
    selector = build_action_selector(checkpoint, env, decode_topk=decode_topk)
    critic = load_value_network(checkpoint, env)
    handler = env.player_action_handler
    movement = handler.movement_slice
    grid = handler._displacements.reshape(-1, 2)  # (n_angles*n_speeds, 2)

    vp = 0.0
    redirects = 0
    approvals = 0
    nominations = 0
    for episode in range(n_episodes):
        observation, _ = env.reset(seed=700000 + episode)
        done = False
        while not done:
            phase = env.game_clock_state.phase
            action = selector.select(observation, env)
            if phase is BattlePhase.movement:
                branch = choose_branch(env, min_stack)
                if branch is not None:
                    nominations += 1
                    donor, target = branch
                    members = [
                        index
                        for index, model in enumerate(env.player_models)
                        if int(model.group_id) == donor and model.is_alive
                    ]
                    if members:
                        centre = np.asarray(
                            env.objectives[target].location, dtype=float
                        )
                        positions = np.array(
                            [env.player_models[i].location for i in members],
                            dtype=float,
                        )
                        centroid = positions.mean(axis=0)
                        # One full move toward the target for the VIRTUAL board.
                        bearing = centre - centroid
                        norm = float(np.linalg.norm(bearing))
                        step = (
                            bearing / norm * float(handler.move_speeds[members[0]])
                            if norm > 1e-9
                            else bearing
                        )
                        saved = [
                            np.array(env.player_models[i].location, copy=True)
                            for i in members
                        ]
                        with torch.no_grad():
                            v0 = army_value(critic, env)
                            for i in members:
                                env.player_models[i].location = (
                                    env.player_models[i].location + step
                                ).astype(env.player_models[i].location.dtype)
                            v1 = army_value(critic, env)
                        for i, loc in zip(members, saved, strict=True):
                            env.player_models[i].location = loc
                        if v1 > v0:
                            approvals += 1
                            # The ONE shared grid cell closest to the target.
                            best = int(
                                np.argmin(
                                    np.linalg.norm(
                                        (centroid[np.newaxis, :] + grid)
                                        - centre[np.newaxis, :],
                                        axis=1,
                                    )
                                )
                            )
                            actions = list(action.actions)
                            for i in members:
                                actions[i] = movement.start + best
                            action = WargameEnvAction(actions=actions)
                            redirects += len(members)
            observation, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        # ⚠ NOT `info["vp_margin"]` -- the step info carries no such key and
        # `dict.get`'s default silently scored every episode 0.0. Caught when a
        # 24-cell screen printed +0.0 in every cell; `measure_charges` reads
        # the env's own counters, so this does too.
        vp += float(env.player_vp - env.opponent_vp)

    print(
        f"  realloc  vp={vp / n_episodes:+8.1f}  nominated/ep={nominations / n_episodes:.2f}  "
        f"approved={approvals}/{nominations}  models_redirected/ep={redirects / n_episodes:.1f}"
    )


def main() -> None:
    """CLI entry point."""
    positionals, overrides = parse_overrides(sys.argv[1:])
    checkpoint = positionals[0]
    config_path = positionals[1]
    n_episodes = int(positionals[2]) if len(positionals) > 2 else 20
    decode_topk = int(positionals[3]) if len(positionals) > 3 else 3
    min_stack = int(positionals[4]) if len(positionals) > 4 else 4
    print(
        f"{config_path}{describe(overrides)}  "
        f"(n={n_episodes}, seeds 700000+, K={decode_topk}, min_stack={min_stack})"
    )
    measure(checkpoint, config_path, n_episodes, decode_topk, min_stack, overrides)


if __name__ == "__main__":
    main()
