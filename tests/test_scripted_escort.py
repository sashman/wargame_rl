"""`scripted_escort`: shoot the blocker off the point, then march -- the C3 bar.

Driven through a real env on the C3 config, because what matters is the order
of events the env sees: the unarmed squads never enter the blockers' reach
while a blocker lives, the armed squad kills every blocker, and with no armed
squad at all the policy is `squad_march_take` to the action.
"""

from __future__ import annotations

import numpy as np

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.baseline.scripted_escort import ScriptedEscortPolicy
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

C2 = "configs/experiments/curriculum/c2.yaml"
C3 = "configs/experiments/curriculum/c3.yaml"


def test_the_seat_declares_that_it_shoots() -> None:
    assert ScriptedEscortPolicy().shoots


def test_unarmed_squads_wait_out_of_reach_until_the_blockers_are_dead() -> None:
    # Arrange: the C3 scenario, our side played by the escort.
    env = create_environment(env_config=load_env_config(C3))
    select = build_action_selector("scripted_escort", env, 1).select
    observation, _ = env.reset(seed=700000)
    armed = np.asarray(env.player_max_ranges) > 0
    blocker_reach = float(np.max(env.opponent_max_ranges))
    closest_unarmed_while_alive = np.inf
    done = False
    # Act
    while not done:
        observation, _reward, done, _trunc, _info = env.step(select(observation, env))
        blockers = [m for m in env.opponent_models if m.is_alive]
        if not blockers:
            continue
        # Assert (every step): no unarmed body inside the blockers' reach.
        enemy = np.array([m.location for m in blockers], dtype=float)
        for i, model in enumerate(env.wargame_models):
            if armed[i] or not model.is_alive:
                continue
            here = np.asarray(model.location, dtype=float)
            closest_unarmed_while_alive = min(
                closest_unarmed_while_alive,
                float(np.linalg.norm(enemy - here, axis=1).min()),
            )
    # Assert: the blockers all died, and no unarmed body ever came inside reach.
    assert all(not m.is_alive for m in env.opponent_models)
    assert closest_unarmed_while_alive > blocker_reach


def test_with_no_armed_squad_it_is_squad_march_take_to_the_action() -> None:
    # Arrange: C2 (nobody of ours is armed), both policies from the same seed.
    trajectories = []
    for name in ("scripted_escort", "squad_march_take"):
        env = create_environment(env_config=load_env_config(C2))
        select = build_action_selector(name, env, 1).select
        observation, _ = env.reset(seed=700001)
        actions = []
        done = False
        # Act
        while not done:
            action = select(observation, env)
            actions.append(list(action.actions))
            observation, _reward, done, _trunc, _info = env.step(action)
        trajectories.append(actions)
    # Assert: identical actions at every step.
    assert trajectories[0] == trajectories[1]
