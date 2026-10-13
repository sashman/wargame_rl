"""The `hold_and_shoot` seat: stands still and fires -- the C2 blocker.

`hold_deployment` shoots at nothing on purpose; this seat is that policy
with `squad_march_shoot`'s target rule bolted on. Driven through a real env
on the C2 config (the opponent unit stands on a point our squads must
reach), because the fact that matters is what the env does with the seat:
the blockers never move, and approaching squads lose bodies.
"""

from __future__ import annotations

import numpy as np

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.baseline.hold_and_shoot import ScriptedHoldAndShootPolicy
from wargame_rl.wargame.envs.baseline.hold_deployment import (
    ScriptedHoldDeploymentPolicy,
)
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.selectors import build_action_selector

C2 = "configs/experiments/curriculum/c2.yaml"


def test_the_seat_declares_that_it_shoots() -> None:
    assert ScriptedHoldAndShootPolicy().shoots
    assert not ScriptedHoldDeploymentPolicy().shoots


def test_the_blockers_never_move_and_the_approaching_squads_lose_bodies() -> None:
    # Arrange: the C2 scenario, our side played by the non-shooting bar.
    env = create_environment(env_config=load_env_config(C2))
    select = build_action_selector("squad_march_take", env, 1).select
    player_dead = 0

    for seed in range(700000, 700005):
        observation, _ = env.reset(seed=seed)
        start = np.array([m.location for m in env.opponent_models], dtype=float)
        done = False
        # Act
        while not done:
            observation, _reward, done, _trunc, _info = env.step(
                select(observation, env)
            )
        # Assert: the blockers ended where they deployed.
        end = np.array([m.location for m in env.opponent_models], dtype=float)
        np.testing.assert_allclose(end, start)
        player_dead += sum(1 for m in env.wargame_models if not m.is_alive)

    # Assert: over five episodes the fire cost us something. A non-shooting
    # blocker (C1) leaves every body alive on every seed.
    assert player_dead > 0
