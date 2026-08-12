"""The floor for a mission that deals you ground at deployment.

The 45 real layouts put a third of their objectives inside each player's own
deployment zone, so a player *starts* holding ~2 of them. That raises a specific
worry — that the mission can be won by standing still — and `random` does not
answer it, because a random walk leaves the objectives it deployed on.
"""

from __future__ import annotations

from typing import cast

from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

CONFIG_PATH = "configs/golden/25v25_shooting_opponent.yaml"


def _env() -> WargameEnv:
    with open(CONFIG_PATH) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    return create_environment(env_config=config)


def test_it_is_registered_under_its_name() -> None:
    assert build_baseline_policy("hold_deployment") is not None


def test_every_model_stays_every_step() -> None:
    """The whole policy: one action, for everyone, forever."""
    # Arrange
    env = _env()
    policy = build_baseline_policy("hold_deployment")
    observation, _ = env.reset(seed=700000)

    # Act -- a few steps, so a policy that only stays on turn one would fail.
    for _ in range(3):
        action = policy.select_action(
            env.wargame_models,
            env,
            observation.action_mask,
        )
        assert action.actions == [STAY_ACTION] * len(env.wargame_models)
        observation, _r, terminated, truncated, _i = env.step(action)
        if terminated or truncated:
            break

    env.close()


def test_nothing_moves() -> None:
    """Behavioural, not structural — the actions could be right and the env still move."""
    # Arrange
    env = _env()
    policy = build_baseline_policy("hold_deployment")
    observation, _ = env.reset(seed=700000)
    before = [tuple(m.location) for m in env.wargame_models]

    # Act
    for _ in range(3):
        observation, _r, terminated, truncated, _i = env.step(
            policy.select_action(
                env.wargame_models,
                env,
                observation.action_mask,
            )
        )
        if terminated or truncated:
            break

    # Assert -- casualties may be removed, but no survivor has moved.
    after = [tuple(m.location) for m in env.wargame_models]
    assert after == before

    env.close()
