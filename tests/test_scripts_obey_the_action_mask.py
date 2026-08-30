"""No scripted policy may emit an action the action mask forbids.

⚠ **This pins a class, not a case.** The agent samples under the mask and
cannot break a rule the mask encodes; `ActionHandler.apply` takes no mask, so a
scripted policy can, and the env accepts it. Every scripted bar in this project
is then measured under different rules from the agent it bars, and the
asymmetry can only flatter the scripts.

`apply`'s own docstring names the hazard for the movement-phase rule -- *"the
action mask already enforces that for a learned policy, but scripted policies
bypass the mask, so honouring `phase` here keeps them on the same footing"* --
and the newer "a declared unit may not stand still" rule went unprotected for a
day. Measured then: **2 of 109 declared model-rows (1.8%)** on the melee config
submitted a mask-forbidden STAY, and **0 of 2,660** on the golden configs, so
the gap was melee-only. ⚠ Two audit panels independently reported **14.7% and
14.8%** for the melee figure; re-measured it is 1.8%.

This test is what stops the next move type reopening it.
"""

from __future__ import annotations

import pytest
import yaml

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

CASES = [
    ("configs/golden/25v25_maps_two_mode.yaml", "squad_march_take"),
    ("configs/golden/25v25_maps_two_mode.yaml", "squad_march_shoot"),
    ("configs/experiments/25v25_maps_melee.yaml", "squad_march_take"),
    ("configs/experiments/25v25_maps_melee.yaml", "squad_march_take_charge"),
]


@pytest.mark.parametrize(("config_path", "policy_name"), CASES)
def test_a_scripted_policy_never_emits_a_masked_out_action(
    config_path: str, policy_name: str
) -> None:
    """One seeded episode is enough: a violation here is systematic, not rare."""
    # Arrange
    with open(config_path) as handle:
        config = WargameEnvConfig(**yaml.safe_load(handle))
    env = create_environment(env_config=config)
    violations: list[str] = []

    # Act
    try:
        observation, _ = env.reset(seed=700000)
        policy = build_baseline_policy(policy_name)
        done = False
        while not done:
            phase = str(env.game_clock_state.phase)
            mask = observation.action_mask
            assert mask is not None, "every config under test masks its actions"
            action = policy.select_action(env.wargame_models, env)
            for index, model in enumerate(env.wargame_models):
                if model.is_alive and not bool(mask[index][action.actions[index]]):
                    violations.append(
                        f"{phase}: model {index} played {action.actions[index]}"
                    )
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    finally:
        env.close()

    # Assert
    assert not violations, (
        f"{policy_name} on {config_path} played "
        f"{len(violations)} action(s) the mask forbids: {violations[:5]}"
    )
