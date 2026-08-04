"""Tests for the diagnostics that make a training run readable.

Their absence is why a degenerate PPO objective, a curriculum that never
reached the rollouts, and a training set of 8 fixed layouts all survived seven
runs without being noticed.
"""

from __future__ import annotations

import math

import pytest

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

EXPECTED_DIAGNOSTICS = {
    "entropy/movement",
    "rollout_phase_index",
    "distinct_layouts_seen",
}


@pytest.fixture
def module() -> PPOLightning:
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=2,
            number_of_objectives=1,
            number_of_battle_rounds=4,
        )
    )
    return PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=2,
    )


def test_rollout_reports_entropy_phase_and_layout_diagnostics(
    module: PPOLightning,
) -> None:
    """A rollout records the diagnostics needed to read the run."""
    module._collect_rollout_parallel()

    assert EXPECTED_DIAGNOSTICS <= set(module._rollout_diagnostics)


def test_movement_entropy_is_raw_nats_not_the_loss_term(
    module: PPOLightning,
) -> None:
    """Entropy is logged un-multiplied by `ent_coef`.

    `loss/entropy_loss` is `-ent_coef * entropy`, which was misread as entropy
    itself and produced a wrong conclusion about the entropy coefficient. An
    untrained policy sits near the uniform ceiling, so the raw value is large
    and positive where the loss term is small and negative.
    """
    module._collect_rollout_parallel()

    movement_entropy = module._rollout_diagnostics["entropy/movement"]
    ceiling = math.log(module.env.n_actions)

    assert 0.0 < movement_entropy <= ceiling + 1e-6
    # An untrained policy is near-uniform, so this should be close to the
    # ceiling. If this ever fails low, the policy learned something at init.
    assert movement_entropy > 0.5 * ceiling


def test_rollout_phase_index_matches_the_rollout_envs(
    module: PPOLightning,
) -> None:
    """The logged phase index is the rollout's, not the eval env's.

    Logging only the eval env's index is what let `reward_phase` report 3
    while every rollout was still rewarding phase 0.
    """
    module._collect_rollout_parallel()

    rollout_envs = module._rollout_envs or []
    assert rollout_envs
    assert module._rollout_diagnostics["rollout_phase_index"] == pytest.approx(
        float(rollout_envs[0].phase_manager.current_phase_index)
    )
