"""Regression tests for how PPO builds and reuses its rollout environments.

Two defects lived here, both invisible without these assertions and both
affecting every training run to date:

- Rollout envs were rebuilt on every `training_step`, and a fresh `WargameEnv`
  starts its own `RewardPhaseManager` at phase 0. Phase advancement only ever
  mutated the eval env, so training reward always came from phase 0 while
  `reward_phase` reported the advanced phase.
- Each rebuilt env was re-seeded to its own index, so every epoch replayed the
  same handful of objective layouts while eval drew random ones.

The default `num_rollout_envs` auto-detects to >1, and the only pre-existing
PPO training test pinned it to 1 — so the path that actually runs was untested.
That is the organisational root cause, hence the parametrisation below.
"""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

# Distinct calculator sets so the reward breakdown names which phase ran.
PHASE_ZERO_CALCULATOR = "closest_objective"
PHASE_ONE_CALCULATOR = "models_at_objectives"


def _two_phase_env() -> WargameEnv:
    """Env whose two phases share no calculator, so the breakdown is diagnostic."""
    config = WargameEnvConfig(
        render_mode=None,
        number_of_wargame_models=2,
        number_of_objectives=1,
        number_of_battle_rounds=4,
        reward_phases=[
            RewardPhaseConfig(
                name="approach",
                reward_calculators=[
                    RewardCalculatorConfig(type=PHASE_ZERO_CALCULATOR, weight=1.0)
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="fraction_at_objectives", params={"min_fraction": 0.5}
                ),
                # Advancement is gated by the phase being left, so these must
                # be relaxed here for a single forced advance to take effect.
                min_epochs=0,
                min_epochs_above_threshold=1,
            ),
            RewardPhaseConfig(
                name="hold",
                reward_calculators=[
                    RewardCalculatorConfig(type=PHASE_ONE_CALCULATOR, weight=1.0)
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="fraction_at_objectives", params={"min_fraction": 0.5}
                ),
            ),
        ],
    )
    return WargameEnv(config=config)


def _make_module(env: WargameEnv, num_rollout_envs: int) -> PPOLightning:
    return PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        log=False,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        num_rollout_envs=num_rollout_envs,
    )


@pytest.mark.parametrize("num_rollout_envs", [1, 2])
def test_rollout_envs_share_the_curriculum_position(num_rollout_envs: int) -> None:
    """Advancing the phase advances it for the envs that generate training data."""
    env = _two_phase_env()
    module = _make_module(env, num_rollout_envs)
    module._ensure_rollout_envs()

    assert env.phase_manager.current_phase_index == 0
    module._advance_reward_phase(1.0)
    assert env.phase_manager.current_phase_index == 1

    for rollout_env in module._rollout_envs or []:
        assert rollout_env.phase_manager.current_phase_index == 1


def test_rollout_reward_follows_the_advanced_phase() -> None:
    """The rollout's reward breakdown names the advanced phase's calculators.

    This is the assertion that would have caught the bug from the logs alone:
    `reward/components/*` is keyed on the rollout env's active phase, so the
    keys never changing as `reward_phase` climbed was a visible tell.
    """
    env = _two_phase_env()
    module = _make_module(env, num_rollout_envs=2)

    module._advance_reward_phase(1.0)
    *_rest, breakdown = module._collect_rollout_parallel()

    assert PHASE_ONE_CALCULATOR in breakdown
    assert PHASE_ZERO_CALCULATOR not in breakdown


def test_rollout_envs_are_built_once_and_reused() -> None:
    """Consecutive rollouts use the same env objects.

    Rebuilding them is what reset the curriculum and the layout stream.
    """
    env = _two_phase_env()
    module = _make_module(env, num_rollout_envs=2)

    first = module._ensure_rollout_envs()
    module._collect_rollout_parallel()
    second = module._ensure_rollout_envs()

    assert first is second


def test_training_sees_new_layouts_across_rollouts() -> None:
    """Two consecutive rollouts do not replay an identical set of layouts.

    Objective placement dominates episode variance; replaying the same layouts
    every epoch turns training into a fit to a handful of maps while eval
    measures generalisation to new ones.
    """
    env = _two_phase_env()
    module = _make_module(env, num_rollout_envs=2)

    def layouts_seen() -> set[tuple[tuple[int, int], ...]]:
        seen: set[tuple[tuple[int, int], ...]] = set()
        for _ in range(6):
            module._collect_rollout_parallel()
            for rollout_env in module._rollout_envs or []:
                seen.add(
                    tuple(
                        (int(o.location[0]), int(o.location[1]))
                        for o in rollout_env.objectives
                    )
                )
        return seen

    first = layouts_seen()
    second = layouts_seen()
    assert first != second


def test_on_train_end_closes_rollout_envs() -> None:
    """The envs held for the run are released when training finishes."""
    env = _two_phase_env()
    module = _make_module(env, num_rollout_envs=2)
    module._ensure_rollout_envs()

    module.on_train_end()

    assert module._rollout_envs is None
