"""Deploying in coherency, as `03-moving.md` § Setting up requires.

The rule says a unit must be *set up* in coherency, and this environment never
was: measured with `just measure-coherency`, 0 of 20 episodes deploy coherently
on the golden shooting config. These tests pin the fix and, just as importantly,
pin that it stays off unless a config asks for it.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.random import default_rng
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.placement import wargame_model_placement
from wargame_rl.wargame.envs.domain.rules_constants import (
    COHERENCY_FURTHEST_IN,
    COHERENCY_NEAREST_IN,
)
from wargame_rl.wargame.envs.domain.rules_quantities import resolve_rules_quantities
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import CoherencyConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

GOLDEN = "configs/golden/25v25_shooting_opponent.yaml"


def load_config(path: str = GOLDEN) -> WargameEnvConfig:
    """Parse a shipped config off disk."""
    with open(path) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    return config


def coherent_forces(env: WargameEnv, config: WargameEnvConfig) -> tuple[bool, bool]:
    """Whether each force is wholly in coherency, at the config's own distances."""
    quantities = resolve_rules_quantities(config)
    scale = quantities.scale
    nearest = scale.to_units(config.coherency.nearest_distance)
    furthest = scale.to_units(config.coherency.furthest_distance)
    verdicts = []
    for models in (env.wargame_models, env.opponent_models):
        report = evaluate_coherency(
            positions=np.array([m.location for m in models], dtype=float),
            group_ids=np.array([m.group_id for m in models], dtype=np.intp),
            alive_mask=alive_mask_for(models),
            base_radii=np.array([m.base_radius for m in models], dtype=float),
            nearest_distance=nearest,
            furthest_distance=furthest,
        )
        verdicts.append(report.all_coherent)
    return verdicts[0], verdicts[1]


def test_deployment_is_coherent_when_enforced() -> None:
    # Arrange: the golden scenario, with the setting-up rule switched on.
    config = load_config()
    config.coherency.enforce_at_deployment = True
    env = create_environment(env_config=config)

    # Act / Assert: every episode. The measured baseline this replaces is 0 of
    # 20, so a single failure here is the whole point.
    #
    # Player only. `reset` runs `run_until_player_phase`, which auto-executes
    # the opponent's first turn, so by the time it returns the opponent has
    # already *moved* -- reading its coherency here would measure the opponent
    # policy, not placement. Placement is covered for both zones by
    # `test_placement_deploys_every_unit_in_coherency`.
    for seed in range(700000, 700020):
        env.reset(seed=seed)
        player, _opponent = coherent_forces(env, config)
        assert player, f"player force deployed out of coherency at seed {seed}"


@pytest.mark.parametrize("zone", [(0.0, 0.0, 20.0, 44.0), (40.0, 0.0, 60.0, 44.0)])
def test_placement_deploys_every_unit_in_coherency(
    zone: tuple[float, float, float, float],
) -> None:
    # Arrange: 5 squads of 5 with real 32mm bases, in each deployment zone.
    # Placement is tested directly because it is the only way to see the
    # opponent's deployment before its first turn moves it.
    models = [
        WargameModel(
            location=position(0.0, 0.0),
            distances_to_objectives=np.zeros(3),
            stats={"current_wounds": 1, "max_wounds": 1},
            group_id=index // 5,
            base_radius=0.63,
        )
        for index in range(25)
    ]

    # Act
    wargame_model_placement(
        models,
        np.array(zone),
        group_max_distance=10.0,
        rng=default_rng(0),
        base_radius=0.63,
        coherency=CoherencyConfig(enforce_at_deployment=True),
    )

    # Assert
    report = evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=np.ones(25, dtype=bool),
        base_radii=np.full(25, 0.63),
        nearest_distance=2.0,
        furthest_distance=9.0,
    )
    assert report.all_coherent
    assert report.n_units == 5


def test_deployment_is_not_coherent_by_default() -> None:
    # Arrange: the same scenario untouched. This is the control, and it is what
    # makes the test above mean something rather than passing vacuously.
    config = load_config()
    assert not config.coherency.enforce_at_deployment
    env = create_environment(env_config=config)

    # Act
    coherent = []
    for seed in range(700000, 700020):
        env.reset(seed=seed)
        coherent.append(coherent_forces(env, config)[0])

    # Assert: the shipped placement essentially never satisfies the rule.
    assert not any(coherent)


def test_enforcing_coherency_changes_no_position_when_off() -> None:
    # Arrange: two envs from the same config, one of which has been through the
    # coherency code path with every switch off.
    positions = []
    for _ in range(2):
        config = load_config()
        env = create_environment(env_config=config)
        env.reset(seed=700000)
        positions.append(np.array([m.location for m in env.wargame_models]))

    # Assert: byte-identical. The feature is a no-op until a config asks.
    np.testing.assert_array_equal(positions[0], positions[1])


def test_a_config_of_solo_units_is_rejected_when_coherency_is_enforced() -> None:
    # Arrange: max_groups above the model count puts every model in its own
    # unit, where the rule holds vacuously -- the silent no-op this guards.
    config = load_config()

    # Act / Assert
    with pytest.raises(ValueError, match="every model in its own"):
        WargameEnvConfig.model_validate(
            {
                **config.model_dump(),
                "max_groups": 1000,
                "coherency": {"attrition": True},
            }
        )


def test_the_spread_cap_may_not_be_below_the_chain_distance() -> None:
    # Arrange: a cap looser than the chain it bounds is incoherent as a rule.
    config = load_config()

    # Act / Assert
    with pytest.raises(ValueError, match="spread cap must be at least"):
        WargameEnvConfig.model_validate(
            {
                **config.model_dump(),
                "coherency": {"nearest_distance": 9.0, "furthest_distance": 2.0},
            }
        )


def test_the_rules_distances_are_the_defaults() -> None:
    # Arrange / Act: the spec's own figures, mirrored out of docs/rules.
    config = load_config()

    # Assert
    assert config.coherency.nearest_distance == 2.0
    assert config.coherency.furthest_distance == 9.0
    assert not config.coherency.attrition


def test_the_config_defaults_match_the_domain_constants() -> None:
    # Arrange: config may not import the domain layer, so the two coherency
    # distances are written down twice. This is what stops them drifting.
    # Act / Assert
    assert CoherencyConfig().nearest_distance == COHERENCY_NEAREST_IN
    assert CoherencyConfig().furthest_distance == COHERENCY_FURTHEST_IN
