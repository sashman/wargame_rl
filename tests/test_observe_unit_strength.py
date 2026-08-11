"""A unit's remaining strength reaches the network.

Shooting names a *unit* and the defender allocates, so how many models a unit
has left decides whether a volley finishes it or is thrown at a full one. Before
this input no tensor carried that count: the shooting head mean-pools opponent
tokens into one token per unit, and a mean is invariant to how many terms it
averages, so five survivors and one survivor produced the same pooled token.

The flag defaults off, and off must stay byte-identical -- every score measured
before it exists was measured without it.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import observation_to_tensor

CONFIG_PATH = "configs/golden/25v25_shooting_opponent.yaml"
# 2 location + 3 objectives x 2 + 5 group one-hot + 1 same-group distance.
STRENGTH_COLUMN = 14


def _env(observe: bool) -> WargameEnv:
    with open(CONFIG_PATH) as handle:
        config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.observe_unit_strength = observe
    return create_environment(env_config=config)


def _player_column(env: WargameEnv) -> np.ndarray:
    return observation_to_tensor(env.observation)[2].cpu().numpy()[:, STRENGTH_COLUMN]


def _opponent_column(env: WargameEnv) -> np.ndarray:
    return observation_to_tensor(env.observation)[3].cpu().numpy()[:, STRENGTH_COLUMN]


class TestTheFlagIsOffByDefault:
    def test_off_leaves_the_per_model_tensor_unchanged(self) -> None:
        """The default must be an exact no-op, not merely a similar one."""
        # Arrange
        off, on = _env(observe=False), _env(observe=True)

        # Act
        off.reset(seed=700000)
        on.reset(seed=700000)
        off_tensors = observation_to_tensor(off.observation)
        on_tensors = observation_to_tensor(on.observation)

        # Assert -- one column wider, and every other column bit-identical.
        assert off_tensors[2].shape[-1] + 1 == on_tensors[2].shape[-1]
        assert off_tensors[3].shape[-1] + 1 == on_tensors[3].shape[-1]
        rebuilt = np.delete(on_tensors[2].cpu().numpy(), STRENGTH_COLUMN, axis=1)
        np.testing.assert_array_equal(rebuilt, off_tensors[2].cpu().numpy())

    def test_the_default_config_carries_no_strength(self) -> None:
        """`None`, not 0.0 -- a zero would read as "this unit is wiped out"."""
        env = _env(observe=False)
        env.reset(seed=700000)

        assert env.observation.wargame_models[0].unit_strength is None
        assert env.observation.opponent_models[0].unit_strength is None


class TestItTracksCasualties:
    def test_a_full_army_reads_one_everywhere(self) -> None:
        # Arrange / Act
        env = _env(observe=True)
        env.reset(seed=700000)

        # Assert
        np.testing.assert_allclose(_player_column(env), 1.0)
        np.testing.assert_allclose(_opponent_column(env), 1.0)

    def test_losses_lower_only_the_unit_that_took_them(self) -> None:
        """The whole point: a depleted unit must look different from a full one."""
        # Arrange -- the config declares five squads of five, ids 0-4.
        env = _env(observe=True)
        env.reset(seed=700000)

        # Act -- kill three models of opponent squad 0.
        for model in env.opponent_models[:3]:
            model.take_damage(99)
        column = _opponent_column(env)

        # Assert
        np.testing.assert_allclose(column[:5], 0.4)
        np.testing.assert_allclose(column[5:], 1.0)

    def test_the_column_is_constant_across_a_unit(self) -> None:
        """Including its dead: a corpse reports its unit, not itself.

        This is what makes one per-model column able to state a per-unit
        quantity -- every member says the same thing, so no member is privileged
        and pooling has nothing to reconcile.
        """
        # Arrange
        env = _env(observe=True)
        env.reset(seed=700000)

        # Act
        env.opponent_models[0].take_damage(99)
        column = _opponent_column(env)

        # Assert -- the dead model's own entry matches its living squadmates'.
        assert column[0] == pytest.approx(0.8)
        assert len(set(column[:5].tolist())) == 1

    def test_a_wiped_unit_reads_zero_rather_than_going_missing(self) -> None:
        # Arrange
        env = _env(observe=True)
        env.reset(seed=700000)

        # Act
        for model in env.opponent_models[:5]:
            model.take_damage(99)

        # Assert
        np.testing.assert_allclose(_opponent_column(env)[:5], 0.0)
