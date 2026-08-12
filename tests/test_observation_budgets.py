"""One network across layouts that carry different numbers of objectives and pieces.

The per-model block is `2 + n_objectives * 2` wide, so objective count is a hard
input dimension: at three objectives a model token is 49 wide, at five 53, at six
55. The 45 real layouts carry five or six objectives and 15 or 16 terrain pieces,
so no single network could span them and no checkpoint trained on the generated
scenario could be scored on any of them.

`objective_budget` / `terrain_budget` pad both to a fixed size. Both default to
None, and off must stay byte-identical -- every score in the repo was measured
without them.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from pydantic_yaml import parse_yaml_raw_as

from scripts.measure_maps import config_for_map
from wargame_rl.wargame.envs.types import TerrainMapConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import TransformerNetwork

CONFIG_PATH = "configs/golden/25v25_shooting_opponent.yaml"
MAPS_DIR = Path("configs/evaluation/maps")
OBJECTIVE_BUDGET = 6
TERRAIN_BUDGET = 16


def _base_config() -> WargameEnvConfig:
    with open(CONFIG_PATH) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    return config


def _budgeted_config() -> WargameEnvConfig:
    config = _base_config()
    config.objective_budget = OBJECTIVE_BUDGET
    config.terrain_budget = TERRAIN_BUDGET
    return config


def _map_env(config: WargameEnvConfig, name: str) -> WargameEnv:
    terrain_map = parse_yaml_raw_as(
        TerrainMapConfig, (MAPS_DIR / f"{name}.yaml").read_text()
    )
    return create_environment(env_config=config_for_map(config, terrain_map))


def _named_map_with(n_objectives: int) -> str:
    """First shipped map carrying exactly `n_objectives`, so the pair is real data."""
    for path in sorted(MAPS_DIR.glob("*.yaml")):
        terrain_map = parse_yaml_raw_as(TerrainMapConfig, path.read_text())
        if terrain_map.objectives and len(terrain_map.objectives) == n_objectives:
            return str(terrain_map.name)
    raise AssertionError(f"no shipped map carries {n_objectives} objectives")


class TestTheBudgetsAreOffByDefault:
    def test_padding_leaves_the_real_slots_bit_identical(self) -> None:
        """Padding must only append -- the same board must read the same way.

        Byte-identity of the *unpadded* build is pinned by
        `test_observation_golden`; what is checked here is the other half, that
        turning the budgets on does not perturb anything already there.
        """
        # Arrange
        off = create_environment(env_config=_base_config())
        on = create_environment(env_config=_budgeted_config())

        # Act
        off.reset(seed=700000)
        on.reset(seed=700000)
        off_tensors = observation_to_tensor(off.observation)
        on_tensors = observation_to_tensor(on.observation)
        n_objectives = len(off.objectives)

        # Assert -- real objective rows, minus the appended `present` column.
        np.testing.assert_array_equal(
            on_tensors[1].cpu().numpy()[:n_objectives, :-1],
            off_tensors[1].cpu().numpy(),
        )
        # And the per-model location plus real distance pairs.
        n_real = 2 + 2 * n_objectives
        np.testing.assert_array_equal(
            on_tensors[2].cpu().numpy()[:, :n_real],
            off_tensors[2].cpu().numpy()[:, :n_real],
        )

    def test_off_carries_no_presence_flags(self) -> None:
        """`None`, not an array of ones -- off must not widen the token at all."""
        env = create_environment(env_config=_base_config())
        env.reset(seed=700000)

        assert env.observation.objectives[0].present is None
        assert env.observation.wargame_models[0].objective_present is None


class TestPaddingIsMarked:
    def test_real_slots_are_flagged_and_padding_is_a_zero_row(self) -> None:
        """The whole row zero is what the network keys on to drop the slot."""
        # Arrange -- five real objectives into a budget of six.
        env = _map_env(_budgeted_config(), _named_map_with(5))

        # Act
        env.reset(seed=700000)
        objectives = observation_to_tensor(env.observation)[1].cpu().numpy()

        # Assert
        assert objectives.shape[0] == OBJECTIVE_BUDGET
        np.testing.assert_array_equal(objectives[:5, -1], 1.0)
        np.testing.assert_array_equal(objectives[5], 0.0)

    def test_a_padding_slot_does_not_read_as_standing_on_an_objective(self) -> None:
        """A padded delta is (0, 0), which is the most emphatic thing it could say."""
        # Arrange
        env = _map_env(_budgeted_config(), _named_map_with(5))

        # Act
        env.reset(seed=700000)
        model = env.observation.wargame_models[0]

        # Assert -- the zero delta is there, and a flag says to ignore it.
        np.testing.assert_array_equal(model.distances_to_objectives[5], 0.0)
        np.testing.assert_array_equal(
            model.objective_present, np.array([1, 1, 1, 1, 1, 0], dtype=np.float32)
        )

    def test_terrain_padding_needs_no_new_column(self) -> None:
        """The vertex count is zero on padding, and no real piece has zero vertices."""
        # Arrange -- a 15-piece layout into a budget of 16.
        config = _budgeted_config()
        env = _map_env(config, "table_01")

        # Act
        env.reset(seed=700000)
        terrain = observation_to_tensor(env.observation)[4].cpu().numpy()

        # Assert
        assert terrain.shape[0] == TERRAIN_BUDGET
        n_real = int((terrain[:, -1] > 0).sum())
        assert n_real == len(env.terrain.footprints)
        np.testing.assert_array_equal(terrain[n_real:], 0.0)


class TestOneNetworkSpansEveryShippedMap:
    def test_all_45_maps_produce_one_tensor_shape(self) -> None:
        # Arrange
        config = _budgeted_config()

        # Act
        widths = set()
        for path in sorted(MAPS_DIR.glob("*.yaml")):
            terrain_map = parse_yaml_raw_as(TerrainMapConfig, path.read_text())
            env = _map_env(config, terrain_map.name)
            env.reset(seed=700000)
            widths.add(
                tuple(int(t.shape[-1]) for t in observation_to_tensor(env.observation))
            )
            env.close()

        # Assert
        assert len(widths) == 1

    def test_a_five_and_a_six_objective_map_batch_together(self) -> None:
        """`observations_to_tensor_batch` stacks, so ragged counts cannot collate."""
        # Arrange
        config = _budgeted_config()
        observations = []
        for n_objectives in (5, 6):
            env = _map_env(config, _named_map_with(n_objectives))
            env.reset(seed=700000)
            observations.append(env.observation)

        # Act
        batch = observations_to_tensor_batch(observations)

        # Assert
        assert tuple(batch[1].shape) == (2, OBJECTIVE_BUDGET, batch[1].shape[-1])
        assert tuple(batch[4].shape) == (2, TERRAIN_BUDGET, batch[4].shape[-1])

    def test_weights_from_the_training_scenario_load_onto_a_real_map(self) -> None:
        """The failure this exists to remove: a size mismatch on `load_state_dict`."""
        # Arrange -- trained on generated terrain at three objectives.
        config = _budgeted_config()
        train_env = create_environment(env_config=config)
        torch.manual_seed(0)
        trained = TransformerNetwork.policy_from_env(train_env)

        # Act -- scored on a six-objective, 16-piece real layout.
        map_env = _map_env(config, _named_map_with(6))
        on_map = TransformerNetwork.policy_from_env(map_env)
        on_map.load_state_dict(trained.state_dict())
        map_env.reset(seed=700000)
        logits = on_map(observation_to_tensor(map_env.observation))

        # Assert
        assert logits.shape[-2] == config.number_of_wargame_models
        assert torch.isfinite(logits).any()


class TestPaddingIsExcludedFromAttention:
    def test_moving_a_padding_slot_cannot_change_the_logits(self) -> None:
        """Behavioural, not structural: a masked token must not reach the output."""
        # Arrange -- a five-objective map leaves one padded slot.
        config = _budgeted_config()
        env = _map_env(config, _named_map_with(5))
        env.reset(seed=700000)
        torch.manual_seed(0)
        net = TransformerNetwork.policy_from_env(env)
        tensors = observation_to_tensor(env.observation)

        # Act -- shove the padding slot to a corner of the board.
        moved = [t.clone() for t in tensors]
        moved[1][OBJECTIVE_BUDGET - 1, 0] = 0.9
        moved[1][OBJECTIVE_BUDGET - 1, 1] = -0.9
        with torch.no_grad():
            before = net(tensors)
            after = net(moved)

        # Assert
        torch.testing.assert_close(before, after)

    def test_moving_a_real_objective_does_change_the_logits(self) -> None:
        """The control -- otherwise the test above would pass on a dead network."""
        # Arrange
        config = _budgeted_config()
        env = _map_env(config, _named_map_with(5))
        env.reset(seed=700000)
        torch.manual_seed(0)
        net = TransformerNetwork.policy_from_env(env)
        tensors = observation_to_tensor(env.observation)

        # Act
        moved = [t.clone() for t in tensors]
        moved[1][0, 0] = 0.9
        moved[1][0, 1] = -0.9
        with torch.no_grad():
            before = net(tensors)
            after = net(moved)

        # Assert
        assert not torch.allclose(before, after)


class TestTheConfigRejectsABudgetThatDoesNotFit:
    def test_objective_budget_below_the_objective_count(self) -> None:
        config = _base_config().model_dump()
        config["objective_budget"] = 2

        with pytest.raises(ValueError, match="objective_budget"):
            WargameEnvConfig(**config)

    def test_terrain_budget_below_the_generated_piece_count(self) -> None:
        config = _base_config().model_dump()
        config["terrain_budget"] = 3

        with pytest.raises(ValueError, match="terrain_budget"):
            WargameEnvConfig(**config)

    def test_a_layout_over_budget_fails_loudly_rather_than_dropping_it(self) -> None:
        """`config_for_map` raises the objective count past a budget set for the scenario.

        Pydantic does not revalidate on assignment, so this one cannot be caught
        at load; the builder refuses instead. Dropping the extra objective would
        be far worse -- the network would score a board it cannot see all of.
        """
        # Arrange -- a budget that fits the scenario but not a six-objective map.
        config = _base_config()
        config.objective_budget = 5
        config.terrain_budget = TERRAIN_BUDGET
        env = _map_env(copy.deepcopy(config), _named_map_with(6))

        # Act / Assert
        with pytest.raises(ValueError, match="objective_budget"):
            env.reset(seed=700000)
