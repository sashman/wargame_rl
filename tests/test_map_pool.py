"""Training on the real tables: a layout is drawn per episode from a pool.

The third terrain mode. `terrain` is one fixed board and `random_terrain` is a
generated one; neither can train on the 45 real layouts, and the generated
mission is not the same mission — the generator can only put objectives in the
contested middle, while the real tables put a third of them inside each player's
own deployment zone.

The pool is loaded and checked once at construction, and a draw is an index.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

from wargame_rl.wargame.envs.map_pool import MapPool
from wargame_rl.wargame.envs.types import (
    MapPoolConfig,
    TerrainMapConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)

CONFIG_PATH = "configs/golden/25v25_shooting_opponent.yaml"
POOL_CONFIG_PATH = "configs/experiments/25v25_real_maps.yaml"
MAPS_DIR = "configs/evaluation/maps"


def _pool_config(
    names: list[str] | None = None,
    objective_budget: int | None = 6,
    terrain_budget: int | None = 16,
) -> WargameEnvConfig:
    with open(CONFIG_PATH) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.random_terrain = None
    config.map_pool = MapPoolConfig(directory=MAPS_DIR, names=names)
    config.objective_budget = objective_budget
    config.terrain_budget = terrain_budget
    config.render_mode = None
    return config


def _env(**kwargs: object) -> WargameEnv:
    return create_environment(env_config=_pool_config(**kwargs))  # type: ignore[arg-type]


class TestTheModesAreMutuallyExclusive:
    def test_a_pool_beside_a_generator_is_rejected(self) -> None:
        """One would silently overwrite the other, exactly as `measure-maps` guards."""
        with open(CONFIG_PATH) as handle:
            data = parse_yaml_raw_as(WargameEnvConfig, handle.read()).model_dump()
        data["map_pool"] = {"directory": MAPS_DIR}

        with pytest.raises(ValueError, match="mutually exclusive"):
            WargameEnvConfig(**data)

    def test_a_pool_alone_is_accepted(self) -> None:
        assert _pool_config().map_pool is not None


class TestTheDrawIsSeeded:
    def test_the_same_seed_draws_the_same_map(self) -> None:
        """A layout that is not a function of the seed makes a run unreproducible."""
        # Arrange
        first, second = _env(), _env()

        # Act
        first.reset(seed=1234)
        second.reset(seed=1234)

        # Assert
        assert first.map_name == second.map_name
        assert len(first.objectives) == len(second.objectives)

    def test_resets_move_through_the_pool(self) -> None:
        """A pool that returned one map would be a fixed layout with extra steps."""
        # Arrange
        env = _env()

        # Act
        drawn = set()
        for seed in range(60):
            env.reset(seed=seed)
            drawn.add(env.map_name)

        # Assert
        assert len(drawn) > 10

    def test_map_name_is_none_without_a_pool(self) -> None:
        with open(CONFIG_PATH) as handle:
            config = cast(
                WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
            )
        config.render_mode = None

        assert create_environment(env_config=config).map_name is None


class TestTheLayoutActuallyChanges:
    def test_terrain_and_objectives_both_come_from_the_drawn_map(self) -> None:
        """Objectives are as much part of a table as its ruins."""
        # Arrange
        env = _env()

        # Act -- collect the boards two different maps produce.
        boards = {}
        for seed in range(40):
            env.reset(seed=seed)
            name = env.map_name
            if name not in boards:
                boards[name] = (
                    len(env.terrain.footprints),
                    tuple(tuple(np.round(o.location, 3)) for o in env.objectives),
                )
            if len(boards) >= 2:
                break

        # Assert -- distinct maps, distinct objectives.
        first, second = list(boards.values())[:2]
        assert first[1] != second[1]

    def test_every_map_in_the_pool_carries_five_objectives(self) -> None:
        """Uniform since the tables became generated from the layout API.

        It was 5-or-6 while they were traced by hand, off six markers per
        layout; the current mission has five, and the collapse of two markers
        onto one ruin never fires on this pool. The budget stays at 6 anyway --
        it is what lets one network span counts the pool does not contain, and
        lowering it would change the observation width and orphan every
        checkpoint.
        """
        # Arrange
        env = _env()

        # Act
        counts = set()
        for seed in range(60):
            env.reset(seed=seed)
            counts.add(len(env.objectives))

        # Assert
        assert counts == {5}

    def test_the_objectives_are_the_maps_own_areas(self) -> None:
        """Not re-derived from the terrain -- that would discard the real placement."""
        # Arrange -- one map, so the expected objectives are knowable.
        env = _env(names=["table_42"])

        # Act
        env.reset(seed=0)

        # Assert
        assert len(env.objectives) == 5
        assert all(objective.is_area for objective in env.objectives)


class TestOneNetworkSpansThePool:
    def test_every_draw_produces_one_tensor_shape(self) -> None:
        # Arrange
        env = _env()

        # Act
        shapes, observations = set(), []
        for seed in range(30):
            observation, _ = env.reset(seed=seed)
            shapes.add(
                tuple(int(t.shape[-1]) for t in observation_to_tensor(observation))
            )
            observations.append(observation)

        # Assert -- and they collate, which is what a rollout batch needs.
        assert len(shapes) == 1
        batch = observations_to_tensor_batch(observations)
        assert tuple(batch[1].shape[:2]) == (30, 6)
        assert tuple(batch[4].shape[:2]) == (30, 16)


class TestThePoolIsCheckedWhenItIsLoaded:
    def test_mixed_counts_without_a_budget_are_rejected(self, tmp_path: Path) -> None:
        """Failing at load beats failing on the episode that first draws a 6.

        The pool has to be built here: the shipped tables are uniform at five
        objectives, so they cannot exercise the check that exists for ragged
        ones. A pool that mixes counts is still the failure being guarded
        against -- it is one upstream layout away, not hypothetical.
        """
        # Arrange -- one four-objective map beside one five-objective map.
        five = parse_yaml_raw_as(
            TerrainMapConfig, (Path(MAPS_DIR) / "table_01.yaml").read_text()
        )
        assert five.objectives is not None and len(five.objectives) == 5
        four = five.model_copy(deep=True)
        four.name = "table_98"
        four.objectives = four.objectives[:4] if four.objectives else None
        for terrain_map in (five, four):
            (tmp_path / f"{terrain_map.name}.yaml").write_text(to_yaml_str(terrain_map))

        # Act / Assert
        config = _pool_config(objective_budget=None)
        config.map_pool = MapPoolConfig(directory=str(tmp_path))
        with pytest.raises(ValueError, match="different widths"):
            create_environment(env_config=config)

    def test_a_budget_below_the_pool_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="over the budget"):
            _env(objective_budget=4)

    def test_an_unknown_map_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            _env(names=["table_01", "table_99"])

    def test_a_missing_directory_is_rejected(self) -> None:
        config = _pool_config()
        config.map_pool = MapPoolConfig(directory="configs/evaluation/nope")

        with pytest.raises(ValueError, match="not a directory"):
            create_environment(env_config=config)

    def test_a_map_off_the_board_is_rejected(self) -> None:
        """The env config's own terrain validator never sees these pieces."""
        config = _pool_config(names=["table_01"])
        config.board_width = 20

        with pytest.raises(ValueError, match="outside the board"):
            create_environment(env_config=config)

    def test_the_names_list_selects_a_subset(self) -> None:
        pool = MapPool.from_config(_pool_config(names=["table_01", "table_02"]))

        assert pool is not None
        assert pool.names == ["table_01", "table_02"]


class TestTheShippedTrainingConfig:
    def test_it_holds_out_every_fifth_table(self) -> None:
        """The split is the point of `names`; training on all 45 leaves no holdout."""
        # Arrange / Act
        with open(POOL_CONFIG_PATH) as handle:
            config = cast(
                WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
            )
        assert config.map_pool is not None
        names = config.map_pool.names or []

        # Assert
        assert len(names) == 36
        held_out = [name for name in names if int(name.split("_")[1]) % 5 == 0]
        assert held_out == []

    def test_it_runs_an_episode(self) -> None:
        # Arrange
        with open(POOL_CONFIG_PATH) as handle:
            config = cast(
                WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
            )
        config.render_mode = None
        env = create_environment(env_config=copy.deepcopy(config))

        # Act
        env.reset(seed=700000)
        _obs, _reward, terminated, truncated, _info = env.step(
            WargameEnvAction(actions=list(env.action_space.sample()))
        )

        # Assert
        assert env.map_name is not None
        assert not (terminated or truncated)
