"""The run base name is what separates one arm's checkpoints from another's.

Everything in it except `config_name` describes the *scenario* — board, force
sizes, phase count, opponent — and the arms of an experiment deliberately share
all of those. Four configs differing only in an observation flag once produced
byte-identical names, so every arm in the batch wrote checkpoints into a single
directory and `measure-checkpoint` scored whichever process saved last.
"""

from pathlib import Path

import pytest
from pydantic_yaml import parse_yaml_raw_as

from train import _build_default_run_base_name
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig


def test_build_default_run_base_name_without_opponent_policy() -> None:
    env_config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=0,
        number_of_objectives=2,
        board_width=60,
        board_height=44,
    )

    run_base_name = _build_default_run_base_name(env_config)

    assert run_base_name == "ppo-transformer-m4-opp0-obj2-b60x44-ph1"


def test_build_default_run_base_name_with_opponent_policy() -> None:
    env_config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        board_width=60,
        board_height=44,
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )

    run_base_name = _build_default_run_base_name(env_config)

    assert (
        run_base_name
        == "ppo-transformer-m4-opp4-obj2-b60x44-ph1-vs-scripted_advance_to_objective"
    )


def test_config_name_leads_the_run_name_when_set() -> None:
    env_config = WargameEnvConfig(
        config_name="my_arm",
        number_of_wargame_models=4,
        number_of_opponent_models=0,
        number_of_objectives=2,
        board_width=60,
        board_height=44,
    )

    run_base_name = _build_default_run_base_name(env_config)

    assert run_base_name == "ppo-transformer-my_arm-m4-opp0-obj2-b60x44-ph1"


ARM_GROUPS = [
    pytest.param(
        [
            "25v25_cover_control.yaml",
            "25v25_shooting_opponent.yaml",
        ],
        id="shooting-scenario",
    ),
]


@pytest.mark.parametrize("config_names", ARM_GROUPS)
def test_arms_of_one_batch_get_distinct_run_names(config_names: list[str]) -> None:
    """Two arms sharing a checkpoint directory silently ruins the experiment.

    Nothing raises: both processes write to the same path, and whichever saves
    last owns the "best" checkpoint for every arm in the batch.
    """
    root = Path("configs") / "golden"
    names = {
        _build_default_run_base_name(
            parse_yaml_raw_as(WargameEnvConfig, (root / name).read_text())
        )
        for name in config_names
    }

    assert len(names) == len(config_names), sorted(names)
