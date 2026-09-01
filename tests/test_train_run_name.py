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

from train import _build_default_run_base_name, _run_paths
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.model.common import init_wandb


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


class TestThePoolSitsBesideTheCheckpoints:
    """The self-play snapshot directory and the run's checkpoints must agree.

    This file already exists because four arms once shared one checkpoint
    directory. The pool repeated it: `snapshot_dir` was named from the run
    *base*, with no timestamp and no `--run-suffix`, while the checkpoint
    directory carries both. So the documented path `checkpoints/<run>/pool/`
    was empty, and every self-play run on one env config wrote the same
    filenames into one shared directory.

    ⚠ The consequence is worse than a misplaced file. A pool entry holds a
    **path**, loaded lazily when the opponent is seated, so two concurrent runs
    would seat each other's weights as their own past selves and nothing would
    raise. Found by following the pre-registration's own check and finding an
    empty directory.
    """

    def test_two_runs_of_one_config_do_not_share_a_pool(self) -> None:
        """The regression. Arms differ by `--run-suffix` and nothing else."""
        _, arm = _run_paths("ppo-transformer-m4-opp0-obj2-b60x44-ph1", "s1-selfplay")
        _, control = _run_paths(
            "ppo-transformer-m4-opp0-obj2-b60x44-ph1", "s2-selfplay"
        )

        assert arm != control

    def test_the_pool_is_a_child_of_the_run_directory(self) -> None:
        """`checkpoints/<run>/pool/` -- the path the pre-registration documents
        and a reader checks the mechanism against."""
        run_name, pool = _run_paths("ppo-transformer-m4-opp0-obj2-b60x44-ph1", "s1")

        assert pool.name == "pool"
        assert pool.parent.name == run_name
        assert pool.parent.parent == Path("checkpoints")

    def test_the_suffix_reaches_the_name(self) -> None:
        run_name, _ = _run_paths(
            "ppo-transformer-m4-opp0-obj2-b60x44-ph1", "s3-control"
        )

        assert run_name.endswith("-s3-control")

    def test_wandb_takes_the_name_it_is_given(self) -> None:
        """The other half of the invariant, and the reason `_run_paths` returns
        the name at all: `make_run_name` stamps wall-clock time, so a second
        call can land in the next second and name a different directory. The
        run and its pool must come from ONE call.
        """
        run_name, _ = _run_paths("ppo-transformer-m4-opp0-obj2-b60x44-ph1", "s1")

        with init_wandb(disabled=True, run_name=run_name) as run:
            assert run.name == run_name
