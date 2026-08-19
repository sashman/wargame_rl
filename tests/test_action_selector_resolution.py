"""One resolver turns a baseline name or a checkpoint path into a selector.

Four near-duplicate implementations of this existed, in `measure_maps`,
`debug`, `measure_paired_policies` and `measure_income_share` -- and **with two
different precedences**: two tried the filesystem first, two tried the registry
first. They disagree only for a baseline name that is also an existing path,
which is why nobody noticed. Consolidating them is what stops a fifth appearing
for the rating arena.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.selectors import build_action_selector, is_checkpoint

POLICY_PREFIX = "ppo_model.policy_network."


def _write_checkpoint(env: WargameEnv, path: Path) -> None:
    """Save a randomly-initialised policy under the prefix PPO writes."""
    torch.manual_seed(0)
    policy = TransformerNetwork.policy_from_env(env=env)
    torch.save(
        {"state_dict": {POLICY_PREFIX + k: v for k, v in policy.state_dict().items()}},
        path,
    )


def test_a_registry_name_resolves_to_a_baseline(env: WargameEnv) -> None:
    resolved = build_action_selector("squad_march_shoot", env)

    assert resolved.kind == "baseline"
    assert resolved.label == "squad_march_shoot"
    assert resolved.network is None


def test_a_checkpoint_path_resolves_to_a_network(
    env: WargameEnv, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "run-2026-08-19-12-00-00-armA" / "last.ckpt"
    checkpoint.parent.mkdir()
    _write_checkpoint(env, checkpoint)

    resolved = build_action_selector(str(checkpoint), env)

    assert resolved.kind == "checkpoint"
    assert resolved.network is not None
    assert resolved.source == str(checkpoint)


def test_a_checkpoint_is_labelled_by_its_run_suffix(
    env: WargameEnv, tmp_path: Path
) -> None:
    """`<scenario>-<timestamp>-<suffix>`: the suffix is the only part that
    identifies which arm of a screen a row belongs to, because the scenario is
    identical across arms."""
    checkpoint = tmp_path / "25v25-2026-08-19-12-00-00-armA" / "last.ckpt"
    checkpoint.parent.mkdir()
    _write_checkpoint(env, checkpoint)

    assert build_action_selector(str(checkpoint), env).label == "armA"


def test_a_selector_plays_a_step(env: WargameEnv) -> None:
    """Resolution is not enough -- the thing it returns has to be playable."""
    resolved = build_action_selector("squad_march", env)
    observation, _info = env.reset(seed=0)

    action = resolved.select(observation, env)

    assert len(action.actions) == len(env.wargame_models)


def test_an_unknown_name_names_the_baselines_it_could_have_been(
    env: WargameEnv,
) -> None:
    with pytest.raises(ValueError, match="squad_march_shoot"):
        build_action_selector("not_a_policy", env)


def test_a_missing_checkpoint_path_is_not_read_as_a_baseline(
    env: WargameEnv, tmp_path: Path
) -> None:
    """A `.ckpt` that does not exist must say so, rather than falling through to
    'unknown baseline' -- the two mistakes need different fixes."""
    with pytest.raises(ValueError, match="no checkpoint"):
        build_action_selector(str(tmp_path / "absent.ckpt"), env)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("squad_march_shoot", False),
        ("checkpoints/run/last.ckpt", True),
        ("some/path.ckpt", True),
    ],
)
def test_is_checkpoint_reads_the_suffix(spec: str, expected: bool) -> None:
    assert is_checkpoint(spec) is expected


def test_resolving_a_baseline_does_not_import_torch() -> None:
    """`debug.py` deliberately does not pay for torch on a scripted session, and
    the deferred import inside the checkpoint branch is what buys that. A plain
    `import torch` at the top of the module would silently undo it, and nothing
    else in the suite would notice.

    Run in a subprocess because the rest of this file has already imported
    torch into the parent interpreter.
    """
    source = (
        "import sys\n"
        "from wargame_rl.wargame.selectors import build_action_selector\n"
        "print('torch' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False"
