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


# ------------------------------------------------------------ the per-model side


def _small_per_model_checkpoint(tmp_path: Path) -> Path:
    from tests.per_model_seats import small_config
    from wargame_rl.wargame.envs.per_model import PerModelEnv
    from wargame_rl.wargame.model.per_model import (
        PerModelPPOConfig,
        SetNetwork,
        SetNetworkConfig,
        save_checkpoint,
    )

    env = PerModelEnv(small_config(opponent_x=22))
    torch.manual_seed(0)
    network = SetNetwork.from_env(
        env, SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)
    )
    path = tmp_path / "per-model-2026-09-10-12-00-00-armP" / "last.pt"
    save_checkpoint(
        path,
        network,
        ppo_config=PerModelPPOConfig(),
        env_config=env.config.model_dump(mode="json"),
        rounds=4,
        seed=0,
        revision="test",
    )
    return path


def test_a_per_model_checkpoint_is_refused_by_the_phase_resolver(
    env: WargameEnv, tmp_path: Path
) -> None:
    path = _small_per_model_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="per-model checkpoint"):
        build_action_selector(str(path), env)


def test_a_whole_phase_checkpoint_is_refused_by_the_per_model_resolver(
    env: WargameEnv, tmp_path: Path
) -> None:
    from tests.per_model_seats import small_config
    from wargame_rl.wargame.envs.per_model import PerModelEnv
    from wargame_rl.wargame.selectors import build_per_model_chooser

    checkpoint = tmp_path / "run" / "last.ckpt"
    checkpoint.parent.mkdir()
    _write_checkpoint(env, checkpoint)
    with pytest.raises(ValueError, match="whole-phase checkpoint"):
        build_per_model_chooser(str(checkpoint), [PerModelEnv(small_config())])


def test_the_per_model_resolver_seats_every_kind_of_policy(tmp_path: Path) -> None:
    from tests.per_model_seats import small_config
    from wargame_rl.wargame.envs.per_model import PerModelEnv
    from wargame_rl.wargame.selectors import build_per_model_chooser

    path = _small_per_model_checkpoint(tmp_path)
    for spec, kind in (
        ("squad_march_take", "baseline"),
        ("random", "random"),
        ("set_network", "set_network"),
        (str(path), "checkpoint"),
    ):
        envs = [PerModelEnv(small_config(opponent_x=22))]
        resolved = build_per_model_chooser(spec, envs, seed=1)
        assert resolved.kind == kind, spec
        observation, _ = envs[0].reset(seed=0)
        action = resolved.choose(envs, [observation])[0]
        assert observation.decision.why_illegal(action) is None
    assert (
        build_per_model_chooser(str(path), [PerModelEnv(small_config())]).label
        == "armP"
    )
    with pytest.raises(ValueError, match="neither a per-model checkpoint"):
        build_per_model_chooser("not_a_policy", [PerModelEnv(small_config())])


def test_evaluate_spec_scores_both_facades_on_one_table(tmp_path: Path) -> None:
    from tests.per_model_seats import small_config
    from wargame_rl.wargame.envs.baseline.evaluate import evaluate_selector
    from wargame_rl.wargame.scoring import evaluate_spec, record_per_model, record_spec

    config = small_config(opponent_x=22)
    seeds = [700000, 700001]
    scripted = evaluate_spec("squad_march_take", config, seeds, "take")
    direct_env = WargameEnv(config)
    direct = evaluate_selector(
        build_action_selector("squad_march_take", direct_env).select,
        direct_env,
        seeds,
        "take",
    )
    assert scripted == direct

    path = _small_per_model_checkpoint(tmp_path)
    learned = evaluate_spec(str(path), config, seeds, "armP", wave_size=2)
    assert learned.n_episodes == 2 and learned.exposure_rate is None
    with pytest.raises(ValueError, match="no joint decode"):
        evaluate_spec(str(path), config, seeds, "armP", decode_topk=3)

    written = record_spec(str(path), config, seeds[0], tmp_path / "pm.jsonl")
    assert written.exists()
    # A baseline name records the phase facade through `record_spec`, and the
    # per-model facade through `record_per_model` -- two different games.
    per_model = record_per_model(
        "squad_march_take",
        config,
        seeds[0],
        tmp_path / "pm_take.jsonl",
        cadence="decision",
    )
    assert per_model.exists()
    with pytest.raises(ValueError, match="phase cadence only"):
        record_spec(
            "squad_march_take",
            config,
            seeds[0],
            tmp_path / "x.jsonl",
            cadence="decision",
        )


def test_resolving_or_scoring_a_baseline_does_not_import_torch() -> None:
    """The scoring service sits beside the resolver and must keep its
    torch-deferral: a scripted score should never pay for the tensor stack."""
    source = (
        "import sys\n"
        "from wargame_rl.wargame.scoring import evaluate_spec\n"
        "from wargame_rl.wargame.selectors import build_per_model_chooser\n"
        "print('torch' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"
