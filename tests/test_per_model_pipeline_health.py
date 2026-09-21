"""Stage 0 of the per-model curriculum: the instruments that say whether the
pipeline works before anything trains on it. The passive fingerprint on the
shared result, sampled play as a diagnostic, the regime made explicit, resume
and warm start, and the health panel's meaning -- an optimizer path that can
overfit one rollout."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic_yaml import to_yaml_str

from tests.per_model_seats import small_config
from train_per_model import affinity_clamp_warning, train
from wargame_rl.wargame.envs.per_model import (
    PerModelAction,
    PerModelEnv,
    PerModelObservation,
    StepKind,
    evaluate_per_model_chooser,
    random_chooser,
)
from wargame_rl.wargame.envs.per_model.random_seat import random_legal_action
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.types import BatchChooser
from wargame_rl.wargame.model.per_model import (
    PerModelPPOConfig,
    Rollout,
    SetAgent,
    SetNetwork,
    SetNetworkConfig,
    collect_rollout,
    compute_gae,
    load_checkpoint,
    load_training_state,
    ppo_update,
    save_checkpoint,
)
from wargame_rl.wargame.model.per_model.ppo import (
    auto_num_rollout_envs,
    rollout_entropy,
)
from wargame_rl.wargame.scoring import evaluate_spec
from wargame_rl.wargame.selectors import build_per_model_chooser

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)
SEEDS = [500000, 500001, 500002]


def _passive_chooser(seed: int) -> BatchChooser:
    """Random legal play, except that every unit opens with declaration 0:
    `stationary` in movement, `hold_fire` in shooting -- the do-nothing
    fingerprint by construction."""
    rng = np.random.default_rng(seed)

    def choose(
        envs: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        actions: list[PerModelAction] = []
        for observation in observations:
            point = observation.decision
            action = random_legal_action(point, rng)
            if point.kind is StepKind.open and point.declaration_mask[action.model][0]:
                action = PerModelAction.open(action.model, 0)
            actions.append(action)
        return actions

    return choose


def _checkpoint(tmp_path: Path, config_seed: int = 0) -> Path:
    env = PerModelEnv(small_config(opponent_x=22))
    torch.manual_seed(config_seed)
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    path = tmp_path / "per-model-2026-09-14-12-00-00-fresh" / "last.pt"
    save_checkpoint(
        path,
        network,
        ppo_config=PerModelPPOConfig(),
        env_config=env.config.model_dump(mode="json"),
        rounds=0,
        seed=0,
        revision="test",
    )
    return path


# ------------------------------------------------------- the passive shares


def test_the_passive_shares_read_one_under_a_do_nothing_seat() -> None:
    env = PerModelEnv(small_config(opponent_x=22))
    result = evaluate_per_model_chooser(_passive_chooser(0), [env], SEEDS, "passive")
    assert result.stationary_share == 1.0
    assert result.hold_fire_share == 1.0


def test_the_passive_shares_are_strictly_inside_the_unit_interval_under_random() -> (
    None
):
    env = PerModelEnv(small_config(opponent_x=22))
    result = evaluate_per_model_chooser(random_chooser(0), [env], SEEDS, "random")
    assert result.stationary_share is not None and 0.0 < result.stationary_share < 1.0
    assert result.hold_fire_share is not None and 0.0 < result.hold_fire_share < 1.0


def test_a_skipped_phase_leaves_its_share_unmeasured() -> None:
    from wargame_rl.wargame.envs.types.game_timing import BattlePhase

    config = small_config(
        opponent_x=22,
        skip_phases=[
            BattlePhase.command,
            BattlePhase.shooting,
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ],
    )
    env = PerModelEnv(config)
    result = evaluate_per_model_chooser(random_chooser(0), [env], SEEDS, "random")
    assert result.stationary_share is not None
    assert result.hold_fire_share is None


def test_the_phase_facade_measures_no_passive_share() -> None:
    result = evaluate_spec("squad_march_take", small_config(opponent_x=22), SEEDS, "t")
    assert result.stationary_share is None and result.hold_fire_share is None


def test_a_per_model_checkpoint_row_carries_the_success_criterion(
    tmp_path: Path,
) -> None:
    """A curriculum rung is decided on the success rate, which only a retimer
    can read off the per-model facade -- without one every `.pt` row printed
    `-` for success and turns, and the rung could not be read at all."""
    result = evaluate_spec(
        str(_checkpoint(tmp_path)), small_config(opponent_x=22), SEEDS, "pt"
    )
    assert result.success_rate is not None
    assert result.mean_turns is not None
    assert len(result.turns_per_episode) == len(SEEDS)


# ---------------------------------------------------- sampled play, seeded


def test_sampled_play_differs_from_greedy_and_reproduces(tmp_path: Path) -> None:
    config = small_config(opponent_x=22)
    path = str(_checkpoint(tmp_path))
    greedy = evaluate_spec(path, config, SEEDS, "g", greedy=True)
    sampled = evaluate_spec(path, config, SEEDS, "s", greedy=False)
    again = evaluate_spec(path, config, SEEDS, "s", greedy=False)
    assert sampled == again
    assert sampled != greedy
    assert greedy.stationary_share is not None
    assert sampled.stationary_share is not None


def test_sampled_play_is_refused_where_there_is_nothing_to_sample(
    tmp_path: Path,
) -> None:
    config = small_config(opponent_x=22)
    with pytest.raises(ValueError, match="sampled play is a per-model checkpoint"):
        evaluate_spec("squad_march_take", config, SEEDS, "t", greedy=False)
    with pytest.raises(ValueError, match="no greedy / sampled distinction"):
        build_per_model_chooser("random", [PerModelEnv(config)], greedy=False)


# ---------------------------------------------------------- the regime


def test_the_affinity_clamp_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    import wargame_rl.wargame.model.per_model.ppo as ppo_module

    cpu = torch.device("cpu")
    monkeypatch.setattr(ppo_module.os, "sched_getaffinity", lambda _pid: {0, 1})
    assert auto_num_rollout_envs(cpu) == 2
    warning = affinity_clamp_warning(2, cpu)
    assert warning is not None
    assert "2 CPUs visible" in warning and "--num-rollout-envs" in warning

    monkeypatch.setattr(ppo_module.os, "sched_getaffinity", lambda _pid: set(range(8)))
    assert auto_num_rollout_envs(cpu) == 4
    assert affinity_clamp_warning(4, cpu) is None


# ------------------------------------------------ the health panel's meaning


def _fixed_rollout() -> tuple[SetNetwork, PerModelPPOConfig, Rollout]:
    torch.manual_seed(0)
    env = PerModelEnv(small_config(opponent_x=22))
    observation = env.reset(seed=100, options={"augment_start": True})[0]
    retimer = PerStepReward(env)
    retimer.reset()
    network = SetNetwork.from_env(env, SMALL_TRUNK)
    # One gradient step per call: one epoch, one minibatch, no entropy bonus
    # pulling the other way.
    config = PerModelPPOConfig(
        rollout_rounds=2, n_epochs=1, batch_size=100_000, ent_coef=0.0
    )
    rollout = collect_rollout(
        [env],
        SetAgent(network),
        [retimer],
        [observation],
        config,
        generator=torch.Generator().manual_seed(0),
    )
    return network, config, rollout


def test_the_optimizer_path_overfits_one_rollout() -> None:
    """Thirty-six gradient steps on a FIXED rollout: the clipped surrogate
    falls without ever rising, the clip fraction climbs off zero, and the
    ratio's 99th percentile leaves the trust region -- exactly what PPO does
    to a batch it is shown again and again. This is what the panel keys
    mean; a run whose panel never looks like this has an optimizer problem
    before it has a learning problem."""
    network, config, rollout = _fixed_rollout()
    returns, advantages = compute_gae(rollout, config)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, eps=1e-5)
    generator = torch.Generator().manual_seed(0)
    policy_losses: list[float] = []
    clip_fractions: list[float] = []
    last = None
    for _ in range(36):
        last = ppo_update(
            network,
            optimizer,
            rollout,
            returns,
            advantages,
            config,
            generator=generator,
        )
        policy_losses.append(last.policy_loss)
        clip_fractions.append(last.clip_fraction)
    assert last is not None
    increases = sum(b > a for a, b in zip(policy_losses, policy_losses[1:]))
    assert policy_losses[-1] < policy_losses[0] - 0.05
    # The per-step rise count is init-seed noise, not a property of the
    # optimizer path: measured 2026-09-21 over init seeds 0..7 it reads
    # 0 / 7 / 6 / 0 / 7 / 7 / 2 / 8 on the pre-commitment-layer network, and
    # the old `<= 3` held on seed 0 alone. Widening the relation vector
    # (#384) shifted the RNG stream and read 6. What the docstring claims
    # and this pins is the FALL and the clip fraction; the rises stay a
    # minority of the 35 comparisons.
    assert increases <= 12
    assert clip_fractions[0] == 0.0 and clip_fractions[-1] > 0.2
    assert last.ratio_p99 > 1.0 + config.eps_clip
    assert last.ratio_p01 < 1.0
    # The pre-normalisation moments are read off the fixed batch, so they do
    # not move; the value moments do, as the critic fits the fixed returns.
    assert last.advantage_std > 0.0 and last.advantage_abs_max >= last.advantage_std
    assert last.return_std > 0.0


def test_the_entropies_are_split_by_head_as_well_as_by_phase() -> None:
    network, config, rollout = _fixed_rollout()
    entropy = rollout_entropy(network, rollout, config.batch_size)
    assert "movement" in entropy.by_phase
    assert {"declaration", "displacement"} <= set(entropy.by_head)
    assert all(value > 0.0 for value in entropy.by_head.values())
    assert entropy.selector > 0.0


# ---------------------------------------------------- resume and warm start


@pytest.fixture
def small_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "small.yaml"
    path.write_text(to_yaml_str(small_config(opponent_x=22)))
    return path


def _train(small_yaml: Path, tmp_path: Path, **overrides: object) -> Path:
    arguments: dict[str, object] = {
        "env_config_path": str(small_yaml),
        "rounds": 4,
        "rollout_rounds": 1,
        "num_rollout_envs": 2,
        "eval_every_rounds": 2,
        "checkpoint_every_rounds": 2,
        "n_eval_episodes": 2,
        "eval_wave_size": 2,
        "n_layers": 1,
        "embedding_size": 32,
        "seed": 1,
        "no_wandb": True,
        "checkpoint_root": str(tmp_path / "ckpt"),
    }
    arguments.update(overrides)
    run_dir = train(**arguments)  # type: ignore[arg-type]
    return Path(run_dir)


def _update_rows(run_dir: Path) -> list[dict[str, float]]:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    return [row for row in rows if "loss/train_loss" in row]


def test_a_resume_continues_the_same_run_in_place(
    small_yaml: Path, tmp_path: Path
) -> None:
    run_dir = _train(small_yaml, tmp_path)
    first = _update_rows(run_dir)
    assert [row["rounds"] for row in first] == [2, 4]
    assert all(row["train/rounds_per_update"] == 2 for row in first)
    assert all("train/gradient_steps_per_round" in row for row in first)
    assert all("train/ratio_p99" in row for row in first)
    # A head's entropy is logged for the updates in which it fired; a random
    # init can declare every unit stationary for an update (measured
    # 2026-09-21, #384), so the key is required on SOME update, not all.
    assert any("train/entropy/head/displacement" in row for row in first)
    assert all("eval/stationary_share" in row for row in first)
    assert all("eval/vp_margin_se" in row for row in first)

    resumed = _train(small_yaml, tmp_path, rounds=8, resume_from=str(run_dir))
    assert resumed == run_dir
    rows = _update_rows(run_dir)
    assert [row["rounds"] for row in rows] == [2, 4, 6, 8]
    # The bar is logged once; the cumulative drift gauge carries across.
    all_rows = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert sum("eval/baseline_random_vp_margin" in line for line in all_rows) == 1
    assert (
        rows[2]["train/approx_kl_cumulative"] >= rows[1]["train/approx_kl_cumulative"]
    )
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["resumed_from"][0]["rounds"] == 4
    assert provenance["driver"]["rounds"] == 8
    loaded = load_checkpoint(run_dir / "last.pt")
    assert loaded.rounds == 8 and loaded.seed == 1
    assert (run_dir / "pm-00000008.pt").exists()


def test_a_resume_refuses_a_changed_knob_a_short_budget_and_another_seed(
    small_yaml: Path, tmp_path: Path
) -> None:
    run_dir = _train(small_yaml, tmp_path)
    with pytest.raises(ValueError, match="would train nothing"):
        _train(small_yaml, tmp_path, rounds=4, resume_from=str(run_dir))
    with pytest.raises(ValueError, match="refuses changed knobs: lr="):
        _train(small_yaml, tmp_path, rounds=8, resume_from=str(run_dir), lr=1e-3)
    with pytest.raises(ValueError, match="different trunk"):
        _train(small_yaml, tmp_path, rounds=8, resume_from=str(run_dir), n_layers=2)
    with pytest.raises(ValueError, match="refuses seed 2"):
        _train(small_yaml, tmp_path, rounds=8, resume_from=str(run_dir), seed=2)
    with pytest.raises(ValueError, match="pass one or the other"):
        _train(
            small_yaml,
            tmp_path,
            rounds=8,
            resume_from=str(run_dir),
            warm_start_from=str(run_dir / "last.pt"),
        )


def test_a_checkpoint_without_training_state_cannot_be_resumed(
    tmp_path: Path,
) -> None:
    path = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="no training state"):
        load_training_state(path)


def test_a_warm_start_carries_the_weights_onto_a_bigger_army(
    small_yaml: Path, tmp_path: Path
) -> None:
    source = _train(small_yaml, tmp_path)
    bigger = tmp_path / "bigger.yaml"
    bigger.write_text(
        to_yaml_str(small_config(opponent_x=22, n_models=9, max_groups=3))
    )
    run_dir = _train(
        bigger,
        tmp_path,
        rounds=2,
        seed=3,
        warm_start_from=str(source / "last.pt"),
        n_layers=None,
        embedding_size=None,
    )
    assert run_dir != source
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["warm_started_from"]["rounds"] == 4
    assert provenance["network"]["n_layers"] == 1
    assert "resumed_from" not in provenance
    started = load_checkpoint(source / "last.pt").network.state_dict()
    trained = load_checkpoint(run_dir / "last.pt").network.state_dict()
    assert started.keys() == trained.keys()
    assert any(not torch.equal(started[k], trained[k]) for k in started)
    with pytest.raises(ValueError, match="different trunk"):
        _train(
            bigger,
            tmp_path,
            rounds=2,
            warm_start_from=str(source / "last.pt"),
            n_layers=2,
        )
