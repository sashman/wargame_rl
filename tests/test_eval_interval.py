"""Evaluation can be run every Nth epoch, but never on a curriculum config.

Evaluation is a per-epoch tax that sits *outside* `perf/epoch_s`: at the 30
episodes the seeded recipes pass, it is 1200 environment steps against the
rollout's 2048 — about 22% of a real epoch. Skipping most of them is the
cheapest wall-clock win available, and it is safe only because nothing about
evaluation feeds back into the weights.

Except on a curriculum config, where it does. `try_advance` counts *consecutive*
epochs above the success threshold, so a coarser evaluation cadence changes which
epoch a phase advances on — and therefore what the run trains. That is rejected
at construction rather than clamped, so a curriculum run cannot quietly train
something different from what was asked for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.ppo.lightning import PPOLightning
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

CONFIG_DIR = Path(__file__).parents[1] / "configs"
SINGLE_PHASE = "golden/25v25_shooting_opponent.yaml"
CURRICULUM = "dev/4v4_two_phases.yaml"


def _env(config_name: str) -> WargameEnv:
    config = parse_yaml_raw_as(WargameEnvConfig, (CONFIG_DIR / config_name).read_text())
    return WargameEnv(config=config)


def _module(config_name: str, eval_every_n_epochs: int) -> PPOLightning:
    env = _env(config_name)
    return PPOLightning(
        env=env,
        ppo_model=PPO_Transformer.from_env(env),
        eval_every_n_epochs=eval_every_n_epochs,
    )


class _FakeTrainer:
    """`current_epoch` is a LightningModule property reading off the trainer.

    Attaching one is the cheapest way to exercise the cadence at a chosen epoch
    without running a fit — the decision under test is pure arithmetic on the
    epoch index and the budget.
    """

    def __init__(self, max_epochs: int | None, current_epoch: int = 0) -> None:
        self.max_epochs = max_epochs
        self.current_epoch = current_epoch


def _should_evaluate_at(
    module: PPOLightning, epoch: int, max_epochs: int | None
) -> bool:
    module._trainer = _FakeTrainer(max_epochs, epoch)  # type: ignore[assignment]
    return module._should_evaluate()


def test_default_evaluates_every_epoch() -> None:
    """The historical behaviour must be untouched unless asked for."""
    module = _module(SINGLE_PHASE, eval_every_n_epochs=1)

    assert all(_should_evaluate_at(module, epoch, 10) for epoch in range(10))


def test_interval_skips_the_epochs_in_between() -> None:
    module = _module(SINGLE_PHASE, eval_every_n_epochs=4)

    evaluated = [epoch for epoch in range(12) if _should_evaluate_at(module, epoch, 12)]

    # Counting from 1, so epochs 3, 7 and 11 (0-indexed) close a group of four.
    assert evaluated == [3, 7, 11]


def test_the_final_epoch_always_evaluates() -> None:
    """A run must not end on a stale score, whatever the cadence.

    `reward/mean_episode_reward` is logged only from the evaluation path and is
    what `get_checkpoint_callback` monitors, so the last epoch must produce it.
    """
    module = _module(SINGLE_PHASE, eval_every_n_epochs=100)

    assert _should_evaluate_at(module, 6, 7)
    assert not _should_evaluate_at(module, 5, 7)


def test_an_unknown_epoch_budget_still_honours_the_cadence() -> None:
    """`max_epochs=None` (open-ended fit) must not disable evaluation."""
    module = _module(SINGLE_PHASE, eval_every_n_epochs=3)

    assert _should_evaluate_at(module, 2, None)
    assert not _should_evaluate_at(module, 3, None)


def test_a_curriculum_config_rejects_an_interval() -> None:
    """The one case where skipping evaluation changes what the run trains."""
    with pytest.raises(ValueError, match="curriculum config"):
        _module(CURRICULUM, eval_every_n_epochs=2)


def test_a_curriculum_config_still_accepts_the_default() -> None:
    """Only the skipping is rejected; curriculum runs must still work."""
    module = _module(CURRICULUM, eval_every_n_epochs=1)

    assert module.eval_every_n_epochs == 1


def test_a_zero_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="eval_every_n_epochs"):
        _module(SINGLE_PHASE, eval_every_n_epochs=0)
