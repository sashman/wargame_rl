"""`last.ckpt` must be the last epoch, and must exist before the run ends.

Two regressions are pinned here, and the second was introduced by the fix for the
first.

1. With a single `ModelCheckpoint` carrying both `monitor` and `save_last=True`,
   `last.ckpt` is only rewritten on epochs that enter the top-k. Across one
   1000-epoch batch of four runs it held epochs 970, 692, 948 and 998 — and every
   score labelled "at 1000 epochs" was really "at whatever epoch this run last
   improved by its own training reward". On 25v25 a 300-epoch difference is worth
   roughly 13 vp_margin, larger than most effects measured here.

2. Splitting that into an unmonitored `ModelCheckpoint(save_top_k=0,
   save_last=True)` fixed completed runs but wrote `last.ckpt` exactly once, from
   `on_train_end` — so a run killed partway left none at all. Runs here are
   routinely killed, and scoring the surviving top-k files reintroduces exactly
   the selection bias of (1).

The tests below are deliberately **behavioural**: they run a real `Trainer.fit`
and look at the files on disk. The previous version of this file asserted only
callback *configuration* (`save_last` is set, `monitor` is None), which stayed
green throughout regression (2) — the configuration was right and the behaviour
it implied never happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from wargame_rl.wargame.model.common.checkpoint_callback import (
    MONITORED_METRIC,
    PeriodicLastCheckpoint,
    get_checkpoint_callback,
)


class _OneItem(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.zeros(1)


class _Tiny(LightningModule):
    """Manual-optimisation module whose reward peaks early, then declines.

    The shape matters: after the peak the monitored callback stops saving, which
    is the condition under which the original bug froze `last.ckpt`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.automatic_optimization = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        optimizer = self.optimizers()
        loss = self.layer(batch).sum()
        optimizer.zero_grad()  # type: ignore[union-attr]
        self.manual_backward(loss)
        optimizer.step()  # type: ignore[union-attr]
        self.log(MONITORED_METRIC, float(3 - abs(self.current_epoch - 3)))

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(_OneItem(), batch_size=1)


class _RecordLastCkptPresence(Callback):
    """Note whether `last.ckpt` was on disk at the end of each epoch."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.seen_at_epoch: list[int] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.path.exists():
            self.seen_at_epoch.append(trainer.current_epoch)


def _fit(
    directory: Path, max_epochs: int, every_n_epochs: int
) -> tuple[_RecordLastCkptPresence, PeriodicLastCheckpoint]:
    last = PeriodicLastCheckpoint(str(directory), every_n_epochs=every_n_epochs)
    spy = _RecordLastCkptPresence(last.checkpoint_path)
    trainer = Trainer(
        accelerator="cpu",
        # `PeriodicLastCheckpoint` is a plain `Callback`, so Lightning sees no
        # user checkpointer and injects its own — and with no logger that one
        # resolves its directory to `default_root_dir/checkpoints`. Left unset
        # it is the cwd, so the repo's own `checkpoints/` collected a stray
        # `epoch=N-step=M.ckpt` per run, `-vN`-suffixed and never overwritten.
        default_root_dir=str(directory),
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[last, spy],
    )
    trainer.fit(_Tiny())
    return spy, last


def test_last_ckpt_exists_during_training_not_only_at_the_end(tmp_path: Path) -> None:
    """The regression that a configuration-only test could not see."""
    spy, last = _fit(tmp_path, max_epochs=10, every_n_epochs=3)

    # Epochs are 0-indexed and the cadence counts from 1, so writes land at the
    # end of epochs 2, 5 and 8 — visible to the spy from epoch 2 onwards.
    assert spy.seen_at_epoch, (
        "last.ckpt never appeared while training was running, so a killed run "
        "would leave only best-by-training-reward checkpoints"
    )
    assert min(spy.seen_at_epoch) == 2
    assert last.checkpoint_path.exists()


def test_last_ckpt_holds_the_final_epoch(tmp_path: Path) -> None:
    """The original bug: it must not be the last epoch that *improved*."""
    _, last = _fit(tmp_path, max_epochs=10, every_n_epochs=3)

    payload = torch.load(last.checkpoint_path, map_location="cpu", weights_only=False)

    # 10 epochs trained, and the monitored metric peaked at epoch 3 — so a
    # `last.ckpt` gated on improvement would read 3 or 4 here.
    assert payload["epoch"] == 10


def test_final_write_happens_even_off_cadence(tmp_path: Path) -> None:
    """`on_train_end` must fire regardless of where the cadence fell."""
    _, last = _fit(tmp_path, max_epochs=7, every_n_epochs=25)

    payload = torch.load(last.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 7


def test_no_staging_file_is_left_behind(tmp_path: Path) -> None:
    """The staged write must be renamed, never left as a partial file.

    A stray `last.ckpt.partial` would be harmless but a leftover would mean the
    rename did not run — i.e. the write was not atomic after all.
    """
    _, last = _fit(tmp_path, max_epochs=4, every_n_epochs=2)

    assert not list(tmp_path.glob("*.partial"))
    assert last.checkpoint_path.exists()


def test_a_zero_cadence_is_rejected_at_construction() -> None:
    """`% 0` would raise per epoch, deep inside the training loop."""
    with pytest.raises(ValueError, match="every_n_epochs"):
        PeriodicLastCheckpoint("./checkpoints/whatever", every_n_epochs=0)


def test_best_by_reward_is_still_kept_separately() -> None:
    """Splitting the roles must not lose model selection."""
    callbacks = get_checkpoint_callback("a-run", filename_prefix="ppo")

    monitored = [c for c in callbacks if getattr(c, "monitor", None) is not None]
    assert len(monitored) == 1
    assert monitored[0].monitor == MONITORED_METRIC
    assert monitored[0].save_top_k == 3
    assert monitored[0].mode == "max"


def test_exactly_one_callback_owns_last_ckpt() -> None:
    """Two writers would race on the same path."""
    callbacks = get_checkpoint_callback("a-run", filename_prefix="ppo")

    owners = [c for c in callbacks if isinstance(c, PeriodicLastCheckpoint)]
    assert len(owners) == 1
    assert not any(getattr(c, "save_last", False) for c in callbacks), (
        "no ModelCheckpoint may also write last.ckpt — it would be gated on the "
        "monitored metric and silently become best.ckpt"
    )


def test_both_callbacks_write_to_the_same_run_directory() -> None:
    """`measure-checkpoint` takes one directory; the two must not diverge.

    Resolved before comparing: `ModelCheckpoint` absolutises `dirpath` on
    construction and `PeriodicLastCheckpoint` keeps what it was given, so the raw
    strings differ while pointing at the same directory.
    """
    callbacks = get_checkpoint_callback("a-run", filename_prefix="ppo")

    directories = {Path(str(c.dirpath)).resolve() for c in callbacks}
    assert len(directories) == 1
    assert str(directories.pop()).endswith("checkpoints/a-run")
