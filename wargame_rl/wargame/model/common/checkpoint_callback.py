"""Checkpoint callbacks for Lightning training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

MONITORED_METRIC = "reward/mean_episode_reward"
# ~148 MB a write, so at this cadence the cost is ~0.014 s per epoch against a
# ~6.5 s epoch. Close enough to free to not be worth a knob.
DEFAULT_LAST_EVERY_N_EPOCHS = 25


class PeriodicLastCheckpoint(Callback):
    """Keep `last.ckpt` on disk *during* training, not just at the end.

    `ModelCheckpoint(save_last=True)` cannot do this job, and the reason is not
    obvious: `_save_last_checkpoint` runs only when
    `_last_global_step_saved == trainer.global_step`, and that field is set by
    `_save_topk_checkpoint`. With `save_top_k=0` it returns early and never sets
    it, so `last.ckpt` is written exactly once, from `on_train_end`.

    For a run that finishes, that is correct and the earlier bug (a *monitored*
    callback writing `last.ckpt` only on improving epochs, so it silently held
    "the last epoch that improved") stays fixed. But runs here are routinely
    stopped before `trainer.fit()` returns -- monitored, judged and killed -- and
    such a run was left with no `last.ckpt` at all, only the best-by-training-
    reward files. Scoring those reintroduces exactly the ~13 vp_margin selection
    bias the split was written to remove.

    The write is staged and renamed rather than written in place. `os.replace` is
    atomic within a filesystem, so a kill landing mid-write leaves the previous
    `last.ckpt` intact instead of truncating the one file this exists to protect.
    """

    def __init__(
        self, dirpath: str, every_n_epochs: int = DEFAULT_LAST_EVERY_N_EPOCHS
    ) -> None:
        if every_n_epochs < 1:
            raise ValueError(f"every_n_epochs must be >= 1, got {every_n_epochs}")
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs

    @property
    def checkpoint_path(self) -> Path:
        return self.dirpath / "last.ckpt"

    def _save(self, trainer: Trainer) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        staging = self.checkpoint_path.with_suffix(".ckpt.partial")
        trainer.save_checkpoint(str(staging))
        os.replace(staging, self.checkpoint_path)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write on the cadence, counting from epoch 1 rather than epoch 0."""
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._save(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Always finish with the true final epoch, whatever the cadence hit."""
        self._save(trainer)

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Salvage the weights on Ctrl-C or a crash.

        Lightning routes `KeyboardInterrupt` here, so an interactively killed run
        keeps its latest epoch. A failure to save must not mask the original
        exception, which is the thing actually worth reading.
        """
        try:
            self._save(trainer)
        except Exception as save_error:  # pragma: no cover - best effort only
            logger.warning("could not save last.ckpt during shutdown: {}", save_error)


def get_checkpoint_callback(
    name: str,
    *,
    filename_prefix: str = "model",
) -> list[Any]:
    """Return the checkpoint callbacks for a run: best-by-reward, and the last epoch.

    Two callbacks rather than one, because a single `ModelCheckpoint` cannot do
    both jobs. With `monitor` set, `save_last=True` only rewrites `last.ckpt` on
    the epochs that actually enter the top-k — so it holds *the last epoch that
    improved*, not the last epoch trained. Measured on a 1000-epoch batch, four
    runs' `last.ckpt` files held epochs 970, 692, 948 and 998, and the epoch-1000
    weights of the 692 run were never written at all.

    That silently turned every "scored at 1000 epochs" comparison into "scored at
    whatever epoch each run last improved by its own training reward" — a spread
    worth ~13 vp_margin on 25v25, and not even a common selection rule across the
    arms of an experiment, since each arm's reward is a different function.

    The last-epoch half is `PeriodicLastCheckpoint` rather than an unmonitored
    `ModelCheckpoint`; see its docstring for why the latter writes only once, at
    the very end, and why that is not good enough for runs that get killed.

    Args:
        name: Run name (used for checkpoint directory).
        filename_prefix: Prefix for checkpoint filenames (e.g. 'ppo').

    Returns:
        The best-by-reward callback and the last-epoch callback.
    """
    directory = f"./checkpoints/{name}"

    # auto_insert_metric_name=False so the "/" in the metric name is substituted
    # as a value rather than turned into a "name=" prefix (which breaks on "/").
    best_by_reward = ModelCheckpoint(
        dirpath=directory,
        filename=f"{filename_prefix}-{{epoch:03d}}-{{{MONITORED_METRIC}:.3f}}",
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor=MONITORED_METRIC,
        mode="max",
        save_last=False,
    )
    return [best_by_reward, PeriodicLastCheckpoint(directory)]
