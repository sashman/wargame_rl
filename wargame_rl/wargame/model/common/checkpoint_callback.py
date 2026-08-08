"""Checkpoint callback factory for Lightning training."""

from pytorch_lightning.callbacks import ModelCheckpoint

MONITORED_METRIC = "reward/mean_episode_reward"


def get_checkpoint_callback(
    name: str,
    *,
    filename_prefix: str = "model",
) -> list[ModelCheckpoint]:
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

    Splitting them makes `last.ckpt` mean what its name says. The top-k files are
    still there under `{prefix}-{epoch}-{reward}.ckpt` when best-by-reward is
    genuinely what is wanted.

    Args:
        name: Run name (used for checkpoint directory).
        filename_prefix: Prefix for checkpoint filenames (e.g. 'dqn' or 'ppo').

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
    # Unmonitored, so it writes every epoch and `last.ckpt` is genuinely the last.
    last_epoch = ModelCheckpoint(dirpath=directory, save_top_k=0, save_last=True)
    return [best_by_reward, last_epoch]
