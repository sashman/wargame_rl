"""Checkpoint callback factory for Lightning training."""

from pytorch_lightning.callbacks import ModelCheckpoint


def get_checkpoint_callback(
    name: str,
    *,
    filename_prefix: str = "model",
) -> list[ModelCheckpoint]:
    """Return a list of ModelCheckpoint callbacks.

    Args:
        name: Run name (used for checkpoint directory).
        filename_prefix: Prefix for checkpoint filenames (e.g. 'dqn' or 'ppo').

    Returns:
        List containing one ModelCheckpoint callback.
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{name}",
        # Metric is logged as "reward/mean_episode_reward" (see lightning_base.py).
        # auto_insert_metric_name=False so the "/" in the metric name is substituted
        # as a value rather than turned into a "name=" prefix (which breaks on "/").
        filename=f"{filename_prefix}-{{epoch:03d}}-{{reward/mean_episode_reward:.3f}}",
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor="reward/mean_episode_reward",
        mode="max",
        save_last=True,
    )
    return [checkpoint_callback]
