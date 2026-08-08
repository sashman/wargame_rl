"""`last.ckpt` must be the last epoch, not the last epoch that improved.

The regression this pins: with a single `ModelCheckpoint` carrying both `monitor`
and `save_last=True`, `last.ckpt` is only rewritten on epochs that enter the
top-k. Across one 1000-epoch batch of four runs it held epochs 970, 692, 948 and
998 — and every score labelled "at 1000 epochs" was really "at whatever epoch this
run last improved by its own training reward". On 25v25 a 300-epoch difference is
worth roughly 13 vp_margin, which is larger than most effects measured here.
"""

from __future__ import annotations

from wargame_rl.wargame.model.common.checkpoint_callback import (
    MONITORED_METRIC,
    get_checkpoint_callback,
)


def test_the_last_epoch_callback_is_not_gated_on_the_monitored_metric() -> None:
    """The whole bug: a monitored callback writes `last.ckpt` only when it saves."""
    callbacks = get_checkpoint_callback("a-run", filename_prefix="ppo")

    writers = [c for c in callbacks if c.save_last]
    assert len(writers) == 1, "exactly one callback may own last.ckpt"
    assert writers[0].monitor is None, (
        "the callback writing last.ckpt must be unmonitored, or last.ckpt "
        "silently becomes best.ckpt"
    )


def test_best_by_reward_is_still_kept_separately() -> None:
    """Splitting the roles must not lose model selection."""
    callbacks = get_checkpoint_callback("a-run", filename_prefix="ppo")

    monitored = [c for c in callbacks if c.monitor is not None]
    assert len(monitored) == 1
    assert monitored[0].monitor == MONITORED_METRIC
    assert monitored[0].save_top_k == 3
    assert monitored[0].mode == "max"


def test_both_callbacks_write_to_the_same_run_directory() -> None:
    """`measure-checkpoint` takes one directory; the two must not diverge."""
    callbacks = get_checkpoint_callback("a-run", filename_prefix="ppo")

    directories = {str(c.dirpath) for c in callbacks}
    assert len(directories) == 1
    assert directories.pop().endswith("checkpoints/a-run")
