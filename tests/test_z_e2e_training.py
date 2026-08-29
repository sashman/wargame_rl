"""End-to-end test: run the training entrypoint (PPO) for one epoch.

Collected last so that faster unit/integration tests run first (file name sorts after test_*).
Uses a minimal env config (2 models, small board) to keep CI fast.

⚠ **It runs from a temporary directory, and that is not tidiness.** Every
checkpoint path in the trainer is *relative to the working directory* --
`./checkpoints/{run}` in `checkpoint_callback.py`, `env_config_callback.py` and
`record_episode_callback.py`, plus Lightning's own `lightning_logs/`. Run from
the repo root this test leaves **~292 MB behind on every invocation** (two 153 MB
checkpoints of a 2-model 20x20 board) and never cleans up. Forty-five runs of it
had accumulated **10.4 GB** in `checkpoints/` alongside the real weights, and
`lightning_logs/` held another 9.1 GB of `epoch=0-step=2.ckpt` plus a ~100 MB
`hparams.yaml` per run -- the whole `WargameEnv` pickled as a Lightning hparam.
That filled the disk and made this test fail with `ENOSPC`.

⚠ **`checkpoints/` is the only copy of any trained weights** and `just clean`
deletes it outright, so debris accumulating *in that directory* is worse than it
sounds: it buries the real runs in a directory nobody can safely bulk-delete.
"""

import os
import sys
from pathlib import Path

import pytest


def test_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: call train() for one epoch with no wandb; assert it completes without raising."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from train import train

    # Absolute, because the next line moves the working directory out from
    # under every relative path in the process.
    env_config_path = os.path.join(project_root, "configs", "dev", "ci_smoke.yaml")
    monkeypatch.chdir(tmp_path)

    train(
        render_mode=None,
        env_config_path=env_config_path,
        record_during_training=False,
        record_after_epoch=10,
        record_every_n_epochs=20,
        max_epochs=1,
        no_wandb=True,
        n_steps=64,
        n_eval_episodes=1,
        # train() is called directly rather than through typer, so every override
        # must be passed explicitly — an unset one arrives as a typer OptionInfo
        # and fails the `is not None` guard.
        gamma=None,
        ent_coef=None,
        num_rollout_envs=None,
        # A SMALL trunk. The default is 8 layers of 256 at 8 heads -- ~12.7M
        # parameters -- and this test ran it on a 2-model 20x20 board, which
        # made it the slowest test in the suite at 114s and wrote two 153 MB
        # checkpoints. What it exists to verify is the `train()` WIRING: config
        # to env to callbacks to Trainer to fit. None of that is a function of
        # trunk depth. `test_network_size` pins that the production default is
        # untouched by this.
        n_layers=2,
        embedding_size=32,
    )

    # The point of the chdir: assert the run actually landed here, so a future
    # change that hardcodes an absolute checkpoint path re-opens the leak
    # loudly instead of silently refilling the repo.
    assert (tmp_path / "checkpoints").is_dir()
    assert not (Path(project_root) / "checkpoints" / "ci_smoke").exists()
