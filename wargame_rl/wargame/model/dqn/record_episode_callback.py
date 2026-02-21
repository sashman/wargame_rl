"""Lightning callback to record a single episode as MP4 during training (async)."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

import multiprocessing
import os
import tempfile
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

import torch
from loguru import logger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import nn

import wandb
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from wargame_rl.wargame.model.dqn.observation import observation_to_tensor


def _run_recording(
    run_name: str,
    epoch: int,
    env_config: WargameEnvConfig,
    policy_state_dict_path: str,
    policy_net_class_name: str,
    checkpoint_dir: str,
    render_fps: int,
) -> None:
    """Run in a separate process: create env with human renderer, run one episode, save MP4.
    Must set SDL_VIDEODRIVER=dummy before any pygame import to avoid EGL conflicts with PyTorch.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    import imageio  # type: ignore[import-untyped]
    import numpy as np
    import torch

    from wargame_rl.wargame.envs.renders.human import HumanRender
    from wargame_rl.wargame.envs.types import WargameEnvAction
    from wargame_rl.wargame.model.dqn.dqn import DQN_MLP, DQN_Transformer, RL_Network
    from wargame_rl.wargame.model.dqn.factory import create_environment

    # Build env with human renderer (same config as training)
    renderer = HumanRender()
    env = create_environment(env_config=env_config, renderer=renderer)
    renderer.setup(env)
    renderer.epoch = epoch

    # Load snapshot from file (avoids pickling tensors across processes)
    policy_state_dict = torch.load(
        policy_state_dict_path, map_location="cpu", weights_only=True
    )
    try:
        if policy_net_class_name == "DQN_Transformer":
            policy_net: RL_Network = DQN_Transformer.from_env(env)
        else:
            policy_net = DQN_MLP.from_env(env)
        policy_net.load_state_dict(policy_state_dict)
        policy_net.eval()
    finally:
        try:
            os.unlink(policy_state_dict_path)
        except OSError:
            pass

    frames: list[np.ndarray] = []

    try:
        observation, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state = observation_to_tensor(observation, policy_net.device)
                q_values = policy_net(state)
                _, action_indexes = q_values.max(axis=-1)
                action = WargameEnvAction(actions=action_indexes.flatten().tolist())
            observation, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            env.render()
            frame = renderer.get_frame_array()
            frames.append(frame)

        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Name consistent with checkpoint style: dqn-epoch-XXX-recording.mp4
        filename = f"dqn-epoch-{epoch:03d}-recording.mp4"
        filepath = out_dir / filename

        if frames:
            writer = imageio.get_writer(
                str(filepath),
                format="FFMPEG",  # type: ignore[arg-type]
                mode="I",
                fps=render_fps,
                codec="libx264",
                output_params=["-pix_fmt", "yuv420p"],
            )
            for f in frames:
                writer.append_data(f)
            writer.close()
    finally:
        renderer.close()
        env.close()


class RecordEpisodeCallback(Callback):
    """Records a single episode as MP4 when a new checkpoint is saved (async).

    Starts only after record_after_epoch epochs. Runs only when the checkpoint
    callback adds or removes a checkpoint file (e.g. new top-k or updated last).
    Uses human-style rendering. Saves to the same directory as checkpoints, with
    names like dqn-epoch-042-recording.mp4. Runs in a separate process (spawn) so
    SDL uses the dummy driver and does not conflict with PyTorch/EGL.
    """

    def __init__(
        self,
        run_name: str,
        env_config: WargameEnvConfig,
        record_during_training: bool = True,
        record_after_epoch: int = 20,
    ) -> None:
        self.run_name = run_name
        self.env_config = env_config
        self.record_during_training = record_during_training
        self.record_after_epoch = record_after_epoch
        self._checkpoint_dir = f"./checkpoints/{run_name}"
        self._last_ckpt_filenames: set[str] | None = None
        self._pending_proc: BaseProcess | None = None
        self._pending_filepath: Path | None = None
        self._logged_videos: set[str] = set()

    def _try_log_pending_video(self) -> None:
        """If a previous recording process finished, log its MP4 to wandb."""
        if self._pending_proc is None:
            return
        if self._pending_proc.is_alive():
            return

        filepath = self._pending_filepath
        self._pending_proc = None
        self._pending_filepath = None

        if filepath is None or not filepath.exists():
            return
        video_key = filepath.name
        if video_key in self._logged_videos:
            return

        if wandb.run is not None:
            logger.info("Logging recorded episode to wandb: {}", filepath.name)
            wandb.log({"episode_recording": wandb.Video(str(filepath), format="mp4")})
            self._logged_videos.add(video_key)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._try_log_pending_video()

        if not self.record_during_training:
            return
        if trainer.current_epoch < self.record_after_epoch:
            return

        # Only record when the checkpoint callback saved (set of .ckpt files changed).
        ckpt_dir = Path(self._checkpoint_dir)
        current_filenames = (
            {p.name for p in ckpt_dir.glob("*.ckpt")} if ckpt_dir.exists() else set()
        )
        if (
            self._last_ckpt_filenames is not None
            and current_filenames == self._last_ckpt_filenames
        ):
            self._last_ckpt_filenames = current_filenames
            return
        self._last_ckpt_filenames = current_filenames

        # If a previous recording is still running, skip this one to avoid piling up.
        if self._pending_proc is not None and self._pending_proc.is_alive():
            logger.debug("Skipping recording — previous recording still in progress")
            return

        model = cast(DQNLightning, pl_module)

        # Save policy snapshot to a temp file so the spawn child can load it (pickling
        # tensors across processes hits shared-memory permission errors).
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False, prefix="record_policy_"
        ) as f:
            policy_state_dict_path = f.name
        orig_net: nn.Module = getattr(model.policy_net, "_orig_mod", model.policy_net)
        torch.save(orig_net.state_dict(), policy_state_dict_path)

        policy_net_class_name = type(orig_net).__name__
        run_name = self.run_name
        env_config = self.env_config
        checkpoint_dir = self._checkpoint_dir
        epoch = trainer.current_epoch
        render_fps = cast(int, model.env.metadata["render_fps"])

        filepath = Path(checkpoint_dir) / f"dqn-epoch-{epoch:03d}-recording.mp4"

        # Run in a separate process so SDL_VIDEODRIVER=dummy is set before any pygame/CUDA
        # init; same process (or thread) hits EGL_BAD_ACCESS when PyTorch already has a context.
        proc = multiprocessing.get_context("spawn").Process(
            target=_run_recording,
            kwargs={
                "run_name": run_name,
                "epoch": epoch,
                "env_config": env_config,
                "policy_state_dict_path": policy_state_dict_path,
                "policy_net_class_name": policy_net_class_name,
                "checkpoint_dir": checkpoint_dir,
                "render_fps": render_fps,
            },
            daemon=True,
        )
        proc.start()
        self._pending_proc = proc
        self._pending_filepath = filepath

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Flush any remaining recording before the wandb run closes."""
        if self._pending_proc is not None:
            self._pending_proc.join(timeout=60)
        self._try_log_pending_video()
