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
from wargame_rl.wargame.model.common.observation import observation_to_tensor


def _run_recording(
    run_name: str,
    epoch: int,
    env_config: WargameEnvConfig,
    policy_state_dict_path: str,
    checkpoint_dir: str,
    render_fps: int,
    filename_prefix: str,
    renderer_name: str = "legacy",
    backend: str = "pygame",
) -> None:
    """Run in a separate process: create env with a renderer, run one episode, save MP4.
    Must set SDL_VIDEODRIVER=dummy before any pygame import to avoid EGL conflicts with PyTorch.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    from typing import cast

    import imageio  # type: ignore[import-untyped]
    import numpy as np
    import torch

    from wargame_rl.wargame.envs.renders.renderer import FrameSource
    from wargame_rl.wargame.envs.renders.v2 import build_renderer
    from wargame_rl.wargame.envs.types import WargameEnvAction
    from wargame_rl.wargame.model.common.factory import create_environment
    from wargame_rl.wargame.model.net import TransformerNetwork

    # Build env with the chosen renderer in headless recording mode. `setup`/`close`
    # come from the Renderer base; `epoch`/`get_frame_array` from the FrameSource
    # protocol both legacy and v2 satisfy.
    renderer = build_renderer(renderer_name, "recording", backend=backend)
    env = create_environment(env_config=env_config, renderer=renderer)
    renderer.setup(env)
    frame_source = cast(FrameSource, renderer)
    frame_source.epoch = epoch

    # Load snapshot from file (avoids pickling tensors across processes)
    policy_state_dict = torch.load(
        policy_state_dict_path, map_location="cpu", weights_only=True
    )
    try:
        policy_net = TransformerNetwork.policy_from_env(env)
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
            frame = frame_source.get_frame_array()
            frames.append(frame)

        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Name consistent with checkpoint style: <prefix>-epoch-XXX-recording.mp4
        filename = f"{filename_prefix}-epoch-{epoch:03d}-recording.mp4"
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
    """Records a single episode as MP4 at regular training intervals (async).

    Starts after record_after_epoch, then records every record_every_n_epochs.
    Uses human-style rendering and stores videos next to checkpoints.
    Runs in a separate process (spawn) so SDL uses the dummy driver and does
    not conflict with PyTorch/EGL.
    """

    def __init__(
        self,
        run_name: str,
        env_config: WargameEnvConfig,
        record_during_training: bool = True,
        record_after_epoch: int = 20,
        record_every_n_epochs: int = 20,
        filename_prefix: str = "ppo",
        renderer_name: str = "legacy",
        backend: str = "pygame",
    ) -> None:
        self.run_name = run_name
        self.env_config = env_config
        self.record_during_training = record_during_training
        self.record_after_epoch = record_after_epoch
        self.record_every_n_epochs = max(1, record_every_n_epochs)
        self.filename_prefix = filename_prefix
        self.renderer_name = renderer_name
        self.backend = backend
        self._checkpoint_dir = f"./checkpoints/{run_name}"
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
            wandb.log(
                {"episode_recording": wandb.Video(str(filepath), format="mp4")},
                commit=False,
            )
            self._logged_videos.add(video_key)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._try_log_pending_video()

        if not self.record_during_training:
            return
        epoch = trainer.current_epoch
        if epoch < self.record_after_epoch:
            return

        epochs_since_start = epoch - self.record_after_epoch
        if (
            epochs_since_start > 0
            and epochs_since_start % self.record_every_n_epochs != 0
        ):
            return

        if self._pending_proc is not None and self._pending_proc.is_alive():
            logger.debug("Skipping recording — previous recording still in progress")
            return

        env = getattr(pl_module, "env", None)
        if env is None or not hasattr(env, "metadata"):
            logger.warning(
                "RecordEpisodeCallback: pl_module has no env with metadata; skipping recording"
            )
            return

        policy_module: nn.Module | None = None
        ppo_model = getattr(pl_module, "ppo_model", None)
        policy_network = (
            getattr(ppo_model, "policy_network", None)
            if ppo_model is not None
            else None
        )
        if isinstance(policy_network, nn.Module):
            policy_module = policy_network

        if policy_module is None:
            logger.warning(
                "RecordEpisodeCallback: pl_module has no ppo_model.policy_network; skipping recording"
            )
            return

        # Save policy snapshot to a temp file so the spawn child can load it (pickling
        # tensors across processes hits shared-memory permission errors).
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False, prefix="record_policy_"
        ) as f:
            policy_state_dict_path = f.name
        orig_net: nn.Module = getattr(policy_module, "_orig_mod", policy_module)
        torch.save(orig_net.state_dict(), policy_state_dict_path)

        run_name = self.run_name
        env_config = self.env_config
        checkpoint_dir = self._checkpoint_dir
        epoch = trainer.current_epoch
        render_fps = cast(int, env.metadata["render_fps"])
        filename_prefix = self.filename_prefix

        filepath = (
            Path(checkpoint_dir) / f"{filename_prefix}-epoch-{epoch:03d}-recording.mp4"
        )

        # Run in a separate process so SDL_VIDEODRIVER=dummy is set before any pygame/CUDA
        # init; same process (or thread) hits EGL_BAD_ACCESS when PyTorch already has a context.
        proc = multiprocessing.get_context("spawn").Process(
            target=_run_recording,
            kwargs={
                "run_name": run_name,
                "epoch": epoch,
                "env_config": env_config,
                "policy_state_dict_path": policy_state_dict_path,
                "checkpoint_dir": checkpoint_dir,
                "render_fps": render_fps,
                "filename_prefix": filename_prefix,
                "renderer_name": self.renderer_name,
                "backend": self.backend,
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
