"""PPO self-play scheduler + Elo logging callback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.model.common.elo import EloRatingSystem
from wargame_rl.wargame.model.ppo.config import SelfPlayConfig
from wargame_rl.wargame.model.ppo.elo_evaluator import (
    OpponentEntry,
    evaluate_elo_ladder,
)


class PPOSelfPlayCallback(Callback):
    """Alternate PPO training epochs between scripted-opponent and self-play."""

    def __init__(
        self,
        run_name: str,
        env_config: WargameEnvConfig,
        self_play_config: SelfPlayConfig,
    ) -> None:
        self.run_name = run_name
        self.env_config = env_config.model_copy(deep=True)
        self.cfg = self_play_config

        if env_config.opponent_policy is None:
            self._base_policy = OpponentPolicyConfig(type="random")
        else:
            self._base_policy = env_config.opponent_policy.model_copy(deep=True)

        self._activation_epoch: int | None = None
        self._snapshot_paths: list[Path] = []
        self._current_mode: str = "update"
        self._current_opponent_name: str = self._base_policy.type

        self._snapshot_dir = Path(f"./checkpoints/{self.run_name}/self_play_snapshots")
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._elo_path = Path(f"./checkpoints/{self.run_name}/elo_ratings.json")
        self._elo = EloRatingSystem(
            initial_rating=self.cfg.elo_initial_rating,
            k_factor=self.cfg.elo_k_factor,
        )
        self._elo.ensure_player("agent_current")

    def mode_for_epoch(self, epoch: int) -> str:
        if self._activation_epoch is None:
            return "update"
        cycle_len = self.cfg.update_epochs + self.cfg.self_play_epochs
        offset = epoch - self._activation_epoch
        if offset < 0:
            return "update"
        if (offset % cycle_len) < self.cfg.update_epochs:
            return "update"
        return "self_play"

    def _should_activate(self, pl_module: LightningModule) -> bool:
        if not self.cfg.enabled:
            return False
        env = getattr(pl_module, "env", None)
        if env is None or env.config.number_of_opponent_models <= 0:
            return False
        if not self.cfg.activate_on_final_reward_phase:
            return True
        return bool(env.phase_manager.is_final_phase)

    def _set_update_mode(self, pl_module: LightningModule) -> None:
        env = pl_module.env  # type: ignore[attr-defined]
        env.set_opponent_policy_config(self._base_policy.model_copy(deep=True))
        self._current_mode = "update"
        self._current_opponent_name = self._base_policy.type

    def _choose_snapshot_path(self, epoch: int) -> Path | None:
        if not self._snapshot_paths:
            return None
        index = epoch % len(self._snapshot_paths)
        return self._snapshot_paths[index]

    def _set_self_play_mode(self, pl_module: LightningModule, epoch: int) -> None:
        env = pl_module.env  # type: ignore[attr-defined]
        snapshot = self._choose_snapshot_path(epoch)
        if snapshot is None:
            self._set_update_mode(pl_module)
            return
        cfg = OpponentPolicyConfig(
            type="model",
            params={
                "checkpoint_path": str(snapshot),
                "deterministic": True,
            },
        )
        env.set_opponent_policy_config(cfg)
        self._current_mode = "self_play"
        self._current_opponent_name = f"snapshot:{snapshot.stem}"

    def _save_snapshot(self, pl_module: LightningModule, epoch: int) -> Path:
        policy_network = pl_module.ppo_model.policy_network  # type: ignore[attr-defined]
        path = self._snapshot_dir / f"ppo_policy_epoch_{epoch:04d}.pt"
        payload: dict[str, Any] = {
            "policy_state_dict": policy_network.state_dict(),
            "epoch": epoch,
            "algorithm": "ppo",
        }
        torch.save(payload, path)
        return path

    def _add_snapshot(self, path: Path) -> None:
        self._snapshot_paths.append(path)
        while len(self._snapshot_paths) > self.cfg.snapshot_pool_size:
            removed = self._snapshot_paths.pop(0)
            if removed.exists():
                removed.unlink()

    def _persist_elo(self, epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "ratings": self._elo.ratings,
            "leaderboard": self._elo.leaderboard(),
        }
        with open(self._elo_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _build_ladder(
        self, *, include_latest_snapshot: bool = False
    ) -> list[OpponentEntry]:
        ladder: list[OpponentEntry] = []
        for opponent_type in self.cfg.scripted_opponents:
            ladder.append(
                OpponentEntry(
                    name=f"policy:{opponent_type}",
                    policy=OpponentPolicyConfig(type=opponent_type),
                )
            )
        snapshots = self._snapshot_paths
        if not include_latest_snapshot and len(snapshots) > 1:
            snapshots = snapshots[:-1]
        for snapshot in snapshots:
            ladder.append(
                OpponentEntry(
                    name=f"snapshot:{snapshot.stem}",
                    policy=OpponentPolicyConfig(
                        type="model",
                        params={
                            "checkpoint_path": str(snapshot),
                            "deterministic": True,
                        },
                    ),
                )
            )
        return ladder

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self.cfg.enabled:
            return

        epoch = trainer.current_epoch
        if self._activation_epoch is None and self._should_activate(pl_module):
            self._activation_epoch = epoch
            logger.info("Self-play activated at epoch {}", epoch)

        mode = self.mode_for_epoch(epoch)
        if mode == "self_play":
            self._set_self_play_mode(pl_module, epoch)
        else:
            self._set_update_mode(pl_module)

        pl_module.log(
            "self_play/mode",
            1.0 if self._current_mode == "self_play" else 0.0,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "self_play/snapshot_pool_size",
            float(len(self._snapshot_paths)),
            prog_bar=False,
            logger=True,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.cfg.enabled:
            return
        if self._activation_epoch is None:
            return

        epoch = trainer.current_epoch
        snapshot = self._save_snapshot(pl_module, epoch)
        self._add_snapshot(snapshot)

        ladder = self._build_ladder(include_latest_snapshot=False)
        if ladder:
            ratings = evaluate_elo_ladder(
                policy_model=pl_module.ppo_model,  # type: ignore[attr-defined]
                env_config=self.env_config,
                opponents=ladder,
                n_episodes=self.cfg.eval_episodes,
                ratings=self._elo,
                agent_name="agent_current",
            )
            pl_module.log(
                "self_play/elo_agent_current",
                float(ratings["agent_current"]),
                prog_bar=False,
                logger=True,
            )
            for name, value in ratings.items():
                if name == "agent_current":
                    continue
                metric = name.replace(":", "_").replace("/", "_")
                pl_module.log(
                    f"self_play/elo_{metric}",
                    float(value),
                    prog_bar=False,
                    logger=True,
                )

        self._persist_elo(epoch)
