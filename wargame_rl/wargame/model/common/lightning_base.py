from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.agent_base import BaseAgent


class WargameLightningBase(LightningModule, ABC):
    """Shared evaluation + reward-phase plumbing for RL Lightning modules."""

    def __init__(
        self,
        env: WargameEnv,
        agent: BaseAgent,
        do_log: bool = True,
        n_episodes: int = 10,
        eval_log_prefix: str = "",
    ):
        super().__init__()
        self.env = env
        self.do_log = do_log
        self.n_episodes = n_episodes
        self.agent = agent
        self.eval_log_prefix = eval_log_prefix
        self.mean_episode_reward = 0.0

    @abstractmethod
    def _policy_model(self) -> nn.Module:
        """Return the policy model used for evaluation."""

    def _run_episode_eval(self, epsilon: float) -> tuple[float, int]:
        """Run a single evaluation episode and return (reward, steps)."""
        reward, steps = self.agent.run_episode(  # type: ignore[attr-defined]
            self._policy_model(),
            epsilon=epsilon,
            render=False,
            save_steps=False,
        )
        return reward, steps

    def _set_policy_mode(self, eval_mode: bool) -> None:
        """Hook to toggle model eval/train mode for evaluation."""
        policy = self._policy_model()
        if eval_mode:
            policy.eval()
        else:
            policy.train()

    def _evaluate_episodes(
        self,
        *,
        n_episodes: int | None = None,
        epsilon: float = 0.0,
        log_prefix: str = "",
    ) -> float:
        """Run evaluation episodes and optionally log common metrics.

        Returns the success rate as a fraction in [0, 1].
        """
        total_episodes = self.n_episodes if n_episodes is None else n_episodes
        steps_s: list[int] = []
        episode_rewards: list[float] = []
        episode_successes: list[bool] = []

        self._set_policy_mode(True)

        with torch.no_grad():
            for _ in range(total_episodes):
                reward, steps = self._run_episode_eval(epsilon)
                episode_rewards.append(reward)
                steps_s.append(steps)

                if self.env.last_step_context is not None:
                    success = self.env.phase_manager.check_success(
                        self.env, self.env.last_step_context
                    )
                    episode_successes.append(success)

        self._set_policy_mode(False)

        self.mean_episode_reward = sum(episode_rewards) / len(episode_rewards)

        if self.do_log:
            prefix = f"{log_prefix}_" if log_prefix else ""
            self.log(
                f"reward/{prefix}mean_episode_reward",
                self.mean_episode_reward,
                prog_bar=False,
            )
            if not log_prefix:
                # Kept for checkpoint monitor compatibility (ModelCheckpoint
                # defaults to monitor="mean_episode_reward").
                self.log(
                    "mean_episode_reward",
                    self.mean_episode_reward,
                    prog_bar=False,
                )
            self.log(
                f"{prefix}mean_episode_steps",
                sum(steps_s) / len(steps_s),
                prog_bar=False,
            )
            self.log(
                f"reward/{prefix}max_episode_reward",
                max(episode_rewards),
                prog_bar=False,
            )
            self.log(
                f"reward/{prefix}min_episode_reward",
                min(episode_rewards),
                prog_bar=False,
            )

        if episode_successes:
            sr = float(np.array(episode_successes, dtype=float).mean())
        else:
            sr = float((np.array(steps_s) < self.env.max_turns).mean())

        if self.do_log:
            prefix = f"{log_prefix}_" if log_prefix else ""
            self.log(f"{prefix}success_rate", sr * 100, prog_bar=False)

        return sr

    def _advance_reward_phase(self, success_rate: float) -> bool:
        advanced = self.env.phase_manager.try_advance(success_rate, self.current_epoch)
        if self.do_log:
            phase_index = int(self.env.phase_manager.current_phase_index)
            self.log(
                "reward_phase",
                float(phase_index),
                prog_bar=False,
            )
            try:
                import wandb

                if wandb.run is not None:  # type: ignore[attr-defined]
                    wandb.log({"reward_phase": phase_index}, step=self.global_step)  # type: ignore[attr-defined]
            except ModuleNotFoundError:
                pass
            if advanced:
                self.log(
                    "phase_advanced_at_epoch",
                    float(self.current_epoch),
                    prog_bar=False,
                )
        return advanced

    def run_episodes(self, n_episodes: int, epsilon: float = 0.0) -> float:
        """Run evaluation episodes and log common metrics."""
        return self._evaluate_episodes(
            n_episodes=n_episodes,
            epsilon=epsilon,
            log_prefix=self.eval_log_prefix,
        )

    def on_train_epoch_end(self) -> None:
        if self.do_log:
            sr = self.run_episodes(self.n_episodes)
            self._advance_reward_phase(sr)
        super().on_train_epoch_end()
