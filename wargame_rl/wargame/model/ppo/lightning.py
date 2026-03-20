from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.lightning_base import WargameLightningBase
from wargame_rl.wargame.model.common.observation import observations_to_tensor_batch
from wargame_rl.wargame.model.ppo.agent import Agent
from wargame_rl.wargame.types import Experience

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.ppo import PPOModel


class _NoOpProgress:
    """No-op progress object when inner progress bars are disabled."""

    def update(self, n: int = 1) -> None:
        pass


class _WargameEnvActionWrapper(gym.ActionWrapper):
    """Convert vector-env Tuple actions into `WargameEnvAction`.

    Gymnasium vector environments expect actions compatible with `action_space`.
    Our `WargameEnv.step()` expects a `WargameEnvAction`, so this wrapper
    bridges the formats for action dispatch.
    """

    def action(self, action: Any) -> WargameEnvAction:  # type: ignore[override]
        if isinstance(action, WargameEnvAction):
            return action
        # Expected shape from vector env: tuple/list/ndarray of length n_models.
        actions_list = [int(x) for x in np.asarray(action).reshape(-1)]
        return WargameEnvAction(actions=actions_list)


class _PPODummyDataset(Dataset[Tensor]):
    """Single-item dataset so Lightning calls training_step once per epoch."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Tensor:
        return torch.tensor(0.0)


class PPOLightning(WargameLightningBase):
    """PPO Lightning Module for training PPO agents."""

    def _largest_divisor_at_most(self, n: int, max_value: int) -> int:
        """Return largest divisor of `n` that is <= `max_value`."""
        candidate = min(n, max_value)
        for v in range(candidate, 0, -1):
            if n % v == 0:
                return v
        return 1

    def _cuda_appears_usable(self) -> bool:
        """Best-effort check that CUDA kernels can execute.

        This is defensive against environments where CUDA is present but
        incompatible with the installed GPU/driver.
        """
        try:
            if not torch.cuda.is_available():
                return False
            if self.ppo_model.device.type != "cuda":
                return False
            # Tiny kernel to force real CUDA initialization / execution.
            _ = torch.empty((1,), device=self.ppo_model.device).sum().item()
            return True
        except Exception:
            return False

    def _auto_detect_num_rollout_envs(self) -> int:
        """Pick a heuristic `num_rollout_envs` based on CPU/GPU availability."""
        # Respect CPU affinity / cgroup limits when possible.
        cpu_count = 1
        try:
            if hasattr(os, "sched_getaffinity"):
                cpu_count = len(os.sched_getaffinity(0))  # type: ignore[arg-type]
            else:
                cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = os.cpu_count() or 1

        # If the model runs on a usable GPU, we can typically afford more envs
        # to amortize inference overhead. On CPU, keep it conservative.
        if self._cuda_appears_usable():
            max_envs = 8
        else:
            max_envs = 4

        heuristic = max(1, min(max_envs, cpu_count))
        # Enforce `n_steps` divisibility so rollout collection never errors.
        return self._largest_divisor_at_most(self.n_steps, heuristic)

    def __init__(
        self,
        env: WargameEnv,
        ppo_model: PPOModel,
        log: bool = True,
        batch_size: int = 1024,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        n_steps: int = 2048,
        num_rollout_envs: int = 0,
        n_episodes: int = 10,
        show_inner_progress: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO Lightning Module.

        Args:
            env: Wargame environment
            ppo_model: PPO policy-value model
            log: Whether to log metrics
            batch_size: Minibatch size for PPO updates (samples per gradient step)
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: Generalized Advantage Estimation lambda
            eps_clip: PPO clipping parameter
            vf_coef: Value function coefficient
            ent_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            n_epochs: Number of epochs for PPO updates
            n_steps: Number of steps to collect before each update
            num_rollout_envs: Number of parallel env instances for rollout
                collection (must be >= 1). When set to 1, rollout collection is
                unchanged.
            n_episodes: Number of episodes to run for evaluation
            show_inner_progress: Whether to show tqdm for rollout and PPO minibatch updates
        """
        super().__init__(env=env, agent=Agent(env), do_log=log, n_episodes=n_episodes)
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.show_inner_progress = show_inner_progress
        self.ppo_model = ppo_model
        self.total_reward = 0
        self.episode_reward = 0
        self.batch_size = batch_size
        self.n_steps = n_steps
        if num_rollout_envs <= 0:
            self.num_rollout_envs = self._auto_detect_num_rollout_envs()
        else:
            self.num_rollout_envs = num_rollout_envs
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.ppo_model.parameters(),
            lr=lr,
            eps=1e-5,
        )

        # Initialize loss components
        self.value_loss_fn = nn.MSELoss()
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def forward(self, x: list[Tensor]) -> Tensor:
        """Forward pass through the policy network.

        Args:
            x: List of input tensors (game state, objectives, models).

        Returns:
            Action logits with shape (batch, n_models, n_actions).
        """
        action_logits: Tensor
        action_logits, _ = self.ppo_model(x)
        return action_logits

    def compute_returns(
        self,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
        last_value: Tensor | None = None,
    ) -> Tensor:
        """Compute returns using Generalized Advantage Estimation.

        Args:
            rewards: Rewards for each step
            dones: Done flags for each step
            values: Values for each step

        Returns:
            Computed returns
        """
        # Compute advantages using GAE.
        # Supports both:
        # - rewards/dones/values with shape (T,)
        # - rewards/dones/values with shape (T, num_envs)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])

        time_steps = rewards.shape[0]
        for t in reversed(range(time_steps)):
            if t == time_steps - 1:
                if last_value is None:
                    next_value = torch.zeros_like(values[t])
                else:
                    next_value = last_value.to(
                        device=rewards.device, dtype=rewards.dtype
                    )
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute returns
        returns = advantages + values
        return returns

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Carry out a single training step.

        Collects n_steps transitions (across one or more episodes), computes
        GAE returns/advantages, then runs n_epochs of minibatch PPO updates.

        Uses manual optimization because PPO runs multiple gradient steps
        per rollout. Observations are converted via ``observations_to_tensor_batch``;
        actions are extracted from ``WargameEnvAction.actions``.

        Args:
            batch: Unused (rollout is collected inline)
            batch_idx: Batch index
        """
        device = self.ppo_model.device
        optimizer = self.optimizers()

        rollout_reward_breakdown: dict[str, float] = {}
        if self.num_rollout_envs == 1:
            experiences, rollout_reward_breakdown = self._collect_experiences()

            state_tensors = observations_to_tensor_batch(
                [exp.state for exp in experiences], device=device
            )
            actions = torch.tensor(
                [exp.action.actions for exp in experiences],
                dtype=torch.long,
                device=device,
            )
            rewards = torch.tensor(
                [exp.reward for exp in experiences],
                dtype=torch.float32,
                device=device,
            )
            dones = torch.tensor(
                [exp.done for exp in experiences],
                dtype=torch.float32,
                device=device,
            )
            old_log_probs = torch.stack(
                [exp.log_prob for exp in experiences]  # type: ignore[misc]
            ).detach()

            _, state_values = self.ppo_model(state_tensors)

            last_done = bool(experiences[-1].done)
            if last_done:
                last_value = torch.tensor(0.0, device=device, dtype=torch.float32)
            else:
                last_state_tensors = observations_to_tensor_batch(
                    [experiences[-1].new_state], device=device
                )
                with torch.no_grad():
                    _, last_state_value = self.ppo_model(last_state_tensors)
                last_value = last_state_value.squeeze(0).detach()

            returns = self.compute_returns(
                rewards,
                dones,
                state_values,
                last_value=last_value,
            ).detach()
            advantages = (returns - state_values).detach()
            n_steps = len(experiences)
        else:
            (
                state_tensors,
                actions,
                rewards_2d,
                dones_2d,
                old_log_probs_2d,
                values_2d,
                last_values,
                rollout_reward_breakdown,
            ) = self._collect_rollout_parallel()

            returns_2d = self.compute_returns(
                rewards_2d,
                dones_2d,
                values_2d,
                last_value=last_values,
            ).detach()
            advantages_2d = (returns_2d - values_2d).detach()

            returns = returns_2d.reshape(-1)
            advantages = advantages_2d.reshape(-1)
            old_log_probs = old_log_probs_2d.reshape(-1).detach()
            n_steps = actions.shape[0]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss_float = 0.0
        n_updates = 0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy_loss = 0.0

        n_minibatches = (n_steps + self.batch_size - 1) // self.batch_size
        total_updates = self.n_epochs * n_minibatches
        pbar_ctx = (
            tqdm(
                total=total_updates,
                desc="PPO",
                unit="upd",
                leave=False,
            )
            if self.show_inner_progress
            else nullcontext(_NoOpProgress())
        )
        with pbar_ctx as pbar:
            for _ in range(self.n_epochs):
                perm = torch.randperm(n_steps, device=device)
                for start in range(0, n_steps, self.batch_size):
                    end = min(start + self.batch_size, n_steps)
                    mb_idx = perm[start:end]

                    mb_state_tensors = [t[mb_idx] for t in state_tensors]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_advantages = advantages[mb_idx]

                    new_logits, new_state_values = self.ppo_model(mb_state_tensors)
                    new_dist = Categorical(logits=new_logits)
                    new_log_probs = new_dist.log_prob(mb_actions).sum(dim=-1)

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (
                        self.value_loss_fn(new_state_values, mb_returns) * self.vf_coef
                    )

                    entropy = new_dist.entropy().sum(dim=-1).mean()
                    entropy_loss = -entropy * self.ent_coef

                    loss = policy_loss + value_loss + entropy_loss

                    optimizer.zero_grad()  # type: ignore[union-attr]
                    self.manual_backward(loss)

                    torch.nn.utils.clip_grad_norm_(
                        self.ppo_model.parameters(), self.max_grad_norm
                    )
                    optimizer.step()  # type: ignore[union-attr]

                    total_loss_float += loss.item()
                    n_updates += 1
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy_loss += entropy_loss.item()
                    pbar.update(1)

        if self.do_log and n_updates > 0:
            self.log(
                "train_loss",
                total_loss_float / n_updates,
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "loss/train_loss",
                total_loss_float / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "policy_loss",
                epoch_policy_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "loss/policy_loss",
                epoch_policy_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "value_loss",
                epoch_value_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "loss/value_loss",
                epoch_value_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "entropy_loss",
                epoch_entropy_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "loss/entropy_loss",
                epoch_entropy_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            for name, value in rollout_reward_breakdown.items():
                self.log(
                    f"reward_components/{name}", value, prog_bar=False, logger=True
                )
                self.log(
                    f"reward/components/{name}", value, prog_bar=False, logger=True
                )
            self.log("env_steps", self.global_step, logger=False, prog_bar=True)

    def _collect_rollout_parallel(
        self,
    ) -> tuple[
        list[Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, float],
    ]:
        """Collect rollouts across multiple env instances.

        This implementation keeps env stepping in Python (single process) but
        batches the policy/value forward pass across environments to reduce
        neural network overhead.
        """
        if self.num_rollout_envs < 1:
            raise ValueError("num_rollout_envs must be >= 1")
        if self.num_rollout_envs == 1:
            raise ValueError("_collect_rollout_parallel called with num_rollout_envs=1")
        if self.n_steps % self.num_rollout_envs != 0:
            raise ValueError(
                "n_steps must be divisible by num_rollout_envs "
                f"({self.n_steps} % {self.num_rollout_envs} != 0)"
            )

        n_envs = self.num_rollout_envs
        t_steps = self.n_steps // n_envs
        device = self.ppo_model.device

        envs: list[WargameEnv] = [
            WargameEnv(self.env.config, renderer=None) for _ in range(n_envs)
        ]
        obs_list: list[Any] = []
        try:
            for env_idx, env in enumerate(envs):
                obs, _ = env.reset(seed=env_idx)
                obs_list.append(obs)

            # Store state_tensors for PPO updates in flattened (T * n_envs) form.
            # The returned order matches a row-major flatten of the 2D rollout arrays.
            state_tensors_per_feature: list[list[Tensor]] = [[] for _ in range(5)]
            n_models = self.env.config.number_of_wargame_models

            actions_2d_np = np.zeros((t_steps, n_envs, n_models), dtype=np.int64)
            rewards_2d_np = np.zeros((t_steps, n_envs), dtype=np.float32)
            dones_2d_np = np.zeros((t_steps, n_envs), dtype=np.float32)
            old_log_probs_2d_np = np.zeros((t_steps, n_envs), dtype=np.float32)
            values_2d_np = np.zeros((t_steps, n_envs), dtype=np.float32)

            pbar_ctx = (
                tqdm(
                    total=self.n_steps,
                    desc="Rollout",
                    unit="step",
                    leave=False,
                )
                if self.show_inner_progress
                else nullcontext(_NoOpProgress())
            )
            breakdown_sums: dict[str, float] = {}
            total_steps = 0
            with pbar_ctx as pbar:
                for t in range(t_steps):
                    state_tensors_batch = observations_to_tensor_batch(
                        obs_list, device=device
                    )
                    for feat_idx, feat_tensor in enumerate(state_tensors_batch):
                        state_tensors_per_feature[feat_idx].append(feat_tensor)

                    with torch.no_grad():
                        logits, state_values = self.ppo_model(state_tensors_batch)
                        dist = Categorical(logits=logits)
                        actions = dist.sample()  # (n_envs, n_models)
                        joint_log_prob = dist.log_prob(actions).sum(dim=-1)  # (n_envs,)

                    actions_np = actions.detach().cpu().numpy()
                    values_np = state_values.detach().cpu().numpy()
                    log_probs_np = joint_log_prob.detach().cpu().numpy()

                    actions_2d_np[t] = actions_np
                    values_2d_np[t] = values_np
                    old_log_probs_2d_np[t] = log_probs_np

                    for env_i, env in enumerate(envs):
                        env_action = WargameEnvAction(
                            actions=[int(a) for a in actions_2d_np[t, env_i]]
                        )
                        next_obs, reward, done, _, _ = env.step(env_action)
                        rewards_2d_np[t, env_i] = float(reward)
                        dones_2d_np[t, env_i] = 1.0 if done else 0.0
                        for key, value in env.last_reward_breakdown.items():
                            breakdown_sums[key] = breakdown_sums.get(key, 0.0) + value
                        total_steps += 1

                        if done:
                            next_obs, _ = env.reset()
                        obs_list[env_i] = next_obs

                    pbar.update(n_envs)

            state_tensors_flat = [
                torch.cat(chunks, dim=0) for chunks in state_tensors_per_feature
            ]
            actions_flat = torch.from_numpy(actions_2d_np.reshape(-1, n_models)).to(
                device=device
            )

            rewards_2d = torch.from_numpy(rewards_2d_np).to(device=device)
            dones_2d = torch.from_numpy(dones_2d_np).to(device=device)
            old_log_probs_2d = torch.from_numpy(old_log_probs_2d_np).to(device=device)
            values_2d = torch.from_numpy(values_2d_np).to(device=device)

            last_state_tensors = observations_to_tensor_batch(obs_list, device=device)
            with torch.no_grad():
                _, last_values = self.ppo_model(last_state_tensors)

            last_values = last_values.detach()
            breakdown_mean = (
                {key: (value / total_steps) for key, value in breakdown_sums.items()}
                if total_steps > 0
                else {}
            )
            return (
                state_tensors_flat,
                actions_flat,
                rewards_2d,
                dones_2d,
                old_log_probs_2d,
                values_2d,
                last_values,
                breakdown_mean,
            )
        finally:
            for env in envs:
                env.close()

    def _collect_experiences(self) -> tuple[list[Experience], dict[str, float]]:
        """Run episodes until n_steps transitions are collected (can span multiple episodes)."""
        rollout: list[Experience] = []
        breakdown_sums: dict[str, float] = {}
        total_steps = 0
        pbar_ctx = (
            tqdm(
                total=self.n_steps,
                desc="Rollout",
                unit="step",
                leave=False,
            )
            if self.show_inner_progress
            else nullcontext(_NoOpProgress())
        )
        with pbar_ctx as pbar:
            while len(rollout) < self.n_steps:
                _reward, _steps, episode_exp = self.agent.run_episode_with_experiences(
                    self.ppo_model,
                    epsilon=1.0,
                    render=False,
                    save_steps=True,
                )
                rollout.extend(episode_exp)
                pbar.update(len(episode_exp))
                episode_steps = len(episode_exp)
                if episode_steps > 0:
                    for key, value in self.agent.last_episode_reward_breakdown.items():
                        breakdown_sums[key] = breakdown_sums.get(key, 0.0) + (
                            value * episode_steps
                        )
                    total_steps += episode_steps
        rollout = rollout[: self.n_steps]
        used_steps = len(rollout)
        if total_steps > 0 and used_steps < total_steps:
            scale = used_steps / total_steps
            for key in breakdown_sums:
                breakdown_sums[key] *= scale
            total_steps = used_steps
        breakdown_mean = (
            {key: (value / total_steps) for key, value in breakdown_sums.items()}
            if total_steps > 0
            else {}
        )
        return rollout, breakdown_mean

    def configure_optimizers(self) -> optim.Optimizer:
        """Initialize optimizer."""
        return self.optimizer

    def train_dataloader(self) -> DataLoader[Tensor]:
        """Return a single-batch loader so Lightning calls training_step once per epoch."""
        return DataLoader(
            dataset=_PPODummyDataset(),
            batch_size=1,
            num_workers=0,
        )

    def _policy_model(self) -> PPOModel:
        return self.ppo_model
