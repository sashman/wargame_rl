from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observations_to_tensor_batch
from wargame_rl.wargame.model.ppo.agent import Agent

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.ppo import PPOModel


class _PPODummyDataset(Dataset[Tensor]):
    """Single-item dataset so Lightning calls training_step once per epoch."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Tensor:
        return torch.tensor(0.0)


class PPOLightning(LightningModule):
    """PPO Lightning Module for training PPO agents."""

    def __init__(
        self,
        env: WargameEnv,
        policy_net: PPOModel,
        log: bool = True,
        batch_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        n_steps: int = 2048,
        n_episodes: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO Lightning Module.

        Args:
            env: Wargame environment
            policy_net: PPO policy network
            log: Whether to log metrics
            batch_size: Batch size for training
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: Generalized Advantage Estimation lambda
            eps_clip: PPO clipping parameter
            vf_coef: Value function coefficient
            ent_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            n_epochs: Number of epochs for PPO updates
            n_steps: Number of steps to collect before each update
            n_episodes: Number of episodes to run for evaluation
        """
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.env = env
        self.policy_net = policy_net
        self.agent = Agent(self.env)
        self.total_reward = 0
        self.episode_reward = 0
        self.do_log = log
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.n_episodes = n_episodes

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
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
        action_logits, _ = self.policy_net(x)
        return action_logits

    def compute_returns(self, rewards: Tensor, dones: Tensor, values: Tensor) -> Tensor:
        """Compute returns using Generalized Advantage Estimation.

        Args:
            rewards: Rewards for each step
            dones: Done flags for each step
            values: Values for each step

        Returns:
            Computed returns
        """
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = torch.tensor(
                    0.0, device=rewards.device, dtype=rewards.dtype
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

        Uses manual optimization because PPO runs multiple gradient steps
        per batch of collected experience.

        Follows the same observation-to-tensor pattern as the DQN pipeline:
        observations are converted via ``observations_to_tensor_batch`` and
        actions are extracted from ``WargameEnvAction.actions``.

        Args:
            batch: Current batch of data (unused -- episodes are collected inline)
            batch_idx: Batch index
        """
        experiences = self._collect_experiences()

        device = self.policy_net.device
        optimizer = self.optimizers()

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

        _, state_values = self.policy_net(state_tensors)

        returns = self.compute_returns(rewards, dones, state_values).detach()
        advantages = (returns - state_values).detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_float = 0.0
        for _ in range(self.n_epochs):
            new_logits, new_state_values = self.policy_net(state_tensors)
            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.value_loss_fn(new_state_values, returns) * self.vf_coef

            entropy = new_dist.entropy().sum(dim=-1).mean()
            entropy_loss = -entropy * self.ent_coef

            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()  # type: ignore[union-attr]
            self.manual_backward(loss)

            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.max_grad_norm
            )
            optimizer.step()  # type: ignore[union-attr]

            total_loss_float += loss.item()

        if self.do_log:
            self.log("train_loss", total_loss_float / self.n_epochs, prog_bar=True)
            self.log("policy_loss", policy_loss.item(), prog_bar=False)
            self.log("value_loss", value_loss.item(), prog_bar=False)
            self.log("entropy_loss", entropy_loss.item(), prog_bar=False)
            self.log("env_steps", self.global_step, logger=False, prog_bar=True)

    def _collect_experiences(self) -> list:
        """Run episodes until at least one step is collected."""
        experiences: list = []
        while len(experiences) == 0:
            _reward, _steps, experiences = self.agent.run_episode(
                self.policy_net,
                epsilon=0.0,
                render=False,
                save_steps=True,
            )
        return experiences

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

    def run_episodes(self, n_episodes: int, epsilon: float = 0.0) -> None:
        """Run episodes for evaluation.

        Args:
            n_episodes: Number of episodes to run
            epsilon: Exploration rate (0 for deterministic)
        """
        steps_s = []
        episode_rewards = []
        self.policy_net.eval()
        with torch.no_grad():
            for _ in range(n_episodes):
                reward, steps, _ = self.agent.run_episode(
                    self.policy_net, epsilon=epsilon, render=False, save_steps=False
                )
                episode_rewards.append(reward)
                steps_s.append(steps)
        self.mean_episode_reward = sum(episode_rewards) / len(episode_rewards)
        if self.do_log:
            self.log("mean_episode_reward", self.mean_episode_reward, prog_bar=False)
            self.log("mean_episode_steps", sum(steps_s) / len(steps_s), prog_bar=False)
            self.log("max_episode_reward", max(episode_rewards), prog_bar=False)
            self.log("min_episode_reward", min(episode_rewards), prog_bar=False)
            success_rate = torch.tensor(steps_s) < self.env.max_turns
            self.log("success_rate", success_rate.float().mean() * 100, prog_bar=False)
        self.policy_net.train()

    def on_train_epoch_end(self) -> None:
        """Run after each training epoch."""
        if self.do_log:
            self.run_episodes(self.n_episodes)
        super().on_train_epoch_end()
