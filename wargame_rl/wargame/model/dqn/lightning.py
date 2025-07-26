from copy import deepcopy

import numpy as np
import torch
from gymnasium import Env
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

import wandb
from wargame_rl.plotting.training import compute_policy_on_grid, plot_policy_on_grid
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.dataset import RLDataset, experience_list_to_batch
from wargame_rl.wargame.model.dqn.dqn import RL_Network
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.types import ExperienceBatch


class DQNLightning(LightningModule):
    def __init__(
        self,
        env: Env,
        net: RL_Network,
        log: bool = True,
        batch_size: int = 16,
        lr: float = 1e-4,
        gamma: float = 0.99,
        replay_size: int = 10000,
        eps_last_epoch: int = 20,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        n_samples_per_epoch: int = 1024,
        weight_decay: float = 1e-4,
        n_episodes: int = 10,
    ) -> None:
        """Basic DQN Model.

        Args:
            env: gym environment tag
            batch_size: size of the batches")
            lr: learning rate
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_epoch: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            n_samples_per_epoch: number of samples per epoch

        """
        super().__init__()
        self.save_hyperparameters()

        self.env = env
        self.net = net
        self.to(net.device)
        self.target_net = deepcopy(net)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate()
        self.loss_fn = nn.MSELoss()

    def populate(self) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        self.run_episodes(self.hparams.n_episodes, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: ExperienceBatch) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss

        """
        batch_states = batch.states
        batch_actions = batch.actions
        batch_rewards = batch.rewards
        batch_dones = batch.dones
        batch_next_states = batch.new_states

        state_action_values = (
            self.net(batch_states)
            .gather(1, batch_actions.long().unsqueeze(-1))
            .squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(batch_next_states).max(1)[0]
            next_state_values[batch_dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            next_state_values * self.hparams.gamma + batch_rewards
        )

        return self.loss_fn(state_action_values, expected_state_action_values)

    def get_epsilon(self, epoch: int) -> float:
        return self.hparams.eps_start - (epoch / self.hparams.eps_last_epoch) * (
            self.hparams.eps_start - self.hparams.eps_end
        )

    def training_step(self, batch: ExperienceBatch, nb_batch) -> Tensor:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss based on
        the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        self.log("train_loss", loss, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        optimizer = Adam(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.n_samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=experience_list_to_batch,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def run_episodes(self, n_episodes: int, epsilon: float | None = None) -> None:
        if epsilon is None:
            epsilon = self.get_epsilon(self.current_epoch)
            self.log("epsilon", epsilon, prog_bar=False)
        self.target_net.load_state_dict(self.net.state_dict())

        steps_s = []
        episode_rewards = []
        for _ in range(n_episodes):
            reward, steps = self.agent.run_episode(
                self.net, epsilon=epsilon, render=False
            )
            episode_rewards.append(reward)
            steps_s.append(steps)
        self.mean_episode_reward = sum(episode_rewards) / len(episode_rewards)
        self.log("mean_episode_reward", self.mean_episode_reward, prog_bar=True)
        self.log("mean_episode_steps", sum(steps_s) / len(steps_s), prog_bar=False)
        self.log("max_episode_reward", max(episode_rewards), prog_bar=False)
        self.log("min_episode_reward", min(episode_rewards), prog_bar=False)
        success_rate = np.array(steps_s) < 40
        self.log("success_rate %", success_rate.mean() * 100, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        self.run_episodes(self.hparams.n_episodes)
        box_agent = self.env.observation_space["agent"]
        values_function, target_state = compute_policy_on_grid(box_agent, self.net)
        fig = plot_policy_on_grid(values_function, target_state)
        if self.hparams.log:
            wandb.log({"Value function": fig})  # type: ignore
        return super().on_train_epoch_end()
