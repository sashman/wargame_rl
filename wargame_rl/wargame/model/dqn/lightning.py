from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

import wandb
from wargame_rl.plotting.training import compute_values_function, plot_policy_on_grid
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.dataset import RLDataset, experience_list_to_batch
from wargame_rl.wargame.model.dqn.dqn import RL_Network
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.types import ExperienceBatch


class DQNLightning(LightningModule):
    def __init__(
        self,
        env: WargameEnv,
        policy_net: RL_Network,
        log: bool = True,
        batch_size: int = 16,
        lr: float = 1e-4,
        gamma: float = 0.99,
        replay_size: int = 10000,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9999,
        sync_rate: int = 32,
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
        self.policy_net = policy_net
        self.policy_net.train()
        self.to(policy_net.device)
        self.target_net = deepcopy(policy_net)
        self.target_net.eval()

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate()
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon_max
        self.optimization_steps = 0

    def populate(self) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        # Just run one episode to start populating the buffer
        self.agent.run_episode(
            self.policy_net, epsilon=1.0, render=False, save_steps=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        output = self.policy_net(x)
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

        # we need to sum over the partial state values to get the total rewards

        state_action_values = (
            self.policy_net(batch_states)
            .gather(1, batch_actions.long().unsqueeze(-1))
            .squeeze(-1)
            .sum(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(batch_next_states).max(-1)[0].sum(-1)
            next_state_values[batch_dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            next_state_values * self.hparams.gamma + batch_rewards
        )

        return self.loss_fn(state_action_values, expected_state_action_values)

    def get_epsilon(self) -> float:
        self.epsilon = max(
            self.epsilon * self.hparams.epsilon_decay, self.hparams.epsilon_min
        )
        return self.epsilon

    def training_step(self, batch: ExperienceBatch, nb_batch) -> Tensor:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss based on
        the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """
        self.optimization_steps += 1
        # run one episode
        epsilon = self.get_epsilon()
        reward, n_steps = self.agent.run_episode(
            self.policy_net, epsilon=epsilon, render=False, save_steps=True
        )
        mean_reward = reward / n_steps
        # calculates training loss
        loss = self.dqn_mse_loss(batch)
        self.log("optimization_step", self.optimization_steps, prog_bar=False)
        self.log("n_steps", n_steps, prog_bar=False)
        self.log("reward", reward, prog_bar=False)
        self.log("mean_reward", mean_reward, prog_bar=True)
        self.log("epsilon", epsilon, prog_bar=False)
        self.log("train_loss", loss, prog_bar=True)
        self.log("env_steps", self.global_step, logger=False, prog_bar=True)
        if self.optimization_steps % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        optimizer = Adam(
            self.policy_net.parameters(),
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
            num_workers=0,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def run_episodes(self, n_episodes: int, epsilon: float = 0.0) -> None:
        steps_s = []
        episode_rewards = []
        self.policy_net.eval()
        with torch.no_grad():
            for _ in range(n_episodes):
                reward, steps = self.agent.run_episode(
                    self.policy_net, epsilon=epsilon, render=False, save_steps=False
                )
                episode_rewards.append(reward)
                steps_s.append(steps)
        self.mean_episode_reward = sum(episode_rewards) / len(episode_rewards)
        self.log("mean_episode_reward", self.mean_episode_reward, prog_bar=False)
        self.log("mean_episode_steps", sum(steps_s) / len(steps_s), prog_bar=False)
        self.log("max_episode_reward", max(episode_rewards), prog_bar=False)
        self.log("min_episode_reward", min(episode_rewards), prog_bar=False)
        success_rate = np.array(steps_s) < self.env.max_turns
        self.log("success_rate", success_rate.mean() * 100, prog_bar=False)
        self.policy_net.train()

    def on_train_epoch_end(self) -> None:
        if self.hparams.log:
            self.run_episodes(self.hparams.n_episodes)
            observation, _ = self.env.reset()
            values_function = compute_values_function(
                observation, self.env.size, self.policy_net
            )
            fig = plot_policy_on_grid(values_function, observation)
            wandb.log({"Value function": fig})  # type: ignore
            plt.close(fig)
        return super().on_train_epoch_end()
