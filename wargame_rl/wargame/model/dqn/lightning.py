from copy import deepcopy

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

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
        self.policy_net: RL_Network = torch.compile(policy_net)  # type: ignore[assignment]
        self.policy_net.train()
        self.to(policy_net.device)
        self.target_net: RL_Network = torch.compile(deepcopy(policy_net))  # type: ignore[assignment]
        self.target_net.eval()

        self.buffer = ReplayBuffer(capacity=self.hparams.replay_size)  # type: ignore
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate()
        self.loss_fn: nn.Module = nn.MSELoss(reduction="mean")
        self.epsilon: float = epsilon_max
        self.optimization_steps = 0

    def populate(self) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        # Just run one episode to start populating the buffer
        for _ in range(200):
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
        output: Tensor = self.policy_net(x)
        return output

    def dqn_mse_loss(self, batch: ExperienceBatch) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss

        """
        batch_states = batch.state_tensors
        batch_actions = batch.actions
        batch_rewards = batch.rewards
        batch_dones = batch.dones
        batch_next_states = batch.new_state_tensors

        batch_size, n_model = batch_actions.shape
        assert batch_rewards.shape == (batch_size,)
        assert batch_dones.shape == (batch_size,)

        # we need to sum over the partial state values to get the total rewards
        index = batch_actions.long().unsqueeze(-1)  # [batch_size, n_model, 1]
        net_output = self.policy_net(batch_states)
        assert net_output.shape[:2] == (batch_size, n_model)
        selected_output = net_output.gather(
            -1, index
        )  # we gather along the last dimension, which is the action dimension
        assert selected_output.shape == (batch_size, n_model, 1)
        state_action_values = selected_output.squeeze(-1).sum(-1)
        assert state_action_values.shape == (batch_size,)

        with torch.no_grad():
            next_q = self.target_net(batch_next_states)
            if batch.next_state_masks.numel() > 0:
                next_q = next_q.masked_fill(~batch.next_state_masks, float("-inf"))
            next_state_values = next_q.max(-1)[0].sum(-1)
            next_state_values[batch_dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            next_state_values * self.hparams.gamma + batch_rewards  # type: ignore
        )

        loss: Tensor = self.loss_fn(state_action_values, expected_state_action_values)
        return loss

    def get_epsilon(self) -> float:
        self.epsilon = max(
            self.epsilon * self.hparams.epsilon_decay,  # type: ignore
            self.hparams.epsilon_min,  # type: ignore
        )
        return self.epsilon

    def training_step(self, batch: ExperienceBatch, nb_batch: int) -> Tensor:
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
        self.log("epsilon", epsilon, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("env_steps", self.global_step, logger=False, prog_bar=True)
        if self.optimization_steps % self.hparams.sync_rate == 0:  # type: ignore
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        optimizer = Adam(
            self.policy_net.parameters(),
            lr=self.hparams.lr,  # type: ignore
            weight_decay=self.hparams.weight_decay,  # type: ignore
        )
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.n_samples_per_epoch)  # type: ignore
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            collate_fn=experience_list_to_batch,
            num_workers=0,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def run_episodes(self, n_episodes: int, epsilon: float = 0.0) -> float:
        """Run evaluation episodes and log metrics.

        Returns the success rate as a fraction in [0, 1].
        """
        steps_s = []
        episode_rewards = []
        episode_successes = []
        self.policy_net.eval()
        with torch.no_grad():
            for _ in range(n_episodes):
                reward, steps = self.agent.run_episode(
                    self.policy_net, epsilon=epsilon, render=False, save_steps=False
                )
                episode_rewards.append(reward)
                steps_s.append(steps)

                if (
                    self.env.phase_manager is not None
                    and self.env.last_step_context is not None
                ):
                    success = self.env.phase_manager.check_success(
                        self.env, self.env.last_step_context
                    )
                    episode_successes.append(success)

        self.mean_episode_reward = sum(episode_rewards) / len(episode_rewards)
        self.log("mean_episode_reward", self.mean_episode_reward, prog_bar=False)
        self.log("mean_episode_steps", sum(steps_s) / len(steps_s), prog_bar=False)
        self.log("max_episode_reward", max(episode_rewards), prog_bar=False)
        self.log("min_episode_reward", min(episode_rewards), prog_bar=False)

        if self.env.phase_manager is not None and episode_successes:
            sr = np.array(episode_successes, dtype=float).mean()
        else:
            sr = float((np.array(steps_s) < self.env.max_turns).mean())

        self.log("success_rate", sr * 100, prog_bar=False)
        self.policy_net.train()
        return float(sr)

    def on_train_epoch_end(self) -> None:
        if self.hparams.log:  # type: ignore
            sr = self.run_episodes(self.hparams.n_episodes)  # type: ignore

            if self.env.phase_manager is not None:
                advanced = self.env.phase_manager.try_advance(sr, self.current_epoch)
                self.log(
                    "reward_phase",
                    float(self.env.phase_manager.current_phase_index),
                    prog_bar=False,
                )
                if advanced:
                    self.log(
                        "phase_advanced_at_epoch",
                        float(self.current_epoch),
                        prog_bar=False,
                    )

        return super().on_train_epoch_end()  # type: ignore
