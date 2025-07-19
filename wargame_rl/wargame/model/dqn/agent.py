import logging
from logging import getLogger
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.experience_replay import ReplayMemory
import gymnasium as gym
import numpy as np

import wargame_rl
from gymnasium.spaces.utils import flatten_space

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml


from datetime import datetime, timedelta
import itertools

import os

logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

# Deep Q-Learning Agent
class Agent():

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.hidden_dim         = hyperparameters['hidden_dim']            # number of nodes in the hidden layer
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.
        
        self.policy_dqn = None

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        
        loss = []
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        logger.info(f"Loading environment {self.env_id}...")
        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = flatten_space(env.observation_space).shape[0]
        # num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        self.policy_dqn = DQN(num_states, num_actions, hidden_dim=self.hidden_dim).to(device)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, hidden_dim=self.hidden_dim).to(device)
            target_dqn.load_state_dict(self.policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a, weight_decay=1e-4)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999
            minimum_loss = 9999999
        else:
            # Load learned policy
            self.policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            self.policy_dqn.eval()

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():

            
            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            tensor_state = self.to_tensor_state(state) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):
                
                # logger.info(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}")

                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # self.policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        # logger.info(f"State input: {state_input}")
                        
                        action = self.policy_dqn(tensor_state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action. Truncated and info is not used.
                new_state, reward, terminated, truncated, info = env.step(action.item())
                # logger.info(f"Action: {action.item()}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_tensor_state = self.to_tensor_state(new_state)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((tensor_state, action, new_tensor_state, reward, terminated))

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                tensor_state = new_tensor_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_training:
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    current_losses = self.optimize(mini_batch, self.policy_dqn, target_dqn)
                    average_loss = np.mean(current_losses)
                    loss.append(average_loss)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(self.policy_dqn.state_dict())
                        step_count=0

                if minimum_loss > average_loss:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New lower loss {average_loss:0.5f} at episode {episode}, saving model... Current reward: {episode_reward:0.1f}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(self.policy_dqn.state_dict(), self.MODEL_FILE)
                    # best_reward = episode_reward
                    minimum_loss = average_loss

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=3):
                    self.save_graph(rewards_per_episode, epsilon_history, loss, env)
                    last_graph_update_time = current_time
                    

    def flatten_state(self, state):
        agent = state["agent"]
        target = state["target"]
        return [agent[0], agent[1], target[0], target[1]]
    
    def to_tensor_state(self, state) -> torch.Tensor:
        agent = state["agent"]
        target = state["target"]
        flatten_state = [agent[0], agent[1], target[0], target[1]]
        norm = 25.0
        tensor_state = (torch.tensor(flatten_state, dtype=torch.float, device=device) - norm) / norm

        return tensor_state
    
    def compute_policy_on_grid(self, env: WargameEnv) -> tuple[np.ndarray, np.ndarray]:
        box_agent = env.observation_space["agent"]
        x_min, y_min = box_agent.low
        x_max, y_max = box_agent.high

        state, _ = env.reset() 

        target_state = state["target"]

        values_function = np.empty([x_max - x_min + 1, y_max - y_min + 1])
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                agent_state = np.array([x, y])
                state = {"agent": agent_state, "target": target_state}
                tensor_state = self.to_tensor_state(state)
                values_function[x, y] = self.policy_dqn(tensor_state).max()

        return values_function, target_state
    
    def save_graph(self, rewards_per_episode, epsilon_history, loss, env):
        # compute all states of the current policy dqn
        values_function, target_state = self.compute_policy_on_grid(env)

        # Save plots
        fig = plt.figure(1, figsize=(12, 8), dpi=100)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(221) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(222) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        # Plot loss (Y-axis) vs episodes (X-axis)
        plt.subplot(2,2,3)
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.plot(loss)
        plt.yscale('log')

        # Plot policy on grid
        plt.subplot(2,2,4)
        plt.imshow(values_function.T, origin='upper')
        plt.plot(target_state[0], target_state[1], "or")
        plt.colorbar()
        
        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            self.policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases
        return loss.item()  # Return the loss value for logging or monitoring

