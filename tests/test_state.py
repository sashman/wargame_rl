import torch

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.dqn.dataset import experience_list_to_batch
from wargame_rl.wargame.model.dqn.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)


def test_observation_to_tensor(experiences):
    states = [experience.state for experience in experiences]
    state_size = states[0].size
    batch_size = len(states)
    state_batch = observations_to_tensor_batch(states)
    assert state_batch.shape == (batch_size, state_size)
    state_batch_2 = torch.cat([observation_to_tensor(state) for state in states], dim=0)
    assert torch.allclose(state_batch, state_batch_2)
    assert state_batch.shape == (batch_size, state_size)


def test_experience_to_batch(experiences):
    wargame_config = WargameEnvConfig()
    n_wargame_models = wargame_config.number_of_wargame_models
    state_size = experiences[0].state.size
    batch_size = len(experiences)
    batch = experience_list_to_batch(experiences)
    assert batch.states.shape == (batch_size, state_size)
    assert batch.actions.shape == (batch_size, n_wargame_models)
    assert batch.rewards.shape == (batch_size,)
    assert batch.dones.shape == (batch_size,)
    assert batch.new_states.shape == (batch_size, state_size)
