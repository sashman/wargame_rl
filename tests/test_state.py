import torch

from wargame_rl.wargame.model.dqn.dataset import experience_list_to_batch
from wargame_rl.wargame.model.dqn.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)


def test_observation_to_tensor(experiences):
    states = [experience.state for experience in experiences]
    state_batch = observations_to_tensor_batch(states)
    state_batch_2 = torch.cat([observation_to_tensor(state) for state in states], dim=0)
    assert torch.allclose(state_batch, state_batch_2)


def test_experience_to_batch(experiences):
    experience_list_to_batch(experiences)
