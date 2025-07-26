import torch

from wargame_rl.wargame.model.dqn.dataset import experience_list_to_batch
from wargame_rl.wargame.model.dqn.state import (
    state_to_tensor_v1,
    state_to_tensor_v1_batch,
)


def test_state_to_tensor_v(experiences):
    states = [experience.state for experience in experiences]
    state_batch = state_to_tensor_v1_batch(states)
    state_batch_2 = torch.cat([state_to_tensor_v1(state) for state in states], dim=0)
    assert torch.allclose(state_batch, state_batch_2)


def test_experience_to_batch(experiences):
    experience_list_to_batch(experiences)
