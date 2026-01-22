import numpy as np
import torch

from wargame_rl.wargame.envs.types import WargameEnvConfig, WargameEnvObservation
from wargame_rl.wargame.model.dqn.dataset import experience_list_to_batch
from wargame_rl.wargame.model.dqn.observation import (
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.types import Experience


def test_observation_to_tensor(experiences: list[Experience]) -> None:
    states: list[WargameEnvObservation] = [
        experience.state for experience in experiences
    ]
    batch_size = len(states)
    state = states[0]
    n_objectives = state.n_objectives
    n_wargame_models = state.n_wargame_models

    dim_location = 2
    dim_distances = dim_location * n_objectives
    dim_model = dim_location + dim_distances

    # Test batch conversion
    state_batch = observations_to_tensor_batch(states)
    state_turn, state_objectives, state_wargame_models = state_batch

    assert state_turn.shape == (batch_size, 1)
    assert state_objectives.shape == (batch_size, n_objectives, dim_location)
    assert state_wargame_models.shape == (batch_size, n_wargame_models, dim_model)

    # Test individual conversion and compare with batch
    individual_tensors = [observation_to_tensor(state) for state in states]
    state_turn_2 = torch.cat(
        [tensor[0].unsqueeze(0) for tensor in individual_tensors], dim=0
    )
    state_objectives_2 = torch.cat(
        [tensor[1].unsqueeze(0) for tensor in individual_tensors], dim=0
    )
    state_wargame_models_2 = torch.cat(
        [tensor[2].unsqueeze(0) for tensor in individual_tensors], dim=0
    )

    assert torch.allclose(state_turn, state_turn_2)
    assert torch.allclose(state_objectives, state_objectives_2)
    assert torch.allclose(state_wargame_models, state_wargame_models_2)


def test_experience_to_batch(experiences: list[Experience]) -> None:
    wargame_config: WargameEnvConfig = WargameEnvConfig()
    batch_size = len(experiences)
    # get one state
    state = experiences[0].state
    n_objectives = state.n_objectives

    dim_location = 2
    dim_distances = dim_location * n_objectives
    dim_model = dim_location + dim_distances

    n_wargame_models = state.n_wargame_models
    size_objectives = state.size_objectives
    size_wargame_models = state.size_wargame_models
    state_size = state.size
    assert wargame_config.number_of_wargame_models == n_wargame_models
    assert wargame_config.number_of_objectives == n_objectives
    assert state_size == sum(size_objectives) + sum(size_wargame_models) + 1

    np.testing.assert_array_equal(
        size_objectives, np.array([dim_location] * n_objectives)
    )
    np.testing.assert_array_equal(
        size_wargame_models, np.array([dim_model] * n_wargame_models)
    )

    batch = experience_list_to_batch(experiences)
    assert batch.actions.shape == (batch_size, n_wargame_models)
    assert batch.rewards.shape == (batch_size,)
    assert batch.dones.shape == (batch_size,)
    state_turn, state_objectives, state_wargame_models = batch.state_tensors
    new_state_turn, new_state_objectives, new_state_wargame_models = (
        batch.new_state_tensors
    )
    assert state_turn.shape == (batch_size, 1)
    assert state_objectives.shape == (batch_size, n_objectives, dim_location)
    assert state_wargame_models.shape == (batch_size, n_wargame_models, dim_model)
    assert new_state_turn.shape == (batch_size, 1)
    assert new_state_objectives.shape == (batch_size, n_objectives, dim_location)
    assert new_state_wargame_models.shape == (batch_size, n_wargame_models, dim_model)
