import torch
from pytorch_lightning import Trainer

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import MovementPhaseActions, WargameEnv
from wargame_rl.wargame.model.dqn.dataset import experience_list_to_batch
from wargame_rl.wargame.model.dqn.dqn import RL_Network
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from wargame_rl.wargame.types import Experience


def test_dqn_forward(
    env: WargameEnv, experiences: list[Experience], dqn_net: RL_Network, n_steps: int
) -> None:
    wargame_config = WargameEnvConfig()
    n_wargame_models = wargame_config.number_of_wargame_models
    n_actions = len(MovementPhaseActions)
    batch = experience_list_to_batch(experiences)
    next_q_values = dqn_net.forward(batch.state_tensors)
    assert next_q_values.shape == (n_steps, n_wargame_models, n_actions)
    assert next_q_values.dtype == torch.float32


def test_dqn_loss(
    env: WargameEnv, dqn_net: RL_Network, replay_buffer: ReplayBuffer
) -> None:
    # set the seed
    torch.manual_seed(42)
    model = DQNLightning(env=env, policy_net=dqn_net)
    batch = experience_list_to_batch(replay_buffer.sample_batch(3))
    loss_initial = model.dqn_mse_loss(batch)
    assert loss_initial.shape == ()
    assert loss_initial.dtype == torch.float32

    optimizer = model.configure_optimizers()
    for _ in range(3):
        optimizer.zero_grad()
        model.dqn_mse_loss(batch).backward()
        optimizer.step()

    loss_final = model.dqn_mse_loss(batch)
    assert loss_final.shape == ()
    assert loss_final.dtype == torch.float32
    assert loss_final < loss_initial

    loss_training = model.training_step(batch, 0)
    for _ in range(10):
        optimizer.zero_grad()
        model.training_step(batch, 0).backward()
        optimizer.step()

    loss_training_final = model.training_step(batch, 0)
    assert loss_training_final.shape == ()
    assert loss_training_final.dtype == torch.float32
    assert loss_training_final < loss_training


def test_dataloaders(env: WargameEnv, dqn_net: RL_Network) -> None:
    batch_size = 5
    observation, _ = env.reset()
    observation.size

    model = DQNLightning(
        env=env, policy_net=dqn_net, batch_size=batch_size, n_samples_per_epoch=35
    )
    dataloader = model.train_dataloader()
    assert len(dataloader) == 35 // 5
    batch = next(iter(dataloader))
    wargame_config = WargameEnvConfig()
    n_wargame_models = wargame_config.number_of_wargame_models
    n_objectives = observation.n_objectives
    dim_location = 2
    dim_distances = dim_location * n_objectives
    max_groups = observation.wargame_models[0].max_groups
    dim_model = (
        dim_location + dim_distances + max_groups + 1
    )  # group_id one-hot + same-group closest dist

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


def test_dqn_training(env: WargameEnv, dqn_net: RL_Network) -> None:
    model = DQNLightning(env=env, policy_net=dqn_net, log=False)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=2,
        val_check_interval=1,
        logger=None,
    )

    trainer.fit(model)
