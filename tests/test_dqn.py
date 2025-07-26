import torch
from pytorch_lightning import Trainer

from wargame_rl.wargame.model.dqn.dataset import experience_list_to_batch
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from wargame_rl.wargame.model.dqn.state import state_to_tensor, state_to_tensor_batch
from wargame_rl.wargame.types import Experience


def test_state_to_tensor(experiences):
    states = [experience.state for experience in experiences]
    state_batch = state_to_tensor_batch(states)
    state_batch_2 = torch.cat([state_to_tensor(state) for state in states], dim=0)
    assert torch.allclose(state_batch, state_batch_2)


def test_dqn_forward(env, experiences: list[Experience], dqn_net: DQN, n_steps: int):
    batch = experience_list_to_batch(experiences)
    n_actions = env.action_space.n
    next_q_values = dqn_net.forward(batch.states)
    assert next_q_values.shape == (n_steps, n_actions)
    assert next_q_values.dtype == torch.float32


def test_dqn_loss(env, dqn_net: DQN, replay_buffer: ReplayBuffer):
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


def test_dataloaders(env, dqn_net: DQN):
    model = DQNLightning(
        env=env, policy_net=dqn_net, batch_size=5, n_samples_per_epoch=35
    )
    dataloader = model.train_dataloader()
    assert len(dataloader) == 35 // 5
    batch = next(iter(dataloader))
    assert batch.states.shape == (5, 4)
    assert batch.actions.shape == (5,)
    assert batch.rewards.shape == (5,)
    assert batch.dones.shape == (5,)
    assert batch.new_states.shape == (5, 4)


def test_dqn_training(env, dqn_net: DQN, replay_buffer: ReplayBuffer):
    model = DQNLightning(env=env, policy_net=dqn_net, log=False)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=2,
        val_check_interval=50,
        logger=None,
    )

    trainer.fit(model)
