import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
from pytorch_lightning import Trainer

from wargame_rl.wargame.model.dqn.callback import get_checkpoint_callback
from wargame_rl.wargame.model.dqn.config import DQNConfig, TrainingConfig, WargameConfig
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from wargame_rl.wargame.model.dqn.wandb import get_logger, init_wandb

if __name__ == "__main__":
    wargame_config = WargameConfig()
    dqn_config = DQNConfig()
    training_config = TrainingConfig()

    assert wargame_config.env_id == "gymnasium_env/Wargame-v0"
    env = gym.make(
        id=wargame_config.env_id, render_mode=None, **wargame_config.env_make_params
    )
    print(env.observation_space)
    obs_size = flatten_space(env.observation_space).shape[0]
    n_actions = env.action_space.n
    net = DQN(obs_size, n_actions)

    model = DQNLightning(env=env, net=net, **dqn_config.model_dump())

    config = {
        "wargame": wargame_config.model_dump(),
        "dqn": dqn_config.model_dump(),
        "training": training_config.model_dump(),
    }

    callbacks = get_checkpoint_callback()

    with init_wandb(config=config) as run:
        logger = get_logger(run)
        trainer = Trainer(
            accelerator="auto",
            max_epochs=training_config.max_epochs,
            val_check_interval=training_config.val_check_interval,
            logger=logger,
            callbacks=callbacks,
        )

        trainer.fit(model)
