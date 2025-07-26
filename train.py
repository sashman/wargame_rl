import gymnasium as gym
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

    net = DQN.from_env(env)
    model = DQNLightning(env=env, net=net, **dqn_config.model_dump())

    config = {
        "wargame": wargame_config.model_dump(),
        "dqn": dqn_config.model_dump(),
        "training": training_config.model_dump(),
    }

    with init_wandb(config=config) as run:
        callbacks = get_checkpoint_callback(run.name)
        logger = get_logger(run)
        trainer = Trainer(
            accelerator="auto",
            max_epochs=training_config.max_epochs,
            val_check_interval=training_config.val_check_interval,
            logger=logger,
            callbacks=callbacks,
        )

        trainer.fit(model)
