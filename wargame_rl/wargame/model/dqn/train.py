import yaml
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.lightning import DQNLightning
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import CSVLogger
import gymnasium as gym


if __name__ == "__main__":
    with open("hyperparameters.yml", "r") as f:
        hyperparameters = yaml.safe_load(f)

    env_id = hyperparameters["wargame"]["env_id"]
    env_make_params = hyperparameters["wargame"]["env_make_params"]

    env = gym.make(env_id, render_mode=None, **env_make_params)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = DQN(obs_size, n_actions)

    model = DQNLightning(env=env, net=net)

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=150,
        val_check_interval=50,
        logger=CSVLogger(save_dir="logs/"),
    )

    trainer.fit(model)
