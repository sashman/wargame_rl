import os

from pydantic_yaml import to_yaml_str
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from wargame_rl.wargame.envs.types import WargameEnvConfig


class EnvConfigCallback(Callback):
    def __init__(self, name: str, env_config: WargameEnvConfig):
        self.env_config = env_config
        self.name = name

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        filepath = f"./checkpoints/{self.name}/env_config.yaml"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(to_yaml_str(self.env_config))
