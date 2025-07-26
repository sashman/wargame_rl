from pydantic import BaseModel


class WargameConfig(BaseModel):
    env_id: str = "gymnasium_env/Wargame-v0"
    env_make_params: dict = {"size": 20}


class DQNConfig(BaseModel):
    batch_size: int = 4
    lr: float = 1e-4
    gamma: float = 0.99
    replay_size: int = 10000
    eps_last_epoch: int = 20
    eps_start: float = 1.0
    eps_end: float = 0.05
    n_samples_per_epoch: int = 1024
    weight_decay: float = 1e-4
    n_episodes: int = 10


class TrainingConfig(BaseModel):
    max_epochs: int = 150
    val_check_interval: int = 1
