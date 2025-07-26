from pydantic import BaseModel


class WargameConfig(BaseModel):
    env_id: str = "gymnasium_env/Wargame-v0"
    env_make_params: dict = {"size": 50}


class DQNConfig(BaseModel):
    batch_size: int = 64
    lr: float = 1e-2
    gamma: float = 0.99
    replay_size: int = 1000
    epsilon_max: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    sync_rate: int = 5
    n_samples_per_epoch: int = 8 * 1024
    weight_decay: float = 1e-5
    n_episodes: int = 20  # just for metrics


class TrainingConfig(BaseModel):
    max_epochs: int = 150
    val_check_interval: int = 1
