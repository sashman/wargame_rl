from pydantic import BaseModel


class DQNConfig(BaseModel):
    batch_size: int = 64
    lr: float = 2e-3
    gamma: float = 0
    replay_size: int = 5000
    epsilon_max: float = 1.0
    epsilon_min: float = 0.2
    epsilon_decay: float = 0.999
    sync_rate: int = 5
    n_samples_per_epoch: int = 8 * 1024
    weight_decay: float = 1e-5
    n_episodes: int = 20  # just for metrics


class TrainingConfig(BaseModel):
    max_epochs: int = 150
    val_check_interval: int = 1
