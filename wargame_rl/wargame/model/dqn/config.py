from dataclasses import field

from pydantic import BaseModel


class WargameConfig(BaseModel):
    env_id: str = "gymnasium_env/Wargame-v0"
    env_make_params: dict = field(default_factory=dict)


class DQNConfig(BaseModel):
    batch_size: int = 16
    lr: float = 1e-3
    gamma: float = 0.99
    sync_rate: int = 10
    replay_size: int = 1000
    warm_start_size: int = 1000
    eps_last_frame: int = 5000
    eps_start: float = 1.0
    eps_end: float = 0.1
    episode_length: int = 1024
    warm_start_steps: int = 1000
    weight_decay: float = 1e-5


class TrainingConfig(BaseModel):
    max_epochs: int = 150
    val_check_interval: int = 50
