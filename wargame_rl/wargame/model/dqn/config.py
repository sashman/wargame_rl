from enum import StrEnum

from pydantic import BaseModel


class NetworkType(StrEnum):
    TRANSFORMER = "transformer"
    MLP = "mlp"


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


class TransformerConfig(BaseModel):
    n_layers: int = 4  # number of layers in the transformer
    n_heads: int = 4  # number of attention heads
    embedding_size: int = 128  # size of the embedding vector
    dropout: float = 0.0  # dropout rate -> 0.0 means no dropout
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal: bool = False  # We don't want causal attention for the WarTransformer
