from __future__ import annotations

from pydantic import BaseModel


class PPOConfig(BaseModel):
    """Configuration for PPO training."""

    # Training parameters
    batch_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.9
    gae_lambda: float = 0.95
    # Controls bias–variance tradeoff in GAE (Generalized Advantage Estimation).
    # λ = 1 → Monte Carlo (low bias, high variance)
    # λ = 0 → TD(0) (higher bias, low variance)
    eps_clip: float = 0.2
    vf_coef: float = 0.5  # value function coefficient
    # Value loss dominates → reduce this
    # Value underfits → increase this
    ent_coef: float = (
        0.02  # entropy coefficient (increase to explore more) -- default is 0.01
    )
    max_grad_norm: float = 0.5  # Gradient Stabilization (prevent exploding gradients)
    n_epochs: int = 5
    n_steps: int = 2048
    n_episodes: int = 10

    # Network parameters
    hidden_size: int = 128
    num_layers: int = 2

    # Training settings
    log: bool = True


class PPOTrainingConfig(BaseModel):
    max_epochs: int = 500
    val_check_interval: int | float = 1
    record_during_training: bool = True
    record_after_epoch: int = 50
