from __future__ import annotations

from pydantic import BaseModel


class PPOConfig(BaseModel):
    """Configuration for PPO training."""

    # Training parameters
    batch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    n_steps: int = 2048
    n_episodes: int = 10

    # Network parameters
    hidden_size: int = 128
    num_layers: int = 2

    # Training settings
    log: bool = True
    sync_rate: int = (
        1000  # How often to sync networks (not used in PPO but kept for consistency)
    )
    replay_size: int = 10000  # Not used in PPO but kept for consistency

    class Config:
        """Pydantic configuration."""

        # Allow extra fields for backward compatibility
        extra = "allow"
