from __future__ import annotations

from pydantic import BaseModel, Field


class PPOConfig(BaseModel):
    """Configuration for PPO training."""

    # Training parameters
    batch_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.9
    gae_lambda: float = 0.95  # 0.9 is quite low, 0.95 - 0.99 is common, above 0.99 is very future oriented
    # Controls bias–variance tradeoff in GAE (Generalized Advantage Estimation).
    # λ = 1 → Monte Carlo (low bias, high variance)
    # λ = 0 → TD(0) (higher bias, low variance)
    eps_clip: float = 0.2
    vf_coef: float = (
        0.3  # value function coefficient (usually between 0 and 2), strictly positive
    )
    # Value loss dominates → reduce this
    # Value underfits → increase this
    ent_coef: float = (
        0.03  # entropy coefficient (increase to explore more) -- default is 0.01
    )
    max_grad_norm: float = 0.5  # Gradient Stabilization (prevent exploding gradients)
    n_epochs: int = 5
    n_steps: int = 2048
    # Parallel rollout collection: number of independent env instances.
    # When set to 1, rollout collection is identical to the existing code.
    # When set to <= 0, an automatic hardware-based selection is used.
    num_rollout_envs: int = 0
    n_episodes: int = 10

    # Network parameters
    hidden_size: int = 128
    num_layers: int = 2
    # When True, PPO uses a shared Transformer trunk for actor and critic.
    # Policy and value keep separate output heads.
    share_transformer: bool = False

    # Training settings
    log: bool = True
    show_inner_progress: bool = False  # Rollout and PPO minibatch tqdm bars


class SelfPlayConfig(BaseModel):
    """Configuration for PPO self-play and Elo evaluation."""

    enabled: bool = True
    activate_on_final_reward_phase: bool = True
    update_epochs: int = Field(default=3, ge=1)
    self_play_epochs: int = Field(default=1, ge=1)
    snapshot_pool_size: int = Field(default=5, ge=1)
    eval_episodes: int = Field(default=1, ge=1)
    elo_initial_rating: float = 1000.0
    elo_k_factor: float = 32.0
    scripted_opponents: list[str] = Field(
        default_factory=lambda: ["random", "scripted_advance_to_objective"]
    )


class PPOTrainingConfig(BaseModel):
    max_epochs: int = 500
    val_check_interval: int | float = 1
    record_during_training: bool = True
    record_after_epoch: int = 50
    record_every_n_epochs: int = 20
    self_play: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
