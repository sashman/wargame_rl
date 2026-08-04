from __future__ import annotations

from pydantic import BaseModel


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
    # Entropy coefficient, applied to the *mean* per-model entropy.
    #
    # It used to multiply the entropy *summed* over models, so its effective
    # magnitude scaled with army size: 0.03 at 25 models was an effective 0.75,
    # which is ~25x a conventional setting and is why a measured 3x change in
    # this value moved total policy entropy by ~1%. 0.75 is set here to hold
    # the pre-change pressure constant at 25 models, so that switching to
    # per-model credit assignment is the only variable under test.
    #
    # 0.03 is the scale-free conventional value and the obvious second arm:
    # `--ent-coef 0.03`. Expect it to matter, since the movement head has been
    # measured sitting at 98.6% of maximum entropy after 945 epochs.
    ent_coef: float = 0.75
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


class PPOTrainingConfig(BaseModel):
    max_epochs: int = 10000
    val_check_interval: int | float = 1
    record_during_training: bool = True
    record_after_epoch: int = 50
    record_every_n_epochs: int = 20
