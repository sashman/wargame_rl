from __future__ import annotations

from pydantic import BaseModel


class PPOConfig(BaseModel):
    """Configuration for PPO training."""

    # Training parameters
    batch_size: int = 128
    lr: float = 3e-4
    # 0.9^40 = 0.015, so under the old default a terminal bonus was worth 0.044
    # at t=0 and the effective horizon was ~10 steps of a 40-step episode --
    # while the decisions that matter here (which objective a squad walks to)
    # pay off 20-35 steps later. With a hard 40-step cap, 1/(1-0.99) = 100 is
    # effectively undiscounted, the standard choice for an episodic task.
    #
    # The earlier "0.99 is worse" finding is retracted as unmeasured: it was
    # taken under a training loop that never applied the reward being tuned.
    # Re-tested as the first screen arm.
    gamma: float = 0.99
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
    # ~25x a conventional setting, which is why a measured 3x change in the
    # nominal value moved total policy entropy by only ~1%.
    #
    # 0.03 is measured, not assumed. Two 30-epoch arms differing in nothing
    # else: at 0.75 movement entropy stayed flat at 4.52 and the agent won 0%
    # of episodes; at 0.03 it fell 4.52 -> 3.95 and win rate reached ~55%.
    ent_coef: float = 0.03
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
