from __future__ import annotations

from pydantic import BaseModel


class PPOConfig(BaseModel):
    """Configuration for PPO training."""

    # Training parameters
    batch_size: int = 128
    lr: float = 3e-4
    # 0.9, kept because it was measured -- twice now, against the theory.
    #
    # The theoretical case for 0.99 is real: 0.9^40 = 0.015, so a terminal
    # bonus is worth 0.044 at t=0 and the effective horizon is ~10 steps of a
    # 40-step episode, while squad-assignment decisions pay off 20-35 steps
    # later. A hard 40-step cap makes 1/(1-0.99) = 100 effectively undiscounted.
    #
    # Measured anyway, two 100-epoch arms on 25v25_single_phase.yaml differing
    # in nothing else, scored on 60 fresh held-out seeds:
    #
    #                win    player VP   opp VP   on objectives
    #   gamma 0.99   0.97       121.2     41.1           0.697
    #   gamma 0.90   0.95       147.4     33.6           0.907
    #
    # Win rate is saturated near the 1.00 ceiling and cannot separate them, but
    # VP margin (+114 vs +80) and occupancy both favour 0.90 clearly. Retest if
    # the reward's time structure changes -- the theoretical argument does not
    # go away, it is simply outweighed here.
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
    # ~25x a conventional setting, which is why a measured 3x change in the
    # nominal value moved total policy entropy by only ~1%.
    #
    # 0.03 is measured, not assumed. Two 30-epoch arms differing in nothing
    # else: at 0.75 movement entropy stayed flat at 4.52 and the agent won 0%
    # of episodes; at 0.03 it fell 4.52 -> 3.95 and win rate reached ~55%.
    ent_coef: float = 0.03
    # Strength of a KL pull back to a frozen REFERENCE policy -- the weights
    # the run warm-started from. 0.0 means no anchor, and at 0.0 no reference
    # network is built at all, so the run is bit-identical to one compiled
    # before the anchor existed.
    #
    # It exists for a measured failure. PPO from a behaviour clone destroys it
    # (docs/melee-teaching-goal.md 47), and the diagnosis is that PPO optimises
    # the policy it *rolls out* -- argmax, undecoded -- while the policy is
    # *scored* through a joint decode. Measured over six paired seeds, PPO
    # gained +19.7 vp in the regime it trains in and lost 34.6 vp of decode
    # headroom doing it. The anchor bounds that drift.
    kl_ref_coef: float = 0.0
    # Target drift, in nats per model, for the adaptive controller. When > 0
    # `kl_ref_coef` stops being a fixed weight and becomes the controller's
    # starting point, doubled or halved each epoch until the measured drift
    # sits near this. That replaces a coefficient nobody can pick a priori
    # with a quantity that has units and a measurement behind it: PPO from a
    # clone drifts to a mean 2.0-2.6 nats per model unpenalised, and destroys
    # 34.6 vp of decode headroom getting there.
    kl_ref_target: float = 0.0
    max_grad_norm: float = 0.5  # Gradient Stabilization (prevent exploding gradients)
    n_epochs: int = 5
    n_steps: int = 2048
    # Parallel rollout collection: number of independent env instances.
    # When set to 1, rollout collection is identical to the existing code.
    # When set to <= 0, an automatic hardware-based selection is used.
    num_rollout_envs: int = 0
    n_episodes: int = 10
    # Evaluation is a per-epoch tax outside `perf/epoch_s`: at the 30 episodes
    # the seeded recipes pass, it is 1200 env steps against the rollout's 2048,
    # i.e. ~22% of a real epoch. 1 keeps the historical every-epoch behaviour.
    # Rejected at construction on curriculum configs -- see `WargameLightningBase`.
    eval_every_n_epochs: int = 1

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
