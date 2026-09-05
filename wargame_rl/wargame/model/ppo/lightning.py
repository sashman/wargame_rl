from __future__ import annotations

import copy
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.lightning_base import WargameLightningBase
from wargame_rl.wargame.model.common.observation import observations_to_tensor_batch
from wargame_rl.wargame.model.common.self_play import OpponentScheduler, SelfPlayConfig
from wargame_rl.wargame.model.ppo.agent import Agent
from wargame_rl.wargame.rating.pool import Snapshot
from wargame_rl.wargame.types import Experience

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.ppo import PPOModel

# Rollout envs are seeded once from this base, then left to run their own RNG
# stream. Kept well below the evaluation seed space so training and evaluation
# never draw the same layouts.
ROLLOUT_SEED_BASE = 0
# Rollout envs opt in to the start-state augmentation; every evaluation path
# leaves it off. Opt-in is the fail-safe direction -- forgetting it here
# trains the bit-identical control, which is obvious, while forgetting to
# disable it at eval would score a different scenario and look plausible.
_AUGMENT_START = {"augment_start": True}


class _NoOpProgress:
    """No-op progress object when inner progress bars are disabled."""

    def update(self, n: int = 1) -> None:
        pass


class _WargameEnvActionWrapper(gym.ActionWrapper):
    """Convert vector-env Tuple actions into `WargameEnvAction`.

    Gymnasium vector environments expect actions compatible with `action_space`.
    Our `WargameEnv.step()` expects a `WargameEnvAction`, so this wrapper
    bridges the formats for action dispatch.
    """

    def action(self, action: Any) -> WargameEnvAction:  # type: ignore[override]
        if isinstance(action, WargameEnvAction):
            return action
        # Expected shape from vector env: tuple/list/ndarray of length n_models.
        actions_list = [int(x) for x in np.asarray(action).reshape(-1)]
        return WargameEnvAction(actions=actions_list)


class _PPODummyDataset(Dataset[Tensor]):
    """Single-item dataset so Lightning calls training_step once per epoch."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Tensor:
        return torch.tensor(0.0)


# Bounds on the adaptive coefficient. The floor keeps a run that has drifted
# back inside its target from setting the weight to zero and losing the anchor
# entirely; the ceiling stops a policy that cannot reach the target from
# doubling the weight without limit until the return is invisible beside it.
_KL_COEF_MIN = 1e-4
_KL_COEF_MAX = 1e4


def masked_categorical_kl(policy_logits: Tensor, reference_logits: Tensor) -> Tensor:
    """Mean per-row exact `KL(policy || reference)` over masked categoricals.

    The exact divergence rather than a sampled estimate: the action space is
    ~150 wide, so summing the whole row costs nothing and removes a variance
    term the anchor would otherwise have to fight.

    ⚠ The network applies the action mask INSIDE its forward pass, so an
    illegal action arrives as `-inf` in both rows and their difference is
    `nan`. The probability there is exactly 0, so the contribution should be
    0 -- but `0 * nan` is `nan`, which would poison the whole loss. Both
    policies are evaluated on the same state and therefore under the same
    mask, so finiteness is a property of the ENTRY: zeroing the difference
    where either row is infinite is exact, not a guard against error.
    """
    policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    both_finite = torch.isfinite(policy_log_probs) & torch.isfinite(reference_log_probs)
    difference = torch.where(
        both_finite,
        policy_log_probs - reference_log_probs,
        torch.zeros_like(policy_log_probs),
    )
    return (policy_log_probs.exp() * difference).sum(dim=-1).mean()


class PPOLightning(WargameLightningBase):
    """PPO Lightning Module for training PPO agents."""

    def _largest_divisor_at_most(self, n: int, max_value: int) -> int:
        """Return largest divisor of `n` that is <= `max_value`."""
        candidate = min(n, max_value)
        for v in range(candidate, 0, -1):
            if n % v == 0:
                return v
        return 1

    def _cuda_appears_usable(self) -> bool:
        """Best-effort check that CUDA kernels can execute.

        This is defensive against environments where CUDA is present but
        incompatible with the installed GPU/driver.
        """
        try:
            if not torch.cuda.is_available():
                return False
            if self.ppo_model.device.type != "cuda":
                return False
            # Tiny kernel to force real CUDA initialization / execution.
            _ = torch.empty((1,), device=self.ppo_model.device).sum().item()
            return True
        except Exception:
            return False

    def _auto_detect_num_rollout_envs(self) -> int:
        """Pick a heuristic `num_rollout_envs` based on CPU/GPU availability."""
        # Respect CPU affinity / cgroup limits when possible.
        cpu_count = 1
        try:
            if hasattr(os, "sched_getaffinity"):
                cpu_count = len(os.sched_getaffinity(0))  # type: ignore[arg-type]
            else:
                cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = os.cpu_count() or 1

        # If the model runs on a usable GPU, we can typically afford more envs
        # to amortize inference overhead. On CPU, keep it conservative.
        if self._cuda_appears_usable():
            max_envs = 8
        else:
            max_envs = 4

        heuristic = max(1, min(max_envs, cpu_count))
        # Enforce `n_steps` divisibility so rollout collection never errors.
        return self._largest_divisor_at_most(self.n_steps, heuristic)

    def __init__(
        self,
        env: WargameEnv,
        ppo_model: PPOModel,
        log: bool = True,
        batch_size: int = 1024,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        kl_ref_coef: float = 0.0,
        kl_ref_target: float = 0.0,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        n_steps: int = 2048,
        num_rollout_envs: int = 0,
        n_episodes: int = 10,
        eval_every_n_epochs: int = 1,
        show_inner_progress: bool = True,
        self_play: SelfPlayConfig | None = None,
        snapshot_dir: Path | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO Lightning Module.

        Args:
            env: Wargame environment
            ppo_model: PPO policy-value model
            log: Whether to log metrics
            batch_size: Minibatch size for PPO updates (samples per gradient step)
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: Generalized Advantage Estimation lambda
            eps_clip: PPO clipping parameter
            vf_coef: Value function coefficient
            ent_coef: Entropy coefficient
            kl_ref_coef: Weight on KL(policy || frozen reference). 0.0 builds
                no reference network and adds no term.
            kl_ref_target: Drift to hold, in nats per model. 0.0 keeps
                `kl_ref_coef` fixed.
            max_grad_norm: Maximum gradient norm
            n_epochs: Number of epochs for PPO updates
            n_steps: Number of steps to collect before each update
            num_rollout_envs: Number of parallel env instances for rollout
                collection (must be >= 1). When set to 1, rollout collection is
                unchanged.
            n_episodes: Number of episodes to run for evaluation
            eval_every_n_epochs: Evaluate every Nth epoch instead of every one.
                Single-phase configs only; raises on a curriculum config.
            show_inner_progress: Whether to show tqdm for rollout and PPO minibatch updates
            self_play: Pool-and-PFSP settings. `None` or disabled means the
                opponent is whatever the env config names, and **no scheduler
                is constructed at all** -- so no stream is drawn from and the
                run is bit-identical to one on a config with no self-play
                block. Deliberately not opt-out; see
                `model/common/self_play.py`.
            snapshot_dir: Where frozen opponents are written. Required when
                self-play is enabled.
            seed: The run seed, used only to offset the opponent stream into
                its own band so it cannot perturb layouts or dice.
        """
        super().__init__(
            env=env,
            agent=Agent(env),
            do_log=log,
            n_episodes=n_episodes,
            eval_every_n_epochs=eval_every_n_epochs,
        )
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.show_inner_progress = show_inner_progress
        self.ppo_model = ppo_model
        self.total_reward = 0
        self.episode_reward = 0
        self.batch_size = batch_size
        self.n_steps = n_steps
        if num_rollout_envs <= 0:
            self.num_rollout_envs = self._auto_detect_num_rollout_envs()
        else:
            self.num_rollout_envs = num_rollout_envs
        # Only the parallel collector passes `augment_start`; the serial path
        # goes through `BaseAgent.reset()`, which takes no options and is shared
        # with evaluation, so plumbing it there would risk the leak this feature
        # is built to prevent. Refuse the combination instead of training the
        # control while the config and the Wandb record both claim otherwise --
        # that failure is silent, and it reads as "the augmentation did nothing"
        # rather than "the augmentation never ran".
        if (
            self.num_rollout_envs == 1
            and self.env.config.start_on_objective_probability > 0.0
        ):
            raise ValueError(
                "start_on_objective_probability="
                f"{self.env.config.start_on_objective_probability} needs the "
                "parallel rollout collector, but num_rollout_envs resolved to 1 "
                "(requested "
                f"{self.hparams.get('num_rollout_envs')}). The serial path cannot "
                "apply the start-state augmentation, so this run would silently "
                "train the un-augmented control. Set num_rollout_envs > 1."
            )
        # Constructed only when self-play is enabled. The `None` is what makes
        # "off changes nothing" checkable by reading rather than by measuring:
        # there is no object to draw a random number from.
        self._opponent_scheduler: OpponentScheduler | None = None
        # This epoch's seating, by rollout-env index, and the finished episodes
        # attributed to it. Both stay empty without a scheduler, so a control
        # run allocates nothing and records nothing.
        self._drawn_opponents: list[Snapshot] | None = None
        self._rollout_margins: dict[str, list[float]] = {}
        if self_play is not None and self_play.enabled:
            if snapshot_dir is None:
                raise ValueError(
                    "self-play needs a snapshot_dir to freeze opponents into"
                )
            self._opponent_scheduler = OpponentScheduler(
                self_play, snapshot_dir, seed=seed
            )
        # Built once, on first use, and kept for the whole run. See
        # _ensure_rollout_envs for why rebuilding them per step was a bug.
        self._rollout_envs: list[WargameEnv] | None = None
        self._rollout_obs: list[Any] = []
        self._rollout_diagnostics: dict[str, float] = {}
        self._layouts_seen: set[tuple[tuple[int, int], ...]] = set()
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.ppo_model.parameters(),
            lr=lr,
            eps=1e-5,
        )

        # Initialize loss components
        self.value_loss_fn = nn.MSELoss()
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.kl_ref_coef = kl_ref_coef
        self.kl_ref_target = kl_ref_target
        # Attached explicitly after any warm start, never here: anchoring to a
        # random initialisation is meaningless, and the weights this pulls
        # towards are the ones the run started FROM.
        self._kl_reference: PPOModel | None = None
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def attach_kl_reference(self) -> None:
        """Freeze the CURRENT policy as the KL anchor for the rest of the run.

        Call once, after any warm start has been applied and before `fit`. A
        no-op at `kl_ref_coef == 0.0`, which is what makes the anchor off a
        provable no-op rather than a measured one -- nothing is copied, no
        parameter is read, and the loss is assembled by the same expression as
        before the feature existed.
        """
        if self.kl_ref_coef <= 0.0:
            return
        reference = copy.deepcopy(self.ppo_model)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self._kl_reference = reference
        logger.info(f"KL reference attached, coef={self.kl_ref_coef}")

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Give a resumed run somewhere to put its saved KL reference.

        The frozen reference is a real submodule, so it is saved with the rest
        of the state dict -- which is correct, because it IS training state: a
        resumed run has to be held to the same anchor it started under, and
        that anchor is not recoverable from anything else in the checkpoint.

        But `attach_kl_reference` only runs on the warm-start path, and resume
        and warm start are mutually exclusive. So on a resume the attribute
        does not exist yet and Lightning's restore dies with `Unexpected
        key(s) in state_dict: "_kl_reference...."` -- **every anchored run was
        un-resumable**, which is the same class of silent-until-you-need-it
        failure `--resume-ckpt-path` already had once. Attaching an
        empty reference here gives the load somewhere to land; the values it
        carries then overwrite it.
        """
        if self._kl_reference is not None:
            return
        state_dict = checkpoint.get("state_dict", {})
        if not any(key.startswith("_kl_reference.") for key in state_dict):
            return
        reference = copy.deepcopy(self.ppo_model)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self._kl_reference = reference
        logger.info("KL reference slot rebuilt for resume")

    def _adapt_kl_coefficient(self, measured_drift: float) -> None:
        """Move the coefficient toward whatever holds drift at the target.

        Schulman's original PPO KL-penalty rule: halve the weight when the
        policy is staying closer than asked, double it when it is drifting
        further. A no-op at `kl_ref_target == 0.0`, which keeps a fixed
        coefficient available as the simpler thing to reason about.

        The band is deliberately wide (1.5x either side). A tight band makes
        the coefficient oscillate by a factor of four every other epoch, and
        the policy then sees a penalty changing faster than it can respond to
        -- which is a different experiment from the one intended.
        """
        if self.kl_ref_target <= 0.0:
            return
        if measured_drift < self.kl_ref_target / 1.5:
            self.kl_ref_coef = max(self.kl_ref_coef / 2.0, _KL_COEF_MIN)
        elif measured_drift > self.kl_ref_target * 1.5:
            self.kl_ref_coef = min(self.kl_ref_coef * 2.0, _KL_COEF_MAX)

    def _kl_divergence_to_reference(
        self, new_logits: Tensor, state_tensors: list[Tensor]
    ) -> Tensor:
        """Mean per-model `KL(policy || reference)` over the minibatch."""
        assert self._kl_reference is not None
        with torch.no_grad():
            reference_logits, _ = self._kl_reference(state_tensors)
        return masked_categorical_kl(new_logits, reference_logits)

    def forward(self, x: list[Tensor]) -> Tensor:
        """Forward pass through the policy network.

        Args:
            x: List of input tensors (game state, objectives, models).

        Returns:
            Action logits with shape (batch, n_models, n_actions).
        """
        action_logits: Tensor
        action_logits, _ = self.ppo_model(x)
        return action_logits

    def compute_returns(
        self,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
        last_value: Tensor | None = None,
    ) -> Tensor:
        """Compute returns using Generalized Advantage Estimation.

        Args:
            rewards: Rewards for each step
            dones: Done flags for each step
            values: Values for each step

        Returns:
            Computed returns
        """
        # Compute advantages using GAE.
        # Supports rewards/values with shape (T,), (T, num_envs) or
        # (T, num_envs, n_models). `dones` carries one flag per env, so it is
        # broadcast across the trailing model axis when present.
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])

        if dones.dim() < rewards.dim():
            dones = dones.unsqueeze(-1).expand_as(rewards)

        time_steps = rewards.shape[0]
        for t in reversed(range(time_steps)):
            if t == time_steps - 1:
                if last_value is None:
                    next_value = torch.zeros_like(values[t])
                else:
                    next_value = last_value.to(
                        device=rewards.device, dtype=rewards.dtype
                    )
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute returns
        returns = advantages + values
        return returns

    def _elapsed_since(self, start: float) -> float:
        """Seconds since `start`, with CUDA work actually finished.

        Without the synchronise, a GPU timing measures how long it took to
        *queue* the kernels, not to run them — which reads as "the update is
        free" and sends the next optimisation at the wrong target.
        """
        if self.ppo_model.device.type == "cuda":
            torch.cuda.synchronize(self.ppo_model.device)
        return time.perf_counter() - start

    def _log_throughput(
        self,
        rollout_seconds: float,
        update_seconds: float,
        n_steps: int,
        n_updates: int,
    ) -> None:
        """Log where the epoch's wall-clock went.

        Logged every epoch, not just when profiling, so a performance
        regression shows up on the same dashboard as a reward regression. The
        GPU spent five months unusable on one machine without anything in the
        metrics saying so.
        """
        metrics = {
            "perf/rollout_s": rollout_seconds,
            "perf/update_s": update_seconds,
            "perf/epoch_s": rollout_seconds + update_seconds,
        }
        if rollout_seconds > 0.0:
            metrics["perf/env_steps_per_s"] = n_steps / rollout_seconds
        if n_updates > 0:
            metrics["perf/update_ms_per_minibatch"] = update_seconds / n_updates * 1000
        for name, value in metrics.items():
            # on_step=False so each timing is one series. PPO runs exactly one
            # `training_step` per epoch, so the default step/epoch pair would be
            # two identical columns.
            self.log(
                name,
                value,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Carry out a single training step.

        Collects n_steps transitions (across one or more episodes), computes
        GAE returns/advantages, then runs n_epochs of minibatch PPO updates.

        Uses manual optimization because PPO runs multiple gradient steps
        per rollout. Observations are converted via ``observations_to_tensor_batch``;
        actions are extracted from ``WargameEnvAction.actions``.

        Args:
            batch: Unused (rollout is collected inline)
            batch_idx: Batch index
        """
        device = self.ppo_model.device
        optimizer = self.optimizers()
        rollout_start = time.perf_counter()

        rollout_reward_breakdown: dict[str, float] = {}
        if self.num_rollout_envs == 1:
            experiences, rollout_reward_breakdown = self._collect_experiences()

            state_tensors = observations_to_tensor_batch(
                [exp.state for exp in experiences], device=device
            )
            actions = torch.tensor(
                [exp.action.actions for exp in experiences],
                dtype=torch.long,
                device=device,
            )
            # (T, n_models): each model is credited with its own contribution
            # plus the shared global terms, rather than an army-wide mean.
            rewards = torch.stack(
                [
                    exp.per_model_reward.to(device)  # type: ignore[union-attr]
                    for exp in experiences
                ]
            ).float()
            dones = torch.tensor(
                [exp.done for exp in experiences],
                dtype=torch.float32,
                device=device,
            )
            old_log_probs = torch.stack(
                [exp.log_prob for exp in experiences]  # type: ignore[misc]
            ).detach()

            _, state_values = self.ppo_model(state_tensors)

            last_done = bool(experiences[-1].done)
            if last_done:
                last_value = torch.zeros_like(state_values[0])
            else:
                last_state_tensors = observations_to_tensor_batch(
                    [experiences[-1].new_state], device=device
                )
                with torch.no_grad():
                    _, last_state_value = self.ppo_model(last_state_tensors)
                last_value = last_state_value.squeeze(0).detach()

            returns = self.compute_returns(
                rewards,
                dones,
                state_values,
                last_value=last_value,
            ).detach()
            advantages = (returns - state_values).detach()
            n_steps = len(experiences)
        else:
            (
                state_tensors,
                actions,
                rewards_2d,
                dones_2d,
                old_log_probs_2d,
                values_2d,
                last_values,
                rollout_reward_breakdown,
            ) = self._collect_rollout_parallel()

            returns_2d = self.compute_returns(
                rewards_2d,
                dones_2d,
                values_2d,
                last_value=last_values,
            ).detach()
            advantages_2d = (returns_2d - values_2d).detach()

            n_models = actions.shape[-1]
            returns = returns_2d.reshape(-1, n_models)
            advantages = advantages_2d.reshape(-1, n_models)
            old_log_probs = old_log_probs_2d.reshape(-1, n_models).detach()
            n_steps = actions.shape[0]

        rollout_seconds = self._elapsed_since(rollout_start)
        update_start = time.perf_counter()

        # How much of the return the critic actually explains. 0 means it is no
        # better than predicting the mean; negative means worse. `advantages`
        # is `returns - values`, i.e. the residual, and must be read here
        # before it is normalised below.
        return_variance = float(returns.var())
        explained_variance = (
            1.0 - float(advantages.var()) / return_variance
            if return_variance > 0.0
            else 0.0
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss_float = 0.0
        n_updates = 0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy_loss = 0.0
        epoch_kl_ref = 0.0
        epoch_clip_fraction = 0.0
        epoch_approx_kl = 0.0
        epoch_grad_norm = 0.0
        epoch_grad_clipped = 0.0

        n_minibatches = (n_steps + self.batch_size - 1) // self.batch_size
        total_updates = self.n_epochs * n_minibatches
        pbar_ctx = (
            tqdm(
                total=total_updates,
                desc="PPO",
                unit="upd",
                leave=False,
            )
            if self.show_inner_progress
            else nullcontext(_NoOpProgress())
        )
        with pbar_ctx as pbar:
            for _ in range(self.n_epochs):
                perm = torch.randperm(n_steps, device=device)
                for start in range(0, n_steps, self.batch_size):
                    end = min(start + self.batch_size, n_steps)
                    mb_idx = perm[start:end]

                    mb_state_tensors = [t[mb_idx] for t in state_tensors]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_advantages = advantages[mb_idx]

                    new_logits, new_state_values = self.ppo_model(mb_state_tensors)
                    new_dist = Categorical(logits=new_logits)
                    # Per-model log-probs, deliberately not summed. Summing
                    # them made one importance ratio for the whole 25-model
                    # joint action, so `eps_clip=0.2` was breached at 0.0073
                    # nats of change per model and roughly half of every
                    # minibatch was clipped flat.
                    new_log_probs = new_dist.log_prob(mb_actions)

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    with torch.no_grad():
                        # The two numbers that say whether the update is
                        # working at all. Their absence is why a degenerate
                        # objective went undiagnosed across seven runs.
                        log_ratio = new_log_probs - mb_old_log_probs
                        epoch_clip_fraction += float(
                            ((ratio - 1.0).abs() > self.eps_clip).float().mean()
                        )
                        epoch_approx_kl += float(((ratio - 1.0) - log_ratio).mean())

                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (
                        self.value_loss_fn(new_state_values, mb_returns) * self.vf_coef
                    )

                    # Mean over models, not sum: with a sum, `ent_coef` was
                    # effectively multiplied by the model count, which is why
                    # a 3x change in the coefficient moved total entropy ~1%.
                    entropy = new_dist.entropy().mean()
                    entropy_loss = -entropy * self.ent_coef

                    loss = policy_loss + value_loss + entropy_loss

                    if self._kl_reference is not None:
                        kl_to_reference = self._kl_divergence_to_reference(
                            new_logits, mb_state_tensors
                        )
                        loss = loss + self.kl_ref_coef * kl_to_reference
                        epoch_kl_ref += float(kl_to_reference)

                    optimizer.zero_grad()  # type: ignore[union-attr]
                    self.manual_backward(loss)

                    # The return value is the norm *before* clipping, and it is
                    # the missing half of the update diagnostics: `max_grad_norm`
                    # is applied to the joint norm across both networks, so if it
                    # binds on most minibatches then it — not `lr` — is setting
                    # the effective step size, and `vf_coef` acts only by taking
                    # a share of that fixed budget. Discarding it left no way to
                    # tell that apart from a genuinely small gradient.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.ppo_model.parameters(), self.max_grad_norm
                    )
                    optimizer.step()  # type: ignore[union-attr]

                    total_loss_float += loss.item()
                    grad_norm_float = float(grad_norm)
                    epoch_grad_norm += grad_norm_float
                    epoch_grad_clipped += float(grad_norm_float > self.max_grad_norm)
                    n_updates += 1
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy_loss += entropy_loss.item()
                    pbar.update(1)

        update_seconds = self._elapsed_since(update_start)
        if self.do_log:
            self._log_throughput(rollout_seconds, update_seconds, n_steps, n_updates)

        if self.do_log and n_updates > 0:
            self.log(
                "loss/train_loss",
                total_loss_float / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "loss/policy_loss",
                epoch_policy_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "loss/value_loss",
                epoch_value_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            self.log(
                "loss/entropy_loss",
                epoch_entropy_loss / n_updates,
                prog_bar=False,
                logger=True,
                on_epoch=True,
            )
            if self._kl_reference is not None:
                # In nats per model. The anchor is doing nothing if this sits
                # at zero, and is overwhelming the return if it dominates the
                # policy loss -- both are visible only if it is logged.
                measured_drift = epoch_kl_ref / n_updates
                self.log(
                    "loss/kl_to_reference",
                    measured_drift,
                    prog_bar=False,
                    logger=True,
                )
                self._adapt_kl_coefficient(measured_drift)
                self.log(
                    "loss/kl_ref_coef",
                    self.kl_ref_coef,
                    prog_bar=False,
                    logger=True,
                )
            # Update health. clip_fraction above ~0.3 means the trust region is
            # being saturated and much of each minibatch contributes nothing;
            # at the bottom of the band it is not binding at all and the step
            # size has headroom. `grad_clipped_fraction` disambiguates the
            # second case: near 1.0 the gradient is being clipped on nearly
            # every minibatch, so `max_grad_norm` is the effective step size and
            # raising `lr` alone will change little.
            for name, value in (
                ("train/clip_fraction", epoch_clip_fraction / n_updates),
                ("train/approx_kl", epoch_approx_kl / n_updates),
                ("train/explained_variance", explained_variance),
                ("train/grad_norm", epoch_grad_norm / n_updates),
                ("train/grad_clipped_fraction", epoch_grad_clipped / n_updates),
            ):
                self.log(name, value, prog_bar=False, logger=True, on_epoch=True)
            for name, value in self._rollout_diagnostics.items():
                self.log(f"train/{name}", value, prog_bar=False, logger=True)
            for name, value in rollout_reward_breakdown.items():
                self.log(
                    f"reward/components/{name}", value, prog_bar=False, logger=True
                )
            self.log("env_steps", self.global_step, logger=False, prog_bar=True)

    def _ensure_rollout_envs(self) -> list[WargameEnv]:
        """Build the rollout environments once and keep them for the whole run.

        Two bugs came from rebuilding them on every `training_step`:

        - A fresh `WargameEnv` builds its own `RewardPhaseManager` starting at
          phase 0, while phase advancement only ever mutated the eval env. So
          training reward always came from phase 0 no matter what
          `reward_phase` reported. Sharing the eval env's `CurriculumPosition`
          removes the possibility rather than papering over it.
        - Each fresh env was re-seeded to its own index, so every epoch
          replayed an identical handful of layouts while eval drew random
          ones. Seeding once and letting `reset()` continue the stream gives
          each env a distinct, non-repeating sequence.
        """
        if self._rollout_envs is not None:
            return self._rollout_envs

        envs = [
            WargameEnv(
                self.env.config,
                renderer=None,
                phase_position=self.env.phase_manager.position,
                # `_collect_rollout_parallel` discards the info dict; building
                # it costs ~0.19 ms of every one of the 2048 steps in an epoch.
                build_info=False,
            )
            for _ in range(self.num_rollout_envs)
        ]
        self._rollout_obs = []
        for env_idx, env in enumerate(envs):
            observation, _ = env.reset(
                seed=ROLLOUT_SEED_BASE + env_idx, options=_AUGMENT_START
            )
            self._rollout_obs.append(observation)
        self._rollout_envs = envs
        return envs

    def _record_rollout_outcome(self, env_index: int, env: WargameEnv) -> None:
        """Bank a finished rollout episode as a rated game against its opponent.

        Read **before** `reset()`, which is the only moment the finished
        episode's victory points are still on the env.

        The rollout is already a rated match and its result was being thrown
        away: no extra games are played for the rating, which is what keeps an
        in-run Elo affordable at all. Costs nothing with self-play off -- there
        is no seating to attribute a result to, so the method returns at once.
        """
        if self._drawn_opponents is None:
            return
        name = self._drawn_opponents[env_index].name
        margin = float(env.player_vp - env.opponent_vp)
        self._rollout_margins.setdefault(name, []).append(margin)

    def on_train_epoch_start(self) -> None:
        """Draw this epoch's opponents, one per rollout env.

        Nothing happens without a scheduler, and there is no scheduler unless
        self-play is enabled -- so a control run does not reach this branch and
        cannot draw from any stream.

        Per epoch rather than per episode, because seating a `model` opponent
        loads a checkpoint and sizes a network; per episode would pay that on
        every reset. Per env rather than per run, so one epoch's batch spans the
        pool instead of betting on a single draw.

        ⚠ **`super()` first, unconditionally.** These hooks are overrides, not
        additions: `WargameLightningBase.on_train_epoch_end` runs the per-epoch
        evaluation and the reward-phase advancement, so an override that returns
        early when self-play is off silently disables the curriculum on **every**
        run. It survived a full suite because the base body is gated on
        `do_log`, which the tests turn off.
        """
        super().on_train_epoch_start()
        if self._opponent_scheduler is None:
            return
        drawn = self._opponent_scheduler.seat(self._ensure_rollout_envs())
        self._drawn_opponents = drawn
        self.log(
            "self_play/pool_size",
            float(len(self._opponent_scheduler.pool.entries)),
            logger=True,
        )
        self.log(
            "self_play/mean_opponent_epoch",
            float(np.mean([entry.epoch for entry in drawn])),
            logger=True,
        )

    def on_train_epoch_end(self) -> None:
        """Freeze the learner into the pool on a snapshot epoch.

        ⚠ A Lightning hook, so **`SIGKILL` writes nothing** -- and SIGKILL is
        the prescribed way to stop these trainers. A pool is routinely up to
        `snapshot_every_n_epochs` behind the run that produced it, exactly as
        `last.ckpt` is up to 25 epochs stale for the same reason.

        ⚠ **`super()` first**, which is where evaluation and reward-phase
        advancement live. See `on_train_epoch_start`.
        """
        super().on_train_epoch_end()
        if self._opponent_scheduler is None:
            return
        # Rate first, snapshot second. A snapshot inherits the learner's rating
        # at the moment it is frozen, so it has to be the rating that includes
        # the epoch just played -- otherwise every pool member is one epoch's
        # results stale and the ladder measures the wrong self.
        if self._rollout_margins:
            self.log(
                "self_play/learner_elo",
                self._opponent_scheduler.record_outcomes(self._rollout_margins),
                logger=True,
            )
            self._rollout_margins = {}
        if self._opponent_scheduler.should_snapshot(self.current_epoch):
            # `self.state_dict()`, not `self.ppo_model.state_dict()`. A snapshot
            # is read back by `NetworkOpponentPolicy`, which goes through
            # `convert_state_dict` -- and that looks for `policy_net.` or
            # `ppo_model.policy_network.`, the prefixes a real Lightning
            # checkpoint carries. Saving the inner module gives bare
            # `policy_network.` keys and raises on load. It raised loudly here
            # only because this path loads STRICT; `_apply_warm_start_weights`
            # uses `strict=False`, where the same mistake loads *nothing* and
            # reports a warm start.
            self._opponent_scheduler.snapshot(self.current_epoch, self.state_dict())

    def on_train_start(self) -> None:
        """Record the resolved rollout-env count before the first epoch.

        `num_rollout_envs` defaults to 0, meaning auto-detect, so the config
        never showed which collection path actually ran — and the parallel
        path was the one carrying two bugs.
        """
        super().on_train_start()
        logger.info(
            "PPO rollout: num_rollout_envs={} (config requested {}), ent_coef={}",
            self.num_rollout_envs,
            self.hparams.get("num_rollout_envs"),
            self.ent_coef,
        )
        if self.do_log:
            self.log(
                "train/num_rollout_envs_resolved",
                float(self.num_rollout_envs),
                prog_bar=False,
            )

    def on_train_end(self) -> None:
        """Close the rollout environments held for the duration of the run."""
        if self._rollout_envs is not None:
            for env in self._rollout_envs:
                env.close()
            self._rollout_envs = None
            self._rollout_obs = []

    def _collect_rollout_parallel(
        self,
    ) -> tuple[
        list[Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, float],
    ]:
        """Collect rollouts across multiple env instances.

        This implementation keeps env stepping in Python (single process) but
        batches the policy/value forward pass across environments to reduce
        neural network overhead.
        """
        if self.num_rollout_envs < 1:
            raise ValueError("num_rollout_envs must be >= 1")
        if self.num_rollout_envs == 1:
            raise ValueError("_collect_rollout_parallel called with num_rollout_envs=1")
        if self.n_steps % self.num_rollout_envs != 0:
            raise ValueError(
                "n_steps must be divisible by num_rollout_envs "
                f"({self.n_steps} % {self.num_rollout_envs} != 0)"
            )

        n_envs = self.num_rollout_envs
        t_steps = self.n_steps // n_envs
        device = self.ppo_model.device

        envs = self._ensure_rollout_envs()
        obs_list = self._rollout_obs
        # Store state_tensors for PPO updates in flattened (T * n_envs) form.
        # The returned order matches a row-major flatten of the 2D rollout arrays.
        state_tensors_per_feature: list[list[Tensor]] = [[] for _ in range(6)]
        n_models = self.env.config.number_of_wargame_models

        actions_2d_np = np.zeros((t_steps, n_envs, n_models), dtype=np.int64)
        # Rewards, values and log-probs carry a per-model axis; `dones` is one
        # flag per env and is broadcast across models in `compute_returns`.
        rewards_2d_np = np.zeros((t_steps, n_envs, n_models), dtype=np.float32)
        dones_2d_np = np.zeros((t_steps, n_envs), dtype=np.float32)
        old_log_probs_2d_np = np.zeros((t_steps, n_envs, n_models), dtype=np.float32)
        values_2d_np = np.zeros((t_steps, n_envs, n_models), dtype=np.float32)

        pbar_ctx = (
            tqdm(
                total=self.n_steps,
                desc="Rollout",
                unit="step",
                leave=False,
            )
            if self.show_inner_progress
            else nullcontext(_NoOpProgress())
        )
        breakdown_sums: dict[str, float] = {}
        # Policy entropy split by battle phase, in raw nats. The aggregate
        # `loss/entropy_loss` is `-ent_coef * entropy`, which was misread as
        # entropy itself; and mixing a heavily-masked shooting phase with an
        # open movement phase makes the average uninterpretable either way.
        entropy_sums: dict[str, float] = {}
        entropy_counts: dict[str, int] = {}
        total_steps = 0
        with pbar_ctx as pbar:
            for t in range(t_steps):
                state_tensors_batch = observations_to_tensor_batch(
                    obs_list, device=device
                )
                for feat_idx, feat_tensor in enumerate(state_tensors_batch):
                    state_tensors_per_feature[feat_idx].append(feat_tensor)

                # Read before stepping, so entropy is bucketed by the phase the
                # action was actually chosen in.
                step_phases = [env.game_clock_state.phase for env in envs]

                with torch.no_grad():
                    logits, state_values = self.ppo_model(state_tensors_batch)
                    dist = Categorical(logits=logits)
                    actions = dist.sample()  # (n_envs, n_models)
                    log_probs = dist.log_prob(actions)  # (n_envs, n_models)
                    step_entropy = dist.entropy()  # (n_envs, n_models)

                for env_i, phase in enumerate(step_phases):
                    key = phase.value if phase is not None else "unknown"
                    entropy_sums[key] = entropy_sums.get(key, 0.0) + float(
                        step_entropy[env_i].mean()
                    )
                    entropy_counts[key] = entropy_counts.get(key, 0) + 1

                actions_np = actions.detach().cpu().numpy()
                values_np = state_values.detach().cpu().numpy()
                log_probs_np = log_probs.detach().cpu().numpy()

                actions_2d_np[t] = actions_np
                values_2d_np[t] = values_np
                old_log_probs_2d_np[t] = log_probs_np

                for env_i, env in enumerate(envs):
                    env_action = WargameEnvAction(
                        actions=[int(a) for a in actions_2d_np[t, env_i]]
                    )
                    next_obs, reward, done, _, _ = env.step(env_action)
                    rewards_2d_np[t, env_i] = env.last_per_model_reward
                    dones_2d_np[t, env_i] = 1.0 if done else 0.0
                    for key, value in env.last_reward_breakdown.items():
                        breakdown_sums[key] = breakdown_sums.get(key, 0.0) + value
                    total_steps += 1

                    if done:
                        self._record_rollout_outcome(env_i, env)
                        next_obs, _ = env.reset(options=_AUGMENT_START)
                    obs_list[env_i] = next_obs

                pbar.update(n_envs)

        state_tensors_flat = [
            torch.cat(chunks, dim=0) for chunks in state_tensors_per_feature
        ]
        actions_flat = torch.from_numpy(actions_2d_np.reshape(-1, n_models)).to(
            device=device
        )

        rewards_2d = torch.from_numpy(rewards_2d_np).to(device=device)
        dones_2d = torch.from_numpy(dones_2d_np).to(device=device)
        old_log_probs_2d = torch.from_numpy(old_log_probs_2d_np).to(device=device)
        values_2d = torch.from_numpy(values_2d_np).to(device=device)

        last_state_tensors = observations_to_tensor_batch(obs_list, device=device)
        with torch.no_grad():
            _, last_values = self.ppo_model(last_state_tensors)

        last_values = last_values.detach()
        breakdown_mean = (
            {key: (value / total_steps) for key, value in breakdown_sums.items()}
            if total_steps > 0
            else {}
        )
        self._rollout_diagnostics = {
            f"entropy/{phase}": entropy_sums[phase] / entropy_counts[phase]
            for phase in entropy_sums
            if entropy_counts[phase] > 0
        }
        # The rollout's own phase index. Logged next to the eval env's
        # `reward_phase` so the two diverging is visible in one chart rather
        # than inferred from which reward component keys appear.
        self._rollout_diagnostics["rollout_phase_index"] = float(
            envs[0].phase_manager.current_phase_index
        )
        # Cumulative distinct layouts across the whole run, not within one
        # rollout: the within-rollout count is trivially the env count and
        # would have read a constant 8 even while every epoch replayed the
        # same 8 maps. This number must keep climbing.
        self._layouts_seen.update(
            tuple((int(o.location[0]), int(o.location[1])) for o in env.objectives)
            for env in envs
        )
        self._rollout_diagnostics["distinct_layouts_seen"] = float(
            len(self._layouts_seen)
        )
        return (
            state_tensors_flat,
            actions_flat,
            rewards_2d,
            dones_2d,
            old_log_probs_2d,
            values_2d,
            last_values,
            breakdown_mean,
        )

    def _collect_experiences(self) -> tuple[list[Experience], dict[str, float]]:
        """Run episodes until n_steps transitions are collected (can span multiple episodes)."""
        rollout: list[Experience] = []
        breakdown_sums: dict[str, float] = {}
        total_steps = 0
        pbar_ctx = (
            tqdm(
                total=self.n_steps,
                desc="Rollout",
                unit="step",
                leave=False,
            )
            if self.show_inner_progress
            else nullcontext(_NoOpProgress())
        )
        with pbar_ctx as pbar:
            while len(rollout) < self.n_steps:
                _reward, _steps, episode_exp = self.agent.run_episode_with_experiences(
                    self.ppo_model,
                    epsilon=1.0,
                    render=False,
                    save_steps=True,
                )
                rollout.extend(episode_exp)
                pbar.update(len(episode_exp))
                episode_steps = len(episode_exp)
                if episode_steps > 0:
                    for key, value in self.agent.last_episode_reward_breakdown.items():
                        breakdown_sums[key] = breakdown_sums.get(key, 0.0) + (
                            value * episode_steps
                        )
                    total_steps += episode_steps
        rollout = rollout[: self.n_steps]
        used_steps = len(rollout)
        if total_steps > 0 and used_steps < total_steps:
            scale = used_steps / total_steps
            for key in breakdown_sums:
                breakdown_sums[key] *= scale
            total_steps = used_steps
        breakdown_mean = (
            {key: (value / total_steps) for key, value in breakdown_sums.items()}
            if total_steps > 0
            else {}
        )
        return rollout, breakdown_mean

    def configure_optimizers(self) -> optim.Optimizer:
        """Initialize optimizer."""
        return self.optimizer

    def train_dataloader(self) -> DataLoader[Tensor]:
        """Return a single-batch loader so Lightning calls training_step once per epoch."""
        return DataLoader(
            dataset=_PPODummyDataset(),
            batch_size=1,
            num_workers=0,
        )

    def _policy_model(self) -> PPOModel:
        return self.ppo_model

    def _batch_greedy_actions(self, state_tensors: list[Tensor]) -> Tensor:
        """Greedy per-model actions for a batch. The net masks internally."""
        logits, _values = self.ppo_model(state_tensors)
        actions: Tensor = logits.argmax(dim=-1)
        return actions
