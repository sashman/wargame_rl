from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pytorch_lightning import LightningModule

from wargame_rl.wargame.envs.baseline.evaluate import evaluate_baseline
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.agent_base import BaseAgent

# The bar to beat (`squad_march`) and the floor (`random`). The middle rungs
# live in scripts/measure_baselines.py; logging two keeps run pages readable.
BASELINE_POLICIES = ("random", "squad_march")
BASELINE_EPISODES = 20
# Held out from ROLLOUT_SEED_BASE so baselines never share training layouts.
BASELINE_SEED_BASE = 10_000


class WargameLightningBase(LightningModule, ABC):
    """Shared evaluation + reward-phase plumbing for RL Lightning modules."""

    def __init__(
        self,
        env: WargameEnv,
        agent: BaseAgent,
        do_log: bool = True,
        n_episodes: int = 10,
        eval_log_prefix: str = "",
    ):
        super().__init__()
        self.env = env
        self.do_log = do_log
        self.n_episodes = n_episodes
        self.agent = agent
        self.eval_log_prefix = eval_log_prefix
        self.mean_episode_reward = 0.0

    @abstractmethod
    def _policy_model(self) -> nn.Module:
        """Return the policy model used for evaluation."""

    def on_train_start(self) -> None:
        """Measure the scripted baselines once and log them as a fixed bar.

        Without a floor and a reference, a `success_rate` or a reward number
        says nothing: a 945-epoch policy scored 17% where a squad-marching
        heuristic scores 80% on the same env. The baselines are pure numpy, so
        measuring them once per run is nearly free, and they do not change
        during training — hence once at the start rather than every epoch.
        """
        if not self.do_log:
            return
        seeds = [BASELINE_SEED_BASE + i for i in range(BASELINE_EPISODES)]
        baseline_env = WargameEnv(self.env.config, renderer=None)
        try:
            for name in BASELINE_POLICIES:
                result = evaluate_baseline(
                    build_baseline_policy(name), baseline_env, seeds
                )
                self.log(f"eval/baseline_{name}_win_rate", result.win_rate * 100)
                self.log(f"eval/baseline_{name}_vp_margin", result.vp_margin)
                self.log(
                    f"eval/baseline_{name}_at_objectives",
                    result.final_fraction_at_objectives,
                )
                logger.info(
                    "Baseline {}: win_rate={:.2f} vp_margin={:.1f} at_objectives={:.3f}",
                    name,
                    result.win_rate,
                    result.vp_margin,
                    result.final_fraction_at_objectives,
                )
        finally:
            baseline_env.close()

    def _run_episode_eval(self, epsilon: float) -> tuple[float, int]:
        """Run a single evaluation episode and return (reward, steps)."""
        reward, steps = self.agent.run_episode(  # type: ignore[attr-defined]
            self._policy_model(),
            epsilon=epsilon,
            render=False,
            save_steps=False,
        )
        return reward, steps

    def _set_policy_mode(self, eval_mode: bool) -> None:
        """Hook to toggle model eval/train mode for evaluation."""
        policy = self._policy_model()
        if eval_mode:
            policy.eval()
        else:
            policy.train()

    def _evaluate_episodes(
        self,
        *,
        n_episodes: int | None = None,
        epsilon: float = 0.0,
        log_prefix: str = "",
    ) -> float:
        """Run evaluation episodes and optionally log common metrics.

        Returns the success rate as a fraction in [0, 1].
        """
        total_episodes = self.n_episodes if n_episodes is None else n_episodes
        steps_s: list[int] = []
        episode_rewards: list[float] = []
        episode_successes: list[bool] = []
        player_vps: list[float] = []
        opponent_vps: list[float] = []

        self._set_policy_mode(True)

        with torch.no_grad():
            for _ in range(total_episodes):
                reward, steps = self._run_episode_eval(epsilon)
                episode_rewards.append(reward)
                steps_s.append(steps)
                player_vps.append(float(self.env.player_vp))
                opponent_vps.append(float(self.env.opponent_vp))

                if self.env.last_step_context is not None:
                    success = self.env.phase_manager.check_success(
                        self.env, self.env.last_step_context
                    )
                    episode_successes.append(success)

        self._set_policy_mode(False)

        self.mean_episode_reward = sum(episode_rewards) / len(episode_rewards)

        if self.do_log:
            prefix = f"{log_prefix}_" if log_prefix else ""
            self.log(
                f"reward/{prefix}mean_episode_reward",
                self.mean_episode_reward,
                prog_bar=False,
            )
            self.log(
                f"{prefix}mean_episode_steps",
                sum(steps_s) / len(steps_s),
                prog_bar=False,
            )
            self.log(
                f"reward/{prefix}max_episode_reward",
                max(episode_rewards),
                prog_bar=False,
            )
            self.log(
                f"reward/{prefix}min_episode_reward",
                min(episode_rewards),
                prog_bar=False,
            )
            # The phase-invariant scoreboard. `success_rate` changes definition
            # at every phase boundary, and reward changes with the calculator
            # set, so neither is comparable across a run or between runs. VP is
            # the game's own measure of winning and never changes meaning.
            mean_player_vp = sum(player_vps) / len(player_vps)
            mean_opponent_vp = sum(opponent_vps) / len(opponent_vps)
            self.log(f"eval/{prefix}vp_player", mean_player_vp, prog_bar=False)
            self.log(f"eval/{prefix}vp_opponent", mean_opponent_vp, prog_bar=False)
            self.log(
                f"eval/{prefix}vp_margin",
                mean_player_vp - mean_opponent_vp,
                prog_bar=False,
            )
            self.log(
                f"eval/{prefix}win_rate",
                100.0
                * sum(1.0 for p, o in zip(player_vps, opponent_vps) if p > o)
                / len(player_vps),
                prog_bar=False,
            )

        # TODO(metrics-trap-A): this fallback publishes a different definition
        # under the same `success_rate` key — "episode ended early" instead of
        # "phase criteria met" — with no signal to the reader. Only reachable if
        # every eval episode ran zero steps, so it is latent rather than active,
        # but a consumer cannot tell which definition produced a given value.
        # Fix: warn, or emit the fallback under a distinct key.
        # See docs/metrics.md § Reading rules.
        if episode_successes:
            sr = float(np.array(episode_successes, dtype=float).mean())
        else:
            sr = float((np.array(steps_s) < self.env.max_turns).mean())

        if self.do_log:
            prefix = f"{log_prefix}_" if log_prefix else ""
            self.log(f"{prefix}success_rate", sr * 100, prog_bar=False)

        return sr

    def _advance_reward_phase(self, success_rate: float) -> bool:
        advanced = self.env.phase_manager.try_advance(success_rate, self.current_epoch)
        if self.do_log:
            phase_index = int(self.env.phase_manager.current_phase_index)
            self.log(
                "reward_phase",
                float(phase_index),
                prog_bar=False,
            )
            # TODO(metrics-trap-C): `reward_phase` is written twice — once above
            # through Lightning and once here through the raw wandb API — on two
            # different step counters, so the same value lands on two rows. Drop
            # one path once it is established which counter consumers should
            # read. See docs/metrics.md § Reading rules.
            try:
                import wandb

                if wandb.run is not None:  # type: ignore[attr-defined]
                    wandb.log({"reward_phase": phase_index}, step=self.global_step)  # type: ignore[attr-defined]
            except ModuleNotFoundError:
                pass
            if advanced:
                self.log(
                    "phase_advanced_at_epoch",
                    float(self.current_epoch),
                    prog_bar=False,
                )
        return advanced

    def run_episodes(self, n_episodes: int, epsilon: float = 0.0) -> float:
        """Run evaluation episodes and log common metrics."""
        return self._evaluate_episodes(
            n_episodes=n_episodes,
            epsilon=epsilon,
            log_prefix=self.eval_log_prefix,
        )

    def on_train_epoch_end(self) -> None:
        if self.do_log:
            sr = self.run_episodes(self.n_episodes)
            self._advance_reward_phase(sr)
        super().on_train_epoch_end()
