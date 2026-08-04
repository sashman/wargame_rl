from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.reward.calculators.base import (
    GlobalRewardCalculator,
    PerModelRewardCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.registry import build_calculator
from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria
from wargame_rl.wargame.envs.reward.criteria.registry import build_criteria
from wargame_rl.wargame.envs.reward.phase import RewardPhaseConfig

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext


@dataclass
class RewardPhase:
    """A fully instantiated reward phase with live calculator/criteria objects."""

    name: str
    per_model_calculators: list[tuple[str, PerModelRewardCalculator]]
    global_calculators: list[tuple[str, GlobalRewardCalculator]]
    criteria: SuccessCriteria
    success_threshold: float
    min_epochs: int
    min_epochs_above_threshold: int
    terminal_success_bonus: float
    terminal_vp_bonus: float
    terminate_on_success: bool


@dataclass
class CurriculumPosition:
    """How far the curriculum has advanced.

    Split out of `RewardPhaseManager` so several environments can share one
    position. Training runs many rollout envs alongside the eval env; each
    needs its *own* calculators, because those carry per-episode state
    (`closest_objective`'s previous distance, `objective_flip_bonus`'s
    potential) that one env resetting would corrupt for the others. But they
    must all reward the phase the curriculum has actually reached.

    Sharing the position rather than propagating an index means there is no
    synchronisation step for a future code path to forget — which is exactly
    how the rollout envs came to train on phase 0 for every run to date while
    `reward_phase` reported otherwise.
    """

    index: int = 0
    epoch_entered: int = 0
    consecutive_epochs_above_threshold: int = 0


@dataclass
class RewardPhaseManager:
    """Manages reward phase progression during training.

    Owns the ordered list of phases, tracks the current phase index,
    and provides the interface for reward calculation, success checking,
    and phase advancement.
    """

    phases: list[RewardPhase]
    position: CurriculumPosition = field(default_factory=CurriculumPosition)
    last_reward_breakdown: dict[str, float] = field(default_factory=dict, init=False)
    last_per_model_reward: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64), init=False
    )

    @classmethod
    def from_configs(
        cls,
        configs: list[RewardPhaseConfig],
        position: CurriculumPosition | None = None,
    ) -> RewardPhaseManager:
        """Build a manager from a list of phase configs.

        Pass `position` to share curriculum progress with another manager.
        """
        if not configs:
            raise ValueError("reward_phases must contain at least one phase")

        phases: list[RewardPhase] = []
        for cfg in configs:
            per_model: list[tuple[str, PerModelRewardCalculator]] = []
            global_: list[tuple[str, GlobalRewardCalculator]] = []
            name_counts: dict[str, int] = {}

            for calc_cfg in cfg.reward_calculators:
                base_name = calc_cfg.type
                name_counts[base_name] = name_counts.get(base_name, 0) + 1
                suffix = name_counts[base_name]
                calc_name = base_name if suffix == 1 else f"{base_name}_{suffix}"
                calc = build_calculator(calc_cfg.type, calc_cfg.weight, calc_cfg.params)
                if isinstance(calc, PerModelRewardCalculator):
                    per_model.append((calc_name, calc))
                else:
                    global_.append((calc_name, calc))

            criteria = build_criteria(
                cfg.success_criteria.type, cfg.success_criteria.params
            )

            phases.append(
                RewardPhase(
                    name=cfg.name,
                    per_model_calculators=per_model,
                    global_calculators=global_,
                    criteria=criteria,
                    success_threshold=cfg.success_threshold,
                    min_epochs=cfg.min_epochs,
                    min_epochs_above_threshold=cfg.min_epochs_above_threshold,
                    terminal_success_bonus=cfg.terminal_success_bonus,
                    terminal_vp_bonus=cfg.terminal_vp_bonus,
                    terminate_on_success=cfg.terminate_on_success,
                )
            )

        return cls(
            phases=phases,
            position=position if position is not None else CurriculumPosition(),
        )

    def reset_episode(self) -> None:
        """Reset per-episode state of all calculators across all phases."""
        for phase in self.phases:
            for _name, pm_calc in phase.per_model_calculators:
                pm_calc.reset_episode()
            for _name, gl_calc in phase.global_calculators:
                gl_calc.reset_episode()

    # -- Properties -----------------------------------------------------------

    @property
    def current_phase(self) -> RewardPhase:
        return self.phases[self.position.index]

    @property
    def current_phase_name(self) -> str:
        return self.current_phase.name

    @property
    def current_phase_index(self) -> int:
        return self.position.index

    @property
    def is_final_phase(self) -> bool:
        return self.position.index >= len(self.phases) - 1

    @property
    def terminate_on_success(self) -> bool:
        """Whether the current phase should terminate on all-at-objectives."""
        return self.current_phase.terminate_on_success

    @property
    def needs_model_model_distances(self) -> bool:
        """True if any calculator in the current phase needs model-model norms."""
        phase = self.current_phase
        for _name, pm_calc in phase.per_model_calculators:
            if pm_calc.needs_model_model_distances:
                return True
        for _name, gl_calc in phase.global_calculators:
            if gl_calc.needs_model_model_distances:
                return True
        return False

    # -- Core methods ---------------------------------------------------------

    def calculate_reward(self, view: BattleView, ctx: StepContext) -> float:
        """Compute the total reward for the current step.

        Per-model rewards are weighted, summed per model, then averaged
        across all models.  Global rewards are weighted and added on top.

        Also records `last_per_model_reward`, the same quantities kept
        undivided per model. The scalar returned here is unchanged; the vector
        exists because averaging 25 models into one number leaves each model's
        action explaining ~4% of the signal it is credited with.
        """
        phase = self.current_phase
        alive_models = [(i, m) for i, m in enumerate(view.player_models) if m.is_alive]
        n_alive = len(alive_models)

        per_model_rewards = np.zeros(len(view.player_models), dtype=np.float64)
        per_model_sums = {name: 0.0 for name, _calc in phase.per_model_calculators}
        per_model_component_sums: dict[str, float] = {}
        for i, model in alive_models:
            for name, pm_calc in phase.per_model_calculators:
                contribution = pm_calc.weight * pm_calc.calculate(i, model, view, ctx)
                per_model_sums[name] += contribution
                per_model_rewards[i] += contribution
                breakdown: dict[str, float] = pm_calc.get_last_breakdown(i)
                if breakdown:
                    for component_name, value in breakdown.items():
                        key = f"{name}/{component_name}"
                        per_model_component_sums[key] = (
                            per_model_component_sums.get(key, 0.0)
                            + pm_calc.weight * value
                        )

        if n_alive > 0:
            for name in per_model_sums:
                per_model_sums[name] /= n_alive
            for name in per_model_component_sums:
                per_model_component_sums[name] /= n_alive

        avg_per_model = sum(per_model_sums.values()) if n_alive > 0 else 0.0

        global_sums = {name: 0.0 for name, _calc in phase.global_calculators}
        for name, gl_calc in phase.global_calculators:
            global_sums[name] += gl_calc.weight * gl_calc.calculate(view, ctx)

        global_total = sum(global_sums.values())

        # Global terms are broadcast whole to each model rather than split
        # between them: they are the part of the outcome that genuinely is not
        # attributable to one model, so every model should see the same signal.
        shared_reward = global_total

        reward = avg_per_model + global_total
        breakdown = {}
        breakdown.update(per_model_sums)
        breakdown.update(per_model_component_sums)
        breakdown.update(global_sums)
        # TODO(metrics-trap-B): the terminal bonuses below enter `breakdown` as
        # once-per-episode values, but every downstream consumer averages this
        # dict over steps (agent_base divides by episode_reward_steps, then the
        # PPO rollout divides by total_steps). The logged
        # `reward/components/terminal_*` therefore scales inversely with episode
        # length and moves when only episode length changed — it is not
        # comparable to the dense components beside it, nor across runs.
        # Fix: emit terminal bonuses on a separate per-episode key rather than
        # folding them into the per-step breakdown. This is a metrics-reporting
        # change only; the reward itself is correct.
        # See docs/metrics.md § Reading rules.
        if ctx.is_terminated and phase.terminal_success_bonus != 0.0:
            if phase.criteria.is_successful(view, ctx):
                # The remaining-turns scaling is a *speed* incentive, and it only
                # means anything when success ends the episode early. With
                # terminate_on_success disabled every episode runs to max_turns,
                # so remaining_frac collapses to 1/max_turns and silently scales
                # the bonus down by that factor (0.05 at 20 rounds) — which made
                # the configured bonus worth ~1% of episode reward and left the
                # success criterion with almost no gradient behind it.
                if phase.terminate_on_success:
                    remaining = max(0.0, float(ctx.max_turns - ctx.current_turn + 1))
                    denom = float(ctx.max_turns) if ctx.max_turns > 0 else 1.0
                    speed_scale = remaining / denom
                else:
                    speed_scale = 1.0
                bonus = phase.terminal_success_bonus * speed_scale
                reward += bonus
                shared_reward += bonus
                if bonus != 0.0:
                    breakdown["terminal_success_bonus"] = bonus
        if ctx.is_terminated and phase.terminal_vp_bonus != 0.0:
            vp_threshold = phase.criteria.vp_threshold_for_terminal_bonus(view)
            if vp_threshold is not None and view.player_vp >= vp_threshold:
                bonus = phase.terminal_vp_bonus
                reward += bonus
                shared_reward += bonus
                if bonus != 0.0:
                    breakdown["terminal_vp_bonus"] = bonus

        if shared_reward != 0.0:
            for i, _model in alive_models:
                per_model_rewards[i] += shared_reward

        self.last_per_model_reward = per_model_rewards
        self.last_reward_breakdown = breakdown
        return reward

    def check_success(self, view: BattleView, ctx: StepContext) -> bool:
        """Evaluate the current phase's success criteria."""
        return self.current_phase.criteria.is_successful(view, ctx)

    def try_advance(self, success_rate: float, current_epoch: int) -> bool:
        """Attempt to advance to the next phase.

        Advancement requires: min_epochs in phase, success_rate >= threshold,
        and success_rate >= threshold for at least min_epochs_above_threshold
        consecutive epochs.

        Returns True if the phase was advanced.
        """
        if self.is_final_phase:
            return False

        phase = self.current_phase
        position = self.position
        epochs_in_phase = current_epoch - position.epoch_entered

        if success_rate < phase.success_threshold:
            position.consecutive_epochs_above_threshold = 0
            return False

        position.consecutive_epochs_above_threshold += 1

        if epochs_in_phase < phase.min_epochs:
            return False

        if (
            position.consecutive_epochs_above_threshold
            < phase.min_epochs_above_threshold
        ):
            return False

        position.index += 1
        position.epoch_entered = current_epoch
        position.consecutive_epochs_above_threshold = 0
        new_phase = self.current_phase
        logger.info(
            "Reward phase advanced: '{}' -> '{}' (success_rate={:.2f}, epoch={})",
            phase.name,
            new_phase.name,
            success_rate,
            current_epoch,
        )
        return True
