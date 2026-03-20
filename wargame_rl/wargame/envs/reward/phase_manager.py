from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


@dataclass
class RewardPhaseManager:
    """Manages reward phase progression during training.

    Owns the ordered list of phases, tracks the current phase index,
    and provides the interface for reward calculation, success checking,
    and phase advancement.
    """

    phases: list[RewardPhase]
    _current_idx: int = field(default=0, init=False)
    _epoch_entered: int = field(default=0, init=False)
    _consecutive_epochs_above_threshold: int = field(default=0, init=False)
    last_reward_breakdown: dict[str, float] = field(default_factory=dict, init=False)

    @classmethod
    def from_configs(cls, configs: list[RewardPhaseConfig]) -> RewardPhaseManager:
        """Build a manager from a list of phase configs."""
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
                )
            )

        return cls(phases=phases)

    # -- Properties -----------------------------------------------------------

    @property
    def current_phase(self) -> RewardPhase:
        return self.phases[self._current_idx]

    @property
    def current_phase_name(self) -> str:
        return self.current_phase.name

    @property
    def current_phase_index(self) -> int:
        return self._current_idx

    @property
    def is_final_phase(self) -> bool:
        return self._current_idx >= len(self.phases) - 1

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
        """
        phase = self.current_phase
        n_models = len(view.player_models)

        per_model_sums = {name: 0.0 for name, _calc in phase.per_model_calculators}
        per_model_component_sums: dict[str, float] = {}
        for i, model in enumerate(view.player_models):
            for name, pm_calc in phase.per_model_calculators:
                per_model_sums[name] += pm_calc.weight * pm_calc.calculate(
                    i, model, view, ctx
                )
                breakdown: dict[str, float] = pm_calc.get_last_breakdown(i)
                if breakdown:
                    for component_name, value in breakdown.items():
                        key = f"{name}/{component_name}"
                        per_model_component_sums[key] = (
                            per_model_component_sums.get(key, 0.0)
                            + pm_calc.weight * value
                        )

        if n_models > 0:
            for name in per_model_sums:
                per_model_sums[name] /= n_models
            for name in per_model_component_sums:
                per_model_component_sums[name] /= n_models

        avg_per_model = sum(per_model_sums.values()) if n_models > 0 else 0.0

        global_sums = {name: 0.0 for name, _calc in phase.global_calculators}
        for name, gl_calc in phase.global_calculators:
            global_sums[name] += gl_calc.weight * gl_calc.calculate(view, ctx)

        global_total = sum(global_sums.values())

        reward = avg_per_model + global_total
        breakdown = {}
        breakdown.update(per_model_sums)
        breakdown.update(per_model_component_sums)
        breakdown.update(global_sums)
        if ctx.is_terminated and view.config.terminal_success_bonus != 0.0:
            if ctx.distance_cache.all_models_at_objectives():
                # Scale terminal bonus by remaining turns to encourage faster success.
                remaining = max(0.0, float(ctx.max_turns - ctx.current_turn + 1))
                denom = float(ctx.max_turns) if ctx.max_turns > 0 else 1.0
                remaining_frac = remaining / denom
                bonus = float(view.config.terminal_success_bonus) * remaining_frac
                reward += bonus
                if bonus != 0.0:
                    breakdown["terminal_success_bonus"] = bonus
        if ctx.is_terminated and view.config.terminal_vp_bonus != 0.0:
            vp_threshold = phase.criteria.vp_threshold_for_terminal_bonus(view)
            if vp_threshold is not None and view.player_vp >= vp_threshold:
                bonus = float(view.config.terminal_vp_bonus)
                reward += bonus
                if bonus != 0.0:
                    breakdown["terminal_vp_bonus"] = bonus
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
        epochs_in_phase = current_epoch - self._epoch_entered

        if success_rate < phase.success_threshold:
            self._consecutive_epochs_above_threshold = 0
            return False

        self._consecutive_epochs_above_threshold += 1

        if epochs_in_phase < phase.min_epochs:
            return False

        if self._consecutive_epochs_above_threshold < phase.min_epochs_above_threshold:
            return False

        self._current_idx += 1
        self._epoch_entered = current_epoch
        self._consecutive_epochs_above_threshold = 0
        new_phase = self.current_phase
        logger.info(
            "Reward phase advanced: '{}' -> '{}' (success_rate={:.2f}, epoch={})",
            phase.name,
            new_phase.name,
            success_rate,
            current_epoch,
        )
        return True
