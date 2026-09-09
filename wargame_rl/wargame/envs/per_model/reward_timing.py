"""The reward paid per DECISION over the per-model facade (issue #286).

`PerModelEnv` pays reward at the phase facade's timing -- one window per
stepped phase, settled where the whole-army step would have ended -- and
`tests/test_per_model_bridge.py` pins that bit for bit. This module is the
other timing, kept out of `env.py` so the two facades stay uncoupled: it reads
what a step DID (`StepEffect`) and the env's public state, and pays every
configured term on the step the design's table names:

- an ACTION term on the step, to the step's actor set (a potential term) or
  to the attackers the step's kills name (the event term), each divided by
  the alive count so a turn's sum is the phase facade's scalar mean and no
  term's per-round total scales with the army;
- a STATE term once per turn on the closing step, as the mean over alive
  models times the number of stepped phases per round -- the phase facade paid
  it once per phase, so the episode total is conserved up to the movement
  between phases within one round (an approximation on a melee config);
- a DELTA global (`vp_gain`, kills, losses, the flip bonus) once at the close,
  unscaled -- `vp_gain` telescopes so its episode total is exact; a STATE
  global (coverage) once at the close, scaled like a state term;
- the terminal bonuses at the terminating close, through the phase manager's
  own `terminal_bonuses`, so the two facades cannot disagree on when one is
  earned.

Every calculator the registry names belongs to exactly one payment class,
keyed by the registry's own string; the module refuses to import if the two
key sets differ, and a phase carrying the refused legacy `closest_objective`
(its potential lives on the model object, which no second instance can
isolate from the env's own windows) is refused at construction by name.
`group_cohesion` is a STATE term here where the #283 table paid it on the
actor's step: paid per action it fines the first mover of a coherent unit for
a gap its squadmates have not yet had a step to close, teaching the selector
to reorder instead of the policy to hold formation.

The retimer owns its OWN `RewardPhaseManager` unless one is passed in:
`closest_objective_v2`, `charge_progress` and `objective_flip_bonus` carry
memory across calls, and sharing instances with the env's windows would let
each call advance the potential the other reads. Numpy only -- `envs/` never
imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.types import (
    PerModelAction,
    PerModelObservation,
    StepEffect,
    StepKind,
)
from wargame_rl.wargame.envs.reward.calculators.base import (
    GlobalRewardCalculator,
    PerModelRewardCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.registry import CALCULATOR_REGISTRY
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView


class PaymentClass(str, Enum):
    """When a term is paid under the per-decision step."""

    # On the step, to the step's actor set.
    potential_action = "potential_action"
    # On the step, to the attackers the step's kills name.
    event_action = "event_action"
    # At the close, mean over alive models, scaled by the phases per round.
    state = "state"
    # At the close, once, unscaled.
    delta_global = "delta_global"
    # At the close, once, scaled by the phases per round.
    state_global = "state_global"
    # Cannot be paid per decision; a config carrying it is refused by name.
    refused = "refused"


# Keyed by the reward registry's own string, and checked against it below so
# a calculator registered without a payment class fails at import, not in a
# training run.
PAYMENT_CLASSES: dict[str, PaymentClass] = {
    "closest_objective": PaymentClass.refused,
    "closest_objective_v2": PaymentClass.potential_action,
    "charge_progress": PaymentClass.potential_action,
    "declared_objective_progress": PaymentClass.potential_action,
    "declared_target_progress": PaymentClass.potential_action,
    "model_kills": PaymentClass.event_action,
    "objective_hold": PaymentClass.state,
    "declared_objective_hold": PaymentClass.state,
    "unit_coherency": PaymentClass.state,
    "group_cohesion": PaymentClass.state,
    "vp_gain": PaymentClass.delta_global,
    "killing": PaymentClass.delta_global,
    "models_lost": PaymentClass.delta_global,
    "objective_flip_bonus": PaymentClass.delta_global,
    "objective_coverage": PaymentClass.state_global,
    "models_at_objectives": PaymentClass.state_global,
}

_unclassified = set(CALCULATOR_REGISTRY) - set(PAYMENT_CLASSES)
_unregistered = set(PAYMENT_CLASSES) - set(CALCULATOR_REGISTRY)
if _unclassified or _unregistered:
    raise ImportError(
        "reward_timing.PAYMENT_CLASSES and CALCULATOR_REGISTRY disagree: "
        f"unclassified {sorted(_unclassified)}, unregistered {sorted(_unregistered)}"
    )

# Exact type, never subclass: the registry keys concrete classes, and a
# subclass registered under its own key carries its own class.
_KEY_BY_CLASS: dict[type, str] = {cls: key for key, cls in CALCULATOR_REGISTRY.items()}


def payment_class_of(calculator: object) -> PaymentClass:
    """The payment class of a calculator instance, by its registry key."""
    key = _KEY_BY_CLASS.get(type(calculator))
    if key is None:
        raise ValueError(
            f"{type(calculator).__name__} is not in CALCULATOR_REGISTRY, so it has "
            "no per-decision payment class"
        )
    return PAYMENT_CLASSES[key]


@dataclass(frozen=True)
class StepPayment:
    """What one step paid: the scalar, its breakdown by term, and whether a
    round boundary lies after it -- a `close_turn`, or any terminating step,
    the only steps the discount applies across."""

    reward: float
    breakdown: dict[str, float]
    is_close: bool


@dataclass(frozen=True)
class _TurnBaseline:
    """The state at the previous close, for the delta globals."""

    player_vp: int
    opponent_vp: int
    player_alive: np.ndarray
    opponent_alive: np.ndarray
    attrition: tuple[int, int]


@dataclass
class _Classified:
    potential: list[tuple[str, PerModelRewardCalculator]] = field(default_factory=list)
    event: list[tuple[str, PerModelRewardCalculator]] = field(default_factory=list)
    state: list[tuple[str, PerModelRewardCalculator]] = field(default_factory=list)
    delta_globals: list[tuple[str, GlobalRewardCalculator]] = field(
        default_factory=list
    )
    state_globals: list[tuple[str, GlobalRewardCalculator]] = field(
        default_factory=list
    )


class _CloseView:
    """The env, with the turn's net VP deltas in place of the window's.

    `_open_window` resets the env's deltas per window, so at the close they
    carry only the last window's scoring; `VPGainCalculator` reads
    `view.player_vp_delta`, and this keeps its arithmetic single-sourced.
    Every other read falls through to the env, as `MirroredEnv` does.
    """

    def __init__(self, env: PerModelEnv, player_delta: int, opponent_delta: int):
        self._env = env
        self._player_delta = player_delta
        self._opponent_delta = opponent_delta

    @property
    def player_vp_delta(self) -> int:
        return self._player_delta

    @property
    def opponent_vp_delta(self) -> int:
        return self._opponent_delta

    def __getattr__(self, name: str) -> Any:
        # `copy.deepcopy` reconstructs without `__init__`, so `self._env`
        # would re-enter here and recurse; `MirroredEnv` documents the same
        # guard as load-bearing.
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


def classify(manager: RewardPhaseManager) -> _Classified:
    """Sort the current phase's calculators into their payment classes.

    Raises for a calculator outside the registry or in the refused class, so
    a config that carries a term this timing cannot pay is refused at
    construction.
    """
    phase = manager.current_phase
    classes = _Classified()
    for name, per_model in phase.per_model_calculators:
        payment = payment_class_of(per_model)
        if payment is PaymentClass.refused:
            raise ValueError(
                f"reward term `{name}` keeps its potential on the model object "
                "and cannot be paid per decision; use closest_objective_v2"
            )
        if payment is PaymentClass.potential_action:
            classes.potential.append((name, per_model))
        elif payment is PaymentClass.event_action:
            classes.event.append((name, per_model))
        elif payment is PaymentClass.state:
            classes.state.append((name, per_model))
        else:
            raise ValueError(f"per-model term `{name}` is classified {payment.value}")
    for name, global_calculator in phase.global_calculators:
        payment = payment_class_of(global_calculator)
        if payment is PaymentClass.delta_global:
            classes.delta_globals.append((name, global_calculator))
        elif payment is PaymentClass.state_global:
            classes.state_globals.append((name, global_calculator))
        else:
            raise ValueError(f"global term `{name}` is classified {payment.value}")
    return classes


class PerStepReward:
    """Pays the configured reward per decision over one `PerModelEnv`.

    Call `reset()` after every `env.reset()` and `on_step(...)` after every
    `env.step(...)`; read `episode_reward` and `episode_breakdown` between.
    `manager` defaults to a fresh `RewardPhaseManager` from the env's config;
    whatever is passed must not be the env's own (see the module docstring).
    """

    def __init__(
        self, env: PerModelEnv, manager: RewardPhaseManager | None = None
    ) -> None:
        if len(env.config.reward_phases) != 1:
            raise ValueError(
                "the per-decision reward supports a single reward phase; "
                f"this config carries {len(env.config.reward_phases)} "
                "(curriculum advancement is a follow-up to #286)"
            )
        if manager is env.phase_manager:
            raise ValueError(
                "the per-decision reward must not share the env's own calculators: "
                "the stateful ones would advance each other's potentials"
            )
        self.env = env
        self.manager = manager or RewardPhaseManager.from_configs(
            env.config.reward_phases
        )
        self.classes = classify(self.manager)
        self.phase_scale = float(env.stepped_phases_per_round)
        self.episode_reward = 0.0
        self.episode_breakdown: dict[str, float] = {}
        self.closes = 0
        self.last_context: StepContext | None = None
        self.last_view: BattleView | None = None
        self._baseline = self._read_baseline()

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Start an episode: call right after `env.reset()` returns, so the
        VP baseline includes round-one command scoring nobody decided."""
        self.manager.reset_episode()
        self.episode_reward = 0.0
        self.episode_breakdown = {}
        self.closes = 0
        self.last_context = None
        self.last_view = None
        self._baseline = self._read_baseline()

    def _read_baseline(self) -> _TurnBaseline:
        env = self.env
        return _TurnBaseline(
            player_vp=int(env.player_vp),
            opponent_vp=int(env.opponent_vp),
            player_alive=alive_mask_for(env.wargame_models).copy(),
            opponent_alive=alive_mask_for(env.opponent_models).copy(),
            attrition=env.attrition_deaths_total,
        )

    # ------------------------------------------------------------------ pay

    def on_step(
        self,
        observation_before: PerModelObservation,
        action: PerModelAction,
        effect: StepEffect,
        terminated: bool,
    ) -> StepPayment:
        """Pay the step just taken; the env is read as it stands after it."""
        is_close = action.kind is StepKind.close_turn or terminated
        breakdown: dict[str, float] = {}
        reward = 0.0
        if action.kind is not StepKind.close_turn:
            reward += self._pay_action_terms(
                observation_before.decision.phase, effect, breakdown
            )
        if is_close:
            reward += self._pay_close(terminated, breakdown)
            self.closes += 1
        self.episode_reward += reward
        for key, value in breakdown.items():
            self.episode_breakdown[key] = self.episode_breakdown.get(key, 0.0) + value
        return StepPayment(reward=reward, breakdown=breakdown, is_close=is_close)

    def _context(
        self,
        *,
        action_phase: BattlePhase | None,
        kills_by_model: np.ndarray | None,
        player_killed: int = 0,
        opponent_killed: int = 0,
        terminated: bool = False,
    ) -> StepContext:
        env = self.env
        alive = alive_mask_for(env.wargame_models)
        cache = compute_distances(
            env.wargame_models,
            env.objectives,
            compute_model_model=self.manager.needs_model_model_distances,
            alive_mask=alive,
        )
        current_round, battle_phase = env.context_clock()
        return StepContext(
            distance_cache=cache,
            current_turn=env.current_turn,
            max_turns=env.max_turns,
            board_width=env.board_width,
            board_height=env.board_height,
            is_terminated=terminated,
            current_round=current_round,
            battle_phase=battle_phase,
            action_phase=action_phase,
            player_models_killed=player_killed,
            opponent_models_killed=opponent_killed,
            player_kills_by_model=kills_by_model,
        )

    def _pay_action_terms(
        self,
        phase: BattlePhase | None,
        effect: StepEffect,
        breakdown: dict[str, float],
    ) -> float:
        classes = self.classes
        actor_set = effect.actor_set
        attackers = tuple(sorted(effect.kills_by_model))
        if not actor_set and not attackers:
            return 0.0
        if not classes.potential and not classes.event:
            return 0.0
        env = self.env
        n_alive = int(alive_mask_for(env.wargame_models).sum())
        if n_alive == 0:
            return 0.0
        n_models = len(env.wargame_models)
        kills = np.zeros(n_models, dtype=np.int64)
        for index, count in effect.kills_by_model.items():
            if index < n_models:
                kills[index] = count
        ctx = self._context(action_phase=phase, kills_by_model=kills)
        view = cast("BattleView", env)
        total = 0.0
        for name, calculator in classes.potential:
            for index in actor_set:
                model = env.wargame_models[index]
                if not model.is_alive:
                    continue
                paid = calculator.weight * calculator.calculate(index, model, view, ctx)
                paid /= n_alive
                total += paid
                breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, calculator in classes.event:
            for index in attackers:
                model = env.wargame_models[index]
                paid = calculator.weight * calculator.calculate(index, model, view, ctx)
                paid /= n_alive
                total += paid
                breakdown[name] = breakdown.get(name, 0.0) + paid
        return total

    def _pay_close(self, terminated: bool, breakdown: dict[str, float]) -> float:
        env = self.env
        classes = self.classes
        before = self._baseline
        player_alive = alive_mask_for(env.wargame_models)
        opponent_alive = alive_mask_for(env.opponent_models)
        # The phase facade's kill rule: casualties since the baseline, minus
        # the attrition deaths the rules give no kill credit for.
        attrition_player, attrition_opponent = env.attrition_deaths_total
        player_lost = max(
            0,
            int((before.player_alive & ~player_alive).sum())
            - (attrition_player - before.attrition[0]),
        )
        opponent_lost = max(
            0,
            int((before.opponent_alive & ~opponent_alive).sum())
            - (attrition_opponent - before.attrition[1]),
        )
        ctx = self._context(
            action_phase=None,
            kills_by_model=None,
            # `StepContext` names these from the PLAYER's point of view as the
            # phase facade does: `player_models_killed` is what the player
            # killed, `opponent_models_killed` what the opponent killed.
            player_killed=opponent_lost,
            opponent_killed=player_lost,
            terminated=terminated,
        )
        view = cast(
            "BattleView",
            _CloseView(
                env,
                int(env.player_vp) - before.player_vp,
                int(env.opponent_vp) - before.opponent_vp,
            ),
        )
        self.last_context = ctx
        self.last_view = view
        total = 0.0
        n_alive = int(player_alive.sum())
        for name, calculator in classes.state:
            if n_alive == 0:
                continue
            summed = 0.0
            for index in np.flatnonzero(player_alive):
                model = env.wargame_models[int(index)]
                summed += calculator.calculate(int(index), model, view, ctx)
            paid = calculator.weight * summed / n_alive * self.phase_scale
            total += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, delta_global in classes.delta_globals:
            paid = delta_global.weight * delta_global.calculate(view, ctx)
            total += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, state_global in classes.state_globals:
            paid = (
                state_global.weight
                * state_global.calculate(view, ctx)
                * self.phase_scale
            )
            total += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        if terminated:
            for key, bonus in self.manager.terminal_bonuses(view, ctx).items():
                total += bonus
                breakdown[key] = breakdown.get(key, 0.0) + bonus
        self._baseline = self._read_baseline()
        return total

    # ------------------------------------------------------------ readouts

    def succeeded(self) -> bool:
        """The phase's success criteria on the terminating context, scored
        against the same view that paid the terminal bonus."""
        ctx = self.last_context
        view = self.last_view
        if ctx is None or view is None:
            return False
        return self.manager.check_success(view, ctx)


__all__ = [
    "PAYMENT_CLASSES",
    "PaymentClass",
    "PerStepReward",
    "StepPayment",
    "classify",
    "payment_class_of",
]
