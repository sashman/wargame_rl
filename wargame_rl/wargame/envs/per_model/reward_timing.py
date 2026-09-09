"""The reward paid per DECISION over the per-model facade (issue #286).

`PerModelEnv` pays reward at the phase facade's timing -- one window per
stepped phase, settled where the whole-army step would have ended -- and
`tests/test_per_model_bridge.py` pins that bit for bit. This module is the
other timing, kept out of `env.py` so the two facades stay uncoupled: it reads
what a step DID (`StepEffect`) and the env's public state, and pays every
configured term on the step the design's table names:

- an ACTION term on the step, to the models the step marked acted (a potential
  term) or to the attackers the step's kills name (the event term), each
  divided by the alive count so a turn's sum is the phase facade's scalar mean
  and no term's per-round total scales with the army;
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

Every registered calculator belongs to exactly one class; one that belongs to
none is a construction error, never a silent zero. `group_cohesion` is a
STATE term here where the #283 table paid it on the actor's step: paid per
action it fines the first mover of a coherent unit for a gap its squadmates
have not yet had a step to close, teaching the selector to reorder instead of
the policy to hold formation. The legacy `closest_objective` keeps its
potential on the model object, which no second calculator instance can
isolate from the env's own windows, so it is refused by name.

The retimer owns its OWN `RewardPhaseManager`: `closest_objective_v2`,
`charge_progress` and `objective_flip_bonus` carry memory across calls, and
sharing instances with the env's windows would let each call advance the
potential the other reads. Numpy only -- `envs/` never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.per_model.env import PerModelEnv, StepEffect
from wargame_rl.wargame.envs.per_model.types import (
    PerModelAction,
    PerModelObservation,
    StepKind,
)
from wargame_rl.wargame.envs.reward.calculators.base import (
    GlobalRewardCalculator,
    PerModelRewardCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.charge_progress import (
    ChargeProgressCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.closest_objective import (
    ClosestObjectiveCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)
from wargame_rl.wargame.envs.reward.calculators.declared_objective_hold import (
    DeclaredObjectiveHoldCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.declared_objective_progress import (
    DeclaredObjectiveProgressCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.declared_target_progress import (
    DeclaredTargetProgressCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.group_cohesion import (
    GroupCohesionCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.killing import KillingReward
from wargame_rl.wargame.envs.reward.calculators.model_kills import ModelKillsCalculator
from wargame_rl.wargame.envs.reward.calculators.models_at_objectives import (
    ModelsAtObjectivesCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.models_lost import ModelsLostPenalty
from wargame_rl.wargame.envs.reward.calculators.objective_coverage import (
    ObjectiveCoverageCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.objective_flip_bonus import (
    ObjectiveFlipBonusCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.objective_hold import (
    ObjectiveHoldCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.unit_coherency import (
    UnitCoherencyCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.vp_gain import VPGainCalculator
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import BattlePhase

# Paid on the step, to the models the step marked acted.
POTENTIAL_ACTION_TERMS: tuple[type[PerModelRewardCalculator], ...] = (
    ClosestObjectiveV2Calculator,
    ChargeProgressCalculator,
    DeclaredObjectiveProgressCalculator,
    DeclaredTargetProgressCalculator,
)
# Paid on the step, to the attackers the step's kills name.
EVENT_ACTION_TERMS: tuple[type[PerModelRewardCalculator], ...] = (ModelKillsCalculator,)
# Paid at the close, mean over alive models, scaled by the phases per round.
STATE_TERMS: tuple[type[PerModelRewardCalculator], ...] = (
    ObjectiveHoldCalculator,
    DeclaredObjectiveHoldCalculator,
    UnitCoherencyCalculator,
    GroupCohesionCalculator,
)
# Paid at the close, once, unscaled.
DELTA_GLOBALS: tuple[type[GlobalRewardCalculator], ...] = (
    VPGainCalculator,
    KillingReward,
    ModelsLostPenalty,
    ObjectiveFlipBonusCalculator,
)
# Paid at the close, once, scaled by the phases per round.
STATE_GLOBALS: tuple[type[GlobalRewardCalculator], ...] = (
    ObjectiveCoverageCalculator,
    ModelsAtObjectivesCalculator,
)
# Refused: its potential lives on the model object, shared with the env's own
# windows, so no second instance can be isolated from them.
REFUSED_TERMS: tuple[type[PerModelRewardCalculator], ...] = (
    ClosestObjectiveCalculator,
)


@dataclass(frozen=True)
class StepPayment:
    """What one step paid: the scalar, its breakdown by term, and whether the
    step closed a turn (the only kind the discount applies across)."""

    reward: float
    breakdown: dict[str, float]
    is_close: bool


@dataclass
class _TurnBaseline:
    """The state at the previous close, for the delta globals."""

    player_vp: int
    opponent_vp: int
    player_alive: np.ndarray
    opponent_alive: np.ndarray


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
        return getattr(self._env, name)


def classify(manager: RewardPhaseManager) -> _Classified:
    """Sort the current phase's calculators into their payment classes.

    Raises for a calculator in no class or in the refused set, so a config
    that carries a term this timing cannot pay is refused at construction.
    """
    phase = manager.current_phase
    classes = _Classified()
    for name, per_model in phase.per_model_calculators:
        if isinstance(per_model, REFUSED_TERMS):
            raise ValueError(
                f"reward term `{name}` keeps its potential on the model object "
                "and cannot be paid per decision; use closest_objective_v2"
            )
        if isinstance(per_model, POTENTIAL_ACTION_TERMS):
            classes.potential.append((name, per_model))
        elif isinstance(per_model, EVENT_ACTION_TERMS):
            classes.event.append((name, per_model))
        elif isinstance(per_model, STATE_TERMS):
            classes.state.append((name, per_model))
        else:
            raise ValueError(
                f"reward term `{name}` ({type(per_model).__name__}) has no "
                "per-decision payment class"
            )
    for name, global_ in phase.global_calculators:
        if isinstance(global_, DELTA_GLOBALS):
            classes.delta_globals.append((name, global_))
        elif isinstance(global_, STATE_GLOBALS):
            classes.state_globals.append((name, global_))
        else:
            raise ValueError(
                f"reward term `{name}` ({type(global_).__name__}) has no "
                "per-decision payment class"
            )
    return classes


class PerStepReward:
    """Pays the configured reward per decision over one `PerModelEnv`.

    Call `reset()` after every `env.reset()` and `on_step(...)` after every
    `env.step(...)`; read `episode_reward` and `episode_breakdown` between.
    """

    def __init__(self, env: PerModelEnv) -> None:
        if len(env.config.reward_phases) != 1:
            raise ValueError(
                "the per-decision reward supports a single reward phase; "
                f"this config carries {len(env.config.reward_phases)} "
                "(curriculum advancement is a follow-up to #286)"
            )
        self.env = env
        self.manager = RewardPhaseManager.from_configs(env.config.reward_phases)
        self.classes = classify(self.manager)
        self.phases_per_round = max(1, env.max_turns // env.n_rounds)
        self.episode_reward = 0.0
        self.episode_breakdown: dict[str, float] = {}
        self.closes = 0
        self.last_context: StepContext | None = None
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
        self._baseline = self._read_baseline()

    def _read_baseline(self) -> _TurnBaseline:
        env = self.env
        return _TurnBaseline(
            player_vp=int(env.player_vp),
            opponent_vp=int(env.opponent_vp),
            player_alive=alive_mask_for(env.wargame_models).copy(),
            opponent_alive=alive_mask_for(env.opponent_models).copy(),
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
        clock = env.game_clock_state
        return StepContext(
            distance_cache=cache,
            current_turn=env.current_turn,
            max_turns=env.max_turns,
            board_width=env.board_width,
            board_height=env.board_height,
            is_terminated=terminated,
            current_round=clock.battle_round or 0,
            battle_phase=clock.phase or BattlePhase.command,
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
        actors = effect.acted
        attackers = tuple(sorted(effect.kills_by_model))
        if not actors and not attackers:
            return 0.0
        if not classes.potential and not classes.event:
            return 0.0
        env = self.env
        n_models = len(env.wargame_models)
        kills = np.zeros(n_models, dtype=np.int64)
        for index, count in effect.kills_by_model.items():
            if index < n_models:
                kills[index] = count
        ctx = self._context(action_phase=phase, kills_by_model=kills)
        n_alive = int(alive_mask_for(env.wargame_models).sum())
        if n_alive == 0:
            return 0.0
        view = cast(Any, env)
        total = 0.0
        for name, calculator in classes.potential:
            for index in actors:
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
        player_lost = int((before.player_alive & ~player_alive).sum())
        opponent_lost = int((before.opponent_alive & ~opponent_alive).sum())
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
        self.last_context = ctx
        view = cast(
            Any,
            _CloseView(
                env,
                int(env.player_vp) - before.player_vp,
                int(env.opponent_vp) - before.opponent_vp,
            ),
        )
        scale = float(self.phases_per_round)
        total = 0.0
        n_alive = int(player_alive.sum())
        for name, calculator in classes.state:
            if n_alive == 0:
                continue
            summed = 0.0
            for index in np.flatnonzero(player_alive):
                model = env.wargame_models[int(index)]
                summed += calculator.calculate(int(index), model, view, ctx)
            paid = calculator.weight * summed / n_alive * scale
            total += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, delta_global in classes.delta_globals:
            paid = delta_global.weight * delta_global.calculate(view, ctx)
            total += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, state_global in classes.state_globals:
            paid = state_global.weight * state_global.calculate(view, ctx) * scale
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
        """The phase's success criteria on the terminating context, if any."""
        ctx = self.last_context
        if ctx is None:
            return False
        return self.manager.check_success(cast(Any, self.env), ctx)


__all__ = [
    "DELTA_GLOBALS",
    "EVENT_ACTION_TERMS",
    "POTENTIAL_ACTION_TERMS",
    "REFUSED_TERMS",
    "STATE_GLOBALS",
    "STATE_TERMS",
    "PerStepReward",
    "StepPayment",
    "classify",
]
