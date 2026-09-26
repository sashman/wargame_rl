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

Two CREDIT modes decide who a payment reaches. `Credit.mean`, the default,
is the accounting above: action terms divided by the alive count, state terms
as the army mean at the close, so a turn's sum is the phase facade's scalar
and the bridge totals hold. `Credit.actor` pays each model its own: the
actor's action term, and a state term's per-model value at the close as a
CREDIT to the model whose position produced it (`StepPayment.credits`, which
the rollout collector lands on that model's own step of the turn) -- every
payment, globals and terminal bonuses included, over the MODEL COUNT, a
constant. Relative to `mean` that leaves the actor's own term where it was
and shrinks every common payment by the army size, which is the treatment:
PPO normalises advantages, so the policy gradient sees only that ratio, and
the constant keeps the return scale of the mean's order so the value loss
does not swamp the clipped gradient (the unscaled first cut had returns 13x
larger and the pre-clip gradient norm 5x, clipped on every step). It exists
because under the mean a model's own move is worth `1/n` of the travel pay
and nothing else it sees varies with what it did -- on a 24-body army the
signal that distinguishes one decision from another is a twenty-fourth of
one term (the A3 speed screen's null and A5b's under-arrival, #340).

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


class Credit(str, Enum):
    """Who a payment reaches under the per-decision step."""

    # Action terms over the alive count, state terms as the army mean: a
    # turn's sum is the phase facade's scalar (the bridge accounting).
    mean = "mean"
    # Every payment over the model count: the actor keeps its action term,
    # state terms are credited per model on the model's own step of the turn,
    # and the common payments shrink by the army size relative to `mean`.
    actor = "actor"


class PlanningCredit(str, Enum):
    """How a close's planning scalar reaches each unit's commitment step (#384).

    `broadcast` (the default): the army's outcome, the same to every unit's
    open span -- a squad stacking on a covered objective and the squad taking
    the empty one are credited alike. `counterfactual` (B6): each unit is
    credited the DIFFERENCE its bodies make to the outcome terms -- the state
    globals and the terminal bonuses re-evaluated with the unit's models
    masked out, subtracted from the same terms with them in -- so a redundant
    squad earns 0 and the squad that covers an empty objective earns what it
    covers. The delta globals (VP, kills) stay broadcast: they are realised
    from the mission's own counts and have no per-unit counterfactual.
    """

    broadcast = "broadcast"
    counterfactual = "counterfactual"


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
    # Not a potential: an occupancy term computed for the mover on its own
    # step from the board after its move, so that a body which ends inside
    # an objective is paid and one which walks out is not (#340, staying).
    "objective_stay": PaymentClass.potential_action,
    "plan_following": PaymentClass.potential_action,
    "objective_hold": PaymentClass.state,
    "declared_objective_hold": PaymentClass.state,
    "unit_coherency": PaymentClass.state,
    "group_cohesion": PaymentClass.state,
    "vp_gain": PaymentClass.delta_global,
    "killing": PaymentClass.delta_global,
    "models_lost": PaymentClass.delta_global,
    "objective_flip_bonus": PaymentClass.delta_global,
    "objective_coverage": PaymentClass.state_global,
    "commitment_coverage": PaymentClass.state_global,
    "commitment_churn": PaymentClass.state_global,
    "models_at_objectives": PaymentClass.state_global,
}

# The TASK each per-decision term serves under `commitments.execution: plan`
# (#384 execution phase): its payment is multiplied by the actor's unit's
# weight for that task. Tasks with no slot yet (charge, attack) weigh 1.0.
TASK_APPROACH = 0
TASK_HOLD = 1
TASK_OF: dict[str, int | None] = {
    "closest_objective_v2": TASK_APPROACH,
    "declared_objective_progress": TASK_APPROACH,
    "objective_stay": TASK_HOLD,
    "charge_progress": None,
    "declared_target_progress": None,
    "model_kills": None,
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
    # Under `Credit.actor`, a close's state terms per model: the collector
    # lands each on that model's own step of the turn (the close's when the
    # model took none). Empty under `Credit.mean`. Counted in `breakdown`.
    credits: dict[int, float] = field(default_factory=dict)
    # Under `streams` (#384 D2): the close's outcome terms -- the delta and
    # state globals and the terminal bonuses -- paid to the PLANNING stream
    # (each unit's open commitment step) instead of to `reward`. 0.0 without
    # streams. Counted in `breakdown` and in `episode_reward`.
    planning: float = 0.0
    # Under `PlanningCredit.counterfactual`: the close's planning credit per
    # UNIT (group id), landed on that unit's open commitment span instead of
    # the broadcast `planning`, which is then 0.0. Empty otherwise.
    planning_credits: dict[int, float] = field(default_factory=dict)


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
        self,
        env: PerModelEnv,
        manager: RewardPhaseManager | None = None,
        credit: Credit = Credit.mean,
        *,
        streams: bool = False,
        planning_credit: PlanningCredit = PlanningCredit.broadcast,
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
        self.credit = Credit(credit)
        # The two reward streams (#384 D2): keyed on the board outcome ->
        # planning (the commitment decision), keyed on the unit's commitment
        # or the member's own step -> execution (the member). Off, one stream.
        self.streams = bool(streams)
        self.planning_credit = PlanningCredit(planning_credit)
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
        credits: dict[int, float] = {}
        reward = 0.0
        planning = 0.0
        planning_credits: dict[int, float] = {}
        if action.kind is not StepKind.close_turn:
            reward += self._pay_action_terms(
                observation_before.decision.phase, effect, breakdown
            )
        if is_close:
            closed, credits, planning, planning_credits = self._pay_close(
                terminated, breakdown
            )
            reward += closed
            self.closes += 1
        self.episode_reward += reward + sum(credits.values()) + planning
        for key, value in breakdown.items():
            self.episode_breakdown[key] = self.episode_breakdown.get(key, 0.0) + value
        return StepPayment(
            reward=reward,
            breakdown=breakdown,
            is_close=is_close,
            credits=credits,
            planning=planning,
            planning_credits=planning_credits,
        )

    def _context(
        self,
        *,
        action_phase: BattlePhase | None,
        kills_by_model: np.ndarray | None,
        player_killed: int = 0,
        opponent_killed: int = 0,
        terminated: bool = False,
        alive_override: np.ndarray | None = None,
    ) -> StepContext:
        env = self.env
        alive = alive_mask_for(env.wargame_models)
        if alive_override is not None:
            # A counterfactual board (#384 B6): the caller's mask decides which
            # of our models the distance cache counts as present.
            alive = alive & alive_override
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
            committed_objective=(
                env.player_commitments.committed_objective_per_model(env.wargame_models)
                if env.config.commitments.enabled
                else None
            ),
            task_weights=(
                env.player_commitments.task_weights_per_model(env.wargame_models)
                if env.config.commitments.plan_weighted
                else None
            ),
            commitment_churn=(
                _churn_share(env) if env.config.commitments.enabled else None
            ),
        )

    @staticmethod
    def _task_weight(ctx: StepContext, name: str, index: int) -> float:
        """The actor's plan weight for the task `name` serves: 1.0 unless the
        plan-weighted execution is on and the term has a task with a slot."""
        weights = ctx.task_weights
        if weights is None:
            return 1.0
        task = TASK_OF.get(name)
        if task is None:
            return 1.0
        return float(weights[index, task])

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
        per_model = self.credit is Credit.actor
        kills = np.zeros(n_models, dtype=np.int64)
        for index, count in effect.kills_by_model.items():
            if index < n_models:
                kills[index] = count
        ctx = self._context(action_phase=phase, kills_by_model=kills)
        view = cast("BattleView", env)
        # The mean accounting shares a term over the ALIVE count; the actor
        # mode over the model count, a constant, so the actor's term does not
        # grow as its army dies while every common payment shrinks by it.
        share = 1.0 / len(env.wargame_models) if per_model else 1.0 / n_alive
        total = 0.0
        for name, calculator in classes.potential:
            for index in actor_set:
                model = env.wargame_models[index]
                if not model.is_alive:
                    continue
                paid = calculator.weight * calculator.calculate(index, model, view, ctx)
                paid *= share * self._task_weight(ctx, name, index)
                total += paid
                breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, calculator in classes.event:
            for index in attackers:
                model = env.wargame_models[index]
                paid = calculator.weight * calculator.calculate(index, model, view, ctx)
                paid *= share * self._task_weight(ctx, name, index)
                total += paid
                breakdown[name] = breakdown.get(name, 0.0) + paid
        return total

    def _pay_close(
        self, terminated: bool, breakdown: dict[str, float]
    ) -> tuple[float, dict[int, float], float, dict[int, float]]:
        """The close's scalar, under `Credit.actor` the per-model credits the
        state terms owe each alive model, under `streams` the planning scalar
        (the outcome terms), which is otherwise 0.0 and inside the first, and
        under `PlanningCredit.counterfactual` the planning credit per unit
        (the scalar is then 0.0)."""
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
        outcome = 0.0
        credits: dict[int, float] = {}
        n_alive = int(player_alive.sum())
        per_model = self.credit is Credit.actor
        # Under `actor` every close payment is over the model count, a
        # constant; under `mean` a state term is the mean over the alive.
        common = 1.0 / len(env.wargame_models) if per_model else 1.0
        for name, calculator in classes.state:
            if n_alive == 0:
                continue
            summed = 0.0
            for index in np.flatnonzero(player_alive):
                model = env.wargame_models[int(index)]
                value = calculator.calculate(int(index), model, view, ctx)
                summed += value
                if per_model:
                    owed = calculator.weight * value * self.phase_scale * common
                    credits[int(index)] = credits.get(int(index), 0.0) + owed
            if per_model:
                paid = calculator.weight * summed * self.phase_scale * common
            else:
                paid = calculator.weight * summed / n_alive * self.phase_scale
                total += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, delta_global in classes.delta_globals:
            paid = delta_global.weight * delta_global.calculate(view, ctx) * common
            outcome += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        for name, state_global in classes.state_globals:
            paid = (
                state_global.weight
                * state_global.calculate(view, ctx)
                * self.phase_scale
                * common
            )
            outcome += paid
            breakdown[name] = breakdown.get(name, 0.0) + paid
        if terminated:
            for key, bonus in self.manager.terminal_bonuses(view, ctx).items():
                outcome += bonus * common
                breakdown[key] = breakdown.get(key, 0.0) + bonus * common
        self._baseline = self._read_baseline()
        if self.streams and self.planning_credit is PlanningCredit.counterfactual:
            planning_credits = self._counterfactual_credits(
                view, ctx, player_alive, terminated, common
            )
            return total, credits, 0.0, planning_credits
        if self.streams:
            return total, credits, outcome, {}
        return total + outcome, credits, 0.0, {}

    def _counterfactual_credits(
        self,
        view: BattleView,
        ctx: StepContext,
        player_alive: np.ndarray,
        terminated: bool,
        common: float,
    ) -> dict[int, float]:
        """Per living unit, the outcome terms with the unit in minus the same
        terms with its models masked out (#384 B6): the state globals and,
        on the terminating close, the terminal bonuses. The delta globals are
        broadcast and carried by nobody here (see `PlanningCredit`)."""
        env = self.env
        classes = self.classes
        with_unit = self._outcome_terms(view, ctx, terminated, common)
        groups = sorted({int(m.group_id) for m in env.wargame_models if m.is_alive})
        credits: dict[int, float] = {}
        for group in groups:
            without = np.array(
                [int(m.group_id) != group for m in env.wargame_models], dtype=bool
            )
            masked_ctx = self._context(
                action_phase=None,
                kills_by_model=None,
                terminated=terminated,
                alive_override=without & player_alive,
            )
            credits[group] = with_unit - self._outcome_terms(
                view, masked_ctx, terminated, common
            )
        del classes
        return credits

    def _outcome_terms(
        self, view: BattleView, ctx: StepContext, terminated: bool, common: float
    ) -> float:
        """The state globals and the terminal bonuses on `ctx`, as `_pay_close`
        sums them into the planning scalar (without the delta globals)."""
        outcome = 0.0
        for _name, state_global in self.classes.state_globals:
            outcome += (
                state_global.weight
                * state_global.calculate(view, ctx)
                * self.phase_scale
                * common
            )
        if terminated:
            for bonus in self.manager.terminal_bonuses(view, ctx).values():
                outcome += bonus * common
        return outcome

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
    "TASK_OF",
    "PaymentClass",
    "PerStepReward",
    "PlanningCredit",
    "StepPayment",
    "classify",
    "payment_class_of",
]


def _churn_share(env: PerModelEnv) -> float:
    """Re-commits before arrival on the turn just closed, over living units."""
    living = {int(m.group_id) for m in env.wargame_models if m.is_alive}
    if not living:
        return 0.0
    return float(env.player_commitments.churn_last_turn) / float(len(living))
