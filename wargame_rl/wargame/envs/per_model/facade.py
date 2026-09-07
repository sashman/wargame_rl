"""The per-model env facade: one step is one model's action.

A second facade over the same ``domain/`` layer (issue #284, parent #283). The
whole-phase facade (``WargameEnv``) is untouched and stays the default; this
one re-sequences the calls the engine already exposes so that:

- **a step is one model's action**, resolved before the next model acts. The
  policy chooses the order through a selector mask under the **unit lock**:
  once a unit's first member acts, only that unit's un-acted members are legal
  until the unit closes — which is the rules' own sequencing (a unit finishes
  before the next is selected), and it is what lets each phase's referee fire
  the moment a unit's move ends;
- **unit-level declarations ride on the unit's opening step** in the phase the
  rules put them in (move type in movement, shoot-or-not in shooting, the
  charge in the charge phase, activation priority in the fight phase). The
  command phase keeps nothing the agent decides and is stepped through without
  an agent step;
- **one turn-closing step per turn cycle**, after the opponent's turn, carries
  no model action and is where the state terms, the global terms and the
  turn's net victory points are paid (`reward_timing.py`, #286); a model step
  pays the action terms for the acting model alone. One scalar reward per
  step;
- **no play-time decode of any kind** — the top-K joint decode, the
  reallocation decode and the charge decode solved the product-policy problem
  this facade removes, and carrying one would hide whether a per-model policy
  has learned formation.

**Two clocks.** ``current_turn`` keeps counting stepped *phases* (so
``max_turns``, termination and every per-phase metric keep their meaning);
``model_steps`` counts per-model steps, and ``phase_model_steps`` counts them
within the current phase.

**The bridge.** A whole-phase script seated here through
``ScriptedPolicyAdapter`` reproduces the whole-phase facade's episode
bit-for-bit on the same layout and dice, wherever ``coherency.enforce_move``
is ``off`` — which is every shipped training config.
``tests/test_per_model_bridge.py`` pins it. On a refereed config the coherency
referee's *timing* is deliberately different: it fires per unit as the unit
closes (the rules' own timing) rather than over the whole force at the phase's
end, so episodes diverge exactly when a revert fires. See
``enforce_unit_after_move``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.coherency_enforcement import (
    CoherencyEnforcement,
    enforce_unit_after_move,
)
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.shooting import resolve_shooting_phase
from wargame_rl.wargame.envs.domain.sight import CLEAR
from wargame_rl.wargame.envs.domain.termination import is_battle_over
from wargame_rl.wargame.envs.domain.turn_execution import run_after_player_action
from wargame_rl.wargame.envs.env_components import compute_distances
from wargame_rl.wargame.envs.env_components.actions import (
    FIGHT_ORDER_LEVELS,
    MOVE_TYPE_ADVANCE,
    STAY_ACTION,
    MovePhaseContext,
)
from wargame_rl.wargame.envs.env_components.observation_builder import (
    compute_player_action_mask,
    update_distances_to_objectives,
)
from wargame_rl.wargame.envs.per_model.reward_timing import (
    PerModelRewardTimer,
    kills_by_model_for_step,
)
from wargame_rl.wargame.envs.per_model.types import (
    ChargeDeclaration,
    MoveDeclaration,
    PerModelAction,
    PerModelObservation,
    ShootDeclaration,
    StepKind,
)
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot
from wargame_rl.wargame.envs.types import (
    BattlePhase,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv

# Phases whose steps are one model's action. Command is deliberately absent —
# it keeps nothing the agent decides here (declarations moved to the phases the
# rules make them in) — and the fight phase is declaration-only: its single
# decision is the unit's activation priority, so a unit costs one step.
_PER_MODEL_PHASES: frozenset[BattlePhase] = frozenset(
    {
        BattlePhase.movement,
        BattlePhase.shooting,
        BattlePhase.charge,
        BattlePhase.pile_in,
        BattlePhase.consolidate,
    }
)


class PerModelEnv(WargameEnv):
    """``WargameEnv`` with per-model stepping. See the module docstring.

    Subclasses the whole-phase facade for its construction, reset, opponent
    execution, ``BattleView`` surface and snapshotting — the *stepping* is what
    this class replaces. The whole-phase ``step`` contract is overridden
    entirely; nothing here changes the parent's behaviour for other callers.
    """

    def __init__(
        self,
        config: WargameEnvConfig,
        record_model_steps: bool = False,
        **kwargs: Any,
    ) -> None:
        if bool(getattr(config, "declare_objectives", False)) or bool(
            getattr(config, "declare_targets", False)
        ):
            # The design leaves where the objective/hunt declaration lands as an
            # open question (they have no rules counterpart and no phase of
            # their own). Refusing is honest; silently dropping them would run
            # a scenario the config does not describe.
            raise ValueError(
                "PerModelEnv does not support declare_objectives / "
                "declare_targets yet — where a plan declaration lands under "
                "per-model stepping is an open design question (issue #283)."
            )
        super().__init__(config, **kwargs)
        n_models = config.number_of_wargame_models
        # Two clocks: `current_turn` (inherited) counts stepped phases; these
        # count per-model steps.
        self.model_steps = 0
        self.phase_model_steps = 0
        self._acted = np.ones(n_models, dtype=bool)
        self._open_unit: int | None = None
        self._declared: dict[int, int] = {}
        self._move_ctx: MovePhaseContext | None = None
        self._cover_alive: np.ndarray | None = None
        self._phase: BattlePhase = BattlePhase.movement
        self._phase_cleared = True
        self._phase_action_vector: list[int] = [STAY_ACTION] * n_models
        self._phase_opp_alive_before: list[bool] = []
        self._phase_player_alive_before: list[bool] = []
        self._fell_back_units: set[int] = set()
        self._intent_totals = (0, 0, 0)
        self._reverted_this_phase = 0
        self._pending_close = False
        self._terminal_close = False
        self._episode_over = False
        self._close_vp_base = (0, 0)
        self._enforce_mode = CoherencyEnforcement(config.coherency.enforce_move)
        # The re-timed reward (#286): action terms pay on the actor's step,
        # state terms / globals / VP on the turn-closing step. Same calculator
        # objects as the whole-phase manager — one set of weights and state.
        # The whole-phase facade steps once per non-skipped player phase, so
        # this is how many times it would evaluate every state term per round.
        stepped_phases_per_round = self.max_turns // config.number_of_battle_rounds
        self._reward_timer = PerModelRewardTimer(
            self.phase_manager, stepped_phases_per_round
        )
        self._cycle_player_damage = 0
        self._cycle_opponent_damage = 0
        self._cycle_player_kills = 0
        self._cycle_opponent_kills = 0
        self._cycle_kills_by_model = np.zeros(n_models, dtype=np.int64)
        self._step_kills = 0
        self._step_damage = 0
        # Recording cadence (#287): per PHASE BOUNDARY by default, which keeps
        # today's snapshot schema semantics; per model step for debugging. A
        # constructor argument, not a config field, for the same reason
        # `ThreatOptions` is not one: it changes what a recording looks like,
        # never what the scenario is.
        self._record_model_steps = bool(record_model_steps)
        self._coherency_nearest = self._rules_quantities.scale.to_units(
            config.coherency.nearest_distance
        )
        self._coherency_furthest = self._rules_quantities.scale.to_units(
            config.coherency.furthest_distance
        )

    # -- Episode flow ---------------------------------------------------------

    def reset(  # type: ignore[override]
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[PerModelObservation, dict[str, Any]]:
        """Start a new episode; land on the first per-model decision point."""
        _obs, info = super().reset(seed=seed, options=options)
        self.model_steps = 0
        self._pending_close = False
        self._terminal_close = False
        self._episode_over = False
        self._close_vp_base = (self.player_vp, self.opponent_vp)
        self._reset_cycle_tallies()
        self._advance_to_decision_point()
        return self._observe(), info

    def step(  # type: ignore[override]
        self, action: PerModelAction
    ) -> tuple[PerModelObservation, float, bool, bool, dict[str, Any]]:
        """One model's action — or, on a closing step, no action at all.

        A model step pays the action terms for the acting model alone; the
        closing step pays the state terms, the globals and the turn's net VP
        (`PerModelRewardTimer`). One scalar reward per step.
        """
        if self._episode_over:
            raise RuntimeError("Episode is over; call reset().")
        if self._pending_close:
            return self._close_turn()

        index = action.model_index
        if index is None:
            raise ValueError("model_index is required outside a turn-closing step.")
        selection = self._selection_mask()
        if not (0 <= index < len(selection)) or not selection[index]:
            raise ValueError(
                f"Model {index} is not selectable: alive un-acted models "
                f"{np.flatnonzero(selection).tolist()} (open unit "
                f"{self._open_unit})."
            )

        self._ensure_phase_cleared()
        model = self.wargame_models[index]
        unit = int(model.group_id)
        consumed_by_declaration: list[int] = []
        if unit not in self._declared:
            declaration = (
                action.declaration
                if action.declaration is not None
                else self._default_declaration()
            )
            consumed_by_declaration = self._apply_unit_declaration(unit, declaration)

        self._step_kills = 0
        self._step_damage = 0
        if not self._acted[index]:
            # A skip declaration (remain stationary / hold fire / decline /
            # fight priority) may have consumed the whole unit, opener
            # included; only a model the declaration left free acts.
            self._resolve_model_action(index, action.action)
            self._acted[index] = True
            self._phase_action_vector[index] = int(action.action)
        self.model_steps += 1
        self.phase_model_steps += 1

        # The opening step pays for every model it resolved: the actor, plus
        # any squadmates its declaration consumed — otherwise a skipped
        # member's action terms (a stay is charged, in this reward stack) are
        # never paid at all, and standing still is systematically cheaper here
        # than under the whole-phase facade.
        actors = [index] + [i for i in consumed_by_declaration if i != index]
        reward, breakdown = self._reward_timer.model_step_reward(
            self, self._model_step_context(index), actors
        )
        self._record_step_reward(reward, breakdown, actor=index)
        if self._record_model_steps:
            self._export_and_render_model_step()

        members = self._members_of(unit)
        if any(self.wargame_models[i].is_alive and not self._acted[i] for i in members):
            self._open_unit = unit
        else:
            self._close_unit(unit, members)
            self._open_unit = None

        if self._phase_exhausted():
            self._complete_player_phase()
            self._advance_to_decision_point()
        return self._observe(), reward, False, False, {}

    def _phase_exhausted(self) -> bool:
        """No decision left this phase: every model has acted or is dead.

        Deliberately NOT ``_acted.all()``: a model that dies un-acted
        mid-phase (no current mechanic does this — melee expansion will) is
        unselectable forever, and completing only on ``acted`` would deadlock
        the episode behind a misleading "not selectable" error.
        """
        return all(
            self._acted[i] or not model.is_alive
            for i, model in enumerate(self.wargame_models)
        )

    def _close_turn(
        self,
    ) -> tuple[PerModelObservation, float, bool, bool, dict[str, Any]]:
        """The turn-closing step: no model action; the turn's VP pays here.

        Placed after the opponent's turn, so `vp_gain` prices the *net* VP
        delta of the whole turn cycle — the view's deltas accumulate between
        closes and are reset only here. The cycle's kill and damage tallies
        feed the global event terms, and the terminal bonuses fire on the last
        round's close.
        """
        self._pending_close = False
        reward, breakdown = self._reward_timer.closing_reward(
            self, self._closing_context()
        )
        self._record_step_reward(reward, breakdown, actor=None)
        if self._record_model_steps:
            self._export_and_render_model_step()
        self._close_vp_base = (self.player_vp, self.opponent_vp)
        self._battle.reset_vp_deltas()
        self._reset_cycle_tallies()
        if self._terminal_close:
            self._episode_over = True
            return self._observe(), reward, True, False, {}
        self._advance_to_decision_point()
        return self._observe(), reward, False, False, {}

    def _advance_to_decision_point(self) -> None:
        """Advance until the policy is owed a decision, or the closing step."""
        while not self._pending_close:
            self._begin_player_phase()
            if self._phase_exhausted():
                # The command phase (no agent decision here), or a phase with
                # no alive model left to act: completes without an agent step.
                self._complete_player_phase()
                continue
            return

    # -- Phase lifecycle ------------------------------------------------------

    def _begin_player_phase(self) -> None:
        """Position the facade at the start of the clock's current player phase."""
        state = self._game_clock.state
        phase = state.phase or BattlePhase.movement
        # The side's turn-start rolls (advance D6, charge 2D6) and per-turn
        # state clear; idempotent, keyed on (round, side), exactly the points
        # the whole-phase facade guarantees them at.
        self._ensure_advance_rolls()
        # Scripts read `model.distances_to_objectives`; the whole-phase facade
        # refreshes them on every observation build, so refresh at the point a
        # policy plans from.
        update_distances_to_objectives(self.wargame_models, self.objectives, None)
        if self.opponent_models:
            update_distances_to_objectives(self.opponent_models, self.objectives, None)

        self._phase = phase
        self._phase_cleared = False
        self._phase_opp_alive_before = [m.is_alive for m in self.opponent_models]
        self._phase_player_alive_before = [m.is_alive for m in self.wargame_models]
        self._acted = np.array(
            [not m.is_alive for m in self.wargame_models], dtype=bool
        )
        self._open_unit = None
        self._declared = {}
        self._phase_action_vector = [STAY_ACTION] * len(self.wargame_models)
        self.phase_model_steps = 0
        self._fell_back_units = set()
        self._intent_totals = (0, 0, 0)
        self._reverted_this_phase = 0
        self._move_ctx = None
        self._cover_alive = None

        if phase not in _PER_MODEL_PHASES and phase is not BattlePhase.fight:
            # The command phase: stepped through without an agent step. Its old
            # action (all-STAY) declared nothing `begin_turn` had not already
            # cleared, so there is nothing to apply — only the phase-step
            # bookkeeping, which `_complete_player_phase` runs.
            self._acted[:] = True
            return
        if self._action_handler.displaces_in(phase):
            self._move_ctx = self._action_handler.begin_move_context(
                phase,
                self.wargame_models,
                self.opponent_models,
                self.board_width,
                self.board_height,
            )
        if phase is BattlePhase.shooting:
            # Cover is judged against the target unit's membership at the
            # phase's start, exactly as the whole-phase facade computes the
            # whole volley's cover in one batch before anything resolves.
            self._cover_alive = np.array(self._phase_opp_alive_before, dtype=bool)

    def _ensure_phase_cleared(self) -> None:
        """Clear the per-step state once, at the phase's first step.

        The whole-phase facade clears at the top of every ``step``; here that
        point is the phase's first model step (or its completion, for the
        command phase), so that between phases the previous boundary's results
        stay readable — which is when a whole-phase script plans.
        """
        if self._phase_cleared:
            return
        self._phase_cleared = True
        # VP deltas deliberately NOT reset here: they accumulate across the
        # turn cycle so the closing step's `vp_gain` prices the whole turn,
        # and `_close_turn` resets them after paying.
        self._last_player_shooting_results = []
        self._last_opponent_shooting_results = []
        self._last_player_fight_results = []
        self._last_opponent_fight_results = []
        self._attrition_deaths_player = 0
        self._attrition_deaths_opponent = 0

    def _complete_player_phase(self) -> None:
        """The phase's boundary: the whole-phase facade's ``step`` tail."""
        self._ensure_phase_cleared()
        phase = self._phase
        handler = self._action_handler

        if phase is BattlePhase.movement:
            handler.intended_coherency_last_move = self._intent_totals
            handler.models_reverted_last_move = self._reverted_this_phase
            self._apply_fall_back_marks()
        if phase in _PER_MODEL_PHASES and handler.displaces_in(phase):
            self._record_coherency(include_intent=phase is BattlePhase.movement)

        self._last_player_action = WargameEnvAction(
            actions=list(self._phase_action_vector)
        )
        self._last_action_phase = phase
        self.current_turn += 1

        previous_round = self._game_clock.state.battle_round
        run_after_player_action(
            self._game_clock,
            self._skip_phases,
            self._player_side,
            self._apply_opponent_action,
            on_before_advance=self._on_before_advance,
        )

        player_alive = alive_mask_for(self.wargame_models)
        needs_mm = self.phase_manager.needs_model_model_distances
        cache = compute_distances(
            self.wargame_models,
            self.objectives,
            compute_model_model=needs_mm,
            alive_mask=player_alive,
        )

        any_player_alive = player_alive.any()
        all_player_eliminated = (
            self.config.terminate_on_player_elimination and not any_player_alive
        )
        all_opponent_eliminated = bool(self.opponent_models) and all(
            not m.is_alive for m in self.opponent_models
        )
        all_eliminated = all_player_eliminated or all_opponent_eliminated

        clock_state = self._game_clock.state
        boundary_phase = clock_state.phase or BattlePhase.command

        p_dmg = sum(
            r.result.damage_dealt for r in self._last_player_shooting_results
        ) + sum(r.result.damage_dealt for r in self._last_player_fight_results)
        o_dmg = sum(
            r.result.damage_dealt for r in self._last_opponent_shooting_results
        ) + sum(r.result.damage_dealt for r in self._last_opponent_fight_results)
        p_kills = max(
            0,
            sum(
                1
                for i, m in enumerate(self.opponent_models)
                if i < len(self._phase_opp_alive_before)
                and self._phase_opp_alive_before[i]
                and not m.is_alive
            )
            - self._attrition_deaths_opponent,
        )
        o_kills = max(
            0,
            sum(
                1
                for i, m in enumerate(self.wargame_models)
                if i < len(self._phase_player_alive_before)
                and self._phase_player_alive_before[i]
                and not m.is_alive
            )
            - self._attrition_deaths_player,
        )
        p_kills_by_model = np.zeros(len(self.wargame_models), dtype=np.int64)
        for shot in self._last_player_shooting_results:
            if shot.killed and shot.attacker_idx < len(p_kills_by_model):
                p_kills_by_model[shot.attacker_idx] += 1
        for blow in self._last_player_fight_results:
            if blow.killed and blow.attacker_idx < len(p_kills_by_model):
                p_kills_by_model[blow.attacker_idx] += 1

        self._cycle_player_damage += p_dmg
        self._cycle_opponent_damage += o_dmg
        self._cycle_player_kills += p_kills
        self._cycle_opponent_kills += o_kills
        self._cycle_kills_by_model += p_kills_by_model

        ctx = StepContext(
            distance_cache=cache,
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            board_width=self.board_width,
            board_height=self.board_height,
            is_terminated=False,
            current_round=clock_state.battle_round or 0,
            battle_phase=boundary_phase,
            action_phase=self._last_action_phase,
            player_damage_dealt=p_dmg,
            opponent_damage_dealt=o_dmg,
            player_models_killed=p_kills,
            opponent_models_killed=o_kills,
            player_kills_by_model=p_kills_by_model,
        )
        succeeded = (
            bool(any_player_alive)
            and self.phase_manager.terminate_on_success
            and self.phase_manager.check_success(self, ctx)
        )
        is_terminated = is_battle_over(
            self._game_clock,
            self.current_turn,
            self.max_turns,
            succeeded,
            all_eliminated=all_eliminated,
        )
        self._last_terminated = is_terminated
        ctx.is_terminated = is_terminated
        self.last_step_context = ctx

        after = self._game_clock.state
        if is_terminated or after.battle_round != previous_round:
            # The player's turn cycle ended (the opponent's turn ran, or the
            # episode did): the turn-closing step is owed before the next
            # decision — and before a terminal episode ends.
            self._pending_close = True
            self._terminal_close = is_terminated

        # Recording cadence: per phase boundary, which keeps today's snapshot
        # schema semantics (#287 makes the cadence a setting).
        if self._state_exporters:
            snapshot = self.to_snapshot()
            for exporter in self._state_exporters:
                exporter.on_step(snapshot)
        if self.renderer is not None:
            self.renderer.render(self)

    # -- Declarations ---------------------------------------------------------

    def declare_for_phase(self, declarations: Mapping[int, int]) -> None:
        """Apply state-carrying declarations for the current phase, up front.

        The scripted adapter's path: a whole-phase script plans from a state in
        which every declaration is already made (the old facade's command step
        precedes its movement plan), so the adapter declares en bloc before
        planning. Game-equivalent to declaring on each unit's opening step —
        a declaration affects only its own unit's members' resolution.

        Only declarations that carry state and skip nobody are accepted
        (normal / advance / shoot / charge); skip declarations still take
        effect on the unit's opening step, where the members they consume are
        accounted for.
        """
        for unit, declaration in declarations.items():
            if unit in self._declared:
                raise ValueError(f"Unit {unit} has already declared this phase.")
            if self._phase is BattlePhase.movement:
                allowed = (MoveDeclaration.normal, MoveDeclaration.advance)
                if MoveDeclaration(declaration) not in allowed:
                    raise ValueError(
                        "declare_for_phase accepts only non-skip declarations."
                    )
            elif self._phase is BattlePhase.shooting:
                if ShootDeclaration(declaration) is not ShootDeclaration.shoot:
                    raise ValueError(
                        "declare_for_phase accepts only non-skip declarations."
                    )
            elif self._phase is BattlePhase.charge:
                if ChargeDeclaration(declaration) is not ChargeDeclaration.charge:
                    raise ValueError(
                        "declare_for_phase accepts only non-skip declarations."
                    )
            else:
                raise ValueError(
                    f"No en-bloc declarations in the {self._phase.value} phase."
                )
            self._apply_unit_declaration(unit, declaration)

    def _default_declaration(self) -> int:
        """What an opening step with no declaration means, per phase.

        Each default is the old facade's STAY behaviour: a normal move, a unit
        free to shoot, no charge, activation priority 0.
        """
        if self._phase is BattlePhase.movement:
            return int(MoveDeclaration.normal)
        if self._phase is BattlePhase.shooting:
            return int(ShootDeclaration.shoot)
        if self._phase is BattlePhase.charge:
            return int(ChargeDeclaration.decline)
        return 0

    def _apply_unit_declaration(self, unit: int, declaration: int) -> list[int]:
        """Record and apply one unit's declaration for the current phase.

        Returns the members the declaration consumed (auto-stayed), so the
        opening step can pay their action terms.
        """
        phase = self._phase
        members = self._members_of(unit)
        consumed: list[int] = []
        if phase is BattlePhase.movement:
            declared = MoveDeclaration(declaration)
            if declared is MoveDeclaration.advance:
                advance_slice = self._action_handler.advance_slice
                if advance_slice is None:
                    raise ValueError(
                        "This scenario registers no advance bins to declare."
                    )
                # The whole-phase facade's own gates, enforced where the COST
                # is paid: a darkened slice, an engaged unit or a roll with no
                # legal rung may not spend the unit's shooting here — the
                # declaration masks say the same, but a mask is advice and
                # this is the till.
                offset = self._action_handler.move_type_offset(MOVE_TYPE_ADVANCE)
                legality = (
                    self.player_declaration_legality[:, offset]
                    if offset is not None
                    else np.zeros(len(self.wargame_models), dtype=bool)
                )
                if BattlePhase.movement not in advance_slice.valid_phases or not any(
                    legality[i] for i in members
                ):
                    raise ValueError(
                        f"Unit {unit} may not declare an advance: the slice is "
                        "darkened, the unit is engaged, or this turn's roll "
                        "leaves no legal rung."
                    )
                # Mirrors `declare_move_types`: the whole unit is bound, and
                # the shooting is spent by the declaration itself.
                for i in members:
                    self.wargame_models[i].declared_advance = True
                    self.wargame_models[i].advanced_this_turn = True
            elif declared is MoveDeclaration.remain_stationary:
                consumed = self._auto_stay_members(members, resolve=True)
        elif phase is BattlePhase.shooting:
            if ShootDeclaration(declaration) is ShootDeclaration.hold_fire:
                consumed = self._auto_stay_members(members, resolve=False)
        elif phase is BattlePhase.charge:
            if ChargeDeclaration(declaration) is ChargeDeclaration.charge:
                for i in members:
                    self.wargame_models[i].declared_charge = True
            else:
                # A non-charging unit's only legal charge-phase action is to
                # stand still; resolving the STAY keeps `previous_location`
                # exactly as the whole-phase facade leaves it.
                consumed = self._auto_stay_members(members, resolve=True)
        elif phase is BattlePhase.fight:
            priority = min(max(int(declaration), 0), FIGHT_ORDER_LEVELS - 1)
            for i in members:
                self.wargame_models[i].fight_priority = priority
            # The priority is the fight phase's whole decision: one step per
            # unit, the engine resolves the blows on the boundary.
            consumed = self._auto_stay_members(members, resolve=False)
        else:
            if declaration != 0:
                raise ValueError(f"No declarations in the {phase.value} phase.")
        self._declared[unit] = int(declaration)
        return consumed

    def _auto_stay_members(self, members: list[int], *, resolve: bool) -> list[int]:
        """Consume a unit's remaining members without agent steps.

        ``resolve=True`` puts each alive member through the same STAY
        resolution the whole-phase facade runs (which writes
        ``previous_location`` — the fall-back inference and the referees read
        it), so a skipped unit's state is bit-identical to one whose members
        each chose STAY. Returns the members consumed.
        """
        consumed: list[int] = []
        for i in members:
            if self._acted[i]:
                continue
            if resolve and self._move_ctx is not None:
                self._action_handler.resolve_one_move(
                    self._move_ctx,
                    STAY_ACTION,
                    i,
                    self.wargame_models,
                    self._action_handler.action_space,
                )
            self._acted[i] = True
            self._phase_action_vector[i] = STAY_ACTION
            consumed.append(i)
        return consumed

    # -- Per-model resolution -------------------------------------------------

    def _resolve_model_action(self, index: int, act: int) -> None:
        """Resolve one model's action for the current phase, immediately."""
        phase = self._phase
        if phase is BattlePhase.shooting:
            self._resolve_model_shot(index, act)
            return
        if phase is BattlePhase.fight:
            # Declaration-only: the unit's priority was applied on its opening
            # step, and members are consumed there. Nothing to resolve.
            return
        if self._move_ctx is not None:
            self._action_handler.resolve_one_move(
                self._move_ctx,
                act,
                index,
                self.wargame_models,
                self._action_handler.action_space,
            )

    def _resolve_model_shot(self, index: int, act: int) -> None:
        """Resolve one model's declared shot, exactly as the batched phase does.

        Non-shooting actions are not shots (the whole-phase facade's decode
        ignores them); a shot at a unit that died earlier in the phase is lost,
        which is the one way the rules lose an attack.
        """
        shooting_slice = self._action_handler.shooting_slice
        if shooting_slice is None:
            return
        if not (shooting_slice.start <= act < shooting_slice.end):
            return
        target_group = act - shooting_slice.start
        shots = [(index, target_group)]
        results = resolve_shooting_phase(
            shots=shots,
            attackers=self.wargame_models,
            targets=self.opponent_models,
            attacker_weapons=[cfg.weapons for cfg in self.config.models or []],
            rng=self._combat_rng,
            cover=self._cover_for_shot(index, target_group),
        )
        self._last_player_shooting_results.extend(results)
        self._step_kills += sum(1 for r in results if r.killed)
        self._step_damage += sum(r.result.damage_dealt for r in results)

    def _cover_for_shot(
        self, attacker_idx: int, target_group: int
    ) -> np.ndarray | None:
        """The whole-phase facade's `_cover_mask`, for one declared pair.

        The target unit's membership is frozen at the phase's start
        (``_cover_alive``): the whole-phase facade traces the entire volley's
        cover in one batch before anything resolves, so a member killed
        mid-volley must not change a later shooter's cover answer.
        """
        targets = self.opponent_models
        if not targets or self._cover_alive is None:
            return None
        groups = np.array([m.group_id for m in targets], dtype=int)
        n_groups = int(groups.max()) + 1 if len(groups) else 0
        if not (0 <= target_group < n_groups):
            return None
        members = (groups == target_group) & self._cover_alive
        if not members.any():
            return None
        candidates = np.zeros((len(self.wargame_models), len(targets)), dtype=bool)
        candidates[attacker_idx, members] = True
        visibility = self.visibility_between(
            np.array([m.location for m in self.wargame_models], dtype=float),
            np.array([m.location for m in targets], dtype=float),
            candidates,
            origin_models=self.wargame_models,
            target_models=list(targets),
        )
        model_in_cover = visibility != CLEAR
        cover = np.zeros((len(self.wargame_models), n_groups), dtype=bool)
        cover[attacker_idx, target_group] = bool(
            model_in_cover[attacker_idx, members].all()
        )
        return cover

    def _close_unit(self, unit: int, members: list[int]) -> None:
        """Referee one unit the moment its last member has acted."""
        phase = self._phase
        if self._move_ctx is not None:
            self._action_handler.referee_unit_move(
                self._move_ctx, self.wargame_models, self.opponent_models, members
            )
        if phase is not BattlePhase.movement:
            return
        # Fall-back moved-ness is judged before the coherency referee edits the
        # board (the whole-phase facade marks before it enforces).
        began = self._move_ctx.began_engaged if self._move_ctx is not None else None
        if began is not None and began.any():
            for i in members:
                model = self.wargame_models[i]
                if not began[i] or not model.is_alive:
                    continue
                previous = getattr(model, "previous_location", None)
                if previous is not None and not np.array_equal(
                    previous, model.location
                ):
                    self._fell_back_units.add(unit)
                    break
        units, coherent, out = self._intent_totals
        unit_count, unit_coherent, unit_out = self._unit_intent(members)
        self._intent_totals = (
            units + unit_count,
            coherent + unit_coherent,
            out + unit_out,
        )
        self._reverted_this_phase += enforce_unit_after_move(
            self.wargame_models,
            members,
            self._coherency_nearest,
            self._coherency_furthest,
            self._enforce_mode,
        )

    def _model_step_context(self, actor: int) -> StepContext:
        """A per-model-step context: this step's own kills, fresh distances."""
        cache = compute_distances(
            self.wargame_models,
            self.objectives,
            compute_model_model=self.phase_manager.needs_model_model_distances,
            alive_mask=alive_mask_for(self.wargame_models),
        )
        state = self._game_clock.state
        return StepContext(
            distance_cache=cache,
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            board_width=self.board_width,
            board_height=self.board_height,
            is_terminated=False,
            current_round=state.battle_round or 0,
            battle_phase=self._phase,
            action_phase=self._phase,
            player_damage_dealt=self._step_damage,
            opponent_damage_dealt=0,
            player_models_killed=self._step_kills,
            opponent_models_killed=0,
            player_kills_by_model=kills_by_model_for_step(
                len(self.wargame_models), actor, self._step_kills
            ),
        )

    def _closing_context(self) -> StepContext:
        """The turn-closing step's context: the whole cycle's tallies."""
        cache = compute_distances(
            self.wargame_models,
            self.objectives,
            compute_model_model=self.phase_manager.needs_model_model_distances,
            alive_mask=alive_mask_for(self.wargame_models),
        )
        state = self._game_clock.state
        return StepContext(
            distance_cache=cache,
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            board_width=self.board_width,
            board_height=self.board_height,
            is_terminated=self._terminal_close,
            current_round=state.battle_round or self.n_rounds,
            battle_phase=state.phase or BattlePhase.command,
            action_phase=self._last_action_phase,
            player_damage_dealt=self._cycle_player_damage,
            opponent_damage_dealt=self._cycle_opponent_damage,
            player_models_killed=self._cycle_player_kills,
            opponent_models_killed=self._cycle_opponent_kills,
            player_kills_by_model=self._cycle_kills_by_model.copy(),
        )

    def _record_step_reward(
        self, reward: float, breakdown: dict[str, float], actor: int | None
    ) -> None:
        """Keep the step's payment where tooling already looks for it."""
        self.last_reward = reward
        self.last_reward_breakdown = dict(breakdown)
        per_model = np.zeros(len(self.wargame_models), dtype=np.float64)
        if actor is not None:
            per_model[actor] = reward
        self.last_per_model_reward = per_model
        for key, value in breakdown.items():
            self.episode_reward_breakdown[key] = (
                self.episode_reward_breakdown.get(key, 0.0) + value
            )
        self.episode_reward_steps += 1
        self.episode_reward += reward

    def _reset_cycle_tallies(self) -> None:
        self._cycle_player_damage = 0
        self._cycle_opponent_damage = 0
        self._cycle_player_kills = 0
        self._cycle_opponent_kills = 0
        self._cycle_kills_by_model = np.zeros(len(self.wargame_models), dtype=np.int64)

    def _apply_fall_back_marks(self) -> None:
        """Mark every unit that began engaged and moved as having fallen back."""
        if not self._fell_back_units:
            return
        for model in self.wargame_models:
            if int(model.group_id) in self._fell_back_units:
                model.fell_back_this_turn = True

    def _unit_intent(self, members: list[int]) -> tuple[int, int, int]:
        """(units, coherent units, models out) for one unit's completed move.

        The per-unit share of the whole-force `_intent_counts`: coherency is a
        per-unit property, so accumulating these over closing units equals the
        whole-force evaluation at the phase's end.
        """
        unit_models = [self.wargame_models[i] for i in members]
        alive = np.array([m.is_alive for m in unit_models], dtype=bool)
        if not alive.any():
            return (0, 0, 0)
        report = evaluate_coherency(
            positions=np.array([m.location for m in unit_models], dtype=float),
            group_ids=np.zeros(len(unit_models), dtype=np.intp),
            alive_mask=alive,
            base_radii=np.array([m.base_radius for m in unit_models], dtype=float),
            nearest_distance=self._coherency_nearest,
            furthest_distance=self._coherency_furthest,
        )
        coherent = sum(1 for unit in report.units if unit.coherent)
        return (len(report.units), coherent, report.n_models_out_of_coherency)

    # -- Selection and observation --------------------------------------------

    def _members_of(self, unit: int) -> list[int]:
        """Model indices belonging to one unit, in canonical order."""
        return [i for i, m in enumerate(self.wargame_models) if int(m.group_id) == unit]

    def _selection_mask(self) -> np.ndarray:
        """Who may act now: alive, un-acted, and inside the open unit if any."""
        mask = alive_mask_for(self.wargame_models) & ~self._acted
        if self._open_unit is not None:
            in_unit = np.array(
                [int(m.group_id) == self._open_unit for m in self.wargame_models],
                dtype=bool,
            )
            mask &= in_unit
        return mask

    def unit_needs_declaration(self, model_index: int) -> bool:
        """True when selecting this model would open its unit's declaration.

        What an agent needs to know to include the declaration factor in a
        step's log-prob: the declaration rides on the unit's opening step, and
        this is the public form of "has this unit declared this phase".
        """
        if self._pending_close or self._episode_over:
            return False
        unit = int(self.wargame_models[model_index].group_id)
        return unit not in self._declared

    def current_action_mask(self) -> np.ndarray:
        """``(n_models, n_actions)`` legality for the current phase.

        Exactly the mask the whole-phase facade puts on its observation — the
        same function computes both.
        """
        return compute_player_action_mask(self, self._action_handler.registry)

    @property
    def turn_cycle_vp_delta(self) -> tuple[int, int]:
        """(player, opponent) VP gained since the last turn-closing step.

        What the closing step will pay on once the per-model reward timing
        lands (#286).
        """
        return (
            self.player_vp - self._close_vp_base[0],
            self.opponent_vp - self._close_vp_base[1],
        )

    def _export_and_render_model_step(self) -> None:
        """The per-model-step recording cadence: one snapshot per model step.

        Additive to the phase-boundary cadence, which still fires — a boundary
        snapshot is the only one that carries the opponent's turn.
        """
        if self._state_exporters:
            snapshot = self.to_snapshot()
            for exporter in self._state_exporters:
                exporter.on_step(snapshot)
        if self.renderer is not None:
            self.renderer.render(self)

    def to_snapshot(self) -> GameStateSnapshot:
        """The whole-phase snapshot plus the second clock (schema 2.8)."""
        snapshot: GameStateSnapshot = (
            super().to_snapshot().model_copy(update={"model_step": self.model_steps})
        )
        return snapshot

    def _observe(self) -> PerModelObservation:
        state = self._game_clock.state
        if self._pending_close or self._episode_over:
            n_models = len(self.wargame_models)
            return PerModelObservation(
                kind=StepKind.turn_close,
                phase=state.phase,
                battle_round=state.battle_round,
                model_steps=self.model_steps,
                phase_model_steps=self.phase_model_steps,
                selection_mask=np.zeros(n_models, dtype=bool),
                acted_mask=self._acted.copy(),
                open_unit=None,
            )
        return PerModelObservation(
            kind=StepKind.model_action,
            phase=self._phase,
            battle_round=state.battle_round,
            model_steps=self.model_steps,
            phase_model_steps=self.phase_model_steps,
            selection_mask=self._selection_mask(),
            acted_mask=self._acted.copy(),
            open_unit=self._open_unit,
        )
