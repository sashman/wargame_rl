"""The per-model facade: one env step is one decision, over the unchanged domain.

`PerModelEnv` is the second application context over `domain/`. It composes
the same aggregate, clock, handlers, reward manager and VP calculator the
phase facade does, and steps them one DECISION at a time: open a unit with a
declaration, act with one model, name a charge target, or close the turn. The
phase programs (`phases.py`) say what each decision does; this class drives
them across the clock, settles the reward windows, runs the boundary hook the
phase facade runs on every phase transition, and offers `BattleView` to reward,
mission and scripts.

Reward keeps the phase facade's TIMING in this stage: one `StepContext`, built
exactly as `WargameEnv.step` builds it, at the point the phase facade's step
would have ended -- the next of our stepped phases opening, or the closing
step after the opponent's turn. Model steps return 0.0. Stage 3 re-times it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np

from wargame_rl.wargame.envs.domain.battle_factory import from_config, unit_count
from wargame_rl.wargame.envs.domain.battlefield.placement import (
    install_layout,
    place_for_episode,
)
from wargame_rl.wargame.envs.domain.battlefield.sight import (
    COVER,
    has_line_of_sight_between_points,
    visibility_matrix,
)
from wargame_rl.wargame.envs.domain.battlefield.terrain import Terrain
from wargame_rl.wargame.envs.domain.battlefield.terrain_placement import (
    generate_terrain,
)
from wargame_rl.wargame.envs.domain.kernel.dice import (
    DiceCall,
    DicePurpose,
    DiceSource,
    RollerAdapter,
)
from wargame_rl.wargame.envs.domain.kernel.entities import (
    WargameModel,
    WargameObjective,
    alive_mask_for,
)
from wargame_rl.wargame.envs.domain.kernel.rules_quantities import (
    RulesQuantities,
    resolve_rules_quantities,
)
from wargame_rl.wargame.envs.domain.kernel.value_objects import BoardDimensions
from wargame_rl.wargame.envs.domain.melee.fight import (
    PASS_RANGE_INCHES,
    FightSide,
    OverrunRules,
    PairedFightResult,
    fight_dragged_in_units,
    fight_eligible_units,
    resolve_fight,
    resolve_fight_step,
)
from wargame_rl.wargame.envs.domain.melee.pile_in import SELECTION_RANGE_INCHES
from wargame_rl.wargame.envs.domain.movement.coherency_enforcement import (
    CoherencyEnforcement,
    apply_attrition,
)
from wargame_rl.wargame.envs.domain.sequencing.game_clock import GameClock
from wargame_rl.wargame.envs.domain.sequencing.termination import is_battle_over
from wargame_rl.wargame.envs.domain.shooting.resolve import PairedShootingResult
from wargame_rl.wargame.envs.domain.shooting.targets import max_weapon_ranges
from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.map_pool import MapPool
from wargame_rl.wargame.envs.mission import build_vp_calculator
from wargame_rl.wargame.envs.opponent.registry import (
    _auto_register,
    build_opponent_policy,
)
from wargame_rl.wargame.envs.per_model.dice_seeded import SeededDice
from wargame_rl.wargame.envs.per_model.observation import build_per_model_observation
from wargame_rl.wargame.envs.per_model.phases import (
    ChargePhase,
    FightPhase,
    MovementPhase,
    PhaseProgram,
    ShootingPhase,
    ShortMovePhase,
)
from wargame_rl.wargame.envs.per_model.scripted import ScriptedSeat
from wargame_rl.wargame.envs.per_model.seat import Seat
from wargame_rl.wargame.envs.per_model.types import (
    FACADE_TAG,
    DecisionPoint,
    FacadeDivergence,
    PerModelAction,
    PerModelObservation,
    PerModelProvenance,
    StepKind,
)
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import (
    BattlePhase,
    PlayerSide,
    TurnOrder,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER, GameState
from wargame_rl.wargame.envs.types.geometry import Polygon

_auto_register()

DiceFactory = Callable[[int, "PerModelEnv"], DiceSource]

# Config switches the per-model facade does not carry yet. Refused rather than
# ignored: a run that silently trained without a declared feature would
# measure a scenario it did not have.
_DEFERRED_SWITCHES = (
    "declare_objectives",
    "declare_targets",
    "hunt_declares_charge",
    "charge_target_binds",
)


@dataclass
class _Window:
    """One of our stepped phases, from its open to the phase facade's step end."""

    phase: BattlePhase
    turn_key: tuple[int, PlayerSide]
    opponent_alive_before: list[bool]
    player_alive_before: list[bool]


@dataclass(frozen=True)
class SettledWindow:
    """What one settled reward window paid, for `info["settled"]`."""

    phase: str
    reward: float
    terminated: bool
    breakdown: dict[str, float]
    per_model: np.ndarray
    # The board as it stood when the window settled -- what the phase facade's
    # step would have returned. Read by the bridge test; the env moves on.
    state: dict[str, Any]


class EpisodeOver(RuntimeError):
    """`step` was called after the episode terminated."""


def default_dice(combat_seed: int, env: PerModelEnv) -> DiceSource:
    """The phase facade's three streams, per-side pre-draw included."""
    return SeededDice(
        combat_seed,
        units_by_side=lambda side: env.seat_for_side(side).units_first_appearance(),
        has_advance=env.player_action_handler.advance_slice is not None,
        has_melee=env.config.melee.enabled,
    )


class PerModelEnv(gym.Env):
    """One decision per step, over the unchanged domain layer."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: WargameEnvConfig,
        *,
        dice_factory: DiceFactory | None = None,
    ) -> None:
        for switch in _DEFERRED_SWITCHES:
            if bool(getattr(config, switch, False)):
                raise ValueError(
                    f"`{switch}` is not carried by the per-model facade yet"
                )
        self.config = config
        self.board_width = config.board_width
        self.board_height = config.board_height
        self.driver_label: str | None = None
        self._skip_phases = frozenset(config.skip_phases)
        self._rules_quantities = resolve_rules_quantities(config)
        self.coherency_mode = CoherencyEnforcement(config.coherency.enforce_move)
        self._coherency_attrition = config.coherency.attrition
        self._dice_factory: DiceFactory = dice_factory or default_dice

        self._action_handler = ActionHandler(
            config,
            n_shoot_targets=unit_count(
                config.number_of_opponent_models,
                config.max_groups,
                config.opponent_models,
            ),
            model_moves=[model.move for model in config.models or ()],
        )
        if config.number_of_opponent_models > 0:
            self._opponent_action_handler = ActionHandler(
                config,
                n_models=config.number_of_opponent_models,
                n_shoot_targets=unit_count(
                    config.number_of_wargame_models,
                    config.max_groups,
                    config.models,
                ),
                model_moves=[model.move for model in config.opponent_models or ()],
            )
        else:
            self._opponent_action_handler = ActionHandler(config, n_models=0)

        self.current_turn = 0
        self.sub_step = 0
        self.episode_step = 0
        self._player_side = self._initial_player_side()
        self._game_clock = GameClock(n_rounds=config.number_of_battle_rounds)
        self._battle = from_config(config)
        self._map_pool = MapPool.from_config(config)
        self._map_name: str | None = None
        if self._map_pool is not None:
            layout = self._map_pool.draw(self.np_random)
            install_layout(self._battle, config, layout)
            self._map_name = layout.name
        if config.random_terrain is not None:
            self._battle.set_terrain(
                generate_terrain(
                    config.random_terrain,
                    BoardDimensions(
                        width=self._battle.board_width,
                        height=self._battle.board_height,
                    ),
                    self.np_random,
                )
            )
        self.wargame_models = self._battle.player_models
        self.objectives = self._battle.objectives
        self.opponent_models = self._battle.opponent_models
        self.deployment_zone = self._battle.deployment_zone
        self.opponent_deployment_zone = self._battle.opponent_deployment_zone
        self._player_max_ranges = max_weapon_ranges(
            config.models, config.number_of_wargame_models
        )
        self._opponent_max_ranges = max_weapon_ranges(
            config.opponent_models, config.number_of_opponent_models
        )
        self._vp_calculator = build_vp_calculator(
            config.mission.type, config.mission.params
        )
        self.phase_manager = RewardPhaseManager.from_configs(config.reward_phases)

        self._player_seat = Seat(
            is_player=True,
            models=self.wargame_models,
            enemies=self.opponent_models,
            handler=self._action_handler,
            ranged_weapons=[cfg.weapons for cfg in config.models or []],
            melee_weapons=[cfg.melee_weapons for cfg in config.models or []],
            max_ranges=self._player_max_ranges,
            enemy_max_ranges=self._opponent_max_ranges,
        )
        self._opponent_seat = Seat(
            is_player=False,
            models=self.opponent_models,
            enemies=self.wargame_models,
            handler=self._opponent_action_handler,
            ranged_weapons=[cfg.weapons for cfg in config.opponent_models or []],
            melee_weapons=[cfg.melee_weapons for cfg in config.opponent_models or []],
            max_ranges=self._opponent_max_ranges,
            enemy_max_ranges=self._player_max_ranges,
        )
        self._opponent_policy = None
        if config.number_of_opponent_models > 0:
            self._opponent_policy = build_opponent_policy(
                config.opponent_policy,  # type: ignore[arg-type]
                cast(Any, self),
            )
            self._opponent_seat.adapter = ScriptedSeat(
                self._opponent_policy, shoots=bool(self._opponent_policy.shoots)
            )

        # Episode state, set on every reset.
        self._dice: DiceSource | None = None
        self._episode_rng_state: dict[str, Any] = {}
        self._episode_combat_seed = 0
        self._episode_seed: int | None = None
        self._rolled_for: tuple[int, PlayerSide] | None = None
        self._planned_command: tuple[int, PlayerSide] | None = None
        self._program: PhaseProgram | None = None
        self._pending: DecisionPoint | None = None
        self._window: _Window | None = None
        self._terminated = False
        self._attrition_deaths_player = 0
        self._attrition_deaths_opponent = 0
        self._fought_by_seat: dict[bool, set[int]] = {}
        self.divergences: list[FacadeDivergence] = []
        self.last_reward: float | None = None
        self.last_reward_breakdown: dict[str, float] = {}
        self.last_per_model_reward = np.zeros(
            config.number_of_wargame_models, dtype=np.float64
        )
        self.last_step_context: StepContext | None = None
        self.episode_reward_breakdown: dict[str, float] = {}
        self.episode_reward_steps = 0
        self.episode_reward = 0.0

    # ------------------------------------------------------------------ seats

    @property
    def player_seat(self) -> Seat:
        return self._player_seat

    @property
    def opponent_seat(self) -> Seat:
        return self._opponent_seat

    def seat_for_side(self, side: PlayerSide) -> Seat:
        """The seat the clock's `side` belongs to this episode."""
        return self._player_seat if side == self._player_side else self._opponent_seat

    def side_of(self, seat: Seat) -> PlayerSide:
        if seat.is_player:
            return self._player_side
        return (
            PlayerSide.player_2
            if self._player_side == PlayerSide.player_1
            else PlayerSide.player_1
        )

    def set_player_planner(self, planner: ScriptedSeat | None) -> None:
        """Let a whole-phase script plan the player's phases (the bridge test).

        The player's decisions still come through `step`; only the planning
        hook at each phase open is taken by the script.
        """
        self._player_seat.adapter = planner

    # ------------------------------------------------------------------ dice

    @property
    def dice(self) -> DiceSource:
        if self._dice is None:
            raise RuntimeError("reset() has not been called")
        return self._dice

    def roller(
        self,
        purpose: DicePurpose,
        seat: Seat,
        *,
        unit: int | None,
        model: int | None,
    ) -> np.random.Generator:
        """A generator-shaped handle on the dice source, tagged with its purpose."""
        state = self._game_clock.state
        call = DiceCall(
            purpose=purpose,
            side=self.side_of(seat),
            battle_round=state.battle_round,
            unit=unit,
            model=model,
        )
        return cast(np.random.Generator, RollerAdapter(self.dice, call))

    def reveal_advance_roll(self, seat: Seat, unit: int) -> None:
        """Materialise the unit's advance D6 on its models."""
        state = self._game_clock.state
        roll = self.dice.advance_roll(
            self.side_of(seat), int(state.battle_round or 0), unit
        )
        for index in seat.unit_members(unit, alive_only=False):
            seat.models[index].advance_roll = roll

    def reveal_charge_roll(self, seat: Seat, unit: int) -> None:
        """Materialise the unit's charge 2D6 on its models."""
        state = self._game_clock.state
        roll = self.dice.charge_roll(
            self.side_of(seat), int(state.battle_round or 0), unit
        )
        for index in seat.unit_members(unit, alive_only=False):
            seat.models[index].charge_roll = roll

    # ------------------------------------------------------------------ reset

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[PerModelObservation, dict[str, Any]]:
        """Start a new episode; the draw order is the phase facade's."""
        super().reset(seed=seed)
        self._episode_rng_state = dict(self.np_random.bit_generator.state)
        self._episode_seed = seed
        derived_combat_seed = int(self.np_random.integers(0, 2**31))
        explicit = (options or {}).get("combat_seed")
        self._episode_combat_seed = (
            derived_combat_seed if explicit is None else int(explicit)
        )
        self._dice = self._dice_factory(self._episode_combat_seed, self)
        for seat in (self._player_seat, self._opponent_seat):
            seat.shooting_results = []
            seat.fight_results = []
        self._rolled_for = None
        self._planned_command = None
        self._program = None
        self._pending = None
        self._window = None
        self._terminated = False
        self._fought_by_seat = {}
        self.divergences = []
        self.current_turn = 0
        self.sub_step = 0
        self.episode_step = 0
        self.last_reward = None
        self.last_step_context = None
        self.last_reward_breakdown = {}
        self.episode_reward_breakdown = {}
        self.episode_reward_steps = 0
        self.episode_reward = 0.0

        self._battle.reset_for_episode()
        self.phase_manager.reset_episode()
        self._resolve_player_side()
        self._game_clock.reset()
        self._game_clock.skip_setup()

        layout = None
        if self._map_pool is not None:
            layout = self._map_pool.draw(self.np_random)
            self._map_name = layout.name
        place_for_episode(
            self._battle,
            self.config,
            self.np_random,
            augment_start=bool((options or {}).get("augment_start", False)),
            layout=layout,
        )
        observation, _reward, _terminated, _truncated, info = self._advance()
        return observation, info

    # ------------------------------------------------------------------ step

    def step(
        self, action: PerModelAction
    ) -> tuple[PerModelObservation, float, bool, bool, dict[str, Any]]:
        """Resolve one decision and run the game to the next one."""
        if self._terminated:
            raise EpisodeOver("the episode has terminated; call reset()")
        point = self._pending
        if point is None:
            raise RuntimeError("reset() has not been called")
        reason = point.why_illegal(action)
        if reason is not None:
            raise ValueError(reason)
        self.episode_step += 1
        if point.kind is StepKind.close_turn:
            settled = self._settle_window()
            if settled.terminated or self._game_clock.is_game_over:
                self._terminated = True
                closing = self._closing_point()
                self._pending = closing
                return (
                    build_per_model_observation(self, closing),
                    settled.reward,
                    True,
                    False,
                    self._info([settled]),
                )
            observation, reward, terminated, truncated, info = self._advance()
            info["settled"] = [settled] + info["settled"]
            info["reward_settled"] = True
            return observation, reward + settled.reward, terminated, truncated, info
        assert self._program is not None
        self._program.apply(point, action)
        self.sub_step += 1
        return self._advance()

    # ------------------------------------------------------------------ driver

    def _advance(
        self,
    ) -> tuple[PerModelObservation, float, bool, bool, dict[str, Any]]:
        reward = 0.0
        settled: list[SettledWindow] = []
        terminated = False
        while True:
            if self._program is not None:
                point = self._program.next_decision()
                if point is not None:
                    seat = (
                        self._player_seat
                        if point.seat_is_player
                        else self._opponent_seat
                    )
                    if not seat.is_player:
                        adapter = seat.adapter
                        if adapter is None:
                            raise RuntimeError("the opponent seat has no policy")
                        self._program.apply(point, adapter.choose(point, seat))
                        continue
                    self._pending = point
                    break
                self._program.close()
                self._program = None
                self._leave_phase()
            self._settle_clock()
            if self._game_clock.is_game_over:
                if self._window is None:
                    terminated = True
                    self._terminated = True
                self._pending = self._closing_point()
                break
            state = self._game_clock.state
            ours = state.active_player == self._player_side
            if ours and self._window is not None:
                if self._window.turn_key != self._turn_key(state):
                    self._pending = self._closing_point()
                    break
                outcome = self._settle_window()
                reward += outcome.reward
                settled.append(outcome)
                if outcome.terminated:
                    terminated = True
                    self._terminated = True
                    self._pending = self._closing_point()
                    break
            if ours:
                self._open_window(state)
            if state.phase is BattlePhase.command:
                self._enter_command(state)
                self._leave_phase()
                continue
            self._program = self._build_program(state)
            self._program.open()
        assert self._pending is not None
        info = self._info(settled)
        info["reward_settled"] = bool(settled)
        return (
            build_per_model_observation(self, self._pending),
            reward,
            terminated,
            False,
            info,
        )

    def _settle_clock(self) -> None:
        """Pass through the phases nobody decides in, until one that is stepped."""
        while not self._game_clock.is_game_over:
            state = self._game_clock.state
            phase = state.phase
            ours = state.active_player == self._player_side
            if phase is BattlePhase.command:
                if ours and phase not in self._skip_phases:
                    return
                self._enter_command(state)
                self._leave_phase()
                continue
            if phase in self._skip_phases:
                self._leave_phase()
                continue
            return

    def _enter_command(self, state: GameState) -> None:
        """A side's turn begins: clear its per-turn state and let a script declare."""
        side = state.active_player
        battle_round = state.battle_round
        if side is None or battle_round is None:
            return
        key = (int(battle_round), side)
        if self._rolled_for != key:
            self._rolled_for = key
            seat = self.seat_for_side(side)
            for model in seat.models:
                model.begin_turn()
            if seat.adapter is not None:
                # A script plans a whole phase against its dice; the phase facade
                # rolled a side's dice at the start of its turn, so a script's
                # seat asks for them now.
                for unit in seat.units_first_appearance():
                    self.reveal_advance_roll(seat, unit)
                    self.reveal_charge_roll(seat, unit)
        if self._planned_command != key:
            self._planned_command = key
            seat = self.seat_for_side(side)
            if seat.adapter is not None:
                seat.adapter.plan(BattlePhase.command, seat, self)

    def _leave_phase(self) -> None:
        state = self._game_clock.state
        self._on_leaving(state)
        if (
            state.active_player == self._player_side
            and state.phase is not None
            and state.phase not in self._skip_phases
        ):
            self.current_turn += 1
        self._game_clock.advance_phase()

    def _build_program(self, state: GameState) -> PhaseProgram:
        assert state.active_player is not None and state.phase is not None
        active = self.seat_for_side(state.active_player)
        other = self._opponent_seat if active.is_player else self._player_seat
        phase = state.phase
        if phase is BattlePhase.movement:
            return MovementPhase(self, active)
        if phase is BattlePhase.shooting:
            return ShootingPhase(self, active)
        if phase is BattlePhase.charge:
            return ChargePhase(self, active)
        if phase in (BattlePhase.pile_in, BattlePhase.consolidate):
            return ShortMovePhase(self, phase, (active, other))
        if phase is BattlePhase.fight:
            return FightPhase(self, (active, other))
        raise ValueError(f"no program for phase {phase}")

    def _turn_key(self, state: GameState) -> tuple[int, PlayerSide]:
        assert state.active_player is not None
        return (int(state.battle_round or 0), state.active_player)

    def _closing_point(self) -> DecisionPoint:
        return DecisionPoint.closing(
            self._player_seat.n_models,
            self._action_handler.n_actions,
            self._player_seat.n_enemy_units(),
        )

    # ------------------------------------------------------------------ hooks

    def _on_leaving(self, state: GameState) -> None:
        """The phase facade's boundary hook, in its order."""
        if state.phase is BattlePhase.fight and BattlePhase.fight in self._skip_phases:
            self._resolve_fight_engine(state)
        if state.phase is BattlePhase.consolidate:
            self._end_fight_phase()
        if state.phase is BATTLE_PHASE_ORDER[-1]:
            self._regain_coherency(state)
        if state.phase is not BattlePhase.command or state.battle_round is None:
            return
        if state.active_player is None:
            return
        vp = self._vp_calculator.compute_vp(
            cast(Any, self), state.active_player, state.battle_round, self._player_side
        )
        if vp <= 0:
            return
        if state.active_player == self._player_side:
            self._battle.add_player_vp(vp)
        else:
            self._battle.add_opponent_vp(vp)

    def _seat_order(self, state: GameState) -> tuple[Seat, Seat]:
        assert state.active_player is not None
        active = self.seat_for_side(state.active_player)
        other = self._opponent_seat if active.is_player else self._player_seat
        return active, other

    def _resolve_fight_engine(self, state: GameState) -> None:
        """The whole fight step by the engine, when `fight` is a skipped phase."""
        if not self.config.melee.enabled or state.active_player is None:
            return
        quantities = self._rules_quantities
        engagement_range = quantities.engagement_range
        base_diameter = 2.0 * quantities.base_radius
        order = self._seat_order(state)
        fought: tuple[set[int], set[int]] = (set(), set())
        if self.config.melee.alternating_activation:
            first, second = order
            eligible = tuple(
                set(
                    fight_eligible_units(
                        seat.models,
                        seat.enemies,
                        engagement_range=engagement_range,
                        base_diameter=base_diameter,
                    )
                )
                for seat in order
            )
            side_results = resolve_fight_step(
                (
                    FightSide(models=first.models, weapons=first.melee_weapons),
                    FightSide(models=second.models, weapons=second.melee_weapons),
                ),
                self.roller(DicePurpose.melee, first, unit=None, model=None),
                engagement_range=engagement_range,
                base_diameter=base_diameter,
                pass_range=quantities.scale.to_units(PASS_RANGE_INCHES),
                started_eligible=(eligible[0], eligible[1]),
                overrun=self._overrun_rules() if self.config.melee.overrun else None,
                record_fought=fought,
            )
            for seat, results in zip(order, side_results, strict=True):
                seat.fight_results.extend(results)
        else:
            for seat in order:
                seat.fight_results.extend(
                    resolve_fight(
                        seat.models,
                        seat.enemies,
                        self.roller(DicePurpose.melee, seat, unit=None, model=None),
                        attacker_weapons=seat.melee_weapons,
                        engagement_range=engagement_range,
                        base_diameter=base_diameter,
                    )
                )
        self.carry_fight_to_consolidate(order, (set(fought[0]), set(fought[1])))

    def _overrun_rules(self) -> OverrunRules:
        quantities = self._rules_quantities
        return OverrunRules(
            pile_in_distance=quantities.scale.to_units(
                self.config.melee.pile_in_distance
            ),
            selection_range=quantities.scale.to_units(SELECTION_RANGE_INCHES),
            base_radius=quantities.base_radius,
            board=(float(self.board_width), float(self.board_height)),
            coherency_nearest=quantities.scale.to_units(
                self.config.coherency.nearest_distance
            ),
            coherency_furthest=quantities.scale.to_units(
                self.config.coherency.furthest_distance
            ),
        )

    def _end_fight_phase(self) -> None:
        """End of the rules' fight PHASE: the charge flags of both forces expire.

        ⚠ `BattlePhase.pile_in`, `fight` and `consolidate` are three STEPS of one
        rules phase (`12-fight-phase.md`) that the clock carries as three phases.
        The phase facade clears `charged_this_turn` when its `fight` phase
        closes -- the end of the fight STEP -- and its consolidate step then reads
        the flag as False, so "made a charge move this turn" never makes a
        unit eligible to consolidate there. Here the flag lives until the
        consolidate step is left, which is where the rules' phase ends. It must
        not survive beyond it: `begin_turn` clears only the side whose turn is
        starting, so the charging force's flag would otherwise reach the
        opponent's fight step and buy a Strikes First it did not earn.
        """
        for seat in (self._player_seat, self._opponent_seat):
            for model in seat.models:
                model.charged_this_turn = False
                # "Was eligible to fight THIS phase" -- a unit that fought in our
                # fight phase is not thereby eligible to consolidate in the
                # opponent's; `begin_turn` would only clear it at our next turn.
                model.fought_this_phase = False

    def carry_fight_to_consolidate(
        self, order: tuple[Seat, Seat], fought: tuple[set[int], set[int]]
    ) -> None:
        """What the consolidate step needs from the fight step.

        Beside the drag-in clause's bookkeeping, every member of a unit that was
        selected to fight is stamped `fought_this_phase` -- the consolidate
        step's "was eligible to fight this phase" reads that flag through
        `short_move_legality`. Stamped here, once, for every route through the
        fight step (an agent's per-model strikes, a script's inline activation,
        an overrun, the engine when `fight` is skipped), so both seats meet the
        same eligibility rule.
        """
        for seat, units in zip(order, fought, strict=True):
            for unit in units:
                for member in seat.unit_members(unit, alive_only=False):
                    seat.models[member].fought_this_phase = True
            self._fought_by_seat[seat.is_player] = set(units)

    def note_divergence(self, rule: str) -> None:
        """Record that a rule the phase facade cannot apply just made a difference."""
        state = self._game_clock.state
        self.divergences.append(
            FacadeDivergence(
                episode_step=self.episode_step,
                battle_round=state.battle_round,
                phase=state.phase,
                rule=rule,
            )
        )

    def fought_units_of(self, seat: Seat) -> set[int]:
        """Units of `seat` selected to fight this fight phase, so far."""
        return self._fought_by_seat.setdefault(seat.is_player, set())

    def drag_in(self, seat: Seat, groups: set[int]) -> None:
        """Engaging consolidation, after moving (`12-fight-phase.md`): each enemy
        unit the move newly engaged that has not been selected to fight this
        phase is selected now by its player and strikes `seat`'s force.

        Resolved at the consolidating unit's close, for that unit's own new
        contacts only -- not batched after both seats have finished, and not
        for a unit an Ongoing move happened to clip, which the rules do not
        grant a swing. The dragged-in unit's strikes are the engine's default
        targets (`DEFERRED: fight.drag_in_choice`), not a decision of its seat.
        """
        other = self._opponent_seat if seat.is_player else self._player_seat
        owed = groups - self.fought_units_of(other)
        if not owed or not self.config.melee.enabled:
            return
        self.fought_units_of(other).update(owed)
        for unit in owed:
            for member in other.unit_members(unit, alive_only=False):
                other.models[member].fought_this_phase = True
        quantities = self._rules_quantities
        other.fight_results.extend(
            fight_dragged_in_units(
                seat.enemies,
                seat.models,
                owed,
                self.roller(DicePurpose.melee, other, unit=None, model=None),
                engagement_range=quantities.engagement_range,
                base_diameter=2.0 * quantities.base_radius,
                attacker_weapons=other.melee_weapons,
            )
        )

    def _engaged_enemy_groups(self, seat: Seat) -> set[int]:
        """Enemy unit ids engaged with any living model of `seat`."""
        mine = [m for m in seat.models if m.is_alive]
        theirs = [m for m in seat.enemies if m.is_alive]
        if not mine or not theirs:
            return set()
        ours = np.array([m.location for m in mine], dtype=float)
        others = np.array([m.location for m in theirs], dtype=float)
        gaps = (
            np.linalg.norm(ours[:, np.newaxis, :] - others[np.newaxis, :, :], axis=2)
            - 2.0 * self._rules_quantities.base_radius
        )
        touching = (gaps <= self._rules_quantities.engagement_range).any(axis=0)
        return {int(theirs[column].group_id) for column in np.flatnonzero(touching)}

    def _regain_coherency(self, state: GameState) -> None:
        """End of Turn: every unit on the board regains coherency, or loses models.

        `03-moving.md` § Regaining coherency: "In the End of Turn step of each
        player's turn, ANY unit on the board that is out of coherency loses
        models". Both forces, not only the side whose turn is ending -- the
        phase facade culls the active side alone, which lets a unit the enemy's
        shooting split take its own movement phase to close up before it is
        ever judged.
        """
        if not self._coherency_attrition or state.active_player is None:
            return
        nearest = self._rules_quantities.scale.to_units(
            self.config.coherency.nearest_distance
        )
        furthest = self._rules_quantities.scale.to_units(
            self.config.coherency.furthest_distance
        )
        active, other = self._seat_order(state)
        for seat in (active, other):
            destroyed = apply_attrition(seat.models, nearest, furthest)
            if destroyed and seat is other:
                self.note_divergence("attrition.every_unit_on_the_board")
            if seat.is_player:
                self._attrition_deaths_player += len(destroyed)
            else:
                self._attrition_deaths_opponent += len(destroyed)

    # ------------------------------------------------------------------ reward

    def _open_window(self, state: GameState) -> None:
        assert state.phase is not None
        self._battle.reset_vp_deltas()
        for seat in (self._player_seat, self._opponent_seat):
            seat.shooting_results = []
            seat.fight_results = []
        self._attrition_deaths_player = 0
        self._attrition_deaths_opponent = 0
        self._window = _Window(
            phase=state.phase,
            turn_key=self._turn_key(state),
            opponent_alive_before=[m.is_alive for m in self.opponent_models],
            player_alive_before=[m.is_alive for m in self.wargame_models],
        )

    def _settle_window(self) -> SettledWindow:
        """`WargameEnv.step`'s tail, for the window that is open."""
        window = self._window
        if window is None:
            raise RuntimeError("no reward window is open")
        self._window = None
        player_alive = alive_mask_for(self.wargame_models)
        cache = compute_distances(
            self.wargame_models,
            self.objectives,
            compute_model_model=self.phase_manager.needs_model_model_distances,
            alive_mask=player_alive,
        )
        any_player_alive = bool(player_alive.any())
        all_player_eliminated = (
            self.config.terminate_on_player_elimination and not any_player_alive
        )
        # `15-missions-and-scoring.md` § Ending the battle: a player with no
        # models left does not lose immediately; both keep taking turns and the
        # survivor keeps scoring. So the opponent's wipe ends nothing here; the
        # player's does only under the config switch, which is a training
        # device. The phase facade ends the battle on either wipe.
        clock_state = self._game_clock.state
        phase = clock_state.phase or BattlePhase.command
        player_shots = self._player_seat.shooting_results
        opponent_shots = self._opponent_seat.shooting_results
        player_blows = self._player_seat.fight_results
        opponent_blows = self._opponent_seat.fight_results
        p_dmg = sum(r.result.damage_dealt for r in player_shots) + sum(
            r.result.damage_dealt for r in player_blows
        )
        o_dmg = sum(r.result.damage_dealt for r in opponent_shots) + sum(
            r.result.damage_dealt for r in opponent_blows
        )
        p_kills = max(
            0,
            sum(
                1
                for i, m in enumerate(self.opponent_models)
                if i < len(window.opponent_alive_before)
                and window.opponent_alive_before[i]
                and not m.is_alive
            )
            - self._attrition_deaths_opponent,
        )
        o_kills = max(
            0,
            sum(
                1
                for i, m in enumerate(self.wargame_models)
                if i < len(window.player_alive_before)
                and window.player_alive_before[i]
                and not m.is_alive
            )
            - self._attrition_deaths_player,
        )
        p_kills_by_model = np.zeros(len(self.wargame_models), dtype=np.int64)
        for shot in player_shots:
            if shot.killed and shot.attacker_idx < len(p_kills_by_model):
                p_kills_by_model[shot.attacker_idx] += 1
        for blow in player_blows:
            if blow.killed and blow.attacker_idx < len(p_kills_by_model):
                p_kills_by_model[blow.attacker_idx] += 1
        ctx = StepContext(
            distance_cache=cache,
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            board_width=self.board_width,
            board_height=self.board_height,
            is_terminated=False,
            current_round=clock_state.battle_round or 0,
            battle_phase=phase,
            action_phase=window.phase,
            player_damage_dealt=p_dmg,
            opponent_damage_dealt=o_dmg,
            player_models_killed=p_kills,
            opponent_models_killed=o_kills,
            player_kills_by_model=p_kills_by_model,
        )
        view = cast(Any, self)
        succeeded = (
            any_player_alive
            and self.phase_manager.terminate_on_success
            and self.phase_manager.check_success(view, ctx)
        )
        terminated = is_battle_over(
            self._game_clock,
            self.current_turn,
            self.max_turns,
            succeeded,
            all_eliminated=all_player_eliminated,
        )
        if not terminated and not any(m.is_alive for m in self.opponent_models):
            self.note_divergence("battle.continues_after_a_wipe")
        ctx.is_terminated = terminated
        self.last_step_context = ctx
        reward = self.phase_manager.calculate_reward(view, ctx)
        self.last_reward = reward
        self.last_reward_breakdown = dict(self.phase_manager.last_reward_breakdown)
        self.last_per_model_reward = self.phase_manager.last_per_model_reward.copy()
        for key, value in self.last_reward_breakdown.items():
            self.episode_reward_breakdown[key] = (
                self.episode_reward_breakdown.get(key, 0.0) + value
            )
        self.episode_reward_steps += 1
        self.episode_reward += reward
        return SettledWindow(
            phase=window.phase.value,
            reward=reward,
            terminated=terminated,
            breakdown=dict(self.last_reward_breakdown),
            per_model=self.last_per_model_reward.copy(),
            state=self._board_state(),
        )

    def _board_state(self) -> dict[str, Any]:
        state = self._game_clock.state
        return {
            "battle_round": state.battle_round,
            "active_player": state.active_player,
            "phase": state.phase,
            "current_turn": self.current_turn,
            "player_side": self._player_side,
            "player_positions": np.array(
                [m.location for m in self.wargame_models], dtype=float
            ),
            "opponent_positions": np.array(
                [m.location for m in self.opponent_models], dtype=float
            ),
            "player_wounds": [
                int(m.stats["current_wounds"]) for m in self.wargame_models
            ],
            "opponent_wounds": [
                int(m.stats["current_wounds"]) for m in self.opponent_models
            ],
            "player_vp": self.player_vp,
            "opponent_vp": self.opponent_vp,
            "charged": any(
                m.charged_this_turn
                for m in (*self.wargame_models, *self.opponent_models)
            ),
            "engaged": bool(self._engaged_enemy_groups(self._player_seat)),
            "dice": getattr(self._dice, "state", None),
        }

    def _info(self, settled: list[SettledWindow]) -> dict[str, Any]:
        state = self._game_clock.state
        pending = self._pending
        program = self._program
        open_unit = None
        if isinstance(program, MovementPhase | ShootingPhase | ChargePhase):
            open_unit = program.activation.open_unit
        return {
            "facade": FACADE_TAG,
            "phase": state.phase.value if state.phase is not None else None,
            "active_seat": (
                "player" if state.active_player == self._player_side else "opponent"
            ),
            "battle_round": state.battle_round,
            "sub_step": self.sub_step,
            "episode_step": self.episode_step,
            "current_turn": self.current_turn,
            "open_unit": open_unit,
            "expected": pending.kind.value if pending is not None else None,
            "reward_settled": bool(settled),
            "settled": list(settled),
        }

    # ------------------------------------------------------------------ misc

    def player_consolidation_modes(self) -> np.ndarray:
        """Per player model, the compulsory consolidation mode this phase (0 = none)."""
        program = self._program
        modes = program.player_consolidation_modes() if program is not None else None
        if modes is None:
            return np.zeros(self._player_seat.n_models, dtype=np.int64)
        return modes

    def _initial_player_side(self) -> PlayerSide:
        if self.config.turn_order == TurnOrder.opponent:
            return PlayerSide.player_2
        return PlayerSide.player_1

    def _resolve_player_side(self) -> None:
        if self.config.turn_order == TurnOrder.player:
            self._player_side = PlayerSide.player_1
        elif self.config.turn_order == TurnOrder.opponent:
            self._player_side = PlayerSide.player_2
        else:
            self._player_side = (
                PlayerSide.player_1
                if self.np_random.random() < 0.5
                else PlayerSide.player_2
            )

    @property
    def provenance(self) -> PerModelProvenance:
        """How to boot this episode again, stamped with this facade's tag."""
        return PerModelProvenance(
            config=self.config.model_dump(mode="json"),
            rng_state=self._episode_rng_state,
            combat_seed=self._episode_combat_seed,
            seed=self._episode_seed,
            driver=self.driver_label,
        )

    @property
    def pending(self) -> DecisionPoint | None:
        """The decision the env expects next, or None before the first reset."""
        return self._pending

    @property
    def max_turns(self) -> int:
        """The phase clock's budget: rounds x our stepped phases, as the phase facade."""
        n_phases = len(BATTLE_PHASE_ORDER) - len(self._skip_phases)
        return self._game_clock.n_rounds * n_phases

    @property
    def n_actions(self) -> int:
        return self._action_handler.n_actions

    @property
    def player_action_handler(self) -> ActionHandler:
        return self._action_handler

    @property
    def opponent_action_handler(self) -> ActionHandler:
        return self._opponent_action_handler

    @property
    def opponent_action_space(self) -> Any:
        return self._opponent_action_handler.action_space

    @property
    def opponent_policy(self) -> Any:
        return self._opponent_policy

    # ------------------------------------------------------------------ BattleView

    @property
    def player_models(self) -> list[WargameModel]:
        return self.wargame_models

    @property
    def deployment_outline(self) -> Polygon | None:
        return self._battle.deployment_outline

    @property
    def opponent_deployment_outline(self) -> Polygon | None:
        return self._battle.opponent_deployment_outline

    @property
    def terrain(self) -> Terrain:
        return self._battle.terrain

    @property
    def map_name(self) -> str | None:
        return self._map_name

    @property
    def rules_quantities(self) -> RulesQuantities:
        return self._rules_quantities

    @property
    def game_clock_state(self) -> GameState:
        return self._game_clock.state

    @property
    def n_rounds(self) -> int:
        return self._game_clock.n_rounds

    @property
    def player_side(self) -> PlayerSide:
        return self._player_side

    @property
    def player_vp(self) -> int:
        return self._battle.player_vp

    @property
    def opponent_vp(self) -> int:
        return self._battle.opponent_vp

    @property
    def player_vp_delta(self) -> int:
        return self._battle.player_vp_delta

    @property
    def opponent_vp_delta(self) -> int:
        return self._battle.opponent_vp_delta

    @property
    def player_max_ranges(self) -> np.ndarray:
        return self._player_max_ranges

    @property
    def opponent_max_ranges(self) -> np.ndarray:
        return self._opponent_max_ranges

    @property
    def last_player_shooting_results(self) -> list[PairedShootingResult]:
        return list(self._player_seat.shooting_results)

    @property
    def last_opponent_shooting_results(self) -> list[PairedShootingResult]:
        return list(self._opponent_seat.shooting_results)

    @property
    def last_player_fight_results(self) -> list[PairedFightResult]:
        return list(self._player_seat.fight_results)

    @property
    def last_opponent_fight_results(self) -> list[PairedFightResult]:
        return list(self._opponent_seat.fight_results)

    @property
    def player_advance_legality(self) -> np.ndarray:
        return self._action_handler.advance_legality(
            self.wargame_models, self.opponent_models
        )

    @property
    def player_charge_legality(self) -> np.ndarray:
        return self._action_handler.charge_legality(
            self.wargame_models, self.opponent_models
        )

    @property
    def player_declaration_legality(self) -> np.ndarray:
        return self._action_handler.declaration_legality(
            self.wargame_models, self.opponent_models
        )

    @property
    def player_objective_target_legality(self) -> np.ndarray:
        return self._action_handler.objective_target_legality(
            self.wargame_models, len(self.objectives)
        )

    @property
    def player_charge_target_legality(self) -> np.ndarray:
        return self._action_handler.charge_target_legality(
            self.wargame_models, self.opponent_models
        )

    @property
    def player_short_move_legality(self) -> np.ndarray:
        return self._action_handler.short_move_legality(
            self.wargame_models,
            self.opponent_models,
            self._game_clock.state.phase or BattlePhase.movement,
        )

    def has_line_of_sight_between_points(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> bool:
        return has_line_of_sight_between_points(
            x0,
            y0,
            x1,
            y1,
            self._battle.terrain,
            self.config.blocking_mask,
            sample_step=self._rules_quantities.los_sample_step,
        )

    def line_of_sight_matrix(
        self,
        origins: np.ndarray,
        targets: np.ndarray,
        candidates: np.ndarray | None = None,
    ) -> np.ndarray:
        centre: np.ndarray = (
            self.visibility_between(origins, targets, candidates, edges=False) >= COVER
        )
        return centre

    def visibility_between(
        self,
        origins: np.ndarray,
        targets: np.ndarray,
        candidates: np.ndarray | None = None,
        *,
        origin_models: list[WargameModel] | None = None,
        target_models: list[WargameModel] | None = None,
        edges: bool = True,
    ) -> np.ndarray:
        return visibility_matrix(
            origins,
            targets,
            self._battle.terrain,
            self.config.blocking_mask,
            sample_step=self._rules_quantities.los_sample_step,
            candidates=candidates,
            origin_radii=_base_radii(origin_models),
            target_radii=_base_radii(target_models) if edges else None,
        )


def _base_radii(models: list[WargameModel] | None) -> np.ndarray | None:
    if models is None:
        return None
    return np.array([m.base_radius for m in models], dtype=float)


__all__ = [
    "EpisodeOver",
    "PerModelEnv",
    "SettledWindow",
    "WargameObjective",
    "default_dice",
]
