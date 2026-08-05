from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.domain.battle_factory import (
    create_objectives as _create_objectives,
)
from wargame_rl.wargame.envs.domain.battle_factory import (
    create_wargame_models as _create_wargame_models,
)
from wargame_rl.wargame.envs.domain.battle_factory import (
    from_config as _battle_from_config,
)
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.game_clock import GameClock
from wargame_rl.wargame.envs.domain.los import has_line_of_sight, iter_los_cells
from wargame_rl.wargame.envs.domain.placement import place_for_episode
from wargame_rl.wargame.envs.domain.shooting import (
    ENGAGEMENT_RANGE,
    DefenderStats,
    PairedShootingResult,
    ShootingResult,
    resolve_shooting,
)
from wargame_rl.wargame.envs.domain.termination import is_battle_over
from wargame_rl.wargame.envs.domain.terrain import Terrain
from wargame_rl.wargame.envs.domain.terrain_placement import generate_terrain
from wargame_rl.wargame.envs.domain.turn_execution import (
    run_after_player_action,
    run_until_player_phase,
)
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.env_components import (
    ActionHandler,
    DistanceCache,
    build_info,
    build_observation,
    compute_distances,
)
from wargame_rl.wargame.envs.env_components.actions import ActionSlice
from wargame_rl.wargame.envs.env_components.exposure import (
    ExposureTracker,
    distances_to_nearest_footprint,
)
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_shooting_masks,
    compute_threat_counts,
    max_weapon_ranges,
)
from wargame_rl.wargame.envs.mission import build_vp_calculator
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import (
    _auto_register,
    build_opponent_policy,
)
from wargame_rl.wargame.envs.renders import renderer
from wargame_rl.wargame.envs.reward.phase_manager import (
    CurriculumPosition,
    RewardPhaseManager,
)
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.state.exporter import StateExporter
from wargame_rl.wargame.envs.state.snapshot import (
    GameStateSnapshot,
    build_snapshot,
    validate_snapshot,
)
from wargame_rl.wargame.envs.types import (
    BattlePhase,
    PlayerSide,
    TurnOrder,
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvInfo,
    WargameEnvObservation,
)
from wargame_rl.wargame.envs.types.config import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import (
    BATTLE_PHASE_ORDER,
    GamePhase,
    GameState,
)
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.envs.wargame_objective import WargameObjective

# Re-export for backward compatibility (tests, dqn import from here)
__all__ = ["WargameEnv", "WargameObjective"]

_auto_register()


class WargameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        config: WargameEnvConfig,
        renderer: renderer.Renderer | None = None,
        state_exporters: list[StateExporter] | None = None,
        phase_position: CurriculumPosition | None = None,
    ):
        """Build the environment.

        `phase_position` shares curriculum progress with another environment.
        Training passes the eval env's position to every rollout env so they
        reward the phase the curriculum has actually reached.
        """
        self.board_width = config.board_width
        self.board_height = config.board_height
        self.window_size = 1024  # The size of the PyGame window
        self.config = config
        self.observation_space = spaces.Dict(
            {
                "current_turn": spaces.Discrete(1),
                "wargame_models": spaces.Tuple(
                    [
                        WargameModel.to_space(
                            board_width=self.board_width,
                            board_height=self.board_height,
                            number_of_objectives=config.number_of_objectives * 2,
                        )
                        for _ in range(config.number_of_wargame_models)
                    ]
                ),
                "objectives": spaces.Sequence(
                    WargameObjective.to_space(
                        board_width=self.board_width, board_height=self.board_height
                    )
                ),
            }
        )

        self._action_handler = ActionHandler(
            config, n_shoot_targets=config.number_of_opponent_models
        )
        self.action_space = self._action_handler.action_space
        self._skip_phases = frozenset(config.skip_phases)

        self.renderer = renderer
        self._state_exporters: list[StateExporter] = state_exporters or []

        self.window = None
        self.clock = None

        self.current_turn = 0
        self._player_side = self._initial_player_side()
        self._game_clock = GameClock(n_rounds=config.number_of_battle_rounds)

        self._battle = _battle_from_config(config)
        # Random terrain is regenerated on every reset, but a layout has to
        # exist before the first one: network sizing reads an observation, and
        # the terrain tensor must already have its final piece count.
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

        # Combat RNG (re-seeded per episode in reset)
        self._combat_rng: np.random.Generator = np.random.default_rng()
        self._last_player_shooting_results: list[PairedShootingResult] = []
        self._last_opponent_shooting_results: list[PairedShootingResult] = []

        # Last actions and termination flag (for snapshot / replay)
        self._last_player_action: WargameEnvAction | None = None
        self._last_opponent_action: WargameEnvAction | None = None
        self._last_action_phase: BattlePhase | None = None
        self._last_terminated: bool = False

        # Last reward from step(); None until first step after reset
        self.last_reward: float | None = None
        self.last_reward_breakdown: dict[str, float] = {}
        # Per-model decomposition of `last_reward`, for algorithms that credit
        # each model's own action rather than a single army-wide scalar.
        self.last_per_model_reward: np.ndarray = np.zeros(
            config.number_of_wargame_models, dtype=np.float64
        )
        self.episode_reward_breakdown: dict[str, float] = {}
        self.episode_reward_steps: int = 0

        # Reward phases (curriculum learning); always used for reward calculation
        self.phase_manager = RewardPhaseManager.from_configs(
            config.reward_phases, position=phase_position
        )

        # Mission VP calculator (scores at end of command phase from round 2)
        self._vp_calculator = build_vp_calculator(
            config.mission.type, config.mission.params
        )

        # Last StepContext from step(); available for post-episode success checks
        self.last_step_context: StepContext | None = None

        # Cover measurement (off by default; costs an extra mask build per
        # shooting phase). Weapon ranges come from config, so they are resolved
        # once here rather than on every shooting phase.
        self._exposure_tracker = ExposureTracker()
        self._opponent_max_ranges = max_weapon_ranges(
            config.opponent_models, config.number_of_opponent_models
        )
        self._player_max_ranges = max_weapon_ranges(
            config.models, config.number_of_wargame_models
        )

        # --- Opponent setup ---
        if config.number_of_opponent_models > 0:
            self._opponent_action_handler = ActionHandler(
                config,
                n_models=config.number_of_opponent_models,
                n_shoot_targets=config.number_of_wargame_models,
            )
            self._opponent_policy: OpponentPolicy | None = build_opponent_policy(
                config.opponent_policy,  # type: ignore[arg-type]
                self,
            )
        else:
            self._opponent_action_handler = ActionHandler(config, n_models=0)
            self._opponent_policy = None

    @property
    def max_turns(self) -> int:
        """Maximum agent steps per episode (game-clock-derived: n_rounds × active phases)."""
        n_phases = len(BATTLE_PHASE_ORDER) - len(self._skip_phases)
        return self._game_clock.n_rounds * n_phases

    @property
    def n_actions(self) -> int:
        """Number of discrete actions per model (including stay)."""
        return self._action_handler.n_actions

    @property
    def player_action_handler(self) -> ActionHandler:
        """Action handler for the player's models (used by baseline policies)."""
        return self._action_handler

    @property
    def opponent_action_handler(self) -> ActionHandler:
        """Action handler for the opponent's models (used by opponent policies)."""
        return self._opponent_action_handler

    @property
    def state_exporters(self) -> list[StateExporter]:
        """Exporters recording this env's steps (empty when not recording)."""
        return list(self._state_exporters)

    @property
    def last_player_shooting_results(self) -> list[PairedShootingResult]:
        """Shots the player resolved during the most recent step."""
        return list(self._last_player_shooting_results)

    @property
    def last_opponent_shooting_results(self) -> list[PairedShootingResult]:
        """Shots the opponent resolved during the most recent step."""
        return list(self._last_opponent_shooting_results)

    @property
    def opponent_action_space(self) -> spaces.Tuple:
        """Action space for opponent models (used by policies)."""
        return self._opponent_action_handler.action_space

    @property
    def terrain(self) -> "Terrain":
        """Read-only access to terrain footprints."""
        return self._battle.terrain

    @property
    def exposure_rate(self) -> float | None:
        """Fraction of alive model-shooting-phases an enemy could see and shoot.

        The measure of cover use: terrain blocks line of sight and nothing else,
        so breaking line of sight is the only thing that lowers this. None when
        `track_exposure` is off.
        """
        return self._exposure_tracker.exposure_rate

    @property
    def terrain_proximity(self) -> float | None:
        """Mean distance from an alive model to the nearest terrain footprint.

        Read next to `exposure_rate`: a policy that is merely out of range keeps
        this high, one that is using ruins pulls it down. None when
        `track_exposure` is off or the board has no terrain.
        """
        return self._exposure_tracker.terrain_proximity

    @property
    def firepower_advantage(self) -> float | None:
        """Mean (enemies we can shoot) − (our models they can shoot) per phase.

        The exchange-ratio measure: positive means the army brings more guns to
        bear than it exposes. Prefer it to `exposure_rate` when the question is
        whether the policy is *choosing* its fights — low exposure alone cannot
        tell manoeuvre apart from hiding. None when `track_exposure` is off.
        """
        return self._exposure_tracker.firepower_advantage

    def _make_is_blocking(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> Callable[[int, int], bool]:
        """Per-query blocking predicate: static blocking_mask OR membership of any
        footprint that contains NEITHER endpoint (10e see-out / see-into rule)."""
        mask = self.config.blocking_mask
        active = self._battle.terrain.blocking_footprints_for_endpoints(x0, y0, x1, y1)

        def is_blocking(x: int, y: int) -> bool:
            if mask is not None and mask[y][x]:
                return True
            return any(fp.contains(x, y) for fp in active)

        return is_blocking

    def has_line_of_sight_between_cells(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> bool:
        """True if LOS is clear between two cells (symmetric: canonical ordering)."""
        (ax, ay), (bx, by) = sorted([(x0, y0), (x1, y1)])
        return has_line_of_sight(
            ax,
            ay,
            bx,
            by,
            self.board_width,
            self.board_height,
            self._make_is_blocking(ax, ay, bx, by),
        )

    def iter_los_cells_between_cells(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> list[tuple[int, int]]:
        """Inclusive Bresenham cells between endpoints; empty if an endpoint is out of bounds."""
        return iter_los_cells(x0, y0, x1, y1, self.board_width, self.board_height)

    # BattleView protocol (read-only battle state for renderers and reward)
    @property
    def player_models(self) -> list[WargameModel]:
        return self.wargame_models

    @property
    def game_clock_state(self) -> GameState:
        return self._game_clock.state

    @property
    def n_rounds(self) -> int:
        return self._game_clock.n_rounds

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

    def _on_before_advance(self, clock: GameClock) -> None:
        """Score VP when leaving command phase from round 2 (mission-driven)."""
        state = clock.state
        if state.phase != BattlePhase.command or state.battle_round is None:
            return
        if state.active_player is None:
            return
        vp = self._vp_calculator.compute_vp(
            self,
            state.active_player,
            state.battle_round,
            self._player_side,
        )
        if vp <= 0:
            return
        if state.active_player == self._player_side:
            self._battle.add_player_vp(vp)
        else:
            self._battle.add_opponent_vp(vp)

    # Backward compat: static factory methods delegate to BattleFactory
    @staticmethod
    def create_wargame_models(config: WargameEnvConfig) -> list[WargameModel]:
        """Build the list of player wargame models from config."""
        return _create_wargame_models(config)

    @staticmethod
    def create_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
        """Build the list of objectives from config."""
        return _create_objectives(config)

    @property
    def observation(self) -> WargameEnvObservation:
        """The observation for the current state, rebuilt on access.

        `reset` and `step` already return this; the property exists for callers
        that mutate state directly and need to re-read it without stepping.
        """
        return self._get_obs()

    def _get_obs(
        self, distance_cache: DistanceCache | None = None
    ) -> WargameEnvObservation:
        """Get the observation for the current state of the environment."""
        return build_observation(
            self,
            distance_cache=distance_cache,
            action_registry=self._action_handler.registry,
        )

    def _get_info(self) -> WargameEnvInfo:
        return build_info(self)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WargameEnvObservation, dict[str, Any]]:
        """Start a new episode.

        `options["combat_seed"]` seeds the dice independently of `seed`, which
        otherwise drives both the layout and the rolls. Varying one with the
        other held fixed is what separates "this scenario is hard" from "the
        dice went badly", and the two are indistinguishable while a single seed
        controls both. Absent, the combat seed is derived from `seed` as before.
        """
        super().reset(seed=seed)

        # The draw happens either way. Skipping it when a combat seed is given
        # would shift every later draw from `np_random`, so the layout would
        # change too and the two sources of variance would still be entangled --
        # the exact thing this option exists to separate.
        derived_combat_seed = int(self.np_random.integers(0, 2**31))
        explicit_combat_seed = (options or {}).get("combat_seed")
        self._combat_rng = np.random.default_rng(
            derived_combat_seed
            if explicit_combat_seed is None
            else int(explicit_combat_seed)
        )
        self._last_player_shooting_results = []
        self._last_opponent_shooting_results = []
        self._last_player_action = None
        self._last_opponent_action = None
        self._last_action_phase = None
        self._last_terminated = False

        self.current_turn = 0
        self.last_reward = None
        self.last_step_context = None
        self.last_reward_breakdown = {}
        self.episode_reward_breakdown = {}
        self.episode_reward_steps = 0
        self._exposure_tracker.reset()

        self._battle.reset_for_episode()
        self.phase_manager.reset_episode()
        self._resolve_player_side()
        self._game_clock.reset()
        self._game_clock.skip_setup()
        # Clock is now at round 1, player_1, command phase

        place_for_episode(self._battle, self.config, self.np_random)

        # If opponent goes first this round, auto-execute their turn and skip to player phase
        run_until_player_phase(
            self._game_clock,
            self._skip_phases,
            self._player_side,
            self._apply_opponent_action,
            on_before_advance=self._on_before_advance,
        )

        cache = compute_distances(self.wargame_models, self.objectives)
        observation = self._get_obs(cache)
        info: WargameEnvInfo = self._get_info()

        if self.renderer is not None:
            self.renderer.setup(self)
            self.renderer.render(self)

        if self._state_exporters:
            snapshot = self.to_snapshot()
            for exporter in self._state_exporters:
                exporter.on_reset(snapshot)

        return observation, info.model_dump()

    def _resolve_shooting_action(
        self,
        action: WargameEnvAction,
        attackers: list[WargameModel],
        targets: list[WargameModel],
        shooting_slice: ActionSlice | None,
        attacker_configs: list[ModelConfig] | None,
    ) -> list[PairedShootingResult]:
        """Resolve shooting for each model in the action against targets.

        Returns one PairedShootingResult per model that actually fired.
        """
        results: list[PairedShootingResult] = []
        if shooting_slice is None:
            return results
        for i, act in enumerate(action.actions):
            if i >= len(attackers):
                continue
            attacker = attackers[i]
            if not attacker.is_alive:
                continue
            if not (shooting_slice.start <= act < shooting_slice.end):
                continue
            target_idx = act - shooting_slice.start
            if target_idx >= len(targets) or not targets[target_idx].is_alive:
                continue
            weapons = (
                attacker_configs[i].weapons
                if attacker_configs and i < len(attacker_configs)
                else []
            )
            if not weapons:
                continue
            w = weapons[0]
            target = targets[target_idx]
            defender = DefenderStats(
                toughness=target.stats["toughness"],
                save=target.stats["save"],
            )
            result = resolve_shooting(w, defender, self._combat_rng)
            killed = False
            if result.damage_dealt > 0:
                targets[target_idx].take_damage(result.damage_dealt)
                # The target was alive at the range check above, so a dead
                # target here means this shot made the kill.
                killed = not targets[target_idx].is_alive
            results.append(
                PairedShootingResult(
                    attacker_idx=i,
                    target_idx=target_idx,
                    result=result,
                    killed=killed,
                )
            )
        return results

    def _apply_player_action(self, action: WargameEnvAction) -> None:
        phase = self._game_clock.state.phase or BattlePhase.movement
        # Captured before the clock advances so snapshots can attribute the
        # recorded actions to the phase that actually executed them.
        self._last_action_phase = phase
        if phase == BattlePhase.shooting:
            self._last_player_shooting_results = self._resolve_shooting_action(
                action,
                self.wargame_models,
                self.opponent_models,
                self._action_handler.shooting_slice,
                self.config.models,
            )
        else:
            self._action_handler.apply(
                action,
                self.wargame_models,
                self.board_width,
                self.board_height,
                self._action_handler.action_space,
                phase=phase,
            )

    def _opponent_action_mask(
        self, phase: BattlePhase, opp_alive: np.ndarray
    ) -> np.ndarray:
        """Legal actions per opponent model, shooting targets included.

        Mirrors the player's mask in `build_observation`, with the sides
        swapped: the opponent's shots are held to the same range, line-of-sight
        and engagement-range rules the player's are. Without this the opponent's
        mask is phase-and-alive only, and any shooting action it emits is
        applied unchecked by `_resolve_shooting_action`.
        """
        handler = self._opponent_action_handler
        mask = handler.registry.get_model_action_masks(
            phase, len(self.opponent_models), alive_mask=opp_alive
        )
        shooting_slice = handler.shooting_slice
        if (
            phase != BattlePhase.shooting
            or shooting_slice is None
            or not self.wargame_models
            or self._opponent_policy is None
            or not self._opponent_policy.shoots
        ):
            return mask

        mask[:, shooting_slice.start : shooting_slice.end] &= compute_shooting_masks(
            np.array([m.location for m in self.opponent_models]),
            np.array([m.location for m in self.wargame_models]),
            opp_alive,
            alive_mask_for(self.wargame_models),
            self._opponent_max_ranges,
            self.has_line_of_sight_between_cells,
            player_advanced=np.array(
                [m.advanced_this_turn for m in self.opponent_models]
            ),
            engagement_range=float(ENGAGEMENT_RANGE),
        )
        return mask

    def _record_exposure(self, opp_alive: np.ndarray) -> None:
        """Sample both sides of the shooting exchange right now.

        Deliberately ignores the engagement-range and advanced gating that
        `_opponent_action_mask` applies. A shooter in base contact cannot fire at
        all, so counting that as safety would score a headlong charge as if it
        were cover — and cover is the thing being measured.

        One scan yields both directions, so `firepower_advantage` costs nothing
        on top of `exposure_rate`.
        """
        player_alive = alive_mask_for(self.wargame_models)
        if not player_alive.any() or not opp_alive.any():
            return
        player_positions = np.array([m.location for m in self.wargame_models])
        threat_to_player, threat_to_opponent = compute_threat_counts(
            player_positions,
            np.array([m.location for m in self.opponent_models]),
            player_alive,
            opp_alive,
            self._player_max_ranges,
            self._opponent_max_ranges,
            self.has_line_of_sight_between_cells,
        )
        self._exposure_tracker.record(
            exposed=threat_to_player > 0,
            alive=player_alive,
            terrain_distances=distances_to_nearest_footprint(
                player_positions, self.terrain.footprints
            ),
            opponents_engaged=int(((threat_to_opponent > 0) & opp_alive).sum()),
        )

    def _apply_opponent_action(self) -> None:
        if self._opponent_policy is None or not self.opponent_models:
            return
        phase = self._game_clock.state.phase or BattlePhase.movement
        opp_alive = alive_mask_for(self.opponent_models)
        if self.config.track_exposure and phase == BattlePhase.shooting:
            self._record_exposure(opp_alive)
        opp_mask = self._opponent_action_mask(phase, opp_alive)
        opp_action = self._opponent_policy.select_action(
            self.opponent_models, self, action_mask=opp_mask
        )
        self._last_opponent_action = opp_action
        if phase == BattlePhase.shooting:
            self._last_opponent_shooting_results = self._resolve_shooting_action(
                opp_action,
                self.opponent_models,
                self.wargame_models,
                self._opponent_action_handler.shooting_slice,
                self.config.opponent_models,
            )
        else:
            self._opponent_action_handler.apply(
                opp_action,
                self.opponent_models,
                self.board_width,
                self.board_height,
                self._opponent_action_handler.action_space,
                phase=phase,
            )

    def _initial_player_side(self) -> PlayerSide:
        """Deterministic side assignment used at __init__ time."""
        if self.config.turn_order == TurnOrder.opponent:
            return PlayerSide.player_2
        return PlayerSide.player_1

    def _resolve_player_side(self) -> None:
        """Set ``_player_side`` based on ``TurnOrder`` (called each reset)."""
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

    def step(
        self, action: WargameEnvAction
    ) -> tuple[WargameEnvObservation, float, bool, bool, dict[str, Any]]:
        self._battle.reset_vp_deltas()
        self._last_player_shooting_results = []
        self._last_opponent_shooting_results = []
        opp_alive_before = [m.is_alive for m in self.opponent_models]
        player_alive_before = [m.is_alive for m in self.wargame_models]

        self._last_player_action = action
        self._apply_player_action(action)

        self.current_turn += 1

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
        phase = clock_state.phase or BattlePhase.command

        p_dmg = sum(r.result.damage_dealt for r in self._last_player_shooting_results)
        o_dmg = sum(r.result.damage_dealt for r in self._last_opponent_shooting_results)
        p_kills = sum(
            1
            for i, m in enumerate(self.opponent_models)
            if i < len(opp_alive_before) and opp_alive_before[i] and not m.is_alive
        )
        o_kills = sum(
            1
            for i, m in enumerate(self.wargame_models)
            if i < len(player_alive_before)
            and player_alive_before[i]
            and not m.is_alive
        )
        # Attribute each kill to the model that fired it, so shooting reward
        # lands on that model's advantage rather than being shared flat.
        p_kills_by_model = np.zeros(len(self.wargame_models), dtype=np.int64)
        for shot in self._last_player_shooting_results:
            if shot.killed and shot.attacker_idx < len(p_kills_by_model):
                p_kills_by_model[shot.attacker_idx] += 1

        # Built before termination is known because the phase's success criteria
        # need a context to evaluate against. No criteria reads `is_terminated`,
        # so the provisional False here cannot change their verdict; it is
        # corrected below before any reward is calculated.
        ctx = StepContext(
            distance_cache=cache,
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            board_width=self.board_width,
            board_height=self.board_height,
            is_terminated=False,
            current_round=clock_state.battle_round or 0,
            battle_phase=phase,
            player_damage_dealt=p_dmg,
            opponent_damage_dealt=o_dmg,
            player_models_killed=p_kills,
            opponent_models_killed=o_kills,
            player_kills_by_model=p_kills_by_model,
        )

        # Consults the *configured* criteria for the active phase rather than a
        # hardcoded all-models-at-objectives test, so a phase gated on a model
        # fraction or on VP can end on its own success.
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
        reward = self.phase_manager.calculate_reward(self, ctx)

        observation = self._get_obs(cache)
        info = self._get_info()

        self.last_reward = reward
        self.last_reward_breakdown = dict(self.phase_manager.last_reward_breakdown)
        self.last_per_model_reward = self.phase_manager.last_per_model_reward.copy()
        for key, value in self.last_reward_breakdown.items():
            self.episode_reward_breakdown[key] = (
                self.episode_reward_breakdown.get(key, 0.0) + value
            )
        self.episode_reward_steps += 1

        if self._state_exporters:
            snapshot = self.to_snapshot()
            for exporter in self._state_exporters:
                exporter.on_step(snapshot)

        return observation, reward, is_terminated, False, info.model_dump()

    def to_snapshot(self) -> GameStateSnapshot:
        """Build a serialisable snapshot of the current game state."""
        ss = self._action_handler.shooting_slice
        return build_snapshot(
            config=self.config,
            step=self.current_turn,
            max_steps=self.max_turns,
            clock_state=self._game_clock.state,
            n_rounds=self._game_clock.n_rounds,
            player_models=self.wargame_models,
            opponent_models=self.opponent_models,
            objectives=self.objectives,
            deployment_zone=self.deployment_zone,
            opponent_deployment_zone=self.opponent_deployment_zone,
            player_vp=self.player_vp,
            opponent_vp=self.opponent_vp,
            player_vp_delta=self.player_vp_delta,
            opponent_vp_delta=self.opponent_vp_delta,
            player_shooting_results=self._last_player_shooting_results,
            opponent_shooting_results=self._last_opponent_shooting_results,
            player_action=self._last_player_action,
            opponent_action=self._last_opponent_action,
            last_reward=self.last_reward,
            reward_breakdown=self.last_reward_breakdown,
            phase_name=self.phase_manager.current_phase_name,
            phase_index=self.phase_manager.current_phase_index,
            is_terminated=self._last_terminated,
            is_truncated=False,
            n_angles=self.config.n_movement_angles,
            n_speed_bins=self.config.n_speed_bins,
            shooting_slice_start=ss.start if ss else None,
            shooting_slice_end=ss.end if ss else None,
            action_phase=(
                self._last_action_phase.value if self._last_action_phase else None
            ),
        )

    def load_state(
        self, snapshot: GameStateSnapshot
    ) -> tuple[WargameEnvObservation, dict[str, Any]]:
        """Restore env from a snapshot. Returns (observation, info) like reset().

        The env is ready for ``step()`` immediately after this call.
        """
        errors = validate_snapshot(snapshot, self.config)
        if errors:
            raise ValueError(
                "Invalid snapshot:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Clock
        clock = snapshot.clock
        self._game_clock.set_state(
            GamePhase(clock.game_phase),
            battle_round=clock.battle_round,
            active_player=(
                PlayerSide(clock.active_player) if clock.active_player else None
            ),
            phase=BattlePhase(clock.battle_phase) if clock.battle_phase else None,
            total_steps=snapshot.step,
        )

        # Player models
        for i, ms in enumerate(snapshot.player_models):
            m = self.wargame_models[i]
            m.location = np.array(ms.location, dtype=np.int32)
            m.previous_location = (
                np.array(ms.previous_location, dtype=np.int32)
                if ms.previous_location is not None
                else None
            )
            m.stats["current_wounds"] = ms.current_wounds
            m.advanced_this_turn = ms.advanced_this_turn
            m.previous_closest_objective_distance = None
            m.best_closest_objective_distance = None
            m.model_rewards_history.clear()

        # Opponent models
        for i, ms in enumerate(snapshot.opponent_models):
            m = self.opponent_models[i]
            m.location = np.array(ms.location, dtype=np.int32)
            m.previous_location = (
                np.array(ms.previous_location, dtype=np.int32)
                if ms.previous_location is not None
                else None
            )
            m.stats["current_wounds"] = ms.current_wounds
            m.advanced_this_turn = ms.advanced_this_turn
            m.previous_closest_objective_distance = None
            m.best_closest_objective_distance = None
            m.model_rewards_history.clear()

        # Objectives
        for i, os_ in enumerate(snapshot.objectives):
            self.objectives[i].location = np.array(os_.location, dtype=np.int32)

        # VP
        self._battle._player_vp = snapshot.player_vp
        self._battle._opponent_vp = snapshot.opponent_vp
        self._battle._player_vp_delta = 0
        self._battle._opponent_vp_delta = 0

        # Env counters
        self.current_turn = snapshot.step
        self._last_terminated = snapshot.is_terminated
        self._last_action_phase = (
            BattlePhase(snapshot.action_phase) if snapshot.action_phase else None
        )

        # Reconstruct last actions
        if snapshot.player_actions is not None:
            self._last_player_action = WargameEnvAction(
                actions=list(snapshot.player_actions)
            )
        else:
            self._last_player_action = None

        if snapshot.opponent_actions is not None:
            self._last_opponent_action = WargameEnvAction(
                actions=list(snapshot.opponent_actions)
            )
        else:
            self._last_opponent_action = None

        # Reconstruct combat results as PairedShootingResult stubs
        self._last_player_shooting_results = [
            PairedShootingResult(
                attacker_idx=cr.attacker_idx,
                target_idx=cr.target_idx,
                result=ShootingResult(
                    hits=cr.hits,
                    wounds=cr.wounds,
                    unsaved=cr.unsaved,
                    damage_dealt=cr.damage_dealt,
                ),
            )
            for cr in snapshot.player_combat_results
        ]
        self._last_opponent_shooting_results = [
            PairedShootingResult(
                attacker_idx=cr.attacker_idx,
                target_idx=cr.target_idx,
                result=ShootingResult(
                    hits=cr.hits,
                    wounds=cr.wounds,
                    unsaved=cr.unsaved,
                    damage_dealt=cr.damage_dealt,
                ),
            )
            for cr in snapshot.opponent_combat_results
        ]

        # Reward
        self.last_reward = snapshot.reward.total
        self.last_reward_breakdown = dict(snapshot.reward.breakdown)
        self.episode_reward_breakdown = {}
        self.episode_reward_steps = 0
        self.last_step_context = None

        # Recompute distances and build observation
        cache = compute_distances(self.wargame_models, self.objectives)
        observation = self._get_obs(cache)
        info = self._get_info()

        return observation, info.model_dump()

    def render(self) -> None:
        if self.renderer is not None:
            self.renderer.render(self)

        return None
