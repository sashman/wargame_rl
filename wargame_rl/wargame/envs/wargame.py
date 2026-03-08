from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.env_components import (
    ActionHandler,
    DistanceCache,
    GameClock,
    VPState,
    build_info,
    build_observation,
    check_max_turns_reached,
    compute_distances,
    compute_primary_vp_earned,
    fixed_objective_placement,
    fixed_wargame_model_placement,
    get_termination,
    objective_placement,
    update_distances_to_objectives,
    wargame_model_placement,
)
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import (
    _auto_register,
    build_opponent_policy,
)
from wargame_rl.wargame.envs.renders import renderer
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.reward.reward import Reward
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import (
    BattlePhase,
    EnvScoreState,
    PlayerSide,
    TurnOrder,
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvInfo,
    WargameEnvObservation,
)
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.envs.wargame_objective import WargameObjective

# Re-export for backward compatibility (tests, dqn import from here)
__all__ = ["WargameEnv", "WargameObjective"]

_auto_register()


class WargameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self, config: WargameEnvConfig, renderer: renderer.Renderer | None = None
    ):
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
                "env_score_state": EnvScoreState.to_space(),
            }
        )

        self._action_handler = ActionHandler(config)
        self.action_space = self._action_handler.action_space
        self._skip_phases = frozenset(config.skip_phases)

        self.renderer = renderer

        self.window = None
        self.clock = None

        self.current_turn = 0
        self._player_side = self._initial_player_side()
        self._game_clock = GameClock(n_rounds=config.number_of_battle_rounds)

        self.wargame_models = self.create_wargame_models(config)
        self.objectives = self.create_objectives(config)

        if config.deployment_zone is not None:
            self.deployment_zone = np.array(config.deployment_zone, dtype=int)
        else:
            self.deployment_zone = np.array(
                [0, 0, self.board_width // 3, self.board_height], dtype=int
            )

        if config.opponent_deployment_zone is not None:
            self.opponent_deployment_zone = np.array(
                config.opponent_deployment_zone, dtype=int
            )
        else:
            self.opponent_deployment_zone = np.array(
                [
                    self.board_width * 2 // 3,
                    0,
                    self.board_width,
                    self.board_height,
                ],
                dtype=int,
            )

        # Last reward from step(); None until first step after reset
        self.last_reward: float | None = None

        # Victory points state (primary mission: 5 VP per objective controlled, cap 15 per turn)
        self._vp_state = VPState()

        # Reward phases (curriculum learning) -- None uses legacy Reward path
        if config.reward_phases is not None:
            self.phase_manager: RewardPhaseManager | None = (
                RewardPhaseManager.from_configs(config.reward_phases)
            )
        else:
            self.phase_manager = None

        # Last StepContext from step(); available for post-episode success checks
        self.last_step_context: StepContext | None = None

        # --- Opponent setup ---
        self.opponent_models: list[WargameModel] = self._create_opponent_models(config)
        if config.number_of_opponent_models > 0:
            self._opponent_action_handler = ActionHandler(
                config, n_models=config.number_of_opponent_models
            )
            self._opponent_policy: OpponentPolicy | None = build_opponent_policy(
                config.opponent_policy,  # type: ignore[arg-type]
                self,
            )
        else:
            self._opponent_action_handler = ActionHandler(config, n_models=0)
            self._opponent_policy = None

    @property
    def player_vp(self) -> int:
        return self._vp_state.player_vp

    @property
    def opponent_vp(self) -> int:
        return self._vp_state.opponent_vp

    @property
    def vp_gained_this_step_player(self) -> int:
        return self._vp_state.vp_gained_this_step_player

    @property
    def vp_gained_this_step_opponent(self) -> int:
        return self._vp_state.vp_gained_this_step_opponent

    @property
    def max_turns(self) -> int:
        """Maximum agent steps per episode.

        When config.max_turns_override is set, returns that value (e.g. 100 for training).
        Otherwise uses game-clock-derived limit (n_rounds × active phases).
        """
        if self.config.max_turns_override is not None:
            return self.config.max_turns_override
        n_phases = len(BATTLE_PHASE_ORDER) - len(self._skip_phases)
        return self._game_clock.n_rounds * n_phases

    @property
    def n_actions(self) -> int:
        """Number of discrete actions per model (including stay)."""
        return self._action_handler.n_actions

    @property
    def opponent_action_space(self) -> spaces.Tuple:
        """Action space for opponent models (used by policies)."""
        return self._opponent_action_handler.action_space

    @staticmethod
    def _build_models(
        n: int,
        model_configs: list[Any] | None,
        n_objectives: int,
        max_groups: int,
    ) -> list[WargameModel]:
        """Shared helper to build a list of WargameModel instances."""
        result: list[WargameModel] = []
        increment = max(1, n // max_groups)
        for i in range(n):
            if model_configs is not None:
                mc = model_configs[i]
                group_id = mc.group_id
                max_wounds = mc.max_wounds
                oc = mc.oc
            else:
                group_id = i // increment
                max_wounds = 100
                oc = 1
            result.append(
                WargameModel(
                    location=np.zeros(2, dtype=int),
                    stats={"max_wounds": max_wounds, "current_wounds": max_wounds},
                    group_id=group_id,
                    distances_to_objectives=np.zeros([n_objectives, 2], dtype=int),
                    oc=oc,
                )
            )
        return result

    @staticmethod
    def create_wargame_models(config: WargameEnvConfig) -> list[WargameModel]:
        """Build the list of player wargame models from config."""
        return WargameEnv._build_models(
            config.number_of_wargame_models,
            config.models,
            config.number_of_objectives,
            config.max_groups,
        )

    @staticmethod
    def _create_opponent_models(config: WargameEnvConfig) -> list[WargameModel]:
        """Build the list of opponent models from config."""
        return WargameEnv._build_models(
            config.number_of_opponent_models,
            config.opponent_models,
            config.number_of_objectives,
            config.max_groups,
        )

    @staticmethod
    def create_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
        """Build the list of objectives from config.

        When ``objectives`` are provided, per-objective radius_size overrides
        the global value if set.
        """
        result: list[WargameObjective] = []
        for i in range(config.number_of_objectives):
            if (
                config.objectives is not None
                and config.objectives[i].radius_size is not None
            ):
                radius = config.objectives[i].radius_size
            else:
                radius = config.objective_radius_size

            result.append(
                WargameObjective(
                    location=np.zeros(2, dtype=int),
                    radius_size=radius,  # type: ignore[arg-type]
                )
            )
        return result

    def _get_obs(
        self, distance_cache: DistanceCache | None = None
    ) -> WargameEnvObservation:
        """Get the observation for the current state of the environment."""
        update_distances_to_objectives(
            self.wargame_models, self.objectives, distance_cache
        )
        if self.opponent_models:
            update_distances_to_objectives(self.opponent_models, self.objectives)

        clock_state = self._game_clock.state
        phase = clock_state.phase or BattlePhase.movement
        action_mask = self._action_handler.registry.get_model_action_masks(
            phase, len(self.wargame_models)
        )
        return build_observation(
            self.current_turn,
            self.wargame_models,
            self.objectives,
            self.config.max_groups,
            self.board_width,
            self.board_height,
            opponent_models=self.opponent_models,
            action_mask=action_mask,
            battle_round=clock_state.battle_round or 1,
            battle_phase_index=list(BattlePhase).index(phase),
            n_rounds=self._game_clock.n_rounds,
            control_range=self.config.objective_control_range,
            env_score_state=EnvScoreState(
                player_vp=self.player_vp,
                opponent_vp=self.opponent_vp,
            ),
        )

    def _get_info(self) -> WargameEnvInfo:
        return build_info(
            self.current_turn,
            self.wargame_models,
            self.objectives,
            (
                int(self.deployment_zone[0]),
                int(self.deployment_zone[1]),
                int(self.deployment_zone[2]),
                int(self.deployment_zone[3]),
            ),
            (
                int(self.opponent_deployment_zone[0]),
                int(self.opponent_deployment_zone[1]),
                int(self.opponent_deployment_zone[2]),
                int(self.opponent_deployment_zone[3]),
            ),
            self.config.max_groups,
            opponent_models=self.opponent_models,
            player_vp=self.player_vp,
            opponent_vp=self.opponent_vp,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WargameEnvObservation, dict[str, Any]]:
        super().reset(seed=seed)

        self.current_turn = 0
        self.last_reward = None
        self.last_step_context = None
        self._vp_state.reset()

        self._resolve_player_side()
        self._game_clock.reset()
        self._game_clock.skip_setup()
        # Clock is now at round 1, player_1, command phase

        # Place player models
        if self.config.has_fixed_model_positions:
            fixed_wargame_model_placement(
                self.wargame_models,
                self.config.models,  # type: ignore[arg-type]
            )
        else:
            wargame_model_placement(
                self.wargame_models,
                self.deployment_zone,
                self.config.group_max_distance,
                self.np_random,
            )

        # Place objectives
        if self.config.has_fixed_objective_positions:
            fixed_objective_placement(
                self.objectives,
                self.config.objectives,  # type: ignore[arg-type]
            )
        else:
            objective_placement(
                self.objectives,
                self.deployment_zone,
                self.board_width,
                self.board_height,
                self.np_random,
                self.opponent_deployment_zone,
            )

        # Place opponent models
        if self.opponent_models:
            if self.config.has_fixed_opponent_positions:
                fixed_wargame_model_placement(
                    self.opponent_models,
                    self.config.opponent_models,  # type: ignore[arg-type]
                )
            else:
                wargame_model_placement(
                    self.opponent_models,
                    self.opponent_deployment_zone,
                    self.config.group_max_distance,
                    self.np_random,
                )

        # If opponent goes first this round, auto-execute their full turn
        if self._is_opponent_active():
            self._execute_opponent_turn()

        self._skip_past_excluded_phases()

        cache = compute_distances(self.wargame_models, self.objectives)
        observation = self._get_obs(cache)
        info: WargameEnvInfo = self._get_info()

        if self.renderer is not None:
            self.renderer.setup(self)
            self.renderer.render(self)

        return observation, info.model_dump()

    def _apply_player_action(self, action: WargameEnvAction) -> None:
        self._action_handler.apply(
            action,
            self.wargame_models,
            self.board_width,
            self.board_height,
            self._action_handler.action_space,
        )

    def _apply_opponent_action(self) -> None:
        if self._opponent_policy is None or not self.opponent_models:
            return
        phase = self._game_clock.state.phase or BattlePhase.movement
        opp_mask = self._opponent_action_handler.registry.get_model_action_masks(
            phase, len(self.opponent_models)
        )
        opp_action = self._opponent_policy.select_action(
            self.opponent_models, self, action_mask=opp_mask
        )
        self._opponent_action_handler.apply(
            opp_action,
            self.opponent_models,
            self.board_width,
            self.board_height,
            self._opponent_action_handler.action_space,
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

    def _is_opponent_active(self) -> bool:
        state = self._game_clock.state
        return (
            state.active_player is not None and state.active_player != self._player_side
        )

    def _compute_primary_vp_earned(self) -> tuple[int, int]:
        """VP earned this scoring moment (delegates to vp module)."""
        return compute_primary_vp_earned(
            self.wargame_models,
            self.opponent_models,
            self.objectives,
            self.config.objective_control_range,
        )

    def _execute_opponent_turn(self) -> None:
        """Auto-execute all phases of the opponent's turn."""
        while not self._game_clock.is_game_over and self._is_opponent_active():
            phase_before = self._game_clock.state.phase or BattlePhase.command
            self._apply_opponent_action()
            self._game_clock.advance_phase()
            # Score opponent VP at end of their Command phase (from round 2)
            if (
                phase_before == BattlePhase.command
                and (self._game_clock.state.battle_round or 0) >= 2
            ):
                _, opp_vp = self._compute_primary_vp_earned()
                self._vp_state.award_opponent(opp_vp)

    def _should_skip_phase(self) -> bool:
        phase = self._game_clock.state.phase
        return phase is not None and phase in self._skip_phases

    def _first_active_phase(self) -> BattlePhase:
        """First battle phase the player actually steps in (not in skip_phases)."""
        for p in BATTLE_PHASE_ORDER:
            if p not in self._skip_phases:
                return p
        return BattlePhase.command  # fallback

    def _skip_past_excluded_phases(self) -> None:
        """Advance past player phases listed in ``skip_phases``."""
        while (
            not self._game_clock.is_game_over
            and not self._is_opponent_active()
            and self._should_skip_phase()
        ):
            self._game_clock.advance_phase()

    def step(
        self, action: WargameEnvAction
    ) -> tuple[WargameEnvObservation, float, bool, bool, dict[str, Any]]:
        self._vp_state.start_step()
        clock = self._game_clock.state
        phase_at_step_start = clock.phase or BattlePhase.command
        round_at_step_start = clock.battle_round or 1
        self._apply_player_action(action)

        if not self._game_clock.is_game_over:
            self._game_clock.advance_phase()

        self.current_turn += 1

        self._skip_past_excluded_phases()

        # Score player VP at the scoring moment (from round 2), before opponent turn.
        # When Command is skipped, the agent only steps in Movement; the board state
        # at start of Movement = end of Command, so we score when we're in the
        # first active phase (Movement by default), not only when phase is Command.
        if (
            phase_at_step_start == self._first_active_phase()
            and round_at_step_start >= 2
        ):
            player_vp_earned, _ = self._compute_primary_vp_earned()
            self._vp_state.award_player(player_vp_earned)

        # If clock rolled over to opponent's turn, auto-execute it
        if not self._game_clock.is_game_over and self._is_opponent_active():
            self._execute_opponent_turn()

        self._skip_past_excluded_phases()

        needs_mm = self.config.group_cohesion_enabled or (
            self.phase_manager is not None
            and self.phase_manager.needs_model_model_distances
        )
        cache = compute_distances(
            self.wargame_models,
            self.objectives,
            compute_model_model=needs_mm,
        )

        clock_state = self._game_clock.state
        phase = clock_state.phase or BattlePhase.command

        if self.phase_manager is not None:
            ctx = StepContext(
                distance_cache=cache,
                current_turn=self.current_turn,
                max_turns=self.max_turns,
                board_width=self.board_width,
                board_height=self.board_height,
                current_round=clock_state.battle_round or 0,
                battle_phase=phase,
                phase_at_step_start=phase_at_step_start,
            )
            self.last_step_context = ctx
            # Terminate when phase success criteria are met and that criteria ends the episode
            goal_reached = self.phase_manager.check_success(self, ctx)
            criteria_terminates = getattr(
                self.phase_manager.current_phase.criteria,
                "terminates_episode",
                True,
            )
            goal_ends_episode = goal_reached and criteria_terminates
            if self.config.max_turns_override is not None:
                is_terminated = (
                    check_max_turns_reached(self.current_turn, self.max_turns)
                    or goal_ends_episode
                )
            else:
                is_terminated = (
                    check_max_turns_reached(self.current_turn, self.max_turns)
                    or self._game_clock.is_game_over
                    or goal_ends_episode
                )
            reward = self.phase_manager.calculate_reward(self, ctx)
        else:
            # Legacy: terminate when all models at objectives (or max_turns / game_over)
            if self.config.max_turns_override is not None:
                is_terminated = check_max_turns_reached(
                    self.current_turn, self.max_turns
                ) or get_termination(cache)
            else:
                is_terminated = (
                    check_max_turns_reached(self.current_turn, self.max_turns)
                    or self._game_clock.is_game_over
                    or get_termination(cache)
                )
            reward = Reward().calculate_reward(self, cache)
            # Legacy terminal success bonus: applied once when all models are at
            # an objective and the episode terminates.
            if is_terminated and self.config.terminal_success_bonus != 0.0:
                if cache.all_models_at_objectives():
                    reward += float(self.config.terminal_success_bonus)

        observation = self._get_obs(cache)
        info = self._get_info()

        self.last_reward = reward
        return observation, reward, is_terminated, False, info.model_dump()

    def render(self) -> None:
        if self.renderer is not None:
            self.renderer.render(self)

        return None
