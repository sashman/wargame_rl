from __future__ import annotations

from typing import Any

import gymnasium as gym
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
from wargame_rl.wargame.envs.domain.game_clock import GameClock
from wargame_rl.wargame.envs.domain.placement import place_for_episode
from wargame_rl.wargame.envs.domain.termination import is_battle_over
from wargame_rl.wargame.envs.domain.turn_execution import (
    run_after_player_action,
    run_until_player_phase,
)
from wargame_rl.wargame.envs.env_components import (
    ActionHandler,
    DistanceCache,
    build_info,
    build_observation,
    compute_distances,
)
from wargame_rl.wargame.envs.mission import build_vp_calculator
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import (
    _auto_register,
    build_opponent_policy,
)
from wargame_rl.wargame.envs.renders import renderer
from wargame_rl.wargame.envs.reward.phase_manager import RewardPhaseManager
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import (
    BattlePhase,
    PlayerSide,
    TurnOrder,
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvInfo,
    WargameEnvObservation,
)
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER, GameState
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

        self._battle = _battle_from_config(config)
        self.wargame_models = self._battle.player_models
        self.objectives = self._battle.objectives
        self.opponent_models = self._battle.opponent_models
        self.deployment_zone = self._battle.deployment_zone
        self.opponent_deployment_zone = self._battle.opponent_deployment_zone

        # Last reward from step(); None until first step after reset
        self.last_reward: float | None = None
        self.last_reward_breakdown: dict[str, float] = {}
        self.episode_reward_breakdown: dict[str, float] = {}
        self.episode_reward_steps: int = 0

        # Reward phases (curriculum learning); always used for reward calculation
        self.phase_manager = RewardPhaseManager.from_configs(config.reward_phases)

        # Mission VP calculator (scores at end of command phase from round 2)
        self._vp_calculator = build_vp_calculator(
            config.mission.type, config.mission.params
        )

        # Last StepContext from step(); available for post-episode success checks
        self.last_step_context: StepContext | None = None

        # --- Opponent setup ---
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
        super().reset(seed=seed)

        self.current_turn = 0
        self.last_reward = None
        self.last_step_context = None
        self.last_reward_breakdown = {}
        self.episode_reward_breakdown = {}
        self.episode_reward_steps = 0

        self._battle.reset_for_episode()
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

    def step(
        self, action: WargameEnvAction
    ) -> tuple[WargameEnvObservation, float, bool, bool, dict[str, Any]]:
        self._battle.reset_vp_deltas()
        self._apply_player_action(action)

        self.current_turn += 1

        run_after_player_action(
            self._game_clock,
            self._skip_phases,
            self._player_side,
            self._apply_opponent_action,
            on_before_advance=self._on_before_advance,
        )

        needs_mm = self.phase_manager.needs_model_model_distances
        cache = compute_distances(
            self.wargame_models,
            self.objectives,
            compute_model_model=needs_mm,
        )

        is_terminated = is_battle_over(
            self._game_clock,
            self.current_turn,
            self.max_turns,
            self.config.max_turns_override,
            cache.all_models_at_objectives(),
        )

        clock_state = self._game_clock.state
        phase = clock_state.phase or BattlePhase.command

        ctx = StepContext(
            distance_cache=cache,
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            board_width=self.board_width,
            board_height=self.board_height,
            is_terminated=is_terminated,
            current_round=clock_state.battle_round or 0,
            battle_phase=phase,
        )
        self.last_step_context = ctx
        reward = self.phase_manager.calculate_reward(self, ctx)

        observation = self._get_obs(cache)
        info = self._get_info()

        self.last_reward = reward
        self.last_reward_breakdown = dict(self.phase_manager.last_reward_breakdown)
        for key, value in self.last_reward_breakdown.items():
            self.episode_reward_breakdown[key] = (
                self.episode_reward_breakdown.get(key, 0.0) + value
            )
        self.episode_reward_steps += 1
        return observation, reward, is_terminated, False, info.model_dump()

    def render(self) -> None:
        if self.renderer is not None:
            self.renderer.render(self)

        return None
