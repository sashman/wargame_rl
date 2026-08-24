from __future__ import annotations

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
from wargame_rl.wargame.envs.domain.battle_factory import unit_count
from wargame_rl.wargame.envs.domain.coherency_enforcement import apply_attrition
from wargame_rl.wargame.envs.domain.consolidate import consolidate_objective
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.domain.fight import (
    PairedFightResult,
    fight_eligible_units,
    resolve_fight,
)
from wargame_rl.wargame.envs.domain.game_clock import GameClock
from wargame_rl.wargame.envs.domain.placement import install_layout, place_for_episode
from wargame_rl.wargame.envs.domain.rules_quantities import (
    RulesQuantities,
    resolve_rules_quantities,
)
from wargame_rl.wargame.envs.domain.shooting import (
    PairedShootingResult,
    resolve_shooting_phase,
)
from wargame_rl.wargame.envs.domain.sight import (
    COVER,
    has_line_of_sight_between_points,
    visibility_matrix,
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
from wargame_rl.wargame.envs.env_components.coherency_tracker import CoherencyTracker
from wargame_rl.wargame.envs.env_components.exposure import (
    ExposureTracker,
    record_shooting_phase,
)
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_unit_shooting_masks,
    max_weapon_ranges,
)
from wargame_rl.wargame.envs.map_pool import MapPool
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
from wargame_rl.wargame.envs.state.restore import (
    restore_clock,
    restore_models,
    restore_objectives,
    restore_shooting_results,
)
from wargame_rl.wargame.envs.state.snapshot import (
    EpisodeProvenance,
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
from wargame_rl.wargame.envs.types.game_timing import BATTLE_PHASE_ORDER, GameState
from wargame_rl.wargame.envs.types.geometry import Polygon
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.envs.wargame_objective import WargameObjective

# Re-export for backward compatibility (tests import from here)
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
        build_info: bool = True,
    ):
        """Build the environment.

        `phase_position` shares curriculum progress with another environment.
        Training passes the eval env's position to every rollout env so they
        reward the phase the curriculum has actually reached.

        `build_info=False` returns an empty info dict from `reset` and `step`.
        The dict costs ~0.19 ms a step -- 50 dataclasses, a Pydantic model and a
        `model_dump()` -- and every caller on the training path discards it, so
        rollout and evaluation envs turn it off. It stays on by default because
        it is part of the Gymnasium contract and `simulate.py`, the baselines
        and the tests all read it.
        """
        self._build_info = build_info
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
                            # The budget, when set, is the width the per-model
                            # distance block actually has -- a pool episode may
                            # carry fewer objectives than the config declares.
                            number_of_objectives=(
                                config.objective_budget or config.number_of_objectives
                            )
                            * 2,
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

        # A weapon names an enemy UNIT, not a model, so the shooting slice is
        # one action per opponent unit -- 5 rather than 25 on a 25v25 board.
        self._action_handler = ActionHandler(
            config,
            n_shoot_targets=unit_count(
                config.number_of_opponent_models,
                config.max_groups,
                config.opponent_models,
            ),
            model_moves=[model.move for model in config.models or ()],
        )
        self.action_space = self._action_handler.action_space
        self._skip_phases = frozenset(config.skip_phases)
        # (battle_round, side) whose advance dice have been rolled. See
        # `_ensure_advance_rolls`.
        self._rolled_for: tuple[int, PlayerSide] | None = None
        # Resolved once, never per call: runtime reads plain floats off this and
        # never divides by the scale.
        self._rules_quantities = resolve_rules_quantities(config)

        self.renderer = renderer
        self._state_exporters: list[StateExporter] = state_exporters or []
        # Free text naming whatever chooses the player's actions, stamped into
        # a recording's provenance. The env cannot know it -- a checkpoint, a
        # baseline name and a human at a keyboard all look the same from here.
        self.driver_label: str | None = None
        # Set on every reset; declared here so `provenance` fails loudly rather
        # than with an AttributeError if it is read before the first one.
        self._episode_rng_state: dict[str, Any] = {}
        self._episode_combat_seed = 0
        self._episode_seed: int | None = None

        self.window = None
        self.clock = None

        self.current_turn = 0
        self._player_side = self._initial_player_side()
        self._game_clock = GameClock(n_rounds=config.number_of_battle_rounds)

        self._battle = _battle_from_config(config)
        # Loaded and checked once: every map parsed, every polygon built, every
        # count held against the observation budgets. A draw is then an index.
        self._coherency_attrition = config.coherency.attrition
        # Models each force destroyed itself this step under the coherency
        # attrition rule. Reset per step; read by `step` when attributing kills.
        self._attrition_deaths_player = 0
        self._attrition_deaths_opponent = 0
        self._map_pool = MapPool.from_config(config)
        self._map_name: str | None = None
        # A drawn layout, like generated terrain below, has to exist before the
        # first reset: network sizing reads an observation, and with a pool that
        # observation must already be at its padded width.
        if self._map_pool is not None:
            layout = self._map_pool.draw(self.np_random)
            install_layout(self._battle, config, layout)
            self._map_name = layout.name
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
        # Separate stream: see `_roll_advance_dice`. Never drawn from unless the
        # scenario registers advance bins, so existing configs are untouched.
        self._advance_rng: np.random.Generator = np.random.default_rng()
        # Same discipline again for the charge's 2D6: drawn only when the
        # scenario fights in melee, so no existing config's dice shift.
        self._charge_rng: np.random.Generator = np.random.default_rng()
        self._last_player_shooting_results: list[PairedShootingResult] = []
        self._last_opponent_shooting_results: list[PairedShootingResult] = []
        # Melee results ride beside the shooting ones but in their own lists:
        # the renderer draws a tracer for every damaging SHOOTING result, and a
        # melee hit at base contact would render as an inch-long stub.
        self._last_player_fight_results: list[PairedFightResult] = []
        self._last_opponent_fight_results: list[PairedFightResult] = []

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
        # Running sum of the per-step totals. Kept beside the breakdown rather
        # than summed from it on demand, so the HUD and a recording report the
        # same number even if a phase's total is not exactly its components.
        self.episode_reward: float = 0.0

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
        self._coherency_tracker = CoherencyTracker()
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
                n_shoot_targets=unit_count(
                    config.number_of_wargame_models,
                    config.max_groups,
                    config.models,
                ),
                # The opponent's own list: one config builds both handlers, so
                # reading `config.models` here would give the enemy our speed.
                model_moves=[model.move for model in config.opponent_models or ()],
            )
            self._opponent_policy: OpponentPolicy | None = build_opponent_policy(
                config.opponent_policy,  # type: ignore[arg-type]
                self,
            )
        else:
            self._opponent_action_handler = ActionHandler(config, n_models=0)
            self._opponent_policy = None

    @property
    def provenance(self) -> EpisodeProvenance:
        """How to boot this episode again. Valid only after `reset`.

        The config rides along in full so a recording is self-contained: a path
        can be edited or deleted between recording a match and wanting to
        recreate it, and a scenario that has drifted reproduces something that
        merely looks like the recording.
        """
        return EpisodeProvenance(
            config=self.config.model_dump(mode="json"),
            rng_state=self._episode_rng_state,
            combat_seed=self._episode_combat_seed,
            seed=self._episode_seed,
            driver=self.driver_label,
        )

    @property
    def opponent_policy(self) -> OpponentPolicy | None:
        """The policy driving the opponent, or None when there are no opponents."""
        return self._opponent_policy

    def set_opponent_policy(self, policy: OpponentPolicy | None) -> None:
        """Replace the opponent's policy for the rest of the episode.

        Exists for the debug session, which wraps the configured policy to let a
        human override individual opponent models. Reaching into
        `_opponent_policy` from outside would work and would silently break the
        first time the attribute moved.
        """
        self._opponent_policy = policy

    def reseed_combat(self, seed: int) -> None:
        """Reseed the dice without touching the layout or the episode.

        `reset(options={"combat_seed": ...})` is the other way to set this, but
        it restarts the episode. Re-running one step against fresh dice — hammer
        redo and watch the outcome spread — needs the dice alone to move, which
        is exactly the separation `measure-noise-floor` is built on.
        """
        self._combat_rng = np.random.default_rng(seed)
        # Offsets so neither move stream is a copy of the combat stream.
        self._advance_rng = np.random.default_rng(
            None if seed is None else seed + 1_000_003
        )
        self._charge_rng = np.random.default_rng(
            None if seed is None else seed + 2_000_003
        )

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
    def deployment_outline(self) -> Polygon | None:
        """The player zone actually deployed into, or None for the rectangle.

        A property rather than an attribute copied at construction, unlike
        `deployment_zone`: a pool draws a different map every episode and
        each brings its own zones, so a snapshot taken once would describe
        the first map forever.
        """
        return self._battle.deployment_outline

    @property
    def opponent_deployment_outline(self) -> Polygon | None:
        """The opponent zone actually deployed into, or None for the rectangle."""
        return self._battle.opponent_deployment_outline

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
    def last_player_fight_results(self) -> list[PairedFightResult]:
        """Melee the player resolved during the most recent step."""
        return list(self._last_player_fight_results)

    @property
    def last_opponent_fight_results(self) -> list[PairedFightResult]:
        """Melee the opponent resolved during the most recent step."""
        return list(self._last_opponent_fight_results)

    @property
    def opponent_action_space(self) -> spaces.Tuple:
        """Action space for opponent models (used by policies)."""
        return self._opponent_action_handler.action_space

    @property
    def terrain(self) -> "Terrain":
        """Read-only access to terrain footprints."""
        return self._battle.terrain

    @property
    def map_name(self) -> str | None:
        """Name of the layout this episode is being played on, or None without a pool.

        The only way to attribute an episode to a map once the pool is drawing
        them, which a per-map breakdown of a training run needs.
        """
        return self._map_name

    @property
    def rules_quantities(self) -> RulesQuantities:
        """Rules distances in board units, resolved once at construction."""
        return self._rules_quantities

    @property
    def player_max_ranges(self) -> np.ndarray:
        """Longest weapon range per player model, resolved once from config.

        Exposed on `BattleView` so the observation builder stops recomputing it
        from config on every step — the values are static for the whole run.
        """
        return self._player_max_ranges

    @property
    def player_advance_legality(self) -> np.ndarray:
        """`(n_models, n_advance_actions)` — which advance rungs this turn allows.

        Exposed on `BattleView` because the observation builder masks with it and
        must not reach into the action handler, and because the mirrored view has
        to answer it for the opponent's own models rather than the player's.
        """
        return self._action_handler.advance_legality(self.wargame_models)

    @property
    def player_charge_legality(self) -> np.ndarray:
        """`(n_models, n_move_actions)` — which charge moves the rules allow.

        On the view for the same reason `player_advance_legality` is: the
        observation builder masks with it and must not reach into the action
        handler, and the mirrored view has to answer for the OPPONENT's models
        when the opponent seat asks.
        """
        return self._action_handler.charge_legality(
            self.wargame_models, self.opponent_models
        )

    @property
    def opponent_max_ranges(self) -> np.ndarray:
        """Longest weapon range per opponent model, resolved once from config.

        The mirror of `player_max_ranges`. It existed only as a private
        attribute, which the threat overlay cannot reach: a renderer reading one
        side off the protocol and the other off `_opponent_max_ranges` would
        draw one army's reach from the engine and the other's from a guess.
        """
        return self._opponent_max_ranges

    @property
    def player_side(self) -> PlayerSide:
        """Which clock seat the player's army occupies this episode.

        Re-resolved every `reset` from `turn_order`, and under `random` it is a
        coin flip — so an episode cannot be attributed to a turn order without
        it. Note that reading it costs a draw from the layout RNG *only* under
        `random`, which is why a config fixing `turn_order` sits on a different
        layout stream from the same config leaving it random.
        """
        return self._player_side

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
    def firepower_ratio(self) -> float | None:
        """(enemies we can shoot) / (our models they can shoot) over the episode.

        The exchange-ratio measure: above 1.0 the army brings more guns to bear
        than it exposes. Prefer it to `exposure_rate` when the question is
        whether the policy is *choosing* its fights — low exposure alone cannot
        tell manoeuvre apart from hiding. None when `track_exposure` is off, or
        when our side was never exposed at all.
        """
        return self._exposure_tracker.firepower_ratio

    @property
    def coherency_rate(self) -> float | None:
        """Share of the player's unit-movement-phases spent in rules coherency.

        Always available -- unlike the exposure metrics there is no config flag,
        because this costs one predicate evaluation per movement phase and a run
        training under the coherency rule with no record of whether it held
        formation is the situation this exists to prevent.

        **Read it with `models_out_of_coherency`.** This number is confounded
        with squad size: a unit shot down to one model is coherent by
        definition, so it rises as an army dies.
        """
        return self._coherency_tracker.coherency_rate

    @property
    def models_out_of_coherency(self) -> float | None:
        """Mean player models outside their unit's coherent body, per phase."""
        return self._coherency_tracker.models_out_of_coherency

    @property
    def intended_coherency_rate(self) -> float | None:
        """Share of unit-samples the POLICY put in coherency, before the revert.

        The number to read when `coherency.enforce_move` is on. `coherency_rate`
        then describes the board the referee left, which is legal by
        construction and says nothing about what was learned -- a policy
        intending 0.630 reads 1.000 there.
        """
        return self._coherency_tracker.intended_coherency_rate

    @property
    def intended_models_out_of_coherency(self) -> float | None:
        """Models per phase the policy left adrift, before the revert."""
        return self._coherency_tracker.intended_models_out_of_coherency

    @property
    def models_reverted_last_move(self) -> int:
        """Models the referee dragged back on the last movement phase.

        The tax, and it was computed every step and read by nothing.
        """
        return self._action_handler.models_reverted_last_move

    def has_line_of_sight_between_points(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> bool:
        """True if sight is clear between two board points.

        Single-pair convenience for the renderer and for tests. Anything asking
        about many pairs should take `line_of_sight_matrix` instead.
        """
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
        """``(P, Q)`` sight between two sets of points, traced in one pass.

        Boolean: True where the target can be shot at *at all* — this is the
        predicate both shooting masks and the exposure scan use. **It is the
        centre ray only**: whether the target is *fully* visible or merely in
        cover is a question about a corridor, which needs the two models' base
        radii, and this seam carries positions. `_unit_cover` asks that question
        separately through `visibility_between`, with the models in hand.

        Only terrain blocks — see `domain/sight.py` for why models do not.
        """
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
        """``(P, Q)`` of HIDDEN / COVER / CLEAR between two sets of points.

        Terrain is the only blocker; models do not occlude (`domain/sight.py`).
        The models are still needed, for their **base radii** — those set the
        width of the corridor traced, and a target only partly visible along it
        is in cover.

        `edges=False` traces only the centre ray, which makes the answer binary
        (CLEAR or HIDDEN, never COVER). For callers that only need "can this be
        shot at", that is a third of the work.
        """
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

    def _resolve_fight_phase(self, state: GameState) -> None:
        """Both sides trade melee blows, on the boundary leaving the fight phase.

        ⚠ **The fight phase carries no agent action, and stays in
        `skip_phases`.** Measured: `registry.get_action_mask(BattlePhase.fight)`
        offers exactly ONE legal action per model (STAY) on every shipped
        config, so stepping it would cost an agent step per round -- +50% of
        episode length on the golden config -- to make a decision with one
        option. `on_before_advance` fires on skipped phases (only the opponent's
        action is guarded), which is how `_regain_coherency` already works.

        Both players act in this phase per `docs/rules/12-fight-phase.md`, so
        both forces swing on the same boundary; the ACTIVE player's units
        resolve first, which is v1's stand-in for alternating activation.

        Dice come from `_combat_rng`. Unlike the advance roll this needs no
        dedicated stream: it draws only when a fight actually resolves, so a
        config with melee off draws nothing and every existing dice sequence is
        untouched.
        """
        if not self.config.melee.enabled:
            return
        if state.phase is not BATTLE_PHASE_ORDER[-1] or state.active_player is None:
            return
        engagement_range = self._rules_quantities.engagement_range
        base_diameter = 2.0 * self._rules_quantities.base_radius
        player_side = (
            self.wargame_models,
            self.opponent_models,
            [cfg.melee_weapons for cfg in self.config.models or []],
            True,
        )
        opponent_side = (
            self.opponent_models,
            self.wargame_models,
            [cfg.melee_weapons for cfg in self.config.opponent_models or []],
            False,
        )
        order = (
            [player_side, opponent_side]
            if state.active_player == self._player_side
            else [opponent_side, player_side]
        )
        # Captured BEFORE anybody swings. `12-fight-phase.md` makes a unit
        # eligible if it "was engaged at the start of this step", and the
        # consolidate step then keys on "was eligible to fight this phase" --
        # both of which are false of a unit whose only contact the first side to
        # fight has already killed.
        eligible = {
            is_player: set(
                fight_eligible_units(
                    attackers,
                    defenders,
                    engagement_range=engagement_range,
                    base_diameter=base_diameter,
                )
            )
            for attackers, defenders, _weapons, is_player in order
        }
        for attackers, defenders, weapons, is_player in order:
            results = resolve_fight(
                attackers,
                defenders,
                self._combat_rng,
                attacker_weapons=weapons,
                engagement_range=engagement_range,
                base_diameter=base_diameter,
            )
            if is_player:
                self._last_player_fight_results.extend(results)
            else:
                self._last_opponent_fight_results.extend(results)
        # The rules' own order: pile-in, fight, THEN consolidate, with every
        # unit fighting before any unit consolidates.
        for attackers, defenders, _weapons, is_player in order:
            self._consolidate(attackers, defenders, eligible[is_player])
        # ⚠ Cleared for BOTH forces, here rather than in `begin_turn`. Both
        # sides fight on the active player's boundary, and only the active
        # player can have charged this turn -- but `begin_turn` clears the side
        # whose turn is starting, so the opposing force's flag would survive
        # into the next turn's fight and buy it a priority it did not earn.
        for models in (self.wargame_models, self.opponent_models):
            for model in models:
                model.charged_this_turn = False

    def _consolidate(
        self,
        models: list[WargameModel],
        enemy_models: list[WargameModel],
        eligible_units: set[int],
    ) -> None:
        """Run the consolidate step in Objective mode for one force.

        ⚠ **Expect this to fire almost never, and that is the rule rather than a
        limitation of the implementation.** The three modes are assessed in order
        and the first match is compulsory, so a unit still in contact is in
        Ongoing mode and a unit with any enemy within 3" is in Engaging mode --
        neither of which reaches Objective. See `domain/consolidate.py` for what
        is deferred and why.
        """
        consolidate_objective(
            models,
            enemy_models,
            self.objectives,
            eligible_units=eligible_units,
            # Reads the board live, and through `compute_distances` -- so "in
            # range of an objective" here is the same test scoring uses, rather
            # than a fourth copy of it.
            objective_offsets=lambda: (
                compute_distances(models, self.objectives).model_obj_norms_offset
            ),
            max_distance=self._rules_quantities.scale.to_units(
                self.config.melee.consolidate_distance
            ),
            engagement_range=self._rules_quantities.engagement_range,
            base_radius=self._rules_quantities.base_radius,
            board=(float(self.board_width), float(self.board_height)),
            coherency_nearest=self._rules_quantities.scale.to_units(
                self.config.coherency.nearest_distance
            ),
            coherency_furthest=self._rules_quantities.scale.to_units(
                self.config.coherency.furthest_distance
            ),
        )

    def _regain_coherency(self, state: GameState) -> None:
        """Apply `03-moving.md` § Regaining coherency to the active player's force.

        There is no End of Turn phase on this clock -- `BattlePhase` is the five
        phases of a turn, and adding a sixth would change `max_turns`, every
        config's `skip_phases` and therefore episode length, voiding every
        measured result before coherency did anything. So the last phase's
        boundary stands in for it: leaving `fight` is the end of that player's
        turn whether or not the phase was skipped, because skipped phases still
        tick.

        The casualties are **not** credited to the other side. The rule says
        they trigger nothing that fires when a model is destroyed, and routing
        them through the kill counter would pay `model_kills` for deaths nobody
        caused.
        """
        if not self._coherency_attrition:
            return
        if state.phase is not BATTLE_PHASE_ORDER[-1] or state.active_player is None:
            return
        models = (
            self.wargame_models
            if state.active_player == self._player_side
            else self.opponent_models
        )
        destroyed = apply_attrition(
            models,
            self._rules_quantities.scale.to_units(
                self.config.coherency.nearest_distance
            ),
            self._rules_quantities.scale.to_units(
                self.config.coherency.furthest_distance
            ),
        )
        # Kept so `step` can take these back out of its alive-diff. Without
        # this the rule's "triggers nothing that fires on a model being
        # destroyed" is documented and not implemented: the diff is taken
        # across the whole step and attrition runs inside it, so the OTHER
        # side is paid for models this force removed itself.
        if state.active_player == self._player_side:
            self._attrition_deaths_player += len(destroyed)
        else:
            self._attrition_deaths_opponent += len(destroyed)

    def _record_coherency(self) -> None:
        """Fold the player's formation into the episode's coherency totals.

        Records BOTH the board after enforcement and the move the policy
        proposed before it. Under `enforce_move` those are different questions
        and only the second says anything about the policy.
        """
        self._coherency_tracker.record_intent(
            self._action_handler.intended_coherency_last_move
        )
        self._coherency_tracker.record(
            positions=np.array([m.location for m in self.wargame_models], dtype=float),
            group_ids=np.array(
                [m.group_id for m in self.wargame_models], dtype=np.intp
            ),
            alive_mask=alive_mask_for(self.wargame_models),
            base_radii=np.array(
                [m.base_radius for m in self.wargame_models], dtype=float
            ),
            nearest_distance=self._rules_quantities.scale.to_units(
                self.config.coherency.nearest_distance
            ),
            furthest_distance=self._rules_quantities.scale.to_units(
                self.config.coherency.furthest_distance
            ),
        )

    def _roll_advance_dice(self, active_side: PlayerSide) -> None:
        """Clear the turn's move state and roll one D6 per unit, for one side.

        The rules make the advance roll "before moving", and the policy chooses
        its move afterwards -- so the roll has to be visible in the observation
        the policy conditions on. It is: `step()` applies the action, THEN runs
        the clock (which calls this on the command -> movement boundary), THEN
        builds the observation. So a roll made here lands in the observation
        that precedes the movement action it governs.

        ⚠ Drawn from a DEDICATED generator, and only when the scenario has
        advance bins. Sharing `np_random` or `_combat_rng` would shift every
        later draw, so adding this feature would silently change the terrain,
        the deployment and every dice roll of every existing config.
        """
        if self._action_handler.advance_slice is None:
            return
        models = self._models_for(active_side)
        # One roll per UNIT, shared by its models -- the rules roll for the
        # unit, not the model.
        rolls: dict[int, float] = {}
        for model in models:
            group = int(model.group_id)
            if group not in rolls:
                rolls[group] = float(self._advance_rng.integers(1, 7))
            model.advance_roll = rolls[group]

    def _roll_charge_dice(self, active_side: PlayerSide) -> None:
        """Roll 2D6 per unit for the charge, at the start of the side's turn.

        `docs/rules/11-charge-phase.md` rolls after the declaration and before
        targets are chosen. Here the roll comes first and is visible in the
        observation the charge action is chosen from -- the same divergence, and
        for the same reason, as the advance roll: legality is gated on the roll,
        so a declaration made before it would have no legal distance to take and
        the policy could not condition on what it is committing to.
        `DEFERRED: charge.blind_declaration`.

        One roll per UNIT, shared by its models, from a dedicated stream drawn
        only when the scenario fights in melee.
        """
        if not self.config.melee.enabled:
            return
        rolls: dict[int, float] = {}
        for model in self._models_for(active_side):
            group = int(model.group_id)
            if group not in rolls:
                rolls[group] = float(
                    self._charge_rng.integers(1, 7) + self._charge_rng.integers(1, 7)
                )
            model.charge_roll = rolls[group]

    def _models_for(self, active_side: PlayerSide) -> list[WargameModel]:
        """The force belonging to `active_side`."""
        return (
            self.wargame_models
            if active_side == self._player_side
            else self.opponent_models
        )

    def _begin_side_turn(self, active_side: PlayerSide) -> None:
        """Clear every model's per-turn state, for one side, unconditionally.

        ⚠ This loop used to live inside `_roll_advance_dice`, BEHIND its
        `advance_slice is None` early return -- so on every config without
        advance rungs, which is most of them, `begin_turn()` was never called at
        all. That was harmless only by coincidence: the sole writer of
        `advanced_this_turn` is `declare_move_types`, which is gated on the same
        condition, so the flags it clears were provably already clear.

        It stops being harmless the moment any OTHER mechanic keeps per-turn
        state. A charge flag hung on `begin_turn()` would be set once and never
        cleared again for the rest of the episode, on 20 of 22 shipped configs,
        with nothing raising. Hoisting it is a no-op today and correct
        tomorrow, which is the only reason to do it before the feature lands
        rather than with it.
        """
        for model in self._models_for(active_side):
            model.begin_turn()

    def _ensure_advance_rolls(self) -> None:
        """Roll each unit's D6 once, at the START of the side's turn.

        ⚠ The roll used to happen on the command→movement boundary, which was
        right while the move type was chosen in the movement phase. It is now
        declared in the **command** phase, so a roll taken on the way out of it
        would leave every unit declaring blind — and, because a rung's legality
        is gated on `M + roll`, would leave no rung legal at all.

        Idempotent and keyed on `(battle_round, active_player)` rather than
        hooked to a phase transition, because the command phase is the FIRST of
        a side's turn: there is no preceding phase within the turn to hang it
        on, and the very first turn of an episode never advances into it.
        """
        state = self._game_clock.state
        if state.active_player is None or state.battle_round is None:
            return
        key = (int(state.battle_round), state.active_player)
        if self._rolled_for == key:
            return
        self._rolled_for = key
        self._begin_side_turn(state.active_player)
        self._roll_advance_dice(state.active_player)
        self._roll_charge_dice(state.active_player)

    def _on_before_advance(self, clock: GameClock) -> None:
        """Resolve the fight, regain coherency, and score VP at the command boundary."""
        state = clock.state
        # ⚠ ORDER IS LOAD-BEARING. Both hang off the same boundary -- leaving
        # `fight`, which is `BATTLE_PHASE_ORDER[-1]` and so stands in for end of
        # turn. You fight, THEN the survivors are culled back into coherency; a
        # unit shredded in melee can lose further models to attrition in the
        # same step, which is the rule. Reversing them would cull first and let
        # models that should have swung die before they did.
        self._resolve_fight_phase(state)
        self._regain_coherency(state)
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
        # Before the observation, never after: the command-phase declaration is
        # chosen from this observation and has to see the turn's roll. Every
        # observation the player acts on comes through here, so this is the one
        # place that guarantees it. Idempotent per (round, side).
        self._ensure_advance_rolls()
        return build_observation(
            self,
            distance_cache=distance_cache,
            action_registry=self._action_handler.registry,
        )

    def _get_info(self) -> WargameEnvInfo:
        return build_info(self)

    def _info_dict(self) -> dict[str, Any]:
        """The Gymnasium info mapping, or an empty one when info is disabled."""
        if not self._build_info:
            return {}
        dumped: dict[str, Any] = self._get_info().model_dump()
        return dumped

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WargameEnvObservation, dict[str, Any]]:
        """Start a new episode.

        `options["combat_seed"]` seeds the dice independently of `seed`, which
        otherwise drives both the layout and the rolls. Varying one with the
        other held fixed is what separates "this scenario is hard" from "the
        dice went badly", and the two are indistinguishable while a single seed
        controls both. Absent, the combat seed is derived from `seed` as before.

        `options["augment_start"]` opts in to `start_on_objective_probability`,
        the training-time start-state augmentation. It is **opt-in rather than
        opt-out on purpose**: a training loop that forgets to ask simply trains
        the control, which is bit-identical and therefore obvious, whereas an
        evaluation that forgot to switch it off would score a different scenario
        and look entirely plausible doing it.
        """
        super().reset(seed=seed)
        # Captured *before* the first draw, so restoring it and resetting again
        # replays every draw this episode makes. A seed cannot stand in for it:
        # a training rollout resets without one, so its layout is a point in a
        # continuing stream that no integer names. See `EpisodeProvenance`.
        self._episode_rng_state = dict(self.np_random.bit_generator.state)
        self._episode_seed = seed

        # The draw happens either way. Skipping it when a combat seed is given
        # would shift every later draw from `np_random`, so the layout would
        # change too and the two sources of variance would still be entangled --
        # the exact thing this option exists to separate.
        derived_combat_seed = int(self.np_random.integers(0, 2**31))
        explicit_combat_seed = (options or {}).get("combat_seed")
        self._episode_combat_seed = (
            derived_combat_seed
            if explicit_combat_seed is None
            else int(explicit_combat_seed)
        )
        self._combat_rng = np.random.default_rng(self._episode_combat_seed)
        # ⚠ These two were NEVER seeded here. `_advance_rng` was built once in
        # `__init__` from OS entropy and only ever re-seeded by `reseed_combat`,
        # which `reset` does not call -- so advance dice were a continuing
        # stream no seed named, and an advance episode could not be reproduced
        # from `seed=` or replayed from `EpisodeProvenance` at all. Derived from
        # the episode's combat seed with distinct offsets, so a seeded episode
        # now reproduces its move dice as well as its shooting.
        self._advance_rng = np.random.default_rng(self._episode_combat_seed + 1_000_003)
        self._charge_rng = np.random.default_rng(self._episode_combat_seed + 2_000_003)
        self._last_player_shooting_results = []
        self._last_opponent_shooting_results = []
        self._last_player_fight_results = []
        self._last_opponent_fight_results = []
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
        self.episode_reward = 0.0
        self._exposure_tracker.reset()
        self._coherency_tracker.reset()

        self._battle.reset_for_episode()
        self.phase_manager.reset_episode()
        self._resolve_player_side()
        self._game_clock.reset()
        self._game_clock.skip_setup()
        # Clock is now at round 1, player_1, command phase

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
        info = self._info_dict()

        if self.renderer is not None:
            self.renderer.setup(self)
            self.renderer.render(self)

        if self._state_exporters:
            snapshot = self.to_snapshot()
            for exporter in self._state_exporters:
                exporter.on_reset(snapshot, self.provenance)

        return observation, info

    def _resolve_shooting_action(
        self,
        action: WargameEnvAction,
        attackers: list[WargameModel],
        targets: list[WargameModel],
        action_handler: ActionHandler,
        attacker_configs: list[ModelConfig] | None,
    ) -> list[PairedShootingResult]:
        """Decode the action into shots, then let the domain resolve them.

        Returns one PairedShootingResult per model that actually fired.

        Cover is resolved here, once for the whole phase, and handed to the
        domain as a mask. Sight is a batch operation; asking it per shot would
        put the environment's most expensive query inside a python loop.
        """
        shots = action_handler.decode_shooting_targets(action, len(attackers))
        return resolve_shooting_phase(
            shots=shots,
            attackers=attackers,
            targets=targets,
            attacker_weapons=[cfg.weapons for cfg in attacker_configs or []],
            rng=self._combat_rng,
            cover=self._cover_mask(shots, attackers, targets),
        )

    def _cover_mask(
        self,
        shots: list[tuple[int, int]],
        attackers: list[WargameModel],
        targets: list[WargameModel],
    ) -> np.ndarray | None:
        """``(n_attackers, n_target_units)`` — True where the *unit* has cover.

        Cover is a unit-level, all-or-nothing property in the rules: a unit has
        it against an attack only when **every** model in it is in a terrain area
        or not fully visible, so *"one model of a unit standing in the open
        denies cover to the whole unit"*. Reducing with `all` rather than `any`
        is that sentence.

        Only the declared (attacker, unit) pairs are traced, expanded to the
        unit's living models -- a handful out of the full product. Returns None
        when nothing was declared, so an empty phase costs nothing.
        """
        if not shots or not attackers or not targets:
            return None
        groups = np.array([m.group_id for m in targets], dtype=int)
        alive = np.array([m.is_alive for m in targets], dtype=bool)
        n_groups = int(groups.max()) + 1 if len(groups) else 0

        candidates = np.zeros((len(attackers), len(targets)), dtype=bool)
        declared = np.zeros((len(attackers), n_groups), dtype=bool)
        for attacker_idx, target_group in shots:
            if 0 <= target_group < n_groups and attacker_idx < len(attackers):
                declared[attacker_idx, target_group] = True
                candidates[attacker_idx, (groups == target_group) & alive] = True
        if not candidates.any():
            return None

        visibility = self.visibility_between(
            np.array([m.location for m in attackers], dtype=float),
            np.array([m.location for m in targets], dtype=float),
            candidates,
            origin_models=attackers,
            target_models=targets,
        )
        model_in_cover = visibility == COVER
        unit_in_cover = np.zeros((len(attackers), n_groups), dtype=bool)
        for group in range(n_groups):
            members = (groups == group) & alive
            if not members.any():
                continue
            # Every living model of the unit must be covered, and only for the
            # attackers that actually declared against it -- an undeclared pair
            # was never traced, so its cells are vacuously True under `all`.
            unit_in_cover[:, group] = (
                model_in_cover[:, members].all(axis=1) & declared[:, group]
            )
        return unit_in_cover

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
                self._action_handler,
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
                enemy_models=self.opponent_models,
            )
            if phase == BattlePhase.movement:
                self._record_coherency()

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
        advance_slice = handler.advance_slice
        if advance_slice is not None and phase is BattlePhase.movement:
            # The same absolute-rung gate the player gets. Leaving it off would
            # let the opponent pick a rung its roll cannot reach and silently
            # receive a shorter move -- the asymmetry that voids a bar.
            mask[:, advance_slice.start : advance_slice.end] &= (
                handler.advance_legality(self.opponent_models)
            )
        if self.config.melee.enabled and phase is BattlePhase.charge:
            # Both seats or neither: an unmasked charge would let the opponent
            # declare one its roll cannot reach, or from a unit the rules make
            # ineligible, and nothing downstream re-checks.
            movement_slice = handler.movement_slice
            mask[:, movement_slice.start : movement_slice.end] &= (
                handler.charge_legality(self.opponent_models, self.wargame_models)
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

        mask[:, shooting_slice.start : shooting_slice.end] &= (
            compute_unit_shooting_masks(
                np.array([m.location for m in self.opponent_models]),
                np.array([m.location for m in self.wargame_models]),
                opp_alive,
                alive_mask_for(self.wargame_models),
                self._opponent_max_ranges,
                self.line_of_sight_matrix,
                np.array([m.group_id for m in self.wargame_models], dtype=int),
                shooting_slice.end - shooting_slice.start,
                player_advanced=np.array(
                    [
                        m.advanced_this_turn or m.fell_back_this_turn
                        for m in self.opponent_models
                    ]
                ),
                engagement_range=self._rules_quantities.engagement_range,
                base_diameter=2.0 * self._rules_quantities.base_radius,
                # ⚠ Both seats or neither. This mask is a hand-written duplicate
                # of the one `build_observation` applies to the player, and a
                # legality overlay added to one and not the other is invisible:
                # nothing downstream re-checks, so the opponent would simply
                # shoot targets the rules forbid. Shooting alone already
                # measures a 24.6 vp seat asymmetry on one golden config.
                exclude_engaged_targets=self.config.melee.enabled,
            )
        )
        return mask

    def _record_exposure(self, opp_alive: np.ndarray) -> None:
        """Sample both sides of the shooting exchange into the exposure tracker."""
        record_shooting_phase(
            self._exposure_tracker,
            self.wargame_models,
            self.opponent_models,
            opp_alive,
            self._player_max_ranges,
            self._opponent_max_ranges,
            self.terrain.footprints,
            self.line_of_sight_matrix,
        )

    def _apply_opponent_action(self) -> None:
        if self._opponent_policy is None or not self.opponent_models:
            return
        # The opponent reads the board directly rather than through `_get_obs`,
        # so its own turn's roll has to be ensured here or it declares blind.
        self._ensure_advance_rolls()
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
                self._opponent_action_handler,
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
                enemy_models=self.wargame_models,
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
        self._last_player_fight_results = []
        self._last_opponent_fight_results = []
        self._attrition_deaths_player = 0
        self._attrition_deaths_opponent = 0
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

        # Damage is damage, whichever phase dealt it -- the shaping terms that
        # read these price wounds, not weapons.
        p_dmg = sum(
            r.result.damage_dealt for r in self._last_player_shooting_results
        ) + sum(r.result.damage_dealt for r in self._last_player_fight_results)
        o_dmg = sum(
            r.result.damage_dealt for r in self._last_opponent_shooting_results
        ) + sum(r.result.damage_dealt for r in self._last_opponent_fight_results)
        # The alive-diff spans the whole step, and coherency attrition runs
        # inside it, so a force's own attrition losses would otherwise read as
        # the other side's kills -- `killing` paying +5 a model for deaths
        # nobody caused, and `models_lost` charging for them. `03-moving.md`
        # § Regaining coherency: these are destroyed but trigger nothing that
        # fires when a model is destroyed.
        p_kills = max(
            0,
            sum(
                1
                for i, m in enumerate(self.opponent_models)
                if i < len(opp_alive_before) and opp_alive_before[i] and not m.is_alive
            )
            - self._attrition_deaths_opponent,
        )
        o_kills = max(
            0,
            sum(
                1
                for i, m in enumerate(self.wargame_models)
                if i < len(player_alive_before)
                and player_alive_before[i]
                and not m.is_alive
            )
            - self._attrition_deaths_player,
        )
        # Attribute each kill to the model that fired it, so shooting reward
        # lands on that model's advantage rather than being shared flat.
        p_kills_by_model = np.zeros(len(self.wargame_models), dtype=np.int64)
        for shot in self._last_player_shooting_results:
            if shot.killed and shot.attacker_idx < len(p_kills_by_model):
                p_kills_by_model[shot.attacker_idx] += 1
        # ⚠ Melee kills count here too. The GLOBAL kill counter is an alive-diff
        # and so picks them up for free, but this vector was built from shooting
        # results alone -- so without this a fight-phase kill would pay the
        # global `killing` calculator and pay `model_kills` nothing. On a
        # lineage where 53.7% of income is already global against a script's
        # 25.8%, and whose standing diagnosis is a difference-reward problem,
        # adding a damage channel that is credited only globally would deepen
        # the exact defect three reward terms have already failed against.
        for blow in self._last_player_fight_results:
            if blow.killed and blow.attacker_idx < len(p_kills_by_model):
                p_kills_by_model[blow.attacker_idx] += 1

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
        info = self._info_dict()

        self.last_reward = reward
        self.last_reward_breakdown = dict(self.phase_manager.last_reward_breakdown)
        self.last_per_model_reward = self.phase_manager.last_per_model_reward.copy()
        for key, value in self.last_reward_breakdown.items():
            self.episode_reward_breakdown[key] = (
                self.episode_reward_breakdown.get(key, 0.0) + value
            )
        self.episode_reward_steps += 1
        self.episode_reward += reward

        if self._state_exporters:
            snapshot = self.to_snapshot()
            for exporter in self._state_exporters:
                exporter.on_step(snapshot)

        return observation, reward, is_terminated, False, info

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
            deployment_outline=self.deployment_outline,
            opponent_deployment_outline=self.opponent_deployment_outline,
            player_vp=self.player_vp,
            opponent_vp=self.opponent_vp,
            player_vp_delta=self.player_vp_delta,
            opponent_vp_delta=self.opponent_vp_delta,
            player_shooting_results=self._last_player_shooting_results,
            opponent_shooting_results=self._last_opponent_shooting_results,
            player_fight_results=self._last_player_fight_results,
            opponent_fight_results=self._last_opponent_fight_results,
            player_action=self._last_player_action,
            opponent_action=self._last_opponent_action,
            last_reward=self.last_reward,
            reward_breakdown=self.last_reward_breakdown,
            episode_reward=self.episode_reward,
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
            terrain=self.terrain,
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

        restore_clock(self._game_clock, snapshot.clock, snapshot.step)
        restore_models(self.wargame_models, snapshot.player_models)
        restore_models(self.opponent_models, snapshot.opponent_models)
        restore_objectives(self.objectives, snapshot.objectives)
        self._battle.restore_victory_points(
            player_vp=snapshot.player_vp,
            opponent_vp=snapshot.opponent_vp,
            player_vp_delta=snapshot.player_vp_delta,
            opponent_vp_delta=snapshot.opponent_vp_delta,
        )

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

        self._last_player_shooting_results = restore_shooting_results(
            snapshot.player_combat_results
        )
        self._last_opponent_shooting_results = restore_shooting_results(
            snapshot.opponent_combat_results
        )

        # Reward
        self.last_reward = snapshot.reward.total
        self.last_reward_breakdown = dict(snapshot.reward.breakdown)
        self.episode_reward_breakdown = {}
        self.episode_reward_steps = 0
        self.episode_reward = snapshot.reward.episode_total or 0.0
        self.last_step_context = None

        # Recompute distances and build observation
        cache = compute_distances(self.wargame_models, self.objectives)
        observation = self._get_obs(cache)
        info = self._info_dict()

        return observation, info

    def render(self) -> None:
        if self.renderer is not None:
            self.renderer.render(self)

        return None


def _base_radii(models: list[WargameModel] | None) -> np.ndarray | None:
    """Base radius per model, which sets how wide a target is to look at."""
    if models is None:
        return None
    return np.array([m.base_radius for m in models], dtype=float)
