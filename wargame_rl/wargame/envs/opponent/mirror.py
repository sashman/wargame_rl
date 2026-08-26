"""The env as the opponent sees it: the two sides swapped, nothing else.

Every attribute a consumer reads off the env is either **shared** — objectives,
terrain, the game clock, the board — or **side-specific**, and only the second
kind is overridden here. Attribute lookup falls through to the real env, so this
stays a mirror rather than a second implementation of `WargameEnv` that has to
be kept in step with it.

`tests/test_scripted_baseline_opponent.py` enumerates the side-specific reads
across the packages that depend on this and fails when a new one appears,
because a consumer reaching for an un-mirrored side attribute would silently
play for the wrong army. That enumeration only catches new *reads of known
names*; the guarantee that the mirrored observation really is the other seat's
is `tests/test_swap_invariance.py`, which compares tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.domain.fight import PairedFightResult
    from wargame_rl.wargame.envs.domain.shooting import PairedShootingResult
    from wargame_rl.wargame.envs.wargame import WargameEnv


class MirroredEnv:
    """A read-only side-swapped view of a live `WargameEnv`."""

    def __init__(self, env: WargameEnv) -> None:
        self._env = env
        # Swapped once at construction rather than per call: the config is
        # immutable for the episode, and `model_copy` re-points fields without
        # re-running validation, which would reject a config whose model list
        # no longer matches its own count field.
        self.config = env.config.model_copy(
            update={
                "number_of_wargame_models": env.config.number_of_opponent_models,
                "number_of_opponent_models": env.config.number_of_wargame_models,
                "models": env.config.opponent_models,
                "opponent_models": env.config.models,
                "deployment_zone": env.config.opponent_deployment_zone,
                "opponent_deployment_zone": env.config.deployment_zone,
            }
        )

    # ---- models ------------------------------------------------------------

    @property
    def wargame_models(self) -> list[WargameModel]:
        """Our models — the opponent's, from the real env's point of view."""
        return self._env.opponent_models

    @property
    def player_models(self) -> list[WargameModel]:
        """`BattleView`'s name for the same list `wargame_models` returns."""
        return self._env.opponent_models

    @property
    def opponent_models(self) -> list[WargameModel]:
        """The enemy — the player's models."""
        return self._env.wargame_models

    # ---- action handlers ---------------------------------------------------

    @property
    def player_action_handler(self) -> Any:
        """The handler that moves *our* models and indexes *their* units."""
        return self._env.opponent_action_handler

    @property
    def opponent_action_handler(self) -> Any:
        """The enemy's handler — the real env's player-side one."""
        return self._env.player_action_handler

    # ---- weapon reach ------------------------------------------------------

    @property
    def player_advance_legality(self) -> np.ndarray:
        """Our models' legal advance rungs — the opponent handler's, not the player's."""
        return self._env.opponent_action_handler.advance_legality(
            self._env.opponent_models, self._env.wargame_models
        )

    @property
    def player_charge_legality(self) -> np.ndarray:
        """Recomputed from the OPPONENT's handler and models, as advance is."""
        return self._env.opponent_action_handler.charge_legality(
            self._env.opponent_models, self._env.wargame_models
        )

    @property
    def player_declaration_legality(self) -> np.ndarray:
        """Recomputed from the OPPONENT's handler and models, as the others are."""
        return self._env.opponent_action_handler.declaration_legality(
            self._env.opponent_models, self._env.wargame_models
        )

    @property
    def player_max_ranges(self) -> np.ndarray:
        """Our weapons' reach. Read by `build_observation` for the shooting mask."""
        return self._env.opponent_max_ranges

    @property
    def opponent_max_ranges(self) -> np.ndarray:
        """The enemy's reach."""
        return self._env.player_max_ranges

    # ---- victory points ----------------------------------------------------
    #
    # All four are read by `build_observation` / `build_info`, and getting one
    # wrong is invisible: the observation is still well formed, the network
    # still runs, and the seat simply reads the wrong side's score.

    @property
    def player_vp(self) -> int:
        """Our score."""
        return self._env.opponent_vp

    @property
    def opponent_vp(self) -> int:
        """Their score."""
        return self._env.player_vp

    @property
    def player_vp_delta(self) -> int:
        """What we scored this step."""
        return self._env.opponent_vp_delta

    @property
    def opponent_vp_delta(self) -> int:
        """What they scored this step."""
        return self._env.player_vp_delta

    # ---- shooting results --------------------------------------------------

    @property
    def last_player_shooting_results(self) -> list[PairedShootingResult]:
        """Our last volley."""
        return self._env.last_opponent_shooting_results

    @property
    def last_opponent_shooting_results(self) -> list[PairedShootingResult]:
        """Theirs."""
        return self._env.last_player_shooting_results

    # ---- melee results -----------------------------------------------------

    @property
    def last_player_fight_results(self) -> list[PairedFightResult]:
        """Our last melee."""
        return self._env.last_opponent_fight_results

    @property
    def last_opponent_fight_results(self) -> list[PairedFightResult]:
        """Theirs."""
        return self._env.last_player_fight_results

    # ---- deployment zones --------------------------------------------------
    #
    # Explicit because these are *instance attributes* on the env, not
    # properties, so `__getattr__` never fires for them and the fall-through
    # would silently return the player's.

    @property
    def deployment_zone(self) -> np.ndarray:
        """Where we deployed."""
        return self._env.opponent_deployment_zone

    @property
    def opponent_deployment_zone(self) -> np.ndarray:
        """Where they deployed."""
        return self._env.deployment_zone

    # ---- everything else ---------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Fall through to the real env for everything not side-specific.

        The `__dict__` lookup rather than `self._env` is load-bearing, and it is
        not defensive programming. `__getattr__` runs for any name normal lookup
        misses, and `copy.deepcopy` reconstructs an instance *without* calling
        `__init__` — so during the copy `_env` is missing, `self._env` re-enters
        here, and the two recurse until the stack ends. Lightning deep-copies the
        env in `save_hyperparameters`, so with the plain version every training
        run on a config using this policy died at startup with `RecursionError`.
        """
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)
