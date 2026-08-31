"""The surplus-reallocation decode, driven through `env.step`.

⚠ Six unit tests once covered a decoder and **none called `env.step`**, so all
six asserted the decoder against its own relaxation; the gap was worth +11.4 vp
when it was finally closed. These go through the env.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.reallocation_decode import apply_reallocation

STAY = 0


def _env() -> WargameEnv:
    """Two 4-model squads, two objectives, one enemy squad holding the second."""
    config = WargameEnvConfig(
        number_of_wargame_models=8,
        number_of_opponent_models=2,
        number_of_objectives=2,
        max_groups=2,
        base_radius=0.0,
        objective_radius_size=3,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(group_id=i // 4) for i in range(8)],
        opponent_models=[ModelConfig(group_id=0) for _ in range(2)],
        skip_phases=[BattlePhase.command, BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=5)
    return env


def _stack_everything_on_objective_zero(env: WargameEnv) -> None:
    """Both player squads pile on objective 0; the enemy holds objective 1."""
    o0 = np.asarray(env.objectives[0].location, dtype=float)
    o1 = np.asarray(env.objectives[1].location, dtype=float)
    for index, model in enumerate(env.wargame_models):
        model.location = (o0 + np.array([0.1 * index, 0.0])).astype(
            model.location.dtype
        )
    for model in env.opponent_models:
        model.location = o1.astype(model.location.dtype)


def test_the_decode_redirects_one_surplus_squad_and_leaves_the_rest_alone() -> None:
    env = _env()
    try:
        _stack_everything_on_objective_zero(env)
        actions = [STAY] * len(env.wargame_models)

        out = apply_reallocation(actions, env)

        changed = [i for i, (a, b) in enumerate(zip(actions, out)) if a != b]
        assert changed, "no squad was redirected on a board with a clear surplus"
        groups = {int(env.wargame_models[i].group_id) for i in changed}
        assert len(groups) == 1, f"more than one squad redirected: {groups}"
        # Rigid: every redirected model gets the SAME action, which is what
        # keeps a coherent squad coherent.
        assert len({out[i] for i in changed}) == 1
    finally:
        env.close()


def test_it_is_a_no_op_when_no_squad_is_surplus() -> None:
    """One squad per objective offers nothing to redistribute."""
    env = _env()
    try:
        o0 = np.asarray(env.objectives[0].location, dtype=float)
        o1 = np.asarray(env.objectives[1].location, dtype=float)
        for index, model in enumerate(env.wargame_models):
            target = o0 if index < 4 else o1
            model.location = (target + np.array([0.1 * index, 0.0])).astype(
                model.location.dtype
            )
        for model in env.opponent_models:
            model.location = (o1 + np.array([20.0, 0.0])).astype(model.location.dtype)
        actions = [STAY] * len(env.wargame_models)

        assert apply_reallocation(actions, env) == actions
    finally:
        env.close()


def test_the_redirect_actually_moves_the_squad_through_env_step() -> None:
    """The decode's output must survive `env.step` — not just the forward model."""
    env = _env()
    try:
        while env.game_clock_state.phase is not BattlePhase.movement:
            env.step(WargameEnvAction(actions=[STAY] * len(env.wargame_models)))
        _stack_everything_on_objective_zero(env)
        target = np.asarray(env.objectives[1].location, dtype=float)
        before = min(
            float(np.linalg.norm(target - np.asarray(m.location, dtype=float)))
            for m in env.wargame_models
            if m.is_alive
        )

        out = apply_reallocation([STAY] * len(env.wargame_models), env)
        env.step(WargameEnvAction(actions=out))

        after = min(
            float(np.linalg.norm(target - np.asarray(m.location, dtype=float)))
            for m in env.wargame_models
            if m.is_alive
        )
        assert after < before, f"redirect closed no distance: {before} -> {after}"
    finally:
        env.close()
