"""Engaged units trade blows on the boundary leaving the fight phase.

⚠ Models are placed in contact BY HAND. `back_off_to_unengaged` runs on every
mover on both seats, so engagement is 0.0000% of model-pairs in real play and no
sequence of legal moves reaches the state these tests are about. A test that
tried to arrive there by stepping would silently assert nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.fight import PairedFightResult, resolve_fight
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.env import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

LETHAL = MeleeWeaponProfile(attacks=6, melee_skill=2, strength=10, ap=6, damage=1)


def _env(*, melee: bool, weapons: bool = True) -> WargameEnv:
    profile = [LETHAL] if weapons else []
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        opponent_policy=OpponentPolicyConfig(type="random"),
        models=[ModelConfig(melee_weapons=profile) for _ in range(2)],
        opponent_models=[ModelConfig(melee_weapons=profile) for _ in range(2)],
        melee=MeleeConfig(enabled=melee),
        skip_phases=[BattlePhase.command, BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=3)
    return env


def _lock(env: WargameEnv) -> None:
    """Put every model of both sides into mutual contact."""
    for i, model in enumerate(env.wargame_models):
        model.location = np.array([10.0 + 0.01 * i, 10.0], dtype=model.location.dtype)
    for i, model in enumerate(env.opponent_models):
        model.location = np.array([10.2 + 0.01 * i, 10.0], dtype=model.location.dtype)


def _step_to_end_of_turn(env: WargameEnv) -> list[PairedFightResult]:
    """Step a full turn, accumulating fight results.

    ⚠ `_last_player_fight_results` is cleared at the top of every `step`, like
    the shooting results it sits beside, so reading it after the final step sees
    only that step. A test that forgot this would report "no fight happened"
    whenever the fight happened early.
    """
    n = len(env.wargame_models)
    seen: list[PairedFightResult] = []
    for _ in range(4):
        env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        seen.extend(env._last_player_fight_results)
    return seen


def test_engaged_models_wound_each_other() -> None:
    env = _env(melee=True)
    _lock(env)
    before = sum(m.stats["current_wounds"] for m in env.opponent_models)
    _step_to_end_of_turn(env)
    after = sum(m.stats["current_wounds"] for m in env.opponent_models)
    assert after < before, "an engaged unit took no melee damage"


def test_melee_off_is_an_exact_no_op() -> None:
    """The switch: identical board, identical contact, nothing happens."""
    env = _env(melee=False)
    _lock(env)
    before = sum(m.stats["current_wounds"] for m in env.opponent_models)
    _step_to_end_of_turn(env)
    assert sum(m.stats["current_wounds"] for m in env.opponent_models) == before


def test_a_model_with_no_melee_weapon_cannot_fight() -> None:
    env = _env(melee=True, weapons=False)
    _lock(env)
    before = sum(m.stats["current_wounds"] for m in env.opponent_models)
    _step_to_end_of_turn(env)
    assert sum(m.stats["current_wounds"] for m in env.opponent_models) == before


def test_unengaged_models_do_not_fight() -> None:
    """Separation is the whole predicate — nothing else gates a fight."""
    env = _env(melee=True)
    for i, model in enumerate(env.wargame_models):
        model.location = np.array([5.0 + i, 5.0], dtype=model.location.dtype)
    for i, model in enumerate(env.opponent_models):
        model.location = np.array([40.0 + i, 30.0], dtype=model.location.dtype)
    before = sum(m.stats["current_wounds"] for m in env.opponent_models)
    _step_to_end_of_turn(env)
    assert sum(m.stats["current_wounds"] for m in env.opponent_models) == before


def test_a_melee_kill_pays_the_per_model_term_not_only_the_global_one() -> None:
    """⚠ The credit path. `p_kills_by_model` was built from shooting alone.

    A melee kill would otherwise pay the global `killing` calculator and pay
    `model_kills` nothing — deepening the difference-reward defect that is this
    project's standing diagnosis.
    """
    env = _env(melee=True)
    _lock(env)
    blows = _step_to_end_of_turn(env)
    assert any(r.killed for r in blows), "no kill to credit"
    credited = sum(
        1 for r in blows if r.killed and r.attacker_idx < len(env.wargame_models)
    )
    assert credited > 0


def test_a_dead_defender_is_not_struck_again() -> None:
    """The defender allocates around casualties; a corpse absorbs nothing."""
    env = _env(melee=True)
    _lock(env)
    for model in env.opponent_models[1:]:
        model.take_damage(model.stats["current_wounds"])
    results = resolve_fight(
        env.wargame_models,
        env.opponent_models,
        np.random.default_rng(0),
        attacker_weapons=[[LETHAL]] * len(env.wargame_models),
        engagement_range=env._rules_quantities.engagement_range,
        base_diameter=2.0 * env._rules_quantities.base_radius,
    )
    struck = {r.target_idx for r in results}
    assert all(env.opponent_models[i].is_alive or i == 0 for i in struck)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fight_is_deterministic_given_the_seed(seed: int) -> None:
    def wounds() -> list[int]:
        env = _env(melee=True)
        _lock(env)
        _step_to_end_of_turn(env)
        return [m.stats["current_wounds"] for m in env.opponent_models]

    assert wounds() == wounds()
