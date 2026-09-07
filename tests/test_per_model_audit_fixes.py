"""Regressions from the 2026-09-07 implementation audit (5 auditors + red
team): the opening step's as-if-declared masks, target-pointer alignment
under gapped unit ids, resolved-stat rows, and the mid-phase-death deadlock
hardening. Each of these failed (or would have failed) on the audited build.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wargame_rl.wargame.envs.per_model import PerModelAction, PerModelEnv
from wargame_rl.wargame.envs.per_model.observation import (
    _stat_rows,
    build_token_observation,
)
from wargame_rl.wargame.envs.per_model.types import MoveDeclaration
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

RIFLE = [WeaponProfile(range=12, attacks=1)]


def _config(**overrides: object) -> WargameEnvConfig:
    squads = [
        ModelConfig(group_id=i // 3, weapons=RIFLE, max_wounds=1) for i in range(6)
    ]
    base: dict[str, Any] = dict(
        render_mode=None,
        board_width=30,
        board_height=30,
        number_of_wargame_models=6,
        number_of_opponent_models=6,
        max_groups=2,
        models=squads,
        opponent_models=list(squads),
        number_of_objectives=2,
        number_of_battle_rounds=3,
        skip_phases=[
            BattlePhase.command,
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
    )
    base.update(overrides)
    return WargameEnvConfig(**base)


def test_the_opening_model_is_offered_its_own_advance_rungs() -> None:
    """The env's mask gates rungs on `declared_advance`, set inside the very
    step that samples the declaration — so the declaring model itself saw 0
    legal rungs (its squadmates, acting after, saw them all): the
    leader-capped-at-M formation trap. The token mask now shows every
    opening candidate the rungs as-if-declared."""
    config = _config(
        n_advance_speed_bins=3,
        skip_phases=[
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ],
    )
    env = PerModelEnv(config, build_info=False)
    observation, _ = env.reset(seed=0)
    tokens = build_token_observation(env, observation)
    handler = env.player_action_handler
    as_if = handler.advance_legality(
        env.wargame_models, env.opponent_models, assume_declared=True
    )
    for i in range(len(env.wargame_models)):
        assert env.unit_needs_declaration(i)
        np.testing.assert_array_equal(tokens.advance_mask[i], as_if[i])
    # The roll gate still binds: at least one model has a rung within M + roll
    # over the seeds this test pins, and every offered rung really is one.
    assert tokens.advance_mask.any(), "no rung offered anywhere — vacuous seed"

    # The env accepts what the mask offered: the opener declares advance and
    # takes a rung in the same step, and travels further than a normal move.
    opener = int(np.flatnonzero(observation.selection_mask)[0])
    rungs = np.flatnonzero(tokens.advance_mask[opener])
    assert rungs.size > 0
    advance_slice = handler.advance_slice
    assert advance_slice is not None
    before = np.array(env.wargame_models[opener].location, dtype=float)
    env.step(
        PerModelAction(
            model_index=opener,
            declaration=int(MoveDeclaration.advance),
            action=advance_slice.start + int(rungs[0]),
        )
    )
    moved = float(
        np.linalg.norm(np.array(env.wargame_models[opener].location) - before)
    )
    assert moved > 0.0


def test_target_columns_follow_sorted_unit_ids_not_slice_positions() -> None:
    """With a gap in opponent group ids the shooting slice (indexed by raw
    id) and the pointer (indexed by sorted-distinct position) disagree; the
    positional copy marked column j with a different unit's legality."""
    opponent_squads = [
        ModelConfig(group_id=0 if i < 3 else 2, weapons=RIFLE, max_wounds=1)
        for i in range(6)
    ]
    config = _config(max_groups=3, opponent_models=opponent_squads)
    env = PerModelEnv(config, build_info=False)
    observation, _ = env.reset(seed=2)
    while observation.phase is not BattlePhase.shooting:
        if observation.selection_mask.any():
            action = PerModelAction(
                model_index=int(np.flatnonzero(observation.selection_mask)[0])
            )
        else:
            action = PerModelAction()
        observation, _r, done, _t, _ = env.step(action)
        assert not done
    tokens = build_token_observation(env, observation)
    unit_ids = sorted({int(m.group_id) for m in env.opponent_models})
    assert unit_ids == [0, 2]
    shooting_slice = env.player_action_handler.shooting_slice
    assert shooting_slice is not None
    full = env.current_action_mask()
    for column, unit_id in enumerate(unit_ids):
        np.testing.assert_array_equal(
            tokens.target_mask[:, 1 + column],
            full[:, shooting_slice.start + unit_id],
            err_msg=f"pointer column {column} does not carry unit {unit_id}",
        )


def test_stat_rows_read_the_resolved_models_not_the_raw_config() -> None:
    """A config that auto-builds its army carries no `models` list; reading
    the raw config zeroed every stat, and T=0 made `expected_damage_matrix`
    report everyone unshootable."""
    config = _config(models=None, opponent_models=None)
    env = PerModelEnv(config, build_info=False)
    env.reset(seed=1)
    stats = _stat_rows(env.wargame_models, None)
    assert (stats[:, 5] > 0).all(), "toughness zeroed — raw config read"
    assert (stats[:, 6] > 0).all(), "save zeroed — raw config read"


def test_a_model_dying_unacted_mid_phase_does_not_deadlock_the_phase() -> None:
    """Phase completion used to be `_acted.all()`; a model dying un-acted
    mid-phase (melee expansion's case) was unselectable forever and the
    episode froze behind a misleading 'not selectable' error."""
    env = PerModelEnv(_config(), build_info=False)
    observation, _ = env.reset(seed=3)
    # Act the first model, then kill an un-acted one directly.
    first = int(np.flatnonzero(observation.selection_mask)[0])
    observation, _r, _d, _t, _ = env.step(PerModelAction(model_index=first))
    victims = [
        i
        for i in range(len(env.wargame_models))
        if env.wargame_models[i].is_alive and i != first
    ]
    victim = victims[-1]
    env.wargame_models[victim].take_damage(10**6)
    assert not env.wargame_models[victim].is_alive
    # The episode must still run to completion without a selection error.
    done = False
    steps = 0
    while not done and steps < 500:
        if observation.selection_mask.any():
            action = PerModelAction(
                model_index=int(np.flatnonzero(observation.selection_mask)[0])
            )
        else:
            action = PerModelAction()
        observation, _r, done, _t, _ = env.step(action)
        steps += 1
    assert done, "episode never terminated after a mid-phase un-acted death"
