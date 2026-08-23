"""The mirrored observation is the other seat's, tensor for tensor.

Every side-carrying field has to swap **together**, or the observation is
internally inconsistent in ways that produce no error at all: decode a shooting
action through the wrong side's `ActionHandler` and the target index still
resolves, still passes the mask, and quietly fires at the wrong army.

Today the shapes coincide because the two armies are the same size, so a shape
error would not catch a seat bug either. The static scan in
`test_scripted_baseline_opponent.py` catches new *reads of known* side-specific
names, but not a newly invented one. This comparison is the guard that cannot be
fooled by either.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.env_components.observation_builder import (
    build_observation,
    update_distances_to_objectives,
)
from wargame_rl.wargame.envs.opponent.mirror import MirroredEnv
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.selectors import build_action_selector

CONFIG = "configs/dev/4v4_two_phases.yaml"
SEED = 900_000


def _config() -> WargameEnvConfig:
    with open(CONFIG) as handle:
        config: WargameEnvConfig = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    return config


def _swapped(config: WargameEnvConfig) -> WargameEnvConfig:
    """The same scenario with the two armies' configs and zones exchanged."""
    swapped: WargameEnvConfig = config.model_copy(deep=True)
    swapped.models, swapped.opponent_models = config.opponent_models, config.models
    swapped.number_of_wargame_models = config.number_of_opponent_models
    swapped.number_of_opponent_models = config.number_of_wargame_models
    swapped.deployment_zone = config.opponent_deployment_zone
    swapped.opponent_deployment_zone = config.deployment_zone
    return swapped


def _mirror_of(env: WargameEnv) -> WargameEnv:
    from typing import cast

    return cast(WargameEnv, MirroredEnv(env))


def _observation_tensors(view: WargameEnv, registry: object) -> list[np.ndarray]:
    observation = build_observation(view, action_registry=registry)  # type: ignore[arg-type]
    # `.cpu()` before `.numpy()`: `observation_to_tensor` places on the default
    # device, so on a GPU machine this raised and the test never ran. CI is
    # CPU-only, which is why it went green while `main` was red on any box with
    # a card in it.
    return [tensor.cpu().numpy() for tensor in observation_to_tensor(observation)]


def _paired_envs(steps: int = 0) -> tuple[WargameEnv, WargameEnv]:
    """`E` and `E'`, where `E'` is `E` with the two armies exchanged.

    Positions are **copied across, not re-drawn**. Seeding both envs alike is
    not enough: `place_for_episode` fills each army into its own deployment
    zone, and the zones differ between the two configs, so the same seed yields
    two different boards rather than one board with the roles exchanged. This
    config fixes its objectives and carries no terrain, so once the models are
    copied the two boards are identical.

    `steps` advances `E` first, so the comparison is not made on the deployment
    alone — a board nobody has moved on would let several kinds of mirror bug
    through.
    """
    base = _config()
    left = WargameEnv(config=base, renderer=None)
    right = WargameEnv(config=_swapped(base), renderer=None)
    observation, _ = left.reset(seed=SEED)
    right.reset(seed=SEED)

    if steps:
        select = build_action_selector("squad_march", left).select
        for _ in range(steps):
            observation, _r, done, truncated, _i = left.step(select(observation, left))
            if done or truncated:
                break

    for destination, source in (
        (right.wargame_models, left.opponent_models),
        (right.opponent_models, left.wargame_models),
    ):
        for target, model in zip(destination, source):
            target.location = model.location.copy()
            target.stats["current_wounds"] = model.stats["current_wounds"]

    # Objective distances are cached ON the models, and `build_observation`
    # recomputes them for the player side only when handed a distance cache --
    # so copying locations without refreshing these leaves `E'` describing where
    # its models used to be.
    for army in (right.wargame_models, right.opponent_models):
        update_distances_to_objectives(army, right.objectives, None)

    # The game tensor carries the round, the phase and both scores, and those
    # are shared board state rather than anything the mirror swaps -- so `E'`
    # has to be moved to the same point in the game, with the two scores
    # exchanged along with the armies.
    state = left.game_clock_state
    right._game_clock.set_state(  # noqa: SLF001 - no public setter; load_state
        state.game_phase,  # would also drop terrain and the combat RNG
        battle_round=state.battle_round,
        active_player=state.active_player,
        phase=state.phase,
    )
    right._battle.restore_victory_points(  # noqa: SLF001
        player_vp=left.opponent_vp,
        opponent_vp=left.player_vp,
        player_vp_delta=left.opponent_vp_delta,
        opponent_vp_delta=left.player_vp_delta,
    )
    return left, right


def test_the_mirror_reports_the_other_seats_models() -> None:
    env, _ = _paired_envs()
    mirror = _mirror_of(env)

    assert mirror.player_models is env.opponent_models
    assert mirror.opponent_models is env.wargame_models
    env.close()


@pytest.mark.parametrize(
    ("mirrored", "real"),
    [
        ("player_vp", "opponent_vp"),
        ("opponent_vp", "player_vp"),
        ("player_vp_delta", "opponent_vp_delta"),
        ("opponent_vp_delta", "player_vp_delta"),
    ],
)
def test_every_victory_point_field_swaps(mirrored: str, real: str) -> None:
    """All four are read by `build_observation` or `build_info`. Getting one
    wrong leaves a well-formed observation reporting the wrong side's score."""
    env, _ = _paired_envs(steps=6)

    assert getattr(_mirror_of(env), mirrored) == getattr(env, real)
    env.close()


def test_the_mirror_reports_the_other_seats_reach_and_zone() -> None:
    env, _ = _paired_envs()
    mirror = _mirror_of(env)

    np.testing.assert_array_equal(mirror.player_max_ranges, env.opponent_max_ranges)
    np.testing.assert_array_equal(mirror.opponent_max_ranges, env.player_max_ranges)
    np.testing.assert_array_equal(mirror.deployment_zone, env.opponent_deployment_zone)
    np.testing.assert_array_equal(mirror.opponent_deployment_zone, env.deployment_zone)
    env.close()


def test_shared_state_still_falls_through() -> None:
    """A mirror that overrode everything would be a second `WargameEnv` to keep
    in step. Objectives, terrain, the clock and the board are one board."""
    env, _ = _paired_envs()
    mirror = _mirror_of(env)

    assert mirror.objectives is env.objectives
    assert mirror.terrain is env.terrain
    assert mirror.board_width == env.board_width
    assert mirror.game_clock_state == env.game_clock_state
    env.close()


def test_the_mirrored_observation_is_the_other_seats() -> None:
    """The load-bearing case.

    Build `E`; build `E'` as `E` with the two armies and zones exchanged; the
    observation the mirror produces on `E` must equal the observation `E'`
    produces from its player seat, tensor for tensor.
    """
    left, right = _paired_envs(steps=6)
    mirror = _mirror_of(left)

    mirrored = _observation_tensors(mirror, left.opponent_action_handler.registry)
    direct = _observation_tensors(right, right.player_action_handler.registry)

    assert len(mirrored) == len(direct)
    for index, (got, want) in enumerate(zip(mirrored, direct)):
        np.testing.assert_array_equal(got, want, err_msg=f"tensor {index} differs")
    left.close()
    right.close()


def test_the_comparison_would_notice_an_unmirrored_side() -> None:
    """The test above is only worth having if it can fail.

    Handing `build_observation` the *real* env where the mirror belongs must
    produce a different observation — otherwise the assertion above is vacuous
    and would pass on a mirror that swapped nothing.
    """
    left, right = _paired_envs(steps=6)

    unmirrored = _observation_tensors(left, left.player_action_handler.registry)
    direct = _observation_tensors(right, right.player_action_handler.registry)

    assert any(
        not np.array_equal(got, want) for got, want in zip(unmirrored, direct)
    ), "the swapped scenario is indistinguishable, so this suite proves nothing"
    left.close()
    right.close()
