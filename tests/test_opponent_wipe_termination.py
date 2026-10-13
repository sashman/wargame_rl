"""The phase facade plays the clock out after an opponent wipe (#317).

`15-missions-and-scoring.md` § Ending the battle: a player with no models left
does not lose immediately; both keep taking turns and the survivor keeps
scoring. The phase facade used to end the battle the moment the opponent was
wiped, unconditionally, which made it disagree with the per-model facade on
any scenario where the opponent can die. Now it ends on an opponent wipe only
under `terminate_on_opponent_elimination`, the mirror of the player-side
switch, and both default to the rules.
"""

from __future__ import annotations

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.wargame import WargameEnv


def _play_out_with_a_wiped_opponent(terminate: bool) -> WargameEnv:
    config = small_config(rounds=3).model_copy(
        update={"terminate_on_opponent_elimination": terminate}
    )
    env = WargameEnv(config)
    env.reset(seed=2)
    for model in env.opponent_models:
        model.stats["current_wounds"] = 0
    done = False
    stay = WargameEnvAction(actions=[STAY_ACTION] * len(env.wargame_models))
    while not done:
        _obs, _reward, done, _trunc, _info = env.step(stay)
    return env


def test_by_default_the_battle_continues_after_the_opponent_is_wiped_out() -> None:
    # Arrange + Act: a wiped opponent from the first step, the rules' default.
    env = _play_out_with_a_wiped_opponent(terminate=False)
    # Assert: the clock ran to its budget and the survivor kept scoring.
    assert env.current_turn == env.max_turns
    assert env.player_vp > 0


def test_the_switch_ends_the_battle_on_the_opponent_wipe() -> None:
    # Arrange + Act: the same wipe under the training shortcut.
    env = _play_out_with_a_wiped_opponent(terminate=True)
    # Assert: the episode ended on the first step.
    assert env.current_turn < env.max_turns
    assert env.current_turn <= 1
