"""Fuzz the per-model facade with random LEGAL decisions.

The bridge test proves a script plays the phase facade's game. It cannot say
whether a decision the phase facade never offers -- a charge target, a strike
target, a unit opened in an order no script would pick -- is safe to take.
This does: a seat that draws a uniformly random legal decision at every point,
over seeded episodes on the golden and melee scenarios, must never raise, must
end the episode at the phase clock's own budget, and must leave the board's
invariants intact -- a decision point that admits an illegal value, or a lock
that deadlocks, shows up here and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_file_as

from tests.per_model_seats import random_legal_action
from wargame_rl.wargame.envs.per_model import PerModelAction, PerModelEnv, StepKind
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

GOLDEN = "configs/golden/25v25_maps_two_mode.yaml"
MELEE = "configs/experiments/25v25_maps_melee.yaml"


def _config(path: str, rounds: int) -> WargameEnvConfig:
    config: WargameEnvConfig = parse_yaml_file_as(WargameEnvConfig, path)
    shortened: WargameEnvConfig = config.model_copy(
        update={"number_of_battle_rounds": rounds}
    )
    return shortened


@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("path", [GOLDEN, MELEE], ids=["golden", "melee"])
def test_random_legal_decisions_never_break_the_game(path: str, seed: int) -> None:
    """Arrange a seeded env; act with random legal decisions to the end; assert
    the episode terminates at the phase clock's budget with sane state."""
    config = _config(path, rounds=4)
    env = PerModelEnv(config)
    rng = np.random.default_rng(seed)
    observation, info = env.reset(seed=seed)
    max_turns = WargameEnv(config).max_turns
    assert env.max_turns == max_turns
    alive_before = sum(m.is_alive for m in env.wargame_models) + sum(
        m.is_alive for m in env.opponent_models
    )
    player_vp, opponent_vp = env.player_vp, env.opponent_vp
    kinds: set[StepKind] = set()
    steps = 0
    done = False
    while not done:
        point = observation.decision
        kinds.add(point.kind)
        action = random_legal_action(point, rng)
        observation, _reward, done, truncated, info = env.step(action)
        assert not truncated
        steps += 1
        alive_now = sum(m.is_alive for m in env.wargame_models) + sum(
            m.is_alive for m in env.opponent_models
        )
        assert alive_now <= alive_before, "a casualty came back to life"
        alive_before = alive_now
        assert env.player_vp >= player_vp and env.opponent_vp >= opponent_vp
        player_vp, opponent_vp = env.player_vp, env.opponent_vp
        assert steps < 20_000, "the episode did not terminate"
    assert env.current_turn == max_turns
    assert StepKind.close_turn in kinds and StepKind.open in kinds
    if config.melee.enabled:
        assert StepKind.act in kinds


def test_an_illegal_decision_is_refused_by_name() -> None:
    """A decision outside the point's masks raises, naming what was wrong."""
    config = _config(GOLDEN, rounds=1)
    env = PerModelEnv(config)
    observation, _ = env.reset(seed=3)
    point = observation.decision
    assert point.kind is StepKind.open
    dead_or_unselectable = (
        int(np.flatnonzero(~point.selector_mask)[0])
        if (~point.selector_mask).any()
        else None
    )
    if dead_or_unselectable is not None:
        with pytest.raises(ValueError, match="not selectable"):
            env.step(PerModelAction.open(dead_or_unselectable, 0))
    with pytest.raises(ValueError, match="expected a open step"):
        env.step(PerModelAction.close_turn())
