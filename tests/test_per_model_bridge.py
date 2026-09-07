"""The bridge: a script seated in the per-model facade reproduces the
whole-phase facade's episode bit-for-bit on the same layout and dice.

This is the identity test issue #283's comparability contract names first —
it is what makes the scripted bar transferable across the two architectures.
It holds wherever ``coherency.enforce_move`` is ``off`` (every shipped
training config); on a refereed config the per-model facade's coherency
referee deliberately fires per unit as the unit closes (the rules' own
timing) rather than over the whole force at the phase's end, so episodes
diverge exactly when a revert fires — see ``enforce_unit_after_move``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.per_model import PerModelEnv, ScriptedPolicyAdapter
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    TurnOrder,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from scenario_overrides import load_env_config  # noqa: E402

PhaseState = tuple[
    int,  # current_turn
    tuple[float, ...],  # player positions, flattened
    tuple[float, ...],  # opponent positions, flattened
    tuple[int, ...],  # player wounds
    tuple[int, ...],  # opponent wounds
    int,  # player VP
    int,  # opponent VP
]


def _capture(env: WargameEnv) -> PhaseState:
    return (
        env.current_turn,
        tuple(float(x) for m in env.wargame_models for x in m.location),
        tuple(float(x) for m in env.opponent_models for x in m.location),
        tuple(int(m.stats["current_wounds"]) for m in env.wargame_models),
        tuple(int(m.stats["current_wounds"]) for m in env.opponent_models),
        int(env.player_vp),
        int(env.opponent_vp),
    )


def _drive_whole_phase(
    config: WargameEnvConfig, policy_name: str, seed: int
) -> dict[int, PhaseState]:
    env = WargameEnv(config, build_info=False)
    policy = build_baseline_policy(policy_name)
    observation, _ = env.reset(seed=seed)
    states: dict[int, PhaseState] = {}
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        observation, _reward, terminated, truncated, _ = env.step(action)
        states[env.current_turn] = _capture(env)
    return states


def _drive_per_model(
    config: WargameEnvConfig, policy_name: str, seed: int
) -> dict[int, PhaseState]:
    env = PerModelEnv(config, build_info=False)
    adapter = ScriptedPolicyAdapter(build_baseline_policy(policy_name))
    observation, _ = env.reset(seed=seed)
    states: dict[int, PhaseState] = {}
    last_turn = env.current_turn
    terminated = False
    while not terminated:
        action = adapter.next_action(env, observation)
        observation, _reward, terminated, _truncated, _ = env.step(action)
        if env.current_turn != last_turn:
            # A phase boundary (possibly several, when the command phase is
            # auto-traversed) completed inside this step; capture the state it
            # left, which is the state the whole-phase facade's step returns.
            states[env.current_turn] = _capture(env)
            last_turn = env.current_turn
    return states


def _assert_bit_identical(
    old: dict[int, PhaseState], new: dict[int, PhaseState]
) -> None:
    # Auto-traversed phases can hide an intermediate boundary from the
    # per-model driver's capture loop; every boundary it did observe must
    # match the whole-phase facade's exactly.
    assert set(new) <= set(old)
    assert max(new) == max(old), "the two facades ended on different boundaries"
    for turn in sorted(new):
        old_state, new_state = old[turn], new[turn]
        for field, (a, b) in enumerate(zip(old_state, new_state, strict=True)):
            assert np.array_equal(np.asarray(a), np.asarray(b)), (
                f"boundary {turn} field {field} diverged:\n"
                f"whole-phase: {a}\nper-model:   {b}"
            )


def _small_shooting_config() -> WargameEnvConfig:
    """A fast scenario that exercises movement AND shooting on both seats.

    Real weapons and one-wound models, so the bridge is held against live
    dice and casualties rather than two armies that never draw blood.
    """
    rifle = [WeaponProfile(range=12, attacks=1)]
    squads = [
        ModelConfig(group_id=i // 4, weapons=rifle, max_wounds=1) for i in range(8)
    ]
    return WargameEnvConfig(
        render_mode=None,
        board_width=36,
        board_height=36,
        number_of_wargame_models=8,
        number_of_opponent_models=8,
        max_groups=2,
        models=squads,
        opponent_models=list(squads),
        number_of_objectives=3,
        number_of_battle_rounds=6,
        skip_phases=[
            BattlePhase.command,
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
    )


@pytest.mark.parametrize("policy_name", ["squad_march_shoot", "squad_march_take"])
@pytest.mark.parametrize("seed", [7, 41])
def test_bridge_small_shooting_scenario(policy_name: str, seed: int) -> None:
    config = _small_shooting_config()
    old = _drive_whole_phase(config, policy_name, seed)
    new = _drive_per_model(config, policy_name, seed)
    _assert_bit_identical(old, new)


@pytest.mark.parametrize("seed", [700001])
def test_bridge_golden_map_pool_config(seed: int) -> None:
    """The config that trains: map pool, shooting script on both seats."""
    config = load_env_config("configs/golden/25v25_maps_two_mode.yaml", rounds="4")
    old = _drive_whole_phase(config, "squad_march_shoot", seed)
    new = _drive_per_model(config, "squad_march_shoot", seed)
    _assert_bit_identical(old, new)


@pytest.mark.parametrize("seed", [700002])
def test_bridge_melee_config(seed: int) -> None:
    """All seven phases: charge declarations re-timed from the command phase
    to the charge phase's opening steps, fight priorities, pile-in and
    consolidate — a charging script on both seats."""
    config = load_env_config("configs/experiments/25v25_maps_melee.yaml", rounds="4")
    old = _drive_whole_phase(config, "squad_march_take_charge", seed)
    new = _drive_per_model(config, "squad_march_take_charge", seed)
    _assert_bit_identical(old, new)


# The three coverage holes the 2026-09-07 audit named: no advance config was
# bridged (the command→movement declaration re-timing, the riskiest in the
# design), no opponent-first turn order (the closing step's placement "after
# the opponent's turn"), and no elimination-terminated episode (the early
# close in `_complete_player_phase`).


@pytest.mark.parametrize("seed", [7, 41, 99])
def test_bridge_advance_config(seed: int) -> None:
    config = _small_shooting_config()
    config.n_advance_speed_bins = 3
    config.skip_phases = [
        BattlePhase.charge,
        BattlePhase.pile_in,
        BattlePhase.fight,
        BattlePhase.consolidate,
    ]
    old = _drive_whole_phase(config, "squad_march_take_advance", seed)
    new = _drive_per_model(config, "squad_march_take_advance", seed)
    _assert_bit_identical(old, new)


@pytest.mark.parametrize("turn_order", [TurnOrder.opponent, TurnOrder.random])
@pytest.mark.parametrize("seed", [7, 41])
def test_bridge_opponent_first_turn_order(turn_order: TurnOrder, seed: int) -> None:
    config = _small_shooting_config()
    config.turn_order = turn_order
    old = _drive_whole_phase(config, "squad_march_shoot", seed)
    new = _drive_per_model(config, "squad_march_shoot", seed)
    _assert_bit_identical(old, new)


@pytest.mark.parametrize("seed", [7, 41])
def test_bridge_elimination_terminated_episode(seed: int) -> None:
    """A game short one army ends early on both facades, at the same state."""
    rifle = [WeaponProfile(range=30, attacks=4)]
    squads = [
        ModelConfig(group_id=i // 2, weapons=rifle, max_wounds=1) for i in range(4)
    ]
    config = _small_shooting_config()
    config.number_of_wargame_models = 4
    config.number_of_opponent_models = 4
    config.models = squads
    config.opponent_models = list(squads)
    config.number_of_battle_rounds = 20
    old = _drive_whole_phase(config, "squad_march_shoot", seed)
    new = _drive_per_model(config, "squad_march_shoot", seed)
    _assert_bit_identical(old, new)
    # The case must actually eliminate: a full-length run proves nothing here.
    final = old[max(old)]
    assert 0 in final[3] or 0 in final[4], "nobody died — elimination untested"
