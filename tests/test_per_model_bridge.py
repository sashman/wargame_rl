"""The bridge: a script through the per-model facade plays the phase facade's game.

This is the first test of the per-model facade (#284) and the one that makes
the scripted bar transferable between the two worlds. A `BaselinePolicy`
seated via `ScriptedSeat` -- planned once per phase, replayed in the engine's
own resolution order -- must reproduce `WargameEnv`'s episode BIT-FOR-BIT on
the same layout and dice: positions, wounds, VP, reward, its per-model vector
and breakdown, and the combat dice stream's position, at every point the
phase facade's step would have returned.

The comparison is per settled reward window on the new side against per step
on the old, which are the same points by construction. Four config families:
the golden map-pool scenario, the random-terrain shooting scenario (whose
opponent draws its targets from `env.np_random` once per phase), the advance
scenario (a stepped command phase and a D6 stream), and a melee scenario --
where the bridge holds only until the first charge that stands, since the
per-model facade moves both seats in pile-in and consolidate and lets a
striker choose its target, neither of which the phase facade does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_file_as

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.domain.engagement import engagement_matrix
from wargame_rl.wargame.envs.per_model import (
    FacadeDivergence,
    PerModelAction,
    PerModelEnv,
    ScriptedSeat,
    StepKind,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

GOLDEN = "configs/golden/25v25_maps_two_mode.yaml"
SHOOTING = "configs/golden/25v25_shooting_opponent.yaml"
ADVANCE = "configs/experiments/25v25_maps_advance.yaml"
MELEE = "configs/experiments/25v25_maps_melee.yaml"

SEEDS = (700001, 700002, 700003)
COMBAT_SEED = 4242


@dataclass(frozen=True)
class Boundary:
    """Everything the phase facade's step returns or leaves on the board."""

    battle_round: int | None
    active_player: str | None
    phase: str | None
    current_turn: int
    player_side: str
    player_positions: np.ndarray
    opponent_positions: np.ndarray
    player_wounds: list[int]
    opponent_wounds: list[int]
    player_vp: int
    opponent_vp: int
    reward: float
    per_model: np.ndarray
    breakdown: dict[str, float]
    dice: dict[str, Any] | None
    charged: bool
    engaged: bool


def _config(path: str, *, rounds: int | None = None) -> WargameEnvConfig:
    config: WargameEnvConfig = parse_yaml_file_as(WargameEnvConfig, path)
    if rounds is not None:
        config = config.model_copy(update={"number_of_battle_rounds": rounds})
    return config


def _policy(name: str) -> tuple[BaselinePolicy, bool]:
    policy = build_baseline_policy(name)
    shoots = type(policy).select_shooting is not BaselinePolicy.select_shooting
    return policy, shoots


def trace_old(config: WargameEnvConfig, name: str, seed: int) -> list[Boundary]:
    """Every step of the phase facade, as a `Boundary`."""
    env = WargameEnv(config)
    policy, _shoots = _policy(name)
    observation, _ = env.reset(seed=seed, options={"combat_seed": COMBAT_SEED})
    boundaries: list[Boundary] = []
    done = False
    while not done:
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        observation, reward, done, _truncated, _info = env.step(action)
        state = env.game_clock_state
        boundaries.append(
            Boundary(
                battle_round=state.battle_round,
                active_player=(
                    state.active_player.value if state.active_player else None
                ),
                phase=state.phase.value if state.phase else None,
                current_turn=env.current_turn,
                player_side=env.player_side.value,
                player_positions=np.array(
                    [m.location for m in env.wargame_models], dtype=float
                ),
                opponent_positions=np.array(
                    [m.location for m in env.opponent_models], dtype=float
                ),
                player_wounds=[
                    int(m.stats["current_wounds"]) for m in env.wargame_models
                ],
                opponent_wounds=[
                    int(m.stats["current_wounds"]) for m in env.opponent_models
                ],
                player_vp=env.player_vp,
                opponent_vp=env.opponent_vp,
                reward=float(reward),
                per_model=env.last_per_model_reward.copy(),
                breakdown=dict(env.last_reward_breakdown),
                dice=dict(env._combat_rng.bit_generator.state),
                charged=any(
                    m.charged_this_turn
                    for m in (*env.wargame_models, *env.opponent_models)
                ),
                engaged=_any_engaged(env),
            )
        )
    return boundaries


@dataclass
class NewTrace:
    """The per-model facade's boundaries, and how many divergences from the phase facade the
    facade had recorded by the time each was settled."""

    boundaries: list[Boundary]
    divergences_seen: list[int]
    divergences: list[FacadeDivergence]


def trace_new(config: WargameEnvConfig, name: str, seed: int) -> NewTrace:
    """Every settled reward window of the per-model facade, as a `Boundary`."""
    env = PerModelEnv(config)
    policy, shoots = _policy(name)
    planner = ScriptedSeat(policy, shoots=shoots)
    env.set_player_planner(planner)
    observation, info = env.reset(seed=seed, options={"combat_seed": COMBAT_SEED})
    boundaries: list[Boundary] = []
    seen: list[int] = []
    # A stepped command phase settles its window before the first decision,
    # so the reset's info carries it, exactly as the phase facade's first step
    # would have returned it.
    settled = _settled_boundaries(info)
    boundaries.extend(settled)
    seen.extend(len(env.divergences) for _ in settled)
    done = False
    while not done:
        point = observation.decision
        if point.kind is StepKind.close_turn:
            action = PerModelAction.close_turn()
        else:
            action = planner.choose(point, env.player_seat)
        observation, _reward, done, _truncated, info = env.step(action)
        settled = _settled_boundaries(info)
        boundaries.extend(settled)
        seen.extend(len(env.divergences) for _ in settled)
    return NewTrace(boundaries, seen, list(env.divergences))


def _divergence_index(divergence: str, old: list[Boundary]) -> int:
    return (
        int(divergence.split(":")[0].split()[-1])
        if "boundary" in divergence
        else len(old)
    )


def _assert_identical_until_the_rules_part(
    old: list[Boundary], new: NewTrace, divergence: str | None
) -> None:
    """Bit-identity is owed up to the first rule the phase facade cannot apply.

    The per-model step honours orderings the whole-army step has no room for
    (`FacadeDivergence` names them); the facade records the first time each
    makes a difference, and from that boundary on the two games are allowed
    to part. Before it, every boundary must agree."""
    if divergence is None:
        return
    index = min(_divergence_index(divergence, old), len(new.divergences_seen) - 1)
    assert new.divergences_seen[index] > 0, (
        f"{divergence}, with no divergence from the phase facade recorded by then"
    )


def _any_engaged(env: WargameEnv) -> bool:
    """Is any living model of either side within engagement range of the other."""
    mine = [m for m in env.wargame_models if m.is_alive]
    theirs = [m for m in env.opponent_models if m.is_alive]
    if not mine or not theirs:
        return False
    quantities = env.rules_quantities
    contacts = engagement_matrix(
        np.array([m.location for m in mine], dtype=float),
        np.array([m.location for m in theirs], dtype=float),
        np.ones(len(theirs), dtype=bool),
        np.ones(len(mine), dtype=bool),
        engagement_range=quantities.engagement_range,
        base_diameter=2.0 * quantities.base_radius,
    )
    return bool(np.asarray(contacts).any())


def _settled_boundaries(info: dict[str, Any]) -> list[Boundary]:
    """The windows a step (or the reset) settled, as boundaries."""
    boundaries: list[Boundary] = []
    for settled in info["settled"]:
        state = settled.state
        dice = state["dice"]
        boundaries.append(
            Boundary(
                battle_round=state["battle_round"],
                active_player=(
                    state["active_player"].value if state["active_player"] else None
                ),
                phase=state["phase"].value if state["phase"] else None,
                current_turn=state["current_turn"],
                player_side=state["player_side"].value,
                player_positions=state["player_positions"],
                opponent_positions=state["opponent_positions"],
                player_wounds=state["player_wounds"],
                opponent_wounds=state["opponent_wounds"],
                player_vp=state["player_vp"],
                opponent_vp=state["opponent_vp"],
                reward=float(settled.reward),
                per_model=settled.per_model,
                breakdown=dict(settled.breakdown),
                dice=dict(dice["combat"]) if dice is not None else None,
                charged=bool(state["charged"]),
                engaged=bool(state["engaged"]),
            )
        )
    return boundaries


def _first_divergence(old: list[Boundary], new: list[Boundary]) -> str | None:
    """Where the two traces first disagree, as a message, or None."""
    for index, (a, b) in enumerate(zip(old, new, strict=False)):
        for field in (
            "battle_round",
            "active_player",
            "phase",
            "current_turn",
            "player_side",
        ):
            if getattr(a, field) != getattr(b, field):
                return f"boundary {index}: {field} {getattr(a, field)!r} != {getattr(b, field)!r}"
        if not np.array_equal(a.player_positions, b.player_positions):
            return f"boundary {index}: player positions differ"
        if not np.array_equal(a.opponent_positions, b.opponent_positions):
            return f"boundary {index}: opponent positions differ"
        if a.player_wounds != b.player_wounds or a.opponent_wounds != b.opponent_wounds:
            return f"boundary {index}: wounds differ"
        if (a.player_vp, a.opponent_vp) != (b.player_vp, b.opponent_vp):
            return f"boundary {index}: VP differ"
        if a.dice != b.dice:
            return f"boundary {index}: combat dice stream position differs"
        if not math.isclose(a.reward, b.reward, rel_tol=1e-12, abs_tol=1e-12):
            return f"boundary {index}: reward {a.reward} != {b.reward}"
        if not np.allclose(a.per_model, b.per_model, rtol=1e-12, atol=1e-12):
            return f"boundary {index}: per-model reward differs"
        if set(a.breakdown) != set(b.breakdown) or any(
            not math.isclose(
                a.breakdown[k], b.breakdown[k], rel_tol=1e-12, abs_tol=1e-12
            )
            for k in a.breakdown
        ):
            return f"boundary {index}: reward breakdown differs"
    if len(old) != len(new):
        return f"trace lengths differ: {len(old)} old, {len(new)} new"
    return None


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("path", "policy"),
    [
        (GOLDEN, "squad_march_take"),
        (GOLDEN, "squad_march_shoot"),
        (SHOOTING, "squad_march_shoot"),
        (ADVANCE, "squad_march_take_advance"),
    ],
    ids=["golden-take", "golden-shoot", "shooting-shoot", "advance-take"],
)
def test_a_script_plays_the_same_game_through_both_facades(
    path: str, policy: str, seed: int
) -> None:
    """Arrange both facades on one layout and dice; act by playing the script
    through each; assert every boundary agrees, bit for bit."""
    config = _config(path)
    old = trace_old(config, policy, seed)
    new = trace_new(config, policy, seed)
    divergence = _first_divergence(old, new.boundaries)
    _assert_identical_until_the_rules_part(old, new, divergence)
    assert len(old) == WargameEnv(config).max_turns or old[-1] is not None


@pytest.mark.parametrize("seed", SEEDS)
def test_the_melee_bridge_holds_until_the_first_standing_charge(seed: int) -> None:
    """The per-model facade diverges from the phase facade only where the rules
    chapter told it to: both seats pile in, and a striker picks its target.
    Until a charge stands, every boundary must agree."""
    config = _config(MELEE)
    old = trace_old(config, "squad_march_take_charge", seed)
    new = trace_new(config, "squad_march_take_charge", seed)
    divergence = _first_divergence(old, new.boundaries)
    if divergence is None:
        return
    index = _divergence_index(divergence, old)
    # A charge that stood is visible either as the flag on our own boundary or,
    # for the opponent's charge inside our window, as the engagement it leaves.
    contact_before = any(b.charged or b.engaged for b in old[: index + 1])
    if not contact_before:
        _assert_identical_until_the_rules_part(old, new, divergence)
