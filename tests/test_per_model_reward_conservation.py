"""Reward conservation across the two facades (issue #286's central claim).

`reward_timing.py` re-times WHEN each term pays; this test pins that on
bridge-identical play the per-EPISODE scalar total of each term matches the
whole-phase facade's. The three mechanisms it guards (each found live by the
2026-09-07 audit): action terms paid undivided where the whole-phase scalar
means over alive (`model_kills` measured 24x heavy), state terms paid once
per round where the whole-phase facade pays once per phase step (halved), and
members consumed by a skip declaration never paying action terms at all.

The opponent here deliberately does not shoot: a player casualty during the
opponent's half-turn changes the alive set between the whole-phase facade's
boundary evaluations and the per-model facade's single close, which turns an
exact identity into an approximate one. `closest_objective_v2` is exempted by
name — its progress/overstack arithmetic reads the board mid-phase under
per-model stepping (each actor is scored as it moves), which is the per-step
credit the design asks for, not a scale error; every other term must agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.per_model import PerModelEnv, ScriptedPolicyAdapter
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import OpponentPolicyConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from scenario_overrides import load_env_config  # noqa: E402

# Mid-phase per-step evaluation is this term's design under per-model credit,
# so its episode total legitimately differs; everything else must conserve.
_EXEMPT = {"closest_objective_v2"}


def _config(rounds: str) -> WargameEnvConfig:
    config: WargameEnvConfig = load_env_config(
        "configs/golden/25v25_maps_two_mode.yaml", rounds=rounds
    )
    config.opponent_policy = OpponentPolicyConfig(type="scripted_advance_to_objective")
    return config


def _whole_phase_breakdown(
    policy_name: str, rounds: str, seed: int
) -> dict[str, float]:
    env = WargameEnv(_config(rounds), build_info=False)
    policy = build_baseline_policy(policy_name)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.select_action(
            env.wargame_models, env, action_mask=observation.action_mask
        )
        observation, _r, terminated, truncated, _ = env.step(action)
    return dict(env.episode_reward_breakdown)


def _per_model_breakdown(policy_name: str, rounds: str, seed: int) -> dict[str, float]:
    env = PerModelEnv(_config(rounds), build_info=False)
    adapter = ScriptedPolicyAdapter(build_baseline_policy(policy_name))
    observation, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        action = adapter.next_action(env, observation)
        observation, _r, terminated, _t, _ = env.step(action)
    return dict(env.episode_reward_breakdown)


# Each case names a term it must exercise NONZERO, so the identity cannot pass
# vacuously on a scenario where the term never fired: `squad_march_shoot` at
# six rounds closes into weapon range (kills — the term the audit measured 24x
# heavy), `split_evenly` breaks formation deterministically (the cohesion
# fine; `random` cannot serve here — its unseeded draw makes the two drives
# play different episodes, so nothing is comparable).
@pytest.mark.parametrize(
    ("policy_name", "rounds", "seed", "must_fire"),
    [
        ("squad_march_shoot", "6", 700001, "model_kills"),
        ("squad_march_shoot", "3", 700002, "objective_hold"),
        ("split_evenly", "3", 700001, "group_cohesion"),
    ],
)
def test_every_terms_episode_total_is_conserved_across_facades(
    policy_name: str, rounds: str, seed: int, must_fire: str
) -> None:
    whole = _whole_phase_breakdown(policy_name, rounds, seed)
    per_model = _per_model_breakdown(policy_name, rounds, seed)
    compared = 0
    for name in sorted(set(whole) | set(per_model)):
        if name in _EXEMPT or "/" in name:  # sub-components sum into parents
            continue
        a, b = whole.get(name, 0.0), per_model.get(name, 0.0)
        assert b == pytest.approx(a, rel=1e-6, abs=1e-9), (
            f"term {name!r}: whole-phase paid {a}, per-model paid {b} — "
            "the re-timing changed a term's mathematics, not just its step"
        )
        compared += 1
    # The identity must actually cover the golden stack, not pass vacuously.
    assert compared >= 4, f"only {compared} terms compared: {sorted(whole)}"
    assert whole.get(must_fire, 0.0) != 0.0, (
        f"{must_fire} never fired — this case is not exercising the term "
        "it exists to pin"
    )
    assert any(name in whole for name in _EXEMPT)
