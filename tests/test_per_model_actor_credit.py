"""`Credit.actor`: each model is paid its own term, undivided, and its own
state credit on its own step of the turn.

Under the default `Credit.mean` (the bridge accounting) a model's own move is
worth `1/n_alive` of the travel pay and every state term reaches it as the
army mean at the close -- on a 24-body army the only thing that distinguishes
one decision from another is a twenty-fourth of one term (#340: the A3 speed
screen's null, A5b's under-arrival). Under `actor` every payment is over the
MODEL COUNT, a constant: the actor's action term stays where the mean put it
(alive count against model count), while the common payments -- globals,
terminal bonuses, and the state terms now returned as per-model credits --
shrink by the army size. Pinned here: the default is unchanged step for step;
the actor's arithmetic against the mean's, term by term; and the rollout
collector lands each credit on the transition its model acted on this turn,
or on the close when it took none, conserving the total.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.domain.kernel.entities import alive_mask_for
from wargame_rl.wargame.envs.per_model import PerModelEnv, StepKind
from wargame_rl.wargame.envs.per_model.random_seat import random_legal_action
from wargame_rl.wargame.envs.per_model.reward_timing import (
    Credit,
    PerStepReward,
    StepPayment,
)
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.per_model import (
    PerModelPPOConfig,
    SetAgent,
    SetNetwork,
    SetNetworkConfig,
    collect_rollout,
)

COMBAT_SEED = 4242
STATE_TERMS = ("objective_hold", "group_cohesion")


def _config(**kwargs: Any) -> WargameEnvConfig:
    phase = RewardPhaseConfig(
        name="retimed",
        reward_calculators=[
            RewardCalculatorConfig(
                type="closest_objective_v2",
                weight=1.0,
                params={"progress_scale": 6.0, "fallback_to_nearest": True},
            ),
            RewardCalculatorConfig(
                type="objective_hold", weight=1.25, params={"crowding_exponent": 1.0}
            ),
            RewardCalculatorConfig(
                type="model_kills", weight=1.0, params={"bonus_per_kill": 2.0}
            ),
            RewardCalculatorConfig(type="group_cohesion", weight=0.3),
            RewardCalculatorConfig(type="vp_gain", weight=2.0),
            RewardCalculatorConfig(type="objective_coverage", weight=0.3),
        ],
        success_criteria=SuccessCriteriaConfig(type="player_ahead_on_vp"),
        terminal_success_bonus=2.0,
    )
    base = small_config(**kwargs)
    return WargameEnvConfig(**{**base.model_dump(), "reward_phases": [phase]})


def _play(
    env: PerModelEnv, retimers: list[PerStepReward], seed: int
) -> list[tuple[StepKind, int, list[StepPayment]]]:
    """One random-legal episode, every retimer paid on every step; returns
    (kind, alive count after the step, payments per retimer) per step."""
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed, options={"combat_seed": COMBAT_SEED})
    for retimer in retimers:
        retimer.reset()
    rows = []
    done = False
    while not done:
        action = random_legal_action(observation.decision, rng)
        before = observation
        observation, _, done, _, info = env.step(action)
        n_alive = int(alive_mask_for(env.wargame_models).sum())
        payments = [r.on_step(before, action, info["effect"], done) for r in retimers]
        rows.append((action.kind, n_alive, payments))
    return rows


def test_the_default_is_the_mean_and_is_unchanged_step_for_step() -> None:
    env = PerModelEnv(_config())
    default, explicit = PerStepReward(env), PerStepReward(env, credit=Credit.mean)
    assert default.credit is Credit.mean
    for _kind, _n, (a, b) in _play(env, [default, explicit], seed=3):
        assert a.reward == b.reward
        assert a.breakdown == b.breakdown
        assert a.credits == {} and b.credits == {}


def test_the_actor_keeps_its_action_term_and_owns_its_state_credit() -> None:
    env = PerModelEnv(_config())
    n_models = len(env.wargame_models)
    mean, actor = PerStepReward(env), PerStepReward(env, credit=Credit.actor)
    saw_credit = saw_action = False
    for kind, n_alive, (m, a) in _play(env, [mean, actor], seed=5):
        if kind is StepKind.close_turn:
            # The close's scalar keeps the common payments over the model
            # count and drops the state terms, which come back as per-model
            # credits summing to n_alive / n_models x the mean's payment.
            state_mean = sum(m.breakdown.get(t, 0.0) for t in STATE_TERMS)
            assert a.reward == pytest.approx((m.reward - state_mean) / n_models)
            assert sum(a.credits.values()) == pytest.approx(
                state_mean * n_alive / n_models
            )
            assert set(a.credits) == set(
                int(i) for i in np.flatnonzero(alive_mask_for(env.wargame_models))
            )
            for term in STATE_TERMS:
                assert a.breakdown[term] == pytest.approx(
                    m.breakdown[term] * n_alive / n_models
                )
            saw_credit = saw_credit or bool(a.credits)
        else:
            # The actor's term: over the model count instead of the alive count.
            assert a.credits == {}
            assert a.reward == pytest.approx(m.reward * n_alive / n_models)
            saw_action = saw_action or a.reward != 0.0
    assert saw_credit and saw_action
    # Both accountings report the whole of what they paid.
    assert actor.episode_reward == pytest.approx(sum(actor.episode_breakdown.values()))


def test_the_collector_lands_each_credit_on_the_models_own_step() -> None:
    torch.manual_seed(0)
    config = _config(rounds=2)
    envs = [PerModelEnv(config), PerModelEnv(config)]

    class Spy(PerStepReward):
        def __init__(self, env: PerModelEnv) -> None:
            super().__init__(env, credit=Credit.actor)
            self.log: list[tuple[tuple[int, ...], StepPayment]] = []

        def on_step(self, before: Any, action: Any, effect: Any, done: bool) -> Any:
            payment = super().on_step(before, action, effect, done)
            self.log.append((effect.actor_set, payment))
            return payment

    retimers = [Spy(env) for env in envs]
    observations = [env.reset(seed=10 + i)[0] for i, env in enumerate(envs)]
    for retimer in retimers:
        retimer.reset()
    agent = SetAgent(
        SetNetwork.from_env(
            envs[0], SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)
        )
    )
    rollout = collect_rollout(
        envs,
        agent,
        retimers,
        observations,
        PerModelPPOConfig(rollout_rounds=1, credit=Credit.actor),
        generator=torch.Generator().manual_seed(0),
    )

    # Rebuild each env's expected reward per transition by the rule: a credit
    # lands where its model last acted this turn, else on the close.
    for env_index, spy in enumerate(retimers):
        got = [t.reward for t in rollout.transitions if t.env_index == env_index]
        assert len(got) == len(spy.log)
        expected = [p.reward for _, p in spy.log]
        acted_at: dict[int, int] = {}
        credited_off_close = 0
        for at, (actors, payment) in enumerate(spy.log):
            for model in actors:
                acted_at[model] = at
            for model, value in payment.credits.items():
                target = acted_at.get(model, at)
                expected[target] += value
                credited_off_close += target != at
            if payment.is_close:
                acted_at = {}
        assert got == pytest.approx(expected)
        assert credited_off_close > 0
        assert sum(got) == pytest.approx(
            sum(p.reward + sum(p.credits.values()) for _, p in spy.log)
        )
