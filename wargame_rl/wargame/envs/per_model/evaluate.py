"""Seeded, batched evaluation of a chooser over the per-model facade.

The per-model counterpart of `envs/baseline/evaluate.py::evaluate_selector`,
producing the same `EvalResult` through the same end-of-episode readouts
(`envs/evaluation/`), so a `.pt` scored here and a script scored there sit on
one table and pair per seed.

Batched in WAVES: `len(envs)` episodes run in lockstep, one `choose` call
per iteration over the envs still playing. A network seat answers the whole
wave in one forward; a scripted seat answers env by env. An env drops out of
the wave when its episode terminates -- under the per-model step episodes
take different numbers of decisions, so a wave is never a fixed number of
steps. Results land by seed index, so the per-episode tuples are in seed
order whatever order the episodes finished in.

Numpy only, and nothing from `baseline/`: that package imports the phase
facade, and the two facades share the domain and nothing about the step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.evaluation import (
    EvalResult,
    mean_of_measured,
    read_end_of_episode,
)
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.random_seat import random_legal_action
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.envs.per_model.scripted import PhasePolicy, ScriptedSeat
from wargame_rl.wargame.envs.per_model.types import (
    BatchChooser,
    PerModelAction,
    PerModelObservation,
    StepKind,
)


@dataclass(frozen=True)
class _Episode:
    """One finished episode's readouts, before aggregation."""

    player_vp: float
    opponent_vp: float
    at_objectives: float
    objectives_held: float
    fraction_alive: float
    worst_cohesion_gap: float
    coherency_rate: float | None
    models_out_of_coherency: float | None
    opponent_coherency_rate: float | None
    opponent_models_out_of_coherency: float | None
    decisions: int
    reward: float | None
    success: bool | None

    @property
    def won(self) -> float:
        return 1.0 if self.player_vp > self.opponent_vp else 0.0


def evaluate_per_model_chooser(
    choose: BatchChooser,
    envs: Sequence[PerModelEnv],
    seeds: Sequence[int],
    name: str,
    *,
    combat_seeds: Sequence[int] | None = None,
    retimers: Sequence[PerStepReward] | None = None,
) -> EvalResult:
    """Run `choose` once per seed over waves of `envs` and aggregate the outcome.

    `retimers`, one per env, supply the per-model-only readouts -- the
    re-timed episode reward and the success criterion; without them those
    columns are `None`, the convention an unmeasured metric follows.
    """
    if not envs:
        raise ValueError("evaluate_per_model_chooser needs at least one env")
    if combat_seeds is not None and len(combat_seeds) != len(seeds):
        raise ValueError(
            f"combat_seeds must match seeds in length: "
            f"{len(combat_seeds)} != {len(seeds)}"
        )
    if retimers is not None and len(retimers) != len(envs):
        raise ValueError(
            f"retimers must match envs in length: {len(retimers)} != {len(envs)}"
        )
    episodes: list[_Episode | None] = [None] * len(seeds)
    wave_size = len(envs)
    for start in range(0, len(seeds), wave_size):
        wave = list(range(start, min(start + wave_size, len(seeds))))
        observations: dict[int, PerModelObservation] = {}
        decisions: dict[int, int] = {}
        for slot, seed_index in enumerate(wave):
            options = (
                None
                if combat_seeds is None
                else {"combat_seed": combat_seeds[seed_index]}
            )
            observation, _ = envs[slot].reset(seed=seeds[seed_index], options=options)
            if retimers is not None:
                retimers[slot].reset()
            observations[slot] = observation
            decisions[slot] = 0
        active = list(range(len(wave)))
        while active:
            actions = choose(
                [envs[slot] for slot in active],
                [observations[slot] for slot in active],
            )
            still_playing: list[int] = []
            for slot, action in zip(active, actions):
                env = envs[slot]
                before = observations[slot]
                observation, _reward, terminated, _truncated, info = env.step(action)
                if retimers is not None:
                    retimers[slot].on_step(before, action, info["effect"], terminated)
                if before.decision.kind is not StepKind.close_turn:
                    decisions[slot] += 1
                observations[slot] = observation
                if terminated:
                    episodes[wave[slot]] = _read_episode(
                        env,
                        None if retimers is None else retimers[slot],
                        decisions[slot],
                    )
                else:
                    still_playing.append(slot)
            active = still_playing
    finished = [episode for episode in episodes if episode is not None]
    assert len(finished) == len(seeds)
    return _aggregate(name, finished)


def _read_episode(
    env: PerModelEnv, retimer: PerStepReward | None, decisions: int
) -> _Episode:
    end = read_end_of_episode(env.wargame_models, env.opponent_models, env.objectives)
    # Intent first, as `evaluate_selector` reads it: under `enforce_move` the
    # realised rate reports the referee. With no referee the two agree.
    return _Episode(
        player_vp=float(env.player_vp),
        opponent_vp=float(env.opponent_vp),
        at_objectives=end.at_objectives,
        objectives_held=end.objectives_held,
        fraction_alive=end.fraction_alive,
        worst_cohesion_gap=end.worst_cohesion_gap,
        coherency_rate=_prefer(env.intended_coherency_rate, env.coherency_rate),
        models_out_of_coherency=_prefer(
            env.intended_models_out_of_coherency, env.models_out_of_coherency
        ),
        opponent_coherency_rate=_prefer(
            env.opponent_intended_coherency_rate, env.opponent_coherency_rate
        ),
        opponent_models_out_of_coherency=_prefer(
            env.opponent_intended_models_out_of_coherency,
            env.opponent_models_out_of_coherency,
        ),
        decisions=decisions,
        reward=None if retimer is None else retimer.episode_reward,
        success=None if retimer is None else retimer.succeeded(),
    )


def _prefer(intended: float | None, realised: float | None) -> float | None:
    return intended if intended is not None else realised


def _aggregate(name: str, episodes: list[_Episode]) -> EvalResult:
    rewards = [e.reward for e in episodes]
    successes = [e.success for e in episodes]
    measured_reward = all(r is not None for r in rewards)
    return EvalResult(
        name=name,
        n_episodes=len(episodes),
        final_fraction_at_objectives=float(
            np.mean([e.at_objectives for e in episodes])
        ),
        win_rate=float(np.mean([e.won for e in episodes])),
        player_vp=float(np.mean([e.player_vp for e in episodes])),
        opponent_vp=float(np.mean([e.opponent_vp for e in episodes])),
        worst_cohesion_gap=float(np.mean([e.worst_cohesion_gap for e in episodes])),
        final_fraction_alive=float(np.mean([e.fraction_alive for e in episodes])),
        # This facade tracks neither exposure nor firepower; None, never 0.0.
        exposure_rate=None,
        terrain_proximity=None,
        firepower_ratio=None,
        objectives_held=float(np.mean([e.objectives_held for e in episodes])),
        coherency_rate=mean_of_measured([e.coherency_rate for e in episodes]),
        models_out_of_coherency=mean_of_measured(
            [e.models_out_of_coherency for e in episodes]
        ),
        opponent_coherency_rate=mean_of_measured(
            [e.opponent_coherency_rate for e in episodes]
        ),
        opponent_models_out_of_coherency=mean_of_measured(
            [e.opponent_models_out_of_coherency for e in episodes]
        ),
        vp_margin_per_episode=tuple(e.player_vp - e.opponent_vp for e in episodes),
        objectives_held_per_episode=tuple(e.objectives_held for e in episodes),
        win_per_episode=tuple(e.won for e in episodes),
        episode_rewards=(
            tuple(float(r) for r in rewards if r is not None)
            if measured_reward
            else None
        ),
        decisions_per_episode=tuple(e.decisions for e in episodes),
        success_per_episode=(
            tuple(bool(s) for s in successes if s is not None)
            if measured_reward
            else None
        ),
    )


def scripted_chooser(policy: PhasePolicy, envs: Sequence[PerModelEnv]) -> BatchChooser:
    """`policy` seated on every env in `envs`, answering decision by decision.

    The seat is installed NOW, before any reset, because a script plans its
    command phase inside `reset`; and each call reads the seat back off the
    env rather than closing over it, so a rewound (deep-copied) env is driven
    by its own copy of the plan.
    """
    for env in envs:
        env.set_player_planner(ScriptedSeat.for_policy(policy))

    def choose(
        envs_: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        actions: list[PerModelAction] = []
        for env, observation in zip(envs_, observations):
            point = observation.decision
            if point.kind is StepKind.close_turn:
                actions.append(PerModelAction.close_turn())
                continue
            planner = env.player_seat.adapter
            if planner is None:
                raise RuntimeError("the env has no scripted seat installed")
            actions.append(planner.choose(point, env.player_seat))
        return actions

    return choose


def random_chooser(seed: int) -> BatchChooser:
    """The random legal seat, one generator for the whole wave."""
    rng = np.random.default_rng(seed)

    def choose(
        envs: Sequence[PerModelEnv], observations: Sequence[PerModelObservation]
    ) -> list[PerModelAction]:
        return [random_legal_action(o.decision, rng) for o in observations]

    return choose


__all__ = ["evaluate_per_model_chooser", "random_chooser", "scripted_chooser"]
