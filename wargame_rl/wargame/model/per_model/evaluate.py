"""Greedy, seeded evaluation of a set network over the per-model facade.

The minimum the training driver needs to select a checkpoint and log the
ladder's readouts: one episode per seed, in seed order, so two runs of the
same seeds pair. Issue #287 owns the batched form, recording and the
`measure-*` audit; this is not that. Every game fact is read off the env
(`objectives_held`, `units_coherent`), never derived here.

`coherency_rate` is the mean over the episode's turn closes of the share of
living units in coherency -- PROVISIONAL: the whole-phase readout samples at
the movement boundary, and #287 reconciles the two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from wargame_rl.wargame.envs.domain.kernel.entities import alive_mask_for
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.per_model.reward_timing import PerStepReward
from wargame_rl.wargame.model.per_model.agent import SetAgent


@dataclass(frozen=True)
class EpisodeResult:
    """One evaluated episode."""

    seed: int
    player_vp: int
    opponent_vp: int
    reward: float
    decision_steps: int
    fraction_alive: float
    objectives_held: int
    success: bool
    coherency_rate: float

    @property
    def vp_margin(self) -> int:
        return self.player_vp - self.opponent_vp

    @property
    def won(self) -> bool:
        return self.player_vp > self.opponent_vp


@dataclass(frozen=True)
class PerModelEvalResult:
    """Means over the episodes, in the whole-phase trainer's vocabulary."""

    episodes: tuple[EpisodeResult, ...]

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def _mean(self, values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @property
    def vp_player(self) -> float:
        return self._mean([e.player_vp for e in self.episodes])

    @property
    def vp_opponent(self) -> float:
        return self._mean([e.opponent_vp for e in self.episodes])

    @property
    def vp_margin(self) -> float:
        return self._mean([e.vp_margin for e in self.episodes])

    @property
    def vp_margin_se(self) -> float | None:
        margins = [float(e.vp_margin) for e in self.episodes]
        if len(margins) < 2:
            return None
        return float(np.std(margins, ddof=1) / np.sqrt(len(margins)))

    @property
    def win_rate(self) -> float:
        return self._mean([float(e.won) for e in self.episodes])

    @property
    def mean_reward(self) -> float:
        return self._mean([e.reward for e in self.episodes])

    @property
    def max_reward(self) -> float:
        return max((e.reward for e in self.episodes), default=0.0)

    @property
    def min_reward(self) -> float:
        return min((e.reward for e in self.episodes), default=0.0)

    @property
    def mean_steps(self) -> float:
        return self._mean([e.decision_steps for e in self.episodes])

    @property
    def fraction_alive(self) -> float:
        return self._mean([e.fraction_alive for e in self.episodes])

    @property
    def objectives_held(self) -> float:
        return self._mean([e.objectives_held for e in self.episodes])

    @property
    def success_rate(self) -> float:
        return self._mean([float(e.success) for e in self.episodes])

    @property
    def coherency_rate(self) -> float:
        return self._mean([e.coherency_rate for e in self.episodes])


def evaluate_per_model(
    env: PerModelEnv,
    agent: SetAgent,
    retimer: PerStepReward,
    seeds: Sequence[int],
    *,
    combat_seeds: Sequence[int] | None = None,
) -> PerModelEvalResult:
    """One greedy episode per seed, in seed order, on `env`."""
    if combat_seeds is not None and len(combat_seeds) != len(seeds):
        raise ValueError("combat_seeds must match seeds in length")
    was_greedy = agent.greedy
    was_training = agent.network.training
    agent.greedy = True
    agent.network.eval()
    episodes: list[EpisodeResult] = []
    try:
        for index, seed in enumerate(seeds):
            options = (
                None if combat_seeds is None else {"combat_seed": combat_seeds[index]}
            )
            episodes.append(_play_episode(env, agent, retimer, seed, options))
    finally:
        agent.greedy = was_greedy
        agent.network.train(was_training)
    return PerModelEvalResult(episodes=tuple(episodes))


def _play_episode(
    env: PerModelEnv,
    agent: SetAgent,
    retimer: PerStepReward,
    seed: int,
    options: dict[str, int] | None,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed, options=options)
    retimer.reset()
    decision_steps = 0
    coherency: list[float] = []
    with torch.no_grad():
        while True:
            decision = agent.act(env, observation)
            before = observation
            observation, _, terminated, _, info = env.step(decision.action)
            payment = retimer.on_step(
                before, decision.action, info["effect"], terminated
            )
            if decision.has_policy:
                decision_steps += 1
            if payment.is_close:
                coherency.append(env.units_coherent())
            if terminated:
                break
    alive = alive_mask_for(env.wargame_models)
    return EpisodeResult(
        seed=seed,
        player_vp=int(env.player_vp),
        opponent_vp=int(env.opponent_vp),
        reward=retimer.episode_reward,
        decision_steps=decision_steps,
        fraction_alive=float(alive.mean()) if alive.size else 0.0,
        objectives_held=env.objectives_held,
        success=retimer.succeeded(),
        coherency_rate=float(np.mean(coherency)) if coherency else 1.0,
    )


__all__ = ["EpisodeResult", "PerModelEvalResult", "evaluate_per_model"]
