"""Evaluation and recording under the per-model step (issue #287): batched
waves that survive unequal episode lengths, the snapshot's second clock, and
the per-model-step recording cadence."""

from __future__ import annotations

import numpy as np
import torch

from wargame_rl.wargame.envs.per_model import PerModelAction, PerModelEnv
from wargame_rl.wargame.envs.state.snapshot import EpisodeProvenance, GameStateSnapshot
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.per_model import SetAgent, SetNetwork, SetNetworkConfig
from wargame_rl.wargame.model.per_model.evaluate import (
    evaluate_per_model,
    evaluate_per_model_batched,
)

SMALL_TRUNK = SetNetworkConfig(embedding_size=32, n_layers=2, n_heads=4)


def _config() -> WargameEnvConfig:
    rifle = [WeaponProfile(range=12, attacks=1)]
    squads = [
        ModelConfig(group_id=i // 3, weapons=rifle, max_wounds=1) for i in range(6)
    ]
    return WargameEnvConfig(
        render_mode=None,
        board_width=30,
        board_height=30,
        number_of_wargame_models=6,
        number_of_opponent_models=6,
        max_groups=2,
        models=squads,
        opponent_models=list(squads),
        number_of_objectives=2,
        number_of_battle_rounds=2,
        skip_phases=[
            BattlePhase.command,
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
    )


def _network() -> SetNetwork:
    torch.manual_seed(0)
    env = PerModelEnv(_config(), build_info=False)
    return SetNetwork.from_env(env, SMALL_TRUNK)


def test_batched_evaluation_equals_sequential() -> None:
    """Waves run until the last env finishes with finished envs masked, and a
    seed's episode comes out identical to the sequential greedy drive — the
    per-model form of the lockstep-eval fix."""
    network = _network()
    seeds = [11, 12, 13, 14, 15]
    sequential = evaluate_per_model(
        PerModelEnv(_config(), build_info=False),
        SetAgent(network),
        seeds,
        "sequential",
    )
    for n_parallel in (1, 3):
        batched = evaluate_per_model_batched(
            _config(), network, seeds, "batched", n_parallel=n_parallel
        )
        assert batched.n_episodes == len(seeds)
        np.testing.assert_array_equal(
            np.array(batched.vp_margin_per_episode),
            np.array(sequential.vp_margin_per_episode),
        )
        np.testing.assert_array_equal(
            np.array(batched.objectives_held_per_episode),
            np.array(sequential.objectives_held_per_episode),
        )


def test_the_snapshot_carries_the_second_clock() -> None:
    env = PerModelEnv(_config(), build_info=False)
    observation, _ = env.reset(seed=3)
    env.step(
        PerModelAction(model_index=int(np.flatnonzero(observation.selection_mask)[0]))
    )
    snapshot = env.to_snapshot()
    assert snapshot.schema_version == "2.8"
    assert snapshot.model_step == env.model_steps == 1
    # `step` keeps counting phases, so every max_turns reader keeps its meaning.
    assert snapshot.step == env.current_turn

    whole_phase = WargameEnv(_config(), build_info=False)
    whole_phase.reset(seed=3)
    assert whole_phase.to_snapshot().model_step is None


class _CountingExporter:
    def __init__(self) -> None:
        self.resets = 0
        self.steps = 0

    def on_reset(
        self,
        snapshot: GameStateSnapshot,
        provenance: EpisodeProvenance | None = None,
    ) -> None:
        self.resets += 1

    def on_step(self, snapshot: GameStateSnapshot) -> None:
        self.steps += 1


def test_recording_cadence_is_a_setting() -> None:
    """Default: one snapshot per phase boundary (today's schema semantics).
    Per-model-step cadence adds one per model step, for debugging."""

    def run(record_model_steps: bool) -> int:
        exporter = _CountingExporter()
        env = PerModelEnv(
            _config(),
            record_model_steps=record_model_steps,
            state_exporters=[exporter],  # type: ignore[list-item]
            build_info=False,
        )
        observation, _ = env.reset(seed=5)
        terminated = False
        while not terminated:
            if observation.selection_mask.any():
                action = PerModelAction(
                    model_index=int(np.flatnonzero(observation.selection_mask)[0])
                )
            else:
                action = PerModelAction()
            observation, _r, terminated, _t, _ = env.step(action)
        return exporter.steps

    phase_cadence = run(record_model_steps=False)
    per_model_cadence = run(record_model_steps=True)
    # 2 rounds x 2 stepped phases = 4 boundaries; per-model adds the 6-model
    # steps of each phase plus the closing steps.
    assert phase_cadence == 4
    assert per_model_cadence == phase_cadence + 2 * 2 * 6 + 2
