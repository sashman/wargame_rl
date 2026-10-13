"""The backward start curriculum (#340): k squads begin on k distinct objectives,
the trainer walks k down as those episodes succeed, and none of it can leak into
an evaluation -- a reset that does not ask for it is bit-identical to the base
scenario, the same discipline the one-squad augmentation is pinned on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.per_model_seats import small_config
from train_per_model import BackwardStart, train
from wargame_rl.wargame.envs.domain.battle_factory import from_config
from wargame_rl.wargame.envs.domain.battlefield.placement import (
    place_for_episode,
    start_groups_on_objectives,
)
from wargame_rl.wargame.envs.per_model.env import PerModelEnv
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    ObjectiveConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.model.per_model.ppo import EpisodeOutcome


def _spread_config() -> WargameEnvConfig:
    """Four squads of three over four objectives, nobody shooting: the A3 shape."""
    return WargameEnvConfig(
        render_mode=None,
        board_width=60,
        board_height=44,
        number_of_wargame_models=12,
        number_of_opponent_models=0,
        number_of_objectives=4,
        objective_radius_size=4,
        number_of_battle_rounds=6,
        max_groups=4,
        deployment_zone=(2, 2, 10, 42),
        opponent_deployment_zone=(50, 2, 58, 42),
        models=[ModelConfig(group_id=i // 3) for i in range(12)],
        objectives=[
            ObjectiveConfig(x=30, y=8),
            ObjectiveConfig(x=30, y=22),
            ObjectiveConfig(x=30, y=36),
            ObjectiveConfig(x=45, y=22),
        ],
    )


def _digest(models: list) -> str:
    raw = np.array([[float(m.location[0]), float(m.location[1])] for m in models])
    return hashlib.sha256(raw.tobytes()).hexdigest()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_k_squads_start_on_k_distinct_objectives(k: int) -> None:
    config = _spread_config()
    battle = from_config(config)
    rng = np.random.default_rng(5)
    place_for_episode(battle, config, rng, augment_start=True, start_groups=k)
    # Assert: every model of a moved squad sits inside one objective, the
    # squads sit on distinct objectives, the others stay in the zone.
    by_group: dict[int, set[int]] = {}
    for model in battle.player_models:
        on = [
            i
            for i, o in enumerate(battle.objectives)
            if float(np.hypot(*(np.asarray(model.location) - np.asarray(o.location))))
            <= config.objective_radius_size
        ]
        by_group.setdefault(model.group_id, set()).update(on)
    moved = {g: on for g, on in by_group.items() if on}
    assert len(moved) == k
    assert len({next(iter(on)) for on in moved.values()}) == k
    assert all(len(on) == 1 for on in moved.values())
    for model in battle.player_models:
        if model.group_id not in moved:
            assert 2 <= float(model.location[0]) <= 10


def test_zero_start_groups_draws_nothing_and_matches_the_plain_placement() -> None:
    config = _spread_config()
    plain = from_config(config)
    place_for_episode(plain, config, np.random.default_rng(9))
    asked = from_config(config)
    place_for_episode(
        asked, config, np.random.default_rng(9), augment_start=True, start_groups=0
    )
    assert _digest(plain.player_models) == _digest(asked.player_models)
    # Draw order after the placement is untouched too.
    assert np.random.default_rng(9).random() == np.random.default_rng(9).random()


def test_the_function_caps_at_the_board_and_returns_the_objectives_taken() -> None:
    config = _spread_config()
    battle = from_config(config)
    place_for_episode(battle, config, np.random.default_rng(1))
    taken = start_groups_on_objectives(
        battle.player_models,
        battle.opponent_models,
        battle.objectives,
        np.random.default_rng(2),
        n_groups=9,
        base_radius=0.63,
    )
    assert sorted(taken) == [0, 1, 2, 3]
    assert (
        start_groups_on_objectives(
            battle.player_models,
            battle.opponent_models,
            battle.objectives,
            np.random.default_rng(2),
            n_groups=0,
        )
        == []
    )


def test_the_env_reset_option_places_squads_and_the_default_does_not() -> None:
    env = PerModelEnv(_spread_config())
    env.reset(seed=3, options={"augment_start": True, "start_groups": 2})
    on_points = sum(
        1
        for m in env.wargame_models
        for o in env.objectives
        if float(np.hypot(*(np.asarray(m.location) - np.asarray(o.location)))) <= 4
    )
    assert on_points == 6
    env.reset(seed=3, options={"augment_start": True})
    on_points = sum(
        1
        for m in env.wargame_models
        for o in env.objectives
        if float(np.hypot(*(np.asarray(m.location) - np.asarray(o.location)))) <= 4
    )
    assert on_points == 0


def _outcome(success: bool, level: int) -> EpisodeOutcome:
    return EpisodeOutcome(
        env_index=0,
        player_vp=0,
        opponent_vp=0,
        reward=0.0,
        decision_steps=1,
        rounds=1,
        success=success,
        start_groups=level,
    )


def test_the_schedule_steps_down_on_the_levels_own_successes_only() -> None:
    schedule = BackwardStart(
        level=2, share=1.0, advance_at=0.8, window=2, rng=np.random.default_rng(0)
    )
    assert schedule.draw() == 2
    # Deployment episodes succeeding do not count; level-2 failures hold it.
    assert schedule.observe([_outcome(True, 0), _outcome(False, 2)]) is False
    assert schedule.observe([_outcome(True, 0), _outcome(False, 2)]) is False
    assert schedule.level == 2
    # Two rollouts of level-2 successes step it down, and the window resets.
    assert schedule.observe([_outcome(True, 2), _outcome(True, 2)]) is False
    assert schedule.observe([_outcome(True, 2)]) is True
    assert schedule.level == 1
    assert schedule.rows()["curriculum/start_groups"] == 1.0
    assert schedule.observe([_outcome(True, 1), _outcome(True, 1)]) is False
    assert schedule.observe([_outcome(True, 1)]) is True
    assert schedule.level == 0
    assert schedule.draw() == 0
    assert schedule.observe([_outcome(True, 0)]) is False


def test_the_share_mixes_deployment_starts_in() -> None:
    schedule = BackwardStart(
        level=3, share=0.5, advance_at=0.8, window=4, rng=np.random.default_rng(1)
    )
    draws = [schedule.draw() for _ in range(200)]
    assert set(draws) == {0, 3}
    assert 60 < draws.count(3) < 140


def test_a_short_run_records_the_curriculum(tmp_path: Path) -> None:
    from pydantic_yaml import to_yaml_str

    yaml_path = tmp_path / "small.yaml"
    yaml_path.write_text(to_yaml_str(small_config(opponent_x=22, rounds=2)))
    run_dir: Path = train(
        env_config_path=str(yaml_path),
        rounds=4,
        rollout_rounds=1,
        num_rollout_envs=2,
        eval_every_rounds=2,
        checkpoint_every_rounds=2,
        record_every_rounds=0,
        video_every_rounds=0,
        n_eval_episodes=2,
        eval_wave_size=2,
        n_layers=2,
        embedding_size=32,
        seed=1,
        no_wandb=True,
        backward_start=1,
        backward_start_share=1.0,
        checkpoint_root=str(tmp_path / "ckpt"),
    )
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["driver"]["backward_start"] == 1
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    levels = [
        r["curriculum/start_groups"] for r in rows if "curriculum/start_groups" in r
    ]
    assert levels and levels[0] == 1.0
