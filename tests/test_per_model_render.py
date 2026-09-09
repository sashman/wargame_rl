"""The eyeball rung, made checkable: frames of the per-model facade with its decisions drawn.

Headless throughout (the recorder, the pillow backend). What a picture cannot
prove is left to the eye; what it can is pinned: a frame is drawn before every
decision, the decision overlay changes the pixels and nothing else does, the
phase cadence draws exactly one frame per settled window, the random seat
plays a whole episode under the renderer, and the frames export to a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.renders.v2.decision import highlight_from
from wargame_rl.wargame.envs.renders.v2.factory import build_backend, resolve_theme
from wargame_rl.wargame.envs.renders.v2.presenters.per_model import PerModelRecorder

pytest.importorskip("PIL")

from play_per_model import build_chooser, play_episode  # noqa: E402


def _recorder() -> PerModelRecorder:
    return PerModelRecorder(build_backend("pillow"), resolve_theme("default"))


def test_a_frame_is_drawn_before_every_decision() -> None:
    """Arrange the small scenario with a scripted seat; act by playing one
    episode at decision cadence; assert one frame per decision taken."""
    env = PerModelEnv(small_config(rounds=1))
    recorder = _recorder()
    chooser = build_chooser(env, "squad_march_take", seed=1)
    play_episode(env, recorder, chooser, seed=1, cadence="decision")
    assert len(recorder.frames) == env.episode_step
    assert all(f.ndim == 3 and f.shape[2] == 3 for f in recorder.frames)


def test_the_decision_overlay_changes_the_frame_and_only_the_overlay_does() -> None:
    """The same board drawn with and without the pending decision differs;
    drawn twice without it, it does not."""
    env = PerModelEnv(small_config(rounds=1))
    observation, _ = env.reset(seed=3)
    recorder = _recorder()
    recorder.setup(env)
    recorder.controls.decision = None
    recorder.render(env)
    recorder.render(env)
    plain_a, plain_b = recorder.frames[-2], recorder.frames[-1]
    recorder.controls.decision = highlight_from(observation.decision, sub_step=0)
    recorder.render(env)
    highlighted = recorder.frames[-1]
    assert np.array_equal(plain_a, plain_b)
    assert not np.array_equal(plain_a, highlighted)


def test_phase_cadence_draws_one_frame_per_settled_window() -> None:
    """Arrange the same episode at phase cadence and, beside it, unrendered;
    assert the frame count is the number of steps that settled a reward
    window, plus the terminal one."""
    config = small_config(rounds=2)
    env = PerModelEnv(config)
    recorder = _recorder()
    play_episode(
        env,
        recorder,
        build_chooser(env, "squad_march_take", 1),
        seed=1,
        cadence="phase",
    )

    twin = PerModelEnv(config)
    chooser = build_chooser(twin, "squad_march_take", 1)
    observation, _ = twin.reset(seed=1)
    settled = 0
    done = False
    while not done:
        observation, _r, done, _t, info = twin.step(chooser(observation))
        settled += bool(info.get("reward_settled") or done)
    assert settled > 0 and len(recorder.frames) == settled


def test_the_random_seat_plays_a_whole_episode_under_the_renderer() -> None:
    env = PerModelEnv(small_config(rounds=1))
    recorder = _recorder()
    chooser = build_chooser(env, "random", seed=5)
    play_episode(env, recorder, chooser, seed=5, cadence="decision")
    assert env.current_turn == env.max_turns
    assert len(recorder.frames) == env.episode_step


def test_the_set_network_plays_a_whole_episode_under_the_renderer() -> None:
    """The stage-2 pipeline -- tokens, network, decode -- through the same
    loop the eyeball rung uses, at fresh weights."""
    pytest.importorskip("torch")
    env = PerModelEnv(small_config(rounds=1))
    recorder = _recorder()
    chooser = build_chooser(env, "set_network", seed=7)
    play_episode(env, recorder, chooser, seed=7, cadence="decision")
    assert env.current_turn == env.max_turns
    assert len(recorder.frames) == env.episode_step


def test_the_frames_export_to_an_mp4(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    env = PerModelEnv(small_config(rounds=1))
    recorder = _recorder()
    chooser = build_chooser(env, "squad_march_take", seed=1)
    play_episode(env, recorder, chooser, seed=1, cadence="phase")
    out = tmp_path / "per_model.mp4"
    recorder.export_mp4(str(out), fps=4)
    assert out.exists() and out.stat().st_size > 0
