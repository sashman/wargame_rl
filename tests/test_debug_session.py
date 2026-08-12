"""Hand-stepping a live match: rewind fidelity, the driving loop, and the keys.

The load-bearing test here is `test_a_restored_env_replays_the_step_identically`.
Everything the debug mode is for — "what if this model had gone left" — rests on
a rewind that changes nothing but the decision, so a restore that quietly alters
the reward or the dice would make every comparison drawn through this tool wrong
while looking perfectly reasonable on screen.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from wargame_rl.wargame.envs.baseline.evaluate import selector_for  # noqa: E402
from wargame_rl.wargame.envs.baseline.registry import (  # noqa: E402
    build_baseline_policy,
)
from wargame_rl.wargame.envs.debug import (  # noqa: E402
    UndoStack,
    capture_state,
    run_session,
)
from wargame_rl.wargame.envs.domain.battle_view import BattleView  # noqa: E402
from wargame_rl.wargame.envs.renders.human import QuitRequested  # noqa: E402
from wargame_rl.wargame.envs.renders.renderer import Renderer  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.factory import build_backend  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.presenters.debug import (  # noqa: E402
    PAUSED_POLL_FPS,
    DebugControls,
    DebugPresenter,
)
from wargame_rl.wargame.envs.types import (  # noqa: E402
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import (  # noqa: E402
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase  # noqa: E402
from wargame_rl.wargame.envs.wargame import WargameEnv  # noqa: E402


def _env(**overrides: Any) -> WargameEnv:
    """A small scenario that actually shoots.

    Dice have to be in play for a rewind to prove anything about the RNG, and
    shooting is skipped by default — so the phases and the weapons are both part
    of the fixture, not decoration.
    """
    rifle = [WeaponProfile(range=30, attacks=2)]
    base: dict[str, Any] = dict(
        board_width=20,
        board_height=20,
        number_of_wargame_models=3,
        number_of_opponent_models=3,
        number_of_objectives=1,
        number_of_battle_rounds=4,
        models=[ModelConfig(x=4, y=5 + i, weapons=rifle) for i in range(3)],
        opponent_models=[ModelConfig(x=15, y=5 + i, weapons=rifle) for i in range(3)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        render_mode=None,
    )
    base.update(overrides)
    return WargameEnv(config=WargameEnvConfig(**base))


def _outcome(env: WargameEnv) -> dict[str, Any]:
    """Everything a step produced that a rewind must reproduce exactly."""
    return {
        "reward": env.last_reward,
        "breakdown": dict(env.last_reward_breakdown),
        "per_model": env.last_per_model_reward.copy(),
        "player_vp": env.player_vp,
        "opponent_vp": env.opponent_vp,
        "player": np.array([m.location for m in env.player_models]),
        "opponent": np.array([m.location for m in env.opponent_models]),
        "wounds": [m.stats["current_wounds"] for m in env.player_models],
        "turn": env.current_turn,
    }


def _assert_same(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left["reward"] == right["reward"]
    assert left["breakdown"] == right["breakdown"]
    np.testing.assert_array_equal(left["per_model"], right["per_model"])
    np.testing.assert_array_equal(left["player"], right["player"])
    np.testing.assert_array_equal(left["opponent"], right["opponent"])
    assert left["wounds"] == right["wounds"]
    assert (left["player_vp"], left["opponent_vp"]) == (
        right["player_vp"],
        right["opponent_vp"],
    )
    assert left["turn"] == right["turn"]


# --- Rewind fidelity --------------------------------------------------------


def test_a_restored_env_replays_the_step_identically() -> None:
    """The guard on the whole undo mechanism.

    Deep-copying rather than round-tripping a `GameStateSnapshot` is a deliberate
    choice: `load_state` never restores the combat RNG or terrain, and
    `state/restore.py` clears the reward-shaping memory on purpose. Any of those
    would make the same action score differently the second time, which is
    precisely the comparison this tool exists to make.
    """
    env = _env()
    env.reset(seed=1234)
    action = selector_for(build_baseline_policy("squad_march_shoot"))(
        env.observation, env
    )

    saved = capture_state(env)
    env.step(action)
    first = _outcome(env)

    replayed = saved
    replayed.step(action)

    _assert_same(first, _outcome(replayed))


def test_rewinding_several_steps_lands_on_the_right_one() -> None:
    """Each pop must return the state from before *that* step, not the newest."""
    env = _env()
    env.reset(seed=99)
    select = selector_for(build_baseline_policy("squad_march_shoot"))
    undo = UndoStack(depth=10)

    positions = []
    for _ in range(3):
        positions.append(np.array([m.location for m in env.player_models]))
        undo.push(env)
        env.step(select(env.observation, env))

    for expected in reversed(positions):
        restored = undo.pop()
        assert restored is not None
        np.testing.assert_array_equal(
            np.array([m.location for m in restored.player_models]), expected
        )
    assert undo.pop() is None


def test_the_undo_stack_is_bounded() -> None:
    """A long episode must not grow the history without limit."""
    env = _env()
    env.reset(seed=7)
    undo = UndoStack(depth=2)

    for _ in range(5):
        undo.push(env)

    assert len(undo) == 2


def test_capturing_leaves_the_renderer_attached() -> None:
    """The renderer is detached only across the copy — a window is not copyable,
    but the live env still needs it afterwards."""
    env = _env()
    env.reset(seed=0)
    renderer = DebugPresenter(build_backend("pillow"), DebugControls())
    env.renderer = renderer

    copied = capture_state(env)

    assert env.renderer is renderer
    assert copied.renderer is None


# --- The driving loop -------------------------------------------------------


class _ScriptedRenderer(Renderer):
    """Replays a list of control mutations, one per rendered frame, then quits.

    Stands in for the window so the loop can be driven deterministically: each
    entry is applied when that frame is rendered, and running out ends the
    session the same way pressing Esc would.
    """

    def __init__(self, controls: DebugControls, script: list[Any]) -> None:
        self._controls = controls
        self._script = list(script)
        self.frames = 0
        self.turns_seen: list[int] = []

    def setup(self, view: BattleView) -> None:
        return None

    def render(self, view: BattleView) -> None:
        self.frames += 1
        self.turns_seen.append(view.current_turn)
        if not self._script:
            raise QuitRequested()
        action = self._script.pop(0)
        if action is not None:
            action(self._controls)

    def close(self) -> None:
        return None


# These mirror what `DebugPresenter._handle_key` does, pausing included — a step
# request means "one step", not "resume from here", and a script that set the
# flag without the pause would exercise a combination no key can produce.
def _step(controls: DebugControls) -> None:
    controls.step_once = True
    controls.paused = True


def _back(controls: DebugControls) -> None:
    controls.step_back = True
    controls.paused = True


def _play(controls: DebugControls) -> None:
    controls.paused = False


def test_a_paused_session_does_not_advance_the_match() -> None:
    """Opening paused is the whole point — frames keep coming, the match does not."""
    env = _env()
    controls = DebugControls()
    renderer = _ScriptedRenderer(controls, [None] * 5)

    ended = run_session(
        env, renderer, controls, selector_for(build_baseline_policy("random")), seed=3
    )

    assert renderer.frames == 6  # five scripted frames, then the one that quits
    assert ended.current_turn == 0
    assert set(renderer.turns_seen) == {0}


def test_one_step_key_advances_exactly_one_step() -> None:
    env = _env()
    controls = DebugControls()
    renderer = _ScriptedRenderer(controls, [_step, None, None])

    ended = run_session(
        env, renderer, controls, selector_for(build_baseline_policy("random")), seed=3
    )

    assert ended.current_turn == 1
    assert controls.step_once is False  # consumed, so it cannot fire again
    assert controls.paused is True


def test_stepping_back_returns_the_match_to_where_it_was() -> None:
    """Step forward twice, back once: the board is the one from after step one."""
    env = _env()
    controls = DebugControls()
    select = selector_for(build_baseline_policy("squad_march_shoot"))
    renderer = _ScriptedRenderer(controls, [_step, _step, _back, None])

    ended = run_session(env, renderer, controls, select, seed=1234)

    assert ended.current_turn == 1
    # Two steps forward, one back, then the frame that quits.
    assert renderer.turns_seen == [0, 1, 2, 1, 1]


def test_stepping_back_with_no_history_is_harmless() -> None:
    """Pressing back on the first frame must not throw or rewind past the reset."""
    env = _env()
    controls = DebugControls()
    renderer = _ScriptedRenderer(controls, [_back, None])

    ended = run_session(
        env, renderer, controls, selector_for(build_baseline_policy("random")), seed=3
    )

    assert ended.current_turn == 0
    assert controls.step_back is False


def test_playing_runs_the_episode_to_the_end_and_holds_there() -> None:
    """Unpaused, the loop steps every frame; at the end it keeps presenting rather
    than falling out of the session, so the terminal state can still be read."""
    env = _env(number_of_battle_rounds=2)
    controls = DebugControls()
    renderer = _ScriptedRenderer(controls, [_play] + [None] * 30)

    ended = run_session(
        env, renderer, controls, selector_for(build_baseline_policy("random")), seed=5
    )

    assert ended.current_turn == ended.max_turns
    assert renderer.frames == 32  # never exited early


def test_a_finished_episode_can_be_stepped_back_into() -> None:
    """The reason the session holds at the end instead of returning."""
    env = _env(number_of_battle_rounds=2)
    controls = DebugControls()
    script: list[Any] = [_play] + [None] * 30 + [_back, None]
    renderer = _ScriptedRenderer(controls, script)

    ended = run_session(
        env, renderer, controls, selector_for(build_baseline_policy("random")), seed=5
    )

    assert ended.current_turn == ended.max_turns - 1


# --- Keys -------------------------------------------------------------------


@pytest.fixture
def presenter_and_env() -> tuple[DebugPresenter, DebugControls, WargameEnv]:
    controls = DebugControls()
    presenter = DebugPresenter(build_backend("pillow"), controls)
    env = _env()
    env.reset(seed=0)
    presenter.setup(env)
    return presenter, controls, env


def _press(presenter: DebugPresenter, env: WargameEnv, key: int) -> None:
    import pygame

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    presenter._process_events(env)


def test_keys_report_intent_rather_than_acting_on_it(
    presenter_and_env: tuple[DebugPresenter, DebugControls, WargameEnv],
) -> None:
    """The presenter must not step anything itself — it only records the ask."""
    import pygame

    presenter, controls, env = presenter_and_env

    _press(presenter, env, pygame.K_SPACE)
    assert controls.paused is False

    _press(presenter, env, pygame.K_PERIOD)
    assert (controls.step_once, controls.paused) == (True, True)

    controls.step_once = False
    _press(presenter, env, pygame.K_COMMA)
    assert (controls.step_back, controls.paused) == (True, True)


def test_speed_keys_stay_inside_their_range(
    presenter_and_env: tuple[DebugPresenter, DebugControls, WargameEnv],
) -> None:
    import pygame

    presenter, _controls, env = presenter_and_env

    for _ in range(80):
        _press(presenter, env, pygame.K_LEFTBRACKET)
    assert presenter._fps == 1

    for _ in range(200):
        _press(presenter, env, pygame.K_RIGHTBRACKET)
    assert presenter._fps == 60


def test_inherited_keys_still_work(
    presenter_and_env: tuple[DebugPresenter, DebugControls, WargameEnv],
) -> None:
    """Deferring unknown keys to `super()` is what keeps [L] and [Tab] alive."""
    import pygame

    presenter, _controls, env = presenter_and_env

    _press(presenter, env, pygame.K_TAB)
    assert presenter._show_keys
    _press(presenter, env, pygame.K_l)
    assert presenter._debug_los


def test_rendering_never_blocks_while_paused(
    presenter_and_env: tuple[DebugPresenter, DebugControls, WargameEnv],
) -> None:
    """`InteractiveRenderer.render` spins here; this one has to return so the
    driver can decide whether the match moves."""
    presenter, controls, env = presenter_and_env
    controls.paused = True

    presenter.render(env)  # would hang if the base class's loop were inherited

    assert presenter._is_paused()
    assert presenter._present_fps() == PAUSED_POLL_FPS


def test_escape_still_quits(
    presenter_and_env: tuple[DebugPresenter, DebugControls, WargameEnv],
) -> None:
    import pygame

    presenter, _controls, env = presenter_and_env

    _press(presenter, env, pygame.K_ESCAPE)
    with pytest.raises(QuitRequested):
        presenter.render(env)


def test_a_checkpointless_session_needs_no_torch() -> None:
    """A scripted name resolves without importing torch, which is why the
    checkpoint branch imports it lazily."""
    import debug as debug_module

    env = _env()
    select, label = debug_module.build_selector_for("squad_march_shoot", env)

    assert label == "squad_march_shoot"
    env.reset(seed=0)
    assert isinstance(select(env.observation, env), WargameEnvAction)
