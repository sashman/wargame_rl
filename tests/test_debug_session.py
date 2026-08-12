"""Hand-stepping a live match: rewind fidelity, the driving loop, and the keys.

The load-bearing test here is `test_a_restored_env_replays_the_step_identically`.
Everything the debug mode is for — "what if this model had gone left" — rests on
a rewind that changes nothing but the decision, so a restore that quietly alters
the reward or the dice would make every comparison drawn through this tool wrong
while looking perfectly reasonable on screen.
"""

from __future__ import annotations

import math
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
from wargame_rl.wargame.envs.debug.session import record_moves  # noqa: E402
from wargame_rl.wargame.envs.domain.battle_view import BattleView  # noqa: E402
from wargame_rl.wargame.envs.renders.human import QuitRequested  # noqa: E402
from wargame_rl.wargame.envs.renders.renderer import Renderer  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.factory import build_backend  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.presenters.debug import (  # noqa: E402
    OPPONENT,
    PANEL_W,
    PAUSED_POLL_FPS,
    PLAYER,
    DebugControls,
    DebugPresenter,
)
from wargame_rl.wargame.envs.renders.v2.presenters.interactive import (  # noqa: E402
    InteractiveRenderer,
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


# --- The inspector panel ----------------------------------------------------


def _px(presenter: DebugPresenter, model: object) -> tuple[int, int]:
    """Window pixel at a model's board location — the inverse of the hit test."""
    location = model.location  # type: ignore[attr-defined]
    return (
        int(presenter._offset_x + location[0] * presenter._scale),
        int(presenter._offset_y + location[1] * presenter._scale),
    )


def test_the_panel_is_reserved_not_drawn_over_the_board() -> None:
    """A debugger that hides what is being debugged is worse than a narrow board.

    Guards the layout hook: without `_reserved_width`, the window is board-width
    and the panel is painted on top of the right-hand third of the table.
    """
    controls = DebugControls()
    presenter = DebugPresenter(build_backend("pillow"), controls)
    plain = InteractiveRenderer(build_backend("pillow"))
    env = _env()
    env.reset(seed=0)
    presenter.setup(env)
    plain.setup(env)

    assert presenter._window_w == plain._window_w + PANEL_W
    assert presenter._canvas_w == plain._canvas_w  # the board is not shrunk
    assert presenter._offset_x + presenter._canvas_w <= presenter._window_w - PANEL_W


def test_clicking_selects_a_model_on_either_side() -> None:
    """Half the reason to open a debugger is to ask what the *opponent* did."""
    controls = DebugControls()
    presenter = DebugPresenter(build_backend("pillow"), controls)
    env = _env()
    env.reset(seed=0)
    presenter.setup(env)

    presenter._handle_click(env, *_px(presenter, env.player_models[1]))
    assert controls.selected == (PLAYER, 1)

    presenter._handle_click(env, *_px(presenter, env.opponent_models[2]))
    assert controls.selected == (OPPONENT, 2)


def test_clicking_the_panel_does_not_select() -> None:
    """The hit test is board-only, so a click on the panel clears the selection."""
    controls = DebugControls(selected=(PLAYER, 0))
    presenter = DebugPresenter(build_backend("pillow"), controls)
    env = _env()
    env.reset(seed=0)
    presenter.setup(env)

    presenter._handle_click(env, presenter._window_w - 10, presenter._top_h + 40)

    assert controls.selected is None


def test_escape_deselects_before_it_quits(
    presenter_and_env: tuple[DebugPresenter, DebugControls, WargameEnv],
) -> None:
    """Otherwise the only way to undo a click is to leave the session."""
    import pygame

    presenter, controls, env = presenter_and_env
    controls.selected = (PLAYER, 0)

    _press(presenter, env, pygame.K_ESCAPE)
    assert controls.selected is None
    assert not presenter._should_quit

    _press(presenter, env, pygame.K_ESCAPE)
    assert presenter._should_quit


def test_the_panel_shows_the_selected_model(
    presenter_and_env: tuple[DebugPresenter, DebugControls, WargameEnv],
) -> None:
    """Selecting must change what the reserved column draws, and two different
    models must not draw the same thing."""
    presenter, controls, env = presenter_and_env
    env.step(
        selector_for(build_baseline_policy("squad_march_shoot"))(env.observation, env)
    )

    def panel() -> np.ndarray:
        frame = presenter._backend.to_rgb_array(presenter._compose_with_tooltip(env))
        return np.asarray(frame[presenter._top_h :, presenter._window_w - PANEL_W :])

    empty = panel()
    controls.selected = (PLAYER, 0)
    first = panel()
    controls.selected = (PLAYER, 1)
    second = panel()

    assert not np.array_equal(empty, first)
    assert not np.array_equal(first, second)


def test_a_move_record_measures_the_gap_from_the_action_it_asked_for() -> None:
    """Intended-versus-actual is the field nothing else in the renderer shows.

    A move the board and its neighbours leave alone lands exactly on the action's
    own vector; `resolve_move` displacement is the difference.
    """
    env = _env()
    env.reset(seed=1234)
    action = selector_for(build_baseline_policy("squad_march"))(env.observation, env)
    before = [(float(m.location[0]), float(m.location[1])) for m in env.player_models]

    env.step(action)
    moves = record_moves(env, action, before)

    assert set(moves) == set(range(len(env.player_models)))
    for index, move in moves.items():
        expected = math.dist(move.intended, move.actual)
        assert move.gap == pytest.approx(expected)
        # Nothing may end further from its intent than one full move.
        assert move.gap <= env.config.max_move_speed
    assert any(m.text != "Stay" for m in moves.values())


def test_stepping_back_restores_the_moves_that_belong_to_that_step() -> None:
    """Move records live on the controls, not in the env, so the rewind has to
    carry them too — otherwise the panel describes a step that was just undone."""
    # Movement-only, so consecutive steps both write records. With the default
    # phases the second step is *shooting*, which writes none — and the test then
    # compares a step against itself and passes for the wrong reason.
    env = _env(
        skip_phases=[
            BattlePhase.command,
            BattlePhase.shooting,
            BattlePhase.charge,
            BattlePhase.fight,
        ]
    )
    controls = DebugControls()
    select = selector_for(build_baseline_policy("squad_march"))
    seen: list[dict[int, object]] = []

    class _Recorder(_ScriptedRenderer):
        def render(self, view: BattleView) -> None:
            seen.append(dict(controls.moves))
            super().render(view)

    renderer = _Recorder(controls, [_step, _step, _back, None])
    run_session(env, renderer, controls, select, seed=1234)

    # Frames: initial, after step 1, after step 2, after the rewind.
    assert seen[1] != seen[2]  # each movement step wrote its own records
    assert seen[3] == seen[1]  # the rewind put step 1's back


def test_a_dead_model_earns_nothing_which_is_why_the_panel_says_so() -> None:
    """Pins the claim the panel prints beside a 0.000.

    `phase_manager` iterates *alive* models, so a model that killed and died in
    the same step earns nothing for the kill. Shown bare that reads as broken
    reward plumbing, so the note is load-bearing — and so is this test, which
    would fail if the reward loop ever started paying casualties.
    """
    env = _env()
    env.reset(seed=1234)
    # The casualty is arranged rather than shot for, so the test asserts the
    # reward rule instead of depending on the dice going a particular way.
    env.player_models[0].stats["current_wounds"] = 0
    assert not env.player_models[0].is_alive

    select = selector_for(build_baseline_policy("squad_march"))
    # Step on until a survivor is actually paid something: on a step where the
    # whole army earns zero, "the dead model earned zero" proves nothing.
    for _ in range(6):
        env.step(select(env.observation, env))
        if any(env.last_per_model_reward[i] != 0.0 for i in (1, 2)):
            break

    assert any(env.last_per_model_reward[i] != 0.0 for i in (1, 2)), "check is vacuous"
    assert env.last_per_model_reward[0] == 0.0


def test_a_checkpointless_session_needs_no_torch() -> None:
    """A scripted name resolves without importing torch, which is why the
    checkpoint branch imports it lazily."""
    import debug as debug_module

    env = _env()
    select, label = debug_module.build_selector_for("squad_march_shoot", env)

    assert label == "squad_march_shoot"
    env.reset(seed=0)
    assert isinstance(select(env.observation, env), WargameEnvAction)
