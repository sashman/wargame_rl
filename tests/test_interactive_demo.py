"""Light tests for interactive_demo.EnvTest (demo harness, not the test runner)."""

from wargame_rl.wargame.envs.interactive_demo import EnvTest


def test_env_test_instantiates_with_render_mode_none() -> None:
    """EnvTest can be created with render_mode=None for headless use."""
    harness = EnvTest(board_width=5, board_height=5, render_mode=None)
    assert harness.env is not None
    assert harness.env.action_space is not None
    assert harness.env.observation_space is not None


def test_env_test_reset() -> None:
    """EnvTest.reset() runs without error."""
    harness = EnvTest(board_width=5, board_height=5, render_mode=None)
    harness.reset()
    assert harness.env.current_turn == 0


def test_env_test_run_actions_no_display() -> None:
    """EnvTest.run_actions() runs without opening a display when render is no-op."""
    harness = EnvTest(board_width=5, board_height=5, render_mode=None)
    harness.env.renderer = None  # avoid HumanRender (pygame) in CI
    harness.run_actions(num_actions=3)
