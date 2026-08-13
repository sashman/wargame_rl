"""Recreating the episode a recording came from, exactly.

Watching a match go wrong and wanting to step through it by hand was not
possible before schema 2.4: a recording carried the full state of every step
and none of the *inputs* that produced it, so replay was a video and nothing
more. These tests pin the round trip — record, rebuild, and get the identical
match back.

The claim is bit-identical, not "close": positions, VP and per-step reward all
come back the same, because anything less makes the debugger lie about the
thing it was opened to investigate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from wargame_rl.wargame.envs.baseline.evaluate import selector_for  # noqa: E402
from wargame_rl.wargame.envs.baseline.registry import (  # noqa: E402
    build_baseline_policy,
)
from wargame_rl.wargame.envs.debug.reproduce import (  # noqa: E402
    build_env,
    provenance_of,
    read_log,
    read_provenance,
    rebuild,
    rebuild_from_recording,
    recorded_actions,
)
from wargame_rl.wargame.envs.debug.reproduce import (  # noqa: E402
    reset_options as reproduce_reset_options,
)
from wargame_rl.wargame.envs.debug.session import run_session  # noqa: E402
from wargame_rl.wargame.envs.debug.undo import capture_state  # noqa: E402
from wargame_rl.wargame.envs.domain.battle_view import BattleView  # noqa: E402
from wargame_rl.wargame.envs.renders.human import QuitRequested  # noqa: E402
from wargame_rl.wargame.envs.renders.renderer import Renderer  # noqa: E402
from wargame_rl.wargame.envs.renders.v2.presenters.debug import (  # noqa: E402
    PLAYER,
    DebugControls,
)
from wargame_rl.wargame.envs.state.codecs import JsonMatchCodec  # noqa: E402
from wargame_rl.wargame.envs.state.events import ResetEvent  # noqa: E402
from wargame_rl.wargame.envs.state.exporter import EventLogExporter  # noqa: E402
from wargame_rl.wargame.envs.types import WargameEnvAction  # noqa: E402
from wargame_rl.wargame.envs.types import WargameEnvConfig  # noqa: E402
from wargame_rl.wargame.envs.types.config import (  # noqa: E402
    ModelConfig,
    OpponentPolicyConfig,
    RandomTerrainConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase  # noqa: E402
from wargame_rl.wargame.envs.wargame import WargameEnv  # noqa: E402


def _config(**overrides: Any) -> WargameEnvConfig:
    """A scenario whose layout is *drawn*, not fixed.

    Regenerated terrain is the case that matters: a fixed layout would be
    reproduced by the config alone and the generator state would go untested.
    """
    rifle = [WeaponProfile(range=30, attacks=2)]
    base: dict[str, Any] = dict(
        board_width=30,
        board_height=30,
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        number_of_battle_rounds=4,
        models=[ModelConfig(weapons=rifle) for _ in range(4)],
        opponent_models=[ModelConfig() for _ in range(4)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
        random_terrain=RandomTerrainConfig(count=4, min_size=3, max_size=5),
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        render_mode=None,
    )
    base.update(overrides)
    return WargameEnvConfig(**base)


def _trace(env: WargameEnv, steps: int = 8) -> list[Any]:
    """Everything a reproduction has to get right, per step."""
    select = selector_for(build_baseline_policy("squad_march_shoot"))
    out = []
    for _ in range(steps):
        env.step(select(env.observation, env))
        out.append(
            (
                env.player_vp,
                env.opponent_vp,
                env.last_reward,
                np.array([m.location for m in env.player_models]).tobytes(),
                np.array([m.location for m in env.opponent_models]).tobytes(),
                [m.stats["current_wounds"] for m in env.player_models],
            )
        )
    return out


def _record(tmp_path: Path, seed: int | None, driver: str = "squad_march_shoot") -> Any:
    exporter = EventLogExporter()
    env = WargameEnv(config=_config(), state_exporters=[exporter])
    env.driver_label = driver
    env.reset(seed=seed)
    original = _trace(env)
    path = tmp_path / "match.jsonl"
    path.write_bytes(JsonMatchCodec().encode(exporter.log))
    return path, original


def test_a_recording_replays_as_the_identical_match(tmp_path: Path) -> None:
    """The whole point. Terrain is regenerated per episode, so this is testing
    the generator state, not just the config."""
    path, original = _record(tmp_path, seed=4242)

    assert _trace(rebuild_from_recording(path)) == original


def test_an_unseeded_episode_is_reproducible_too(tmp_path: Path) -> None:
    """Why the provenance stores a generator state and not a seed.

    A training rollout resets without one, so its layout is a point in a
    continuing stream that no integer names. Storing the state names it anyway.
    """
    path, original = _record(tmp_path, seed=None)
    assert read_provenance(path).seed is None

    assert _trace(rebuild_from_recording(path)) == original


def test_the_session_reset_path_reproduces_it(tmp_path: Path) -> None:
    """`run_session` owns the reset, so `debug.py` installs the state and lets
    it reset once. Resetting in both places would draw from the stream twice and
    land on a different layout — the failure this splits `build_env` to avoid."""
    path, original = _record(tmp_path, seed=7)
    provenance = read_provenance(path)

    env = build_env(provenance)
    env.reset(seed=None, options=reproduce_reset_options(provenance))

    assert _trace(env) == original


def test_the_recording_carries_the_config_not_a_path(tmp_path: Path) -> None:
    """A path can be edited or deleted between recording a match and wanting it
    back; a scenario that has drifted would reproduce something that merely
    looks like the recording."""
    path, _ = _record(tmp_path, seed=1)

    provenance = read_provenance(path)

    assert provenance.config["board_width"] == 30
    assert WargameEnvConfig(**provenance.config).number_of_wargame_models == 4


def test_the_driver_is_recorded_so_the_debugger_can_default_to_it(
    tmp_path: Path,
) -> None:
    """The env cannot know what chose the player's actions — a checkpoint, a
    baseline and a human all look the same from inside — so whoever drives it
    says so."""
    path, _ = _record(tmp_path, seed=1, driver="checkpoints/run/last.ckpt")

    assert read_provenance(path).driver == "checkpoints/run/last.ckpt"


def test_provenance_is_header_metadata_not_game_state(tmp_path: Path) -> None:
    """It says how the recording was made, not what was true at a point in the
    battle — and a snapshot is a value object compared for equality, so a field
    on the recorded copy that a live snapshot lacks would break the "replayed
    equals live" invariant `test_v9_milestone_validation` pins."""
    exporter = EventLogExporter()
    env = WargameEnv(config=_config(), state_exporters=[exporter])
    env.reset(seed=3)
    _trace(env, steps=3)

    reset_event = exporter.log.events[0]
    assert isinstance(reset_event, ResetEvent)
    assert exporter.log.provenance is not None
    assert not hasattr(reset_event.snapshot, "provenance")
    # The recorded reset snapshot still matches what a seek reconstructs.
    assert reset_event.snapshot == exporter.log.snapshot_at(0)


def test_a_recording_without_provenance_says_so(tmp_path: Path) -> None:
    """ "Cannot be reproduced" and "reproduction failed" are worth telling
    apart — the first is not fixable by trying again."""
    path, _ = _record(tmp_path, seed=1)
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"provenance"', '"_dropped"')
    stripped = tmp_path / "old.jsonl"
    stripped.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="no provenance"):
        read_provenance(stripped)


def test_an_unusable_gpu_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """`torch.cuda.is_available()` answers the wrong question.

    A card whose compute capability predates the torch build reports available,
    accepts `.to("cuda")`, and fails on the *first forward* with `no kernel
    image is available` — so every checkpoint path died at inference rather
    than at startup, reading as a bug in whatever was being run. Measured on a
    GTX 1080 Ti (sm_61) against a build listing sm_70+.
    """
    import torch

    from wargame_rl.wargame.model.common.device import auto_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_90"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (6, 1))
    auto_device.cache_clear()
    try:
        assert auto_device().type == "cpu"

        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        auto_device.cache_clear()
        assert auto_device().type == "cuda"
    finally:
        auto_device.cache_clear()


class _ScriptedRenderer(Renderer):
    """Replays control mutations, one per rendered frame, then quits.

    A local copy of the one in `test_debug_session`: importing across test
    modules to share a five-line stand-in couples two files that otherwise have
    nothing to say to each other.
    """

    def __init__(self, controls: DebugControls, script: list[Any]) -> None:
        self._controls = controls
        self._script = list(script)

    def setup(self, view: BattleView) -> None:
        return None

    def render(self, view: BattleView) -> None:
        if not self._script:
            raise QuitRequested()
        action = self._script.pop(0)
        if action is not None:
            action(self._controls)

    def close(self) -> None:
        return None


# --- Following the recording ------------------------------------------------


def _follow_one_step(env: WargameEnv, actions: list[list[int] | None]) -> None:
    """Step the env with the recording's action for whatever turn it is on."""
    recorded = actions[env.current_turn + 1]
    assert recorded is not None
    env.step(WargameEnvAction(actions=list(recorded)))


def test_following_a_recording_reproduces_the_match_without_a_driver(
    tmp_path: Path,
) -> None:
    """Why following exists at all.

    A *training* recording's actions came from a network mid-training that was
    never saved, so no driver can reproduce its decisions. Replaying the
    recording's own actions does — and a scripted driver deliberately unlike the
    recorded one proves the actions, not the driver, are what is steering.
    """
    path, original = _record(tmp_path, seed=4242)
    log = read_log(path)
    env = rebuild(provenance_of(log, path))
    actions = recorded_actions(log)

    replayed = []
    for _ in range(len(original)):
        _follow_one_step(env, actions)
        replayed.append(
            (
                env.player_vp,
                env.opponent_vp,
                env.last_reward,
                np.array([m.location for m in env.player_models]).tobytes(),
                np.array([m.location for m in env.opponent_models]).tobytes(),
                [m.stats["current_wounds"] for m in env.player_models],
            )
        )

    assert replayed == original


def test_the_action_index_is_the_turn_so_a_rewind_needs_no_bookkeeping(
    tmp_path: Path,
) -> None:
    """`env.current_turn` is restored along with the env, so stepping back and
    forward again follows the recording without the loop tracking anything."""
    path, original = _record(tmp_path, seed=4242)
    log = read_log(path)
    actions = recorded_actions(log)
    env = rebuild(provenance_of(log, path))

    for _ in range(3):
        _follow_one_step(env, actions)
    saved = capture_state(env)
    for _ in range(2):
        _follow_one_step(env, actions)
    after = np.array([m.location for m in env.player_models])

    # Rewind, then follow again from the restored turn.
    restored = saved
    assert restored.current_turn == 3
    for _ in range(2):
        _follow_one_step(restored, actions)

    np.testing.assert_array_equal(
        np.array([m.location for m in restored.player_models]), after
    )


def test_the_session_follows_then_hands_over_when_you_diverge(tmp_path: Path) -> None:
    """An action recorded for a state you have since altered is not that match
    any more, it is a coincidence — so following stops the moment you author
    something, and resumes if you rewind back past it."""
    path, original = _record(tmp_path, seed=4242)
    log = read_log(path)
    actions = recorded_actions(log)
    controls = DebugControls()
    select = selector_for(build_baseline_policy("random"))

    def _step(c: DebugControls) -> None:
        c.step_once = True
        c.paused = True

    def _order(c: DebugControls) -> None:
        c.selected = (PLAYER, 0)
        c.order_at = (PLAYER, 0, 1.0, 1.0)

    seen: list[Any] = []

    class _Recorder(_ScriptedRenderer):
        def render(self, view: BattleView) -> None:
            seen.append(np.array([m.location for m in view.player_models]).tobytes())
            super().render(view)

    env = build_env(provenance_of(log, path))
    script = [_step, _step, _order, _step, None]
    run_session(
        env,
        _Recorder(controls, script),
        controls,
        select,
        seed=None,
        reset_options=reproduce_reset_options(provenance_of(log, path)),
        recorded=actions,
    )

    # The two followed steps match the recording; the authored one does not.
    assert seen[1] == original[0][3]
    assert seen[2] == original[1][3]
    assert seen[-1] != original[2][3]
