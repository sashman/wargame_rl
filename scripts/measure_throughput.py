"""Where a training epoch's wall-clock actually goes.

The repo had no timing tooling at all — `just profile` runs pyinstrument over a
whole training run, and every `scripts/measure_*` script measures *behaviour*.
So the one performance question that matters ("what is slow?") was only ever
answered by guesswork, and the guess was wrong: the neural network is not the
bottleneck on a modern GPU. Measured on the 25v25 configs, **~85% of `env.step()`
is reward calculation**, and two of the six calculators are ~80% of the step,
because each recomputes a model-independent quantity once per model.

Timing is done with `time.perf_counter` wrappers rather than cProfile on
purpose: the reward loop makes ~100 `calculate()` calls per step, and cProfile's
per-call overhead inflates exactly that shape by ~3x, which would point the
optimisation at the wrong place.

Usage: just measure-throughput <env_config> [n_steps] [engaged]

`engaged` forces every model within weapon range of every opponent before
stepping. Line of sight is only ~0.5% of a step under random play, but it is
O(n^2 * line_length * n_terrain) with no cache, so it grows toward ~32 ms/step as
a policy learns to close. That mode quotes the ceiling rather than today's floor.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

DEFAULT_STEPS = 400
SEED = 4242
COMBAT_SEED = 99


class _Timings:
    """Accumulated seconds per named section, plus a call count."""

    def __init__(self) -> None:
        self.total: defaultdict[str, float] = defaultdict(float)
        self.calls: defaultdict[str, int] = defaultdict(int)

    def wrap(self, owner: Any, method: str, label: str) -> None:
        """Replace `owner.method` with a timed passthrough recorded under `label`."""
        original = getattr(owner, method)

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.total[label] += time.perf_counter() - start
                self.calls[label] += 1

        setattr(owner, method, timed)


def _instrument_line_of_sight(env: WargameEnv, timings: _Timings) -> None:
    """Time whole LOS queries, not just the predicate's construction.

    `_make_is_blocking` returns a closure that is then called once per cell along
    the Bresenham line, so timing the factory alone would miss almost all of the
    cost. Wrapping the returned closure as well makes "line of sight" the real
    per-query total, and its call count is `los_queries_per_step` — the number
    that says whether the quadratic LOS scan is worth optimising yet.
    """
    original = env._make_is_blocking

    def timed_factory(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        predicate = original(*args, **kwargs)
        timings.total["line of sight"] += time.perf_counter() - start
        timings.calls["line of sight"] += 1

        def timed_predicate(x: int, y: int) -> bool:
            inner = time.perf_counter()
            try:
                return bool(predicate(x, y))
            finally:
                timings.total["line of sight"] += time.perf_counter() - inner

        return timed_predicate

    env._make_is_blocking = timed_factory  # type: ignore[method-assign]


def _instrument(env: WargameEnv, timings: _Timings) -> None:
    """Wrap the sections of `step()` worth attributing time to.

    Each calculator is wrapped individually — the per-calculator split is the
    whole point, since the aggregate "reward is slow" is not actionable.
    """
    timings.wrap(env.phase_manager, "calculate_reward", "reward")
    timings.wrap(env, "_get_obs", "observation build")
    timings.wrap(env, "_get_info", "info build (discarded)")
    timings.wrap(env, "_apply_opponent_action", "opponent turn")
    _instrument_line_of_sight(env, timings)

    for phase in env.phase_manager.phases:
        calculators = list(phase.per_model_calculators) + list(phase.global_calculators)
        for name, calculator in calculators:
            timings.wrap(calculator, "calculate", f"  reward/{name}")


def _sample_actions(
    mask: np.ndarray | None, rng: np.random.Generator
) -> WargameEnvAction:
    """One valid action per model, from an explicitly seeded generator."""
    if mask is None:
        raise ValueError("the observation carries no action mask")
    return WargameEnvAction(actions=[int(rng.choice(np.flatnonzero(r))) for r in mask])


def _force_engagement(env: WargameEnv) -> None:
    """Pack both armies into one blob so every pair is in range and LOS runs.

    This is the regime a competent policy converges toward, and the regime the
    uncached line-of-sight scan is quadratic in.
    """
    centre_x = env.config.board_width // 2
    centre_y = env.config.board_height // 2
    for index, model in enumerate(env.wargame_models):
        model.location = np.array(
            [centre_x + index % 3, centre_y + index // 3 % 3], dtype=np.int64
        )
    for index, model in enumerate(env.opponent_models):
        model.location = np.array(
            [centre_x + 2 + index % 3, centre_y + index // 3 % 3], dtype=np.int64
        )


def _run(
    config_path: Path, n_steps: int, engaged: bool
) -> tuple[float, float, _Timings, WargameEnv, Any]:
    config = parse_yaml_raw_as(WargameEnvConfig, config_path.read_text())
    env = WargameEnv(config=config)
    timings = _Timings()

    observation, _ = env.reset(seed=SEED, options={"combat_seed": COMBAT_SEED})
    rng = np.random.default_rng(SEED)

    # Instrument after reset so construction and the first layout are excluded.
    _instrument(env, timings)

    reset_seconds = 0.0
    resets = 0
    step_seconds = 0.0

    for _ in range(n_steps):
        if engaged:
            _force_engagement(env)
        action = _sample_actions(observation.action_mask, rng)
        start = time.perf_counter()
        observation, _, terminated, truncated, _ = env.step(action)
        step_seconds += time.perf_counter() - start
        if terminated or truncated:
            start = time.perf_counter()
            observation, _ = env.reset(options={"combat_seed": COMBAT_SEED})
            reset_seconds += time.perf_counter() - start
            resets += 1

    mean_reset = reset_seconds / resets if resets else float("nan")
    return step_seconds / n_steps, mean_reset, timings, env, observation


def _time_observation_tensor(env: WargameEnv, observation: Any) -> float:
    """Cost of turning one observation into the numpy blocks the network eats.

    Reported separately from `env.step()` because it is paid once per rollout
    step too, lives a layer above the env, and is dominated by an expected-damage
    matrix that is constant for an entire run.
    """
    from wargame_rl.wargame.model.common.observation import _observation_to_numpy

    repeats = 50
    start = time.perf_counter()
    for _ in range(repeats):
        _observation_to_numpy(observation)
    return (time.perf_counter() - start) / repeats


def _print_report(
    config_path: Path,
    env: WargameEnv,
    n_steps: int,
    engaged: bool,
    mean_step: float,
    mean_reset: float,
    timings: _Timings,
    tensor_seconds: float,
) -> None:
    print(f"config          {config_path}")
    print(
        f"army            {env.config.number_of_wargame_models}v"
        f"{env.config.number_of_opponent_models}   "
        f"board {env.config.board_width}x{env.config.board_height}   "
        f"max_turns {env.max_turns}"
    )
    print(f"mode            {'engaged (LOS ceiling)' if engaged else 'random play'}")
    print(f"steps           {n_steps}")
    print()
    print(f"env.step()      {mean_step * 1000:8.3f} ms   ({1.0 / mean_step:6.0f} /s)")
    print(f"env.reset()     {mean_reset * 1000:8.3f} ms")
    print()

    def share(seconds: float) -> str:
        return f"{seconds / (mean_step * n_steps) * 100:5.1f}%"

    for label in sorted(timings.total, key=lambda k: (k.startswith("  "), k)):
        seconds = timings.total[label]
        per_step = seconds / n_steps * 1000
        calls = timings.calls[label] / n_steps
        print(f"  {label:<34} {per_step:8.3f} ms  {share(seconds)}  {calls:6.1f} calls")

    print()
    print(
        f"obs -> numpy    {tensor_seconds * 1000:8.3f} ms   (per rollout step, on top)"
    )
    per_rollout_step = mean_step + tensor_seconds
    print(
        f"rollout step    {per_rollout_step * 1000:8.3f} ms   "
        f"-> {2048 * per_rollout_step:5.1f} s per 2048-step PPO epoch"
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    config_path = Path(argv[0])
    n_steps = int(argv[1]) if len(argv) > 1 and argv[1] else DEFAULT_STEPS
    engaged = len(argv) > 2 and argv[2].lower() in {"engaged", "true", "1"}

    mean_step, mean_reset, timings, env, observation = _run(
        config_path, n_steps, engaged
    )
    tensor_seconds = _time_observation_tensor(env, observation)
    _print_report(
        config_path,
        env,
        n_steps,
        engaged,
        mean_step,
        mean_reset,
        timings,
        tensor_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
