# Training throughput

Where a training epoch's wall-clock goes, how to measure it, and what is left to
do. Everything here is measured, not modelled — except where it says otherwise,
which is only the GPU section.

## The short version

**Training speed is an environment problem, not a GPU problem.** The network is
12.86M parameters over a 61-token sequence; on a modern GPU the whole 80-gradient-step
PPO update is a couple of seconds. Environment stepping is 2048 sequential Python
`env.step()` calls, and it used to cost 23 s per epoch.

`RewardPhaseManager` calls every per-model calculator once per model. Two
calculators were recomputing a quantity that does not depend on the model being
scored, so each was doing 25× the necessary work — together ~80% of a 25v25 step.

| per rollout step (`env.step` + observation→numpy) | before | after |
|---|---|---|
| `25v25_single_phase.yaml` | 11.34 ms | **2.26 ms** |
| `25v25_shooting_opponent.yaml` | 10.72 ms | **2.54 ms** |
| `25v25_cover_control.yaml` | 10.36 ms | **2.44 ms** |

On the control config that is **23.2 s → 4.6 s** of environment time per
2048-step PPO epoch, with bit-identical behaviour.

## Measuring

```
just measure-throughput examples/env_config/25v25_single_phase.yaml [n_steps] [engaged]
```

Reports the per-section and per-calculator split of `env.step()`, `reset()`, and
the observation→numpy conversion. It measures with `build_info=False`, matching
what the rollout and evaluation pools actually run.

Timing uses `time.perf_counter` wrappers rather than cProfile deliberately: the
reward loop makes ~100 `calculate()` calls per step, and cProfile's per-call
overhead inflates exactly that shape by ~3×, which points the optimisation at the
wrong target.

Every training run also logs, every epoch:

| metric | meaning |
|---|---|
| `perf/rollout_s` | collecting `n_steps` transitions, including GAE |
| `perf/update_s` | the `n_epochs × n_minibatches` gradient steps |
| `perf/eval_s` | the per-epoch evaluation episodes |
| `perf/epoch_s` | rollout + update |
| `perf/env_steps_per_s` | rollout throughput |
| `perf/update_ms_per_minibatch` | the number to watch when changing the network |

These synchronise the device before stopping the clock. Without that, a GPU
timing measures how long it took to *queue* kernels, not to run them — which
reads as "the update is free". That failure is not hypothetical: one machine
trained on CPU for five months with a live CUDA device and nothing in the metrics
said so.

## The invariant: reward changes must be bit-identical

`tests/test_reward_golden.py` pins per-step reward, per-model reward, the reward
breakdown, VP and every model position against recorded trajectories for three
configs, using `assert_array_equal` — **never `assert_allclose`**. These values
feed published reports in `reports/`, and a tolerance waves through exactly the
float-reassociation regression a vectorisation is most likely to introduce. The
gate is verified sensitive: perturbing `min_distances_to_same_group` by one ULP
fails it.

Regenerate only when the reward *should* change, never to make a red test green:

```
uv run python -m tests.test_reward_golden --regenerate
```

`tests/test_reward_memoisation.py` additionally pins each rewrite against a
reference copy of the code it replaced. The golden trajectories only cover the
states they happen to reach; the reference tests cover all-dead groups, singleton
groups and every objective control-state count pair.

## Rules for the reward hot path

- **Anything a per-model calculator computes that does not depend on `model_idx`
  must be memoised.** That single mistake was ~80% of a 25v25 step.
- **Key the memo on `ctx` identity.** A fresh `StepContext` is built every step
  and held by the env, so `ctx is self._cached_ctx` is safe and needs no
  invalidation logic. `objective_hold` and `group_cohesion` are the reference
  patterns.
- **Give every cached quantity its own key field.** Sharing one key between two
  quantities computed at different points in a step freezes the later one at its
  first value — `objective_hold._player_occupancy` documents this trap, which
  would otherwise price a whole episode at the opening crowd.
- Prefer a *selection* rewrite to an arithmetic one when you need bit-identity.
  `min_distances_to_same_group` vectorises safely precisely because filling
  non-candidates with `inf` and taking a row-wise `min` picks the same element
  without performing any arithmetic.

## Current step budget (25v25, after the above)

```
env.step()                 1.72 ms
  reward                   0.63 ms   36%   <- closest_objective_v2 is 0.30 of this
  opponent turn            0.45 ms   26%
  observation build        0.39 ms   23%
  line of sight            0.06 ms    4%   <- 10.6 queries/step under random play
obs -> numpy               0.54 ms
```

## What is left, ranked

### 1. Parallel rollout — the largest remaining win

`ppo/lightning.py` steps envs in a serial Python loop; only the network forward is
batched. The env is pure Python and GIL-bound, so **processes are the only lever**
— raising `num_rollout_envs` improves forward batching and nothing else.

`gymnasium.vector.AsyncVectorEnv` is not viable as-is: `WargameEnv.observation_space`
does not describe what `step()` actually returns (a Pydantic object), and the
rollout loop reads five things off the env that no vector-env API exposes
(`last_per_model_reward`, `last_reward_breakdown`, `game_clock_state.phase`,
`phase_manager.current_phase_index`, `objectives`). A custom spawn worker pool
that does the observation→numpy conversion **inside the worker** is the shape that
fits — it moves that 0.54 ms off the main process and sends ~15 KB of numpy over
the pipe instead of a pickled Pydantic object.

**The trap, and it has already cost seven runs:** rollout envs share
`phase_manager.position` *by object identity*, which breaks silently across a
process boundary. Push `set_reward_phase(...)` to every worker each
`training_step`, have workers echo their resolved phase index back, and assert it
in the parent. Keep evaluation in-process — it is ~400 steps against the
rollout's 2048, so it is not worth the risk, and its fixed seeding and env-0
`StateExporter` then need no changes at all.

### 2. GPU settings — measure on the 4090, do not port findings from the dev box

The dev workstation is a GTX 1080 Ti (sm_61), where TF32, bf16, Flash attention
and `torch.compile` are **all unavailable**. On sm_89 they are all real. Untested
here, in rough order of expected value:

- `torch.set_float32_matmul_precision("high")` — TF32, needs sm_80+.
- bf16 autocast — supported on Ada, unlike Pascal.
- `torch.compile` — Triton requires compute capability ≥ 7.
- `dqn/layers.py` passes an explicit **bool** `attn_mask` to
  `scaled_dot_product_attention`, which disqualifies the Flash backend. Converting
  it to an additive float mask is ~5 lines. It is worth ~1% on the dev box; it may
  be worth real time on a 4090. It cannot simply be dropped — it is the
  dead-model key-padding mask.
- `cudnn.benchmark` is a no-op regardless: the model has zero convolutions.

Take a `perf/*` baseline on the 4090 *before* changing any of these.

### 3. `share_transformer` — an experiment, not a speedup

`share_transformer=True` is measured at 12.86M → 6.51M parameters and 2.1× on the
update. But on a 4090 the update is a small share of the epoch, so it buys single-digit
percent of wall-clock. Its real value is halved checkpoints and whatever it does
to sample efficiency, and it couples value-loss gradients into the policy trunk —
so it needs a proper A/B, two seeds minimum per `just measure-noise-floor`.

Two things must land *with* the flag or the A/B is not one:

- `train.py` calls `PPO_Transformer.from_env(env)` without the config, so
  `PPOConfig` is silently ignored and only the *default* takes effect.
- `state_dict()` does not deduplicate shared parameters, so a checkpoint loads
  across the sharing boundary without error — but loading an *unshared* checkpoint
  into a *shared* model applies the policy trunk and then overwrites it with the
  value trunk, silently, because `_apply_warm_start_weights` uses `strict=False`.

Also worth doing here: `PPOConfig.hidden_size` and `num_layers` are **dead config
for the transformer path** — `net.py` hardcodes `TransformerConfig()` — and they
have been reported into every run's wandb config blob as though they described
the network. If sizes become tunable, the architecture must be persisted beside
the weights, or `RecordEpisodeCallback`, `simulate.py` and
`scripts/measure_checkpoint.py` will all rebuild with defaults and die on a shape
mismatch inside a spawned subprocess that swallows the traceback.

### 4. Line of sight — conditional, watch the trigger

LOS is uncached: every query rebuilds a filtered footprint list and walks the
Bresenham cells through a Python closure. Under random play it is 3.6% of a step
and ~10 queries. Run `just measure-throughput <config> 400 engaged` and it becomes
**24% and ~151 queries**, because a fully engaged 25×25 pays a scan three times per
step (player mask, opponent mask, and exposure when `track_exposure` is set).

So it is cheap now and expensive later, and the trigger is a policy that closes.
Watch `los_queries_per_step` in the harness against a trained checkpoint. The fix
is a per-cell **bitset of covering footprints** cached on `Terrain`, not a plain
boolean grid — the blocking predicate is endpoint-dependent (the see-out rule) and
a bitset is also exact when footprints overlap, which fixed YAML `terrain:` lists
do not forbid. Invalidation is free: `Battle.set_terrain` replaces the object on
every reset.

### 5. Dev-box CUDA and dependency pinning

`pyproject.toml` pins `torch` with **no version constraint** and `uv.lock` is
gitignored, so every `just dev-sync` re-resolves the whole graph and there is no
record of what any published result was built with. The 1080 Ti workstation lost
Pascal support in that drift and has been training on CPU since; `torch
2.8.0+cu126` is the same version with Pascal kernels and would fix it. Pinning
`torch==2.8.0`, adding the cu126 index with `explicit = true`, and **committing
`uv.lock`** prevents a repeat on any machine. `Dockerfile` already runs `uv sync
--frozen` against a lockfile the repo does not ship.

Worth adding at the same time: a startup preflight that runs a real kernel and
either raises with `get_arch_list()` vs `get_device_capability()` spelled out or
falls back to CPU loudly. The current `_cuda_appears_usable()` probe has it
inverted — it quietly halves `num_rollout_envs` while the model still moves to
CUDA and crashes on the first forward.
