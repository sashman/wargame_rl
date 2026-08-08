# Training throughput

Where a training epoch's wall-clock goes, how to measure it, and what is left to
do. Everything here is measured, not modelled.

## The short version

**Training speed *was* an environment problem. On the 4090 it is now split
roughly evenly between the environment and the update.** Environment stepping is
2048 sequential Python `env.step()` calls and used to cost 23 s per epoch; that
is now 3.8 s. The 80-gradient-step PPO update is 2.8 s beside it — so with the
environment fixed, neither half caps the epoch on its own, and the GPU levers
matter more than an earlier draft of this page predicted.

**A ranked plan for what to do next lives in
[reports/2026-08-08-throughput-review.md](../reports/2026-08-08-throughput-review.md).**
It also records five premises this page previously got wrong, including two that
invert conclusions drawn from historical run timings — every wandb record predates
the environment work above and was produced under a ~5x slower environment.

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

## Epoch budget on the training GPU (RTX 4090, sm_89)

**Do not combine this table with the step budget above it — they are from
different machines.** The step budget is dev-box scale; deriving "the forward is
X% of rollout" from the two together yields a negative number.

**`perf/epoch_s` excludes evaluation**, and the eval figure below used the
`PPOConfig` default of 10 episodes. Every seeded recipe passes
`--n-eval-episodes 30` (`Justfile:91,105,128`), and `max_turns` = 20 rounds x 2
active phases = 40 — so a real experiment epoch runs **1200 eval env steps against
2048 rollout steps** and costs **~8.4 s, of which eval is ~22%**. Percentages
quoted against 6.55 s are optimistic by about a quarter.

Measured on `25v25_shooting_opponent.yaml`, median of six epochs, 2048 rollout
steps and 80 minibatches. Sequence length is **T = 83** on this config (1 game + 3
objectives + 25 player + 25 opponent + **29 terrain**) — the "61-token" figure
quoted elsewhere belongs to `25v25_single_phase.yaml`, which has 7 fixed terrain
pieces. Terrain is 35% of the token budget here, and trunk cost is linear in T:

| | fp32 (`--no-tf32`) | **TF32 (default)** | TF32 + `bf16-mixed` |
|---|---|---|---|
| `perf/rollout_s` | 3.86 | 3.75 | 4.02 |
| `perf/update_s` | 3.66 | **2.76** | 1.55 |
| `perf/update_ms_per_minibatch` | 45.7 | **34.5** | 19.3 |
| `perf/epoch_s` | 7.57 | **6.55** | 5.61 |

**Correction to the section below, which was written on the dev box: the update
is not a small share of the epoch here — it was 49% of it.** That was a
projection from a machine where the update ran on a CPU, and it under-rated
every GPU lever. Rollout and update are now within ~35% of each other, so
neither one alone caps the epoch.

Note bf16 makes the *rollout* slightly slower (3.75 → 4.02 s). Rollout forwards
are one observation at a time; at that size the cast costs more than the tensor
cores return. The gain is entirely in the batched update.

### TF32 is on by default

`configure_matmul_precision` (`model/common/performance.py`) is called once from
`train.py`, before any model is built, and enables TF32 wherever the device is
sm_80 or newer. `--no-tf32` restores full fp32.

This drops matmul mantissa precision from 24 bits to 11, so a run before this
change and a run after are not bit-identical. That is well below anything this
project can resolve — win rate cannot separate differences under ~7pp, and
`vp_margin` under ~10 — and the environment and reward are untouched, being numpy
on the CPU. But it does mean "training is deterministic given seed + config +
code" holds only within one setting of this flag.

### bf16 is opt-in, and its effect on *learning* is unmeasured

`--precision bf16-mixed` is 2.4x on the update and 1.35x on the epoch. It is not
the default because only its **speed** has been measured. Before trusting a run
under it, A/B it over two seeds per `just measure-noise-floor` — this project has
a standing rule that no single-seed difference under ~7pp win rate is readable,
and a precision change is exactly the kind of thing that would move results
without moving throughput metrics.

One guard already landed with the flag, and the A/B is not valid without it:
`PPOModel.forward` casts both heads back to float32. PPO's importance ratio is
`exp(new_log_prob − old_log_prob)`, resolving per-model changes of ~0.007 nats,
and these log-probs sit near −4.8 where bf16 spaces values 0.0156 apart. That
change does not survive the round trip at all — it collapses to exactly zero for
70% of base values and inflates to a whole step for the rest, never landing
within 10% of its true size. Left in bf16, the surrogate objective would read a
ratio of 1 and train on nothing, at full speed, with no metric saying so.
`tests/test_precision.py` pins both the cast and the quantisation that motivates
it.

### `torch.compile` is measured but not wired

1.42x on the update alone, **3.26x** stacked with bf16 (14.0 ms per minibatch).

**Corrected 2026-08-08 — an earlier version of this section overstated the
blocker.** It claimed a compiled checkpoint would silently load as nothing in
`simulate`, `measure-checkpoint` and warm starts alike. In fact
`convert_state_dict` (`net.py:677-678`) already strips `_orig_mod.` and raises
`ValueError` when no key matches, so `simulate`, `record-sim`,
`measure-checkpoint` and `measure-phase-gates` are safe and fail loudly, and
`trainer.fit(ckpt_path=...)` is strict by default. **Only
`_apply_warm_start_weights` (`train.py:139`, `strict=False`) is silently holed.**

And the problem is avoidable rather than fixable: **`torch.compile(model.forward)`
— the bound method rather than the module — leaves `state_dict` byte-identical**
(verified: compiling a module yields `_orig_mod.0.weight`; compiling its
`forward` yields `0.weight`). Wire it that way and no checkpoint consumer is
touched at all.

Worth doing only on top of bf16: the gain is 2.8% of an epoch at TF32 but 7.5% at
bf16, because eager dispatch only starts to bind once the minibatch is down to
~19 ms. Use `mode="max-autotune-no-cudagraphs"`, explicitly **not**
`"reduce-overhead"` — CUDA graphs target launch overhead, and this loop is
bandwidth-bound, not launch-bound.

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

### 2. GPU settings — done, see the epoch budget above

Measured on the 4090. TF32 shipped on by default, bf16 shipped opt-in pending a
learning A/B, `torch.compile` measured at 3.26x and blocked on the `state_dict`
prefix. `cudnn.benchmark` remains a no-op: the model has zero convolutions.

**The attention-mask item was wrong on both counts and is closed.** The claim was
that the explicit **bool** `attn_mask` in `dqn/layers.py` disqualifies the Flash
backend and that an additive float mask would recover it. Flash supports **no
`attn_mask` at all** — the dispatcher says so directly ("Flash Attention does not
support non-null attn_mask"), so the mask's *dtype* was never what excluded it.
Both forms dispatch to exactly the same backends (`EFFICIENT_ATTENTION` and
`MATH` in fp32, plus `CUDNN_ATTENTION` in bf16).

The float mask *is* faster in isolation — 77.1 → 68.8 µs per call in fp32 and
34.5 → 26.0 µs in bf16, at the real `(B, 1, 1, T)` broadcast shape. It measured
**zero** at model level: 34.24 vs 34.17 ms per minibatch. At 61 tokens and 256
embedding dims the projections and MLP dominate, and attention is too small a
share for a 25% saving on it to appear. Do not spend the ~5 lines.

The dev workstation is a GTX 1080 Ti (sm_61) where none of this is available, so
its numbers still do not transfer in the other direction.

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
