# Throughput review — what to try next, ranked

Point-in-time review of training wall-clock across five areas (GPU/kernels, rollout
parallelism, MLOps/orchestration, sample efficiency, observation pipeline). Findings
were produced independently and are consolidated here; where two disagreed, the
disagreement is recorded rather than resolved by preference.

**Everything below is analysis and arithmetic. Nothing here has been trained.**

---

## 0. Corrections that change the ranking — read these first

Five premises this project has been working from are wrong. Four were verified
directly against the code or a probe.

### 0.1 The sequence is 83 tokens, not 61 — verified

`docs/training-throughput.md` and this repo's PR history describe "12.86M parameters
over a 61-token sequence". The epoch numbers were measured on
`25v25_shooting_opponent.yaml`, which sets `random_terrain.count: 29`. So
T = 1 game + 3 objectives + 25 player + 25 opponent + **29 terrain = 83**.

61 belongs to `25v25_single_phase.yaml` (7 fixed terrain pieces). **Terrain is 35% of
the token budget, not 11%** — which promotes "do we need 29 terrain tokens?" from a
rounding error to the second-largest single lever on the model side.

### 0.2 Eval is ~22% of a real epoch and is not in `perf/epoch_s` — verified

`perf/epoch_s` is `rollout + update` only. `perf/eval_s` is logged separately.

The 0.65 s eval in the recorded baseline used the `PPOConfig` default of
`n_episodes=10`. **Every seeded recipe passes `--n-eval-episodes 30`**
(`Justfile:91,105,128`), and `max_turns` = 20 rounds x 2 active phases = 40. So a
real experiment epoch runs 30 x 40 = **1200 eval env steps against 2048 rollout
steps** — 37% of all environment stepping.

True epoch under the recipes actually used: **~8.4 s, of which eval is ~22%.**
Every percentage quoted against 6.55 s is therefore optimistic by about a quarter.

### 0.3 The `torch.compile` blocker is smaller than documented, and avoidable — verified

Documented as: a compiled checkpoint would load into `simulate`,
`measure-checkpoint` or a warm start matching no keys, silently. Actually:

- `convert_state_dict` (`net.py:677-678`) **already strips `_orig_mod.`**, and raises
  `ValueError` when nothing matches. `simulate`, `measure-checkpoint` and
  `measure-phase-gates` all route through it — they are safe and fail loudly.
- `trainer.fit(ckpt_path=...)` is strict by default — also loud.
- Only `_apply_warm_start_weights` (`train.py:139`, `strict=False`) is silently holed.

And the whole problem is avoidable: **`torch.compile(model.forward)` — the bound
method rather than the module — leaves `state_dict` byte-identical.** Verified:
compiling a module yields `_orig_mod.0.weight`; compiling its `forward` yields
`0.weight`. No checkpoint consumer is affected at all.

### 0.4 `last.ckpt` is written once, at `on_train_end` — verified by probe

The 2026-08-08 split into two callbacks fixed the original bug for runs that
**complete**: `last.ckpt` holds the true final epoch. But a 10-epoch probe with the
real `get_checkpoint_callback` shows the unmonitored callback calls
`_save_checkpoint` **exactly once**, at the end — `last.ckpt` does not exist at any
point during training.

Cause: `ModelCheckpoint._save_last_checkpoint` runs only when
`_last_global_step_saved == trainer.global_step`, and with `save_top_k=0`
`_save_topk_checkpoint` returns without ever setting it.

**Consequence: a killed run leaves no `last.ckpt` at all** — only best-by-training-
reward files, which is precisely the ~13 vp_margin selection bias the split removed.
`EventLogCallback`'s own docstring says runs here are "routinely stopped before
`trainer.fit()` returns — monitored, judged, and killed".

`tests/test_checkpoint_callback.py` asserts callback *configuration*
(`save_last`, `monitor is None`), so it passes while the behaviour it names does not
occur. A behavioural test that counts writes across a multi-epoch fit is needed.

### 0.5 Every historical wandb timing is pre-optimisation — and this inverts two conclusions

The env hot-path work (11.34 -> 2.26 ms/step) is on the current branch and has never
been in a completed run. So **all 358 local run records were produced under an
environment ~5x slower than the current code.**

Two conclusions drawn from that data invert:

- *"The GPU is idle, so extra gradient work is nearly free."* True at ~22.5 s/epoch
  where the env dominated. **False now**: update is 2.76 s of a 6.55 s solo epoch
  (42%). Doubling `n_epochs` 5 -> 10 adds ~2.76 s, i.e. **+42% per epoch**, not
  "nearly free".
- *"2 -> 4 concurrent runs is 1.87x throughput."* Measured on a CPU-bound-env
  workload. Post-optimisation the GPU share is much larger, so 4-wide scaling will
  be worse. The *direction* is probably still right; the magnitude must be
  re-measured before anyone plans a batch around it.

**Nothing derived from historical run timings should be trusted until re-measured on
this branch.**

---

## 1. Ranked candidates

Percentages are against the **8.4 s real epoch** from §0.2 unless stated. "Pure"
means bit-identical or numerics-only; "learning" means it needs a 2-seed A/B and is
not a speedup claim.

| # | Change | Kind | Est. gain | Confidence |
|---|---|---|---|---|
| 1 | Parallel rollout worker pool | pure | **−2.5 s (−30%)** | med-high |
| 2 | Gate evaluation (interval, or n=10) | pure* | **−1.4 s (−17%)** | high |
| 3 | `--precision bf16-mixed` | learning | −0.94 s (−11%) | high (speed) |
| 4 | `torch.compile(model.forward)` | learning | −0.2 to −0.4 s | high |
| 5 | Observation encoder: hoist static columns | pure | −0.5 to −0.7 s | med-high |
| 6 | `env.step` hoists (4 sites) | pure | −0.4 to −0.6 s | high |
| 7 | Drop/demote the 29 terrain tokens | learning | −1.0 s | high (arith) |
| 8 | `share_transformer=True` | learning | −0.6 s | med |
| 9 | Rollout GPU-sync cleanup | pure | −0.1 s | high |
| 10 | Skip the value trunk in eval | pure | −0.06 s | high |

### 1. Parallel rollout worker pool — the largest single lever

Rollout is 3.75 s of a 6.55 s solo epoch and is pure GIL-bound Python. Processes are
the only lever; raising `num_rollout_envs` improves forward batching and nothing else.

`gymnasium.vector.AsyncVectorEnv` does not fit, for four independent reasons:
`observation_space` is a `spaces.Dict` containing a `spaces.Sequence` (no shared-memory
representation) while `step()` returns a Pydantic/dataclass object; the rollout loop
reads five attributes off the env that no vector API exposes; routing those through
`info` reintroduces the ~0.19 ms/step cost that `build_info=False` exists to avoid;
and Gymnasium 1.x autoreset would silently break the GAE bootstrap alignment the
hand-rolled reset currently preserves.

Custom spawn pool instead. `wargame_rl/wargame/envs/` imports torch nowhere, so a
worker importing only `envs.*` + numpy starts in ~0.2 s and never initialises CUDA.
Do the observation->numpy conversion **in the worker**: ~13.6 KB/env/step, so plain
pickling is ~80 ms/epoch — build it with pipes first and only add
`multiprocessing.shared_memory` if `perf/rollout_s` says it matters. Set
`OMP_NUM_THREADS=1` before importing numpy in the worker.

**The trap, which has already cost seven runs:** rollout envs share
`phase_manager.position` *by object identity*, which breaks silently across a process
boundary. Push `set_phase` every `training_step` and have workers echo back
`(index, phase_name, n_per_model_calculators, n_global_calculators)`. **Assert all
four** — the index alone is insufficient, because a config that failed to pickle
identically would give a matching index over a different calculator list, which is
exactly the original failure. Turn the existing `rollout_phase_index` log into an
assert; a human noticing two diverging lines is what failed last time.

Keep evaluation in-process (see #2 instead) — its fixed seeding and env-0
`StateExporter` then need no changes.

Estimate: rollout 3.75 -> ~1.2 s. **Caveat:** `train-multi-seeds` runs 4 trainings at
once; 4 x 8 workers on 24 cores is oversubscribed. Either divide the worker count by a
concurrency env var or document `--num-rollout-envs 4` for multi-arm batches.

### 2. Gate evaluation — the cheapest large win

1200 eval env steps per epoch against 2048 rollout steps. Evaluating every 4th epoch
cuts total env stepping by ~28%; dropping to n=10 every epoch cuts ~17%.

Safe here because `_advance_reward_phase` is the only consumer needing per-epoch eval,
and it is a **no-op on single-phase configs** — `try_advance` returns immediately at
`is_final_phase`. `25v25_shooting_opponent.yaml` has one phase.

*Gate the interval on `len(reward_phases) > 1`.* On curriculum configs
`try_advance` counts **consecutive** epochs above threshold, so skipping epochs
changes curriculum timing — that is the one place this is not free.

Costs nothing in policy quality: eval never touches the weights, and the repo already
forbids reading a single-epoch value (50-epoch buckets are the reading unit).

### 3. `--precision bf16-mixed` — already written, needs the A/B not the code

Measured: update 2.76 -> 1.55 s, rollout 3.75 -> 4.02 s, net **−0.94 s**.

The `_as_float32` guard fixes *representation* of the log-prob delta. It does not fix
the other half: a bf16 trunk perturbs the logits themselves at ~1e-2, the same order
as the 0.007-nat signal the ratio must resolve.

**Cheap pre-check that costs no training run:** load an existing checkpoint, run one
batch of 128 real observations through `PPOModel.forward` twice — plain, and under
`torch.autocast("cuda", bfloat16)` — and report the per-model `|Δ log_prob|`
distribution against 0.007. If the median is at or above 0.007, the A/B would be
measuring precision noise rather than policy. Minutes, not hours.

**Do not try to recover the +0.27 s rollout regression by running the rollout in
fp32.** Lightning wraps the whole `training_step`, rollout included. Splitting them
would make `old_log_probs` fp32 and `new_log_probs` bf16, so the importance ratio
would be systematically != 1 at the first minibatch. That is a silent trainer bug for
a 4% gain.

### 4. `torch.compile(model.forward)` — only on top of bf16

Bound-method compile, so `state_dict` never changes (§0.3). Route only the update-loop
call through it; leave rollout, bootstrap and eval eager (different batch shapes, each
would trigger its own compile for work that is not the bottleneck).

Shapes are static — T=83 always, `n_steps` 2048 / `batch_size` 128 = exactly 16
minibatches of 128 — so one graph, no recompiles.

Gain is asymmetric and that asymmetry is the tell: **2.8% on TF32, 7.5% on bf16.** At
34.5 ms eager dispatch is hidden; at 19.3 ms it starts to bind. **Not worth wiring
unless #3 lands.** If tried, use `mode="max-autotune-no-cudagraphs"`, explicitly not
`"reduce-overhead"` — CUDA graphs target launch overhead, which is not the constraint
here (see §2 below).

### 5-6. Observation encoder and `env.step` hoists

Of the 49 per-model feature columns, **38 are static for the entire run** — group
one-hot, `max_wounds_norm`, 7 combat stats, and the 25-wide expected-damage block.
All are sourced from `ModelConfig`, and `expected_damage` never reads
`current_wounds`. Today ~34 Python list-comprehensions per observation rebuild them,
plus two `np.unique(axis=0)` lexsorts on data that cannot change.

Four `env.step` hoists, all bit-identical by construction:
- Opponent distances-to-objectives falls into a Python double loop while the player
  side uses the vectorised distance cache (~125 us/step). Pure integer arithmetic, so
  no float reassociation is possible.
- Terrain observation rebuilds 29 arrays + 29 dataclasses per step for a quantity that
  changes only on reset (~55 us/step).
- `compute_distances(opponent_models, ...)` runs **three times per step** on identical
  positions, from three different calculators (~60 us/step). (Two further calls inside
  `DefaultVPCalculator` fire at a different point in the step and must be left alone.)
- `objective_hold.calculate` computes `values / occupancy**crowding_exponent` **once
  per model**, 25x/step, though every operand is model-independent (~48 us/step). Both
  operands are already ctx-cached; the division is not. Use a **third** ctx key field —
  sharing `_cached_ctx` would freeze it at step one, the trap that file documents.

**Prerequisite — DONE (2026-08-09).** `tests/test_reward_golden.py` does not pin the
observation feature arrays, so a wrong feature column would have passed every test in
the repo. `tests/test_observation_golden.py` now pins them, verified sensitive against
a one-ULP change and an appended column.

It also turned up something that changes how the column trap must be tested: the
column after `alive` is `wound_ratio`, and at `max_wounds: 1` the two are
**bit-identical**, so an off-by-one read is invisible unless a model is wounded but
alive. The first version of that test passed with the trap injected.

### 7. The 29 terrain tokens

Trunk FLOPs are linear in T (attention's quadratic term is only ~5% of the model).
83 -> 54 tokens is a **35% cut** to every forward — update, rollout and eval alike.

Two variants: delete them outright — justified by this repo's own repeatedly
confirmed finding that the agent does not use terrain for cover, and if that is right
this is free; or keep the information and make the 33 context-only tokens
key/value-only, since only player and opponent latents are ever read out. The second
preserves information and is a larger code change.

Either is a **learning** change. If terrain is genuinely null input, the first variant
is also the cleanest experiment that would finally prove it.

### 8. `share_transformer=True`

**Two documented blockers are stale** — `train.py:408` does pass `ppo_config`, and
`nn.Module.parameters()` de-duplicates by identity so the optimiser sees the shared
trunk once.

**The real landmine, not previously flagged:** sharing the trunk turns `vf_coef` from
near-inert into a live hyperparameter. With separate trunks and Adam, `vf_coef` is
almost a no-op (Adam's m/sqrt(v) is scale-invariant per parameter). Shared, policy and
value gradients mix in the same parameters — and the two losses are scaled differently
by dead models, since `policy_loss` is diluted by the alive fraction (falling to ~0.6
late in an episode) while `value_loss` is not. That makes the policy/value gradient
ratio survival-dependent and time-varying.

So this is **not "the same training with half the parameters"** and must not be
reported as one. Any A/B must sweep `vf_coef` within the shared arm or it is
uninterpretable.

Still holed: `_apply_warm_start_weights` uses `strict=False`, so loading an *unshared*
checkpoint into a *shared* model applies the policy trunk then overwrites it with the
value trunk, silently. Fix that first — the same fix `torch.compile` would have needed.

---

## 2. The update is bandwidth-bound, not launch-bound

This kills several plausible-sounding candidates, so the reasoning is recorded.

Kernel-launch overhead is invariant to numeric precision. The measured 45.7 -> 34.5 ->
19.3 ms swing comes from changing *only* matmul precision. A launch-bound loop cannot
do that.

Decomposition: GEMM math ~11.2 ms at TF32, leaving **~23 ms of non-GEMM work**;
independently, ~15 GB of activation traffic per minibatch / 34.5 ms = ~435 GB/s,
roughly half a 4090's achievable bandwidth against ~21% of TF32 peak FLOPs. The model
predicts bf16 at ~19.6 ms against **19.3 measured**, which is why it is trusted.

Occupancy is fine: the largest GEMM is ~5 waves of tiles across 128 SMs, the smallest
~1.3. Nothing is starved.

**Therefore:** fusion helps (bytes), precision helps (bytes and math), and CUDA graphs,
multi-stream, grouped-GEMM trunk merging and bigger minibatches do not. Activation
checkpointing would be strictly worse.

---

## 3. Sample efficiency — the update looks throttled

From learning curves of three completed 1000-epoch arms (in-run eval at **fixed**
seeds, so these carry no sampling noise):

**86% of the total gain lands in the first 100 epochs.** Epochs 100 -> 1000 buy about
+10.8 vp over 900 epochs.

Across the entire run: `clip_fraction` 0.09-0.14 and `approx_kl` 0.008-0.020 — the
**bottom** of the band `docs/metrics.md` itself calls healthy (0.1-0.3, 0.01-0.03).
Movement entropy flatlines at ~2.24 nats from epoch 100 to 1000 and never moves again.
`explained_variance` reaches 0.80-0.88, so the critic is not the bottleneck.

That reads as a policy taking roughly half the step PPO would permit, then idling
under a constant LR and constant entropy bonus for 900 epochs.

**The one-line diagnostic that settles it, at zero training cost:**
`clip_grad_norm_` (`ppo/lightning.py:490`) **returns** the pre-clip total norm and the
return value is discarded. Log it. `max_grad_norm=0.5` is applied to the *joint* norm
across both 6.5M networks; if that norm routinely exceeds 0.5, then `max_grad_norm` is
the de-facto learning rate and both `lr` and `vf_coef` are partly inert — the most
economical explanation for a thousand epochs of flat `approx_kl`. Five epochs with
`--no-wandb` answers it.

If confirmed, the candidates are `lr` 3e-4 -> 1e-3 (zero wall-clock cost), or
`n_steps` 2048 -> 1024 (halves data staleness). **Note `n_epochs` 5 -> 10 is no longer
cheap** — see §0.5; it now costs ~+42% per epoch.

Two protocol notes:

- **In-run `eval/vp_margin` has no sampling noise.** Eval seeds are fixed at 500000+
  and the env is fully seeded. Arm-vs-arm on those same 30 episodes is a **paired**
  comparison and is sharper than the "SE ~8-9 at n=30" rule, which describes
  generalisation to *new* layouts. Keep n=100 on held-out seeds 700000+ for published
  effect sizes; use the paired in-run curve for screening. Conflating them is why the
  screen looks noisier than it is.
- **A 150-epoch screen may be enough to reject clear losers** — the `flat` disaster arm
  was already 33 vp behind at epochs 50-99 against a final gap of 45. But the
  counter-evidence must travel with it: `share` vs `share_soft` were indistinguishable
  at 300 epochs (+12.3 vs +10.4) and separated by 11.7 vp at 1000. A short screen
  rejects losers; it cannot rank close arms or quote effect sizes. This can be settled
  for free from existing wandb history.

---

## 4. Not worth doing

Recorded so they are not re-derived.

- **CUDA graphs / `reduce-overhead` / multi-stream / grouped-GEMM across the two
  trunks / bigger minibatches** — all target launch overhead or occupancy; §2 shows
  neither binds. Bigger minibatches also change a hyperparameter while posing as a
  speedup.
- **Activation checkpointing** — trades bandwidth for compute in a bandwidth-bound
  loop. Strictly worse.
- **Anything to attention itself** (~5% of FLOPs) or to the policy head (**0.14%** of a
  forward). The 128x25x122 output shape looks alarming and is one 256->122 GEMM over
  25 rows. The `-inf` masking machinery is ~0.1%.
- **`set_to_none=True`** — already the default in both torch and Lightning.
- **`nn.Dropout` at p=0.0** — ATen early-returns. Already free.
- **Value-function work** (separate LR, value clipping, return normalisation) —
  `explained_variance` is 0.80-0.88 and rising. Note it is *inflated*: dead models get
  exactly zero reward, so their rows are trivially predictable. Read it as "the critic
  is not the problem", not as an absolute.
- **Shrinking the network purely for speed** — and note `PPOConfig.hidden_size` /
  `num_layers` are **dead config** for the transformer path (`net.py` hardcodes
  `TransformerConfig()`). Delete or wire them; leaving them is how someone eventually
  sweeps a parameter that does nothing.
- **A cheap warm-start scenario with fewer models** — structurally impossible. Transfer
  requires identical `n_objectives`, `max_groups`, `n_opponent_models`,
  `n_movement_angles`, `n_speed_bins` and `observe_objective_control`, because both the
  feature width and `n_actions` depend on them. But env step cost is dominated by the
  25x25 distance/LOS/damage work — exactly what the shape constraint forbids shrinking.
- **Reducing env steps via `skip_phases` / frame skip** — adding `shooting` to
  `skip_phases` deletes the agent's shooting decisions outright. Frame skip does not
  typecheck: consecutive steps alternate movement and shooting phases with disjoint
  valid action slices.
- **Vectorising reward *arithmetic* across models** — `phase_manager` accumulates in
  model order and the golden gate is one-ULP sensitive. Restrict changes to *selection*
  and *hoisting*.
- **`val_check_interval` / `log_every_n_steps=1` / dataloader workers / progress bars /
  the start-of-run baseline sweep** — all verified to cost approximately nothing, and
  `log_every_n_steps=1` is *required* since PPO runs one `training_step` per epoch.

---

## 5. Suggested order

1. **Add the observation-tensor golden test.** Unblocks #5 and #6 and closes a real
   hole — a wrong feature column currently passes the entire suite.
2. **Fix `last.ckpt` for killed runs** (§0.4) and make the test behavioural. This is
   correctness, not speed, and it silently corrupts scoring comparisons.
3. **Log the pre-clip gradient norm** (§3). One line, no run, and it decides whether
   the sample-efficiency work has a target.
4. **Gate evaluation** (#2). Largest pure win per line of code.
5. **Parallel rollout pool** (#1) with the four-field phase assert.
6. **The pre-check for bf16** (§3 of #3), then the A/B, then `torch.compile` on top.
7. Observation encoder and `env.step` hoists (#5, #6), behind the golden test.

Items 1-5 are pure or correctness work and need no A/B budget. Everything from 6 on
changes trained numerics and must be measured at 2 seeds, n=100, on held-out layouts.
