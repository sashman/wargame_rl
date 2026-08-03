# Metrics Reference

Semantics of every metric logged to Wandb, written to be usable as an **evaluation framework
for an LLM agent** reading runs through the W&B MCP server or API.

The catalogue below states, for each metric, where it is emitted, how often, what its units
are, and — critically — what it does *not* mean. Several metrics in this project are means of
means, or change definition silently; an agent that reads them naively will draw confident
wrong conclusions. Those traps are called out inline and collected in
[Reading rules](#reading-rules).

- Related: [reward-phases.md](reward-phases.md) (curriculum config) · [missions-and-vp.md](missions-and-vp.md) (VP scoring) · [game-state-io.md](game-state-io.md) (per-match event logs, a finer-grained alternative to these aggregates)

---

## 1. Where metrics come from

Two independent emission paths, on two different step counters. **No single W&B step carries
both**, which is why history rows look ragged.

```
                   on_train_epoch_end (lightning_base.py:155)
                              │
      ┌───────────────────────┴───────────────────────┐
      │                                               │
  run_episodes(n_episodes)                    _advance_reward_phase(sr)
  → _evaluate_episodes                        → reward_phase
  → success_rate, mean_episode_steps,           phase_advanced_at_epoch
    reward/{mean,max,min}_episode_reward

                   PPO training step (ppo/lightning.py:400)
                              │
                  loss/*, reward/components/*
```

Evaluation episodes are **separate rollouts** run under `torch.no_grad()` with the policy in
eval mode (`lightning_base.py:71–85`). They are not the training rollouts. So
`reward/mean_episode_reward` (eval) and `reward/components/*` (training rollout) describe
different episodes and need not agree.

### Episode count

`n_episodes` defaults to **10** for PPO (`ppo/config.py:33`) and **20** for DQN
(`dqn/config.py:22`). This sets the resolution of `success_rate`: with 10 episodes it can only
ever be a multiple of 10. **A change from 80% to 90% is one episode.** Do not read
single-epoch movements as signal; require a trend across epochs.

---

## 2. Metric catalogue

### Episode-level (per epoch, from eval rollouts)

| Key | Source | Units / range | Meaning |
|---|---|---|---|
| `success_rate` | `lightning_base.py:119` | **0–100** | Percent of eval episodes where the current phase's `success_criteria` held on the final step. See trap Ⓐ. |
| `mean_episode_steps` | `lightning_base.py:96` | steps | Mean eval episode length. Compare against `max_turns`; equality means episodes never terminate early. |
| `reward/mean_episode_reward` | `lightning_base.py:91` | reward units | Mean total (undiscounted, summed) reward per eval episode. This is the checkpoint selection metric (`checkpoint_callback.py:28`). |
| `reward/max_episode_reward` | `lightning_base.py:101` | reward units | Best single eval episode. Gap vs. mean indicates variance across seeds/placements. |
| `reward/min_episode_reward` | `lightning_base.py:106` | reward units | Worst single eval episode. Persistently negative alongside a healthy mean signals a failure mode not captured by `success_rate`. |

### Curriculum

| Key | Source | Units / range | Meaning |
|---|---|---|---|
| `reward_phase` | `lightning_base.py:127` and `:136` | index, 0-based | Index into `reward_phases`. Emitted twice on two step counters — see trap Ⓒ. |
| `phase_advanced_at_epoch` | `lightning_base.py:140` | epoch | Logged **only on the epoch an advance happens.** Absent for the whole run means the agent never cleared phase 0. |

Advancement requires all three of (`phase_manager.py:211–240`): `success_rate/100 >=
success_threshold`, at least `min_epochs` spent in the phase, and the threshold held for
`min_epochs_above_threshold` **consecutive** epochs (any miss resets the counter to 0).

### Losses (per epoch, mean over the epoch's optimizer updates)

| Key | Source | Sign convention |
|---|---|---|
| `loss/train_loss` | `ppo/lightning.py:401` | Total: policy + `vf_coef`·value + `ent_coef`·entropy. Routinely negative — the entropy term is negative. **Negative is not an error.** |
| `loss/policy_loss` | `ppo/lightning.py:408` | Clipped surrogate. Near zero at convergence; magnitude says how much the update moved. |
| `loss/value_loss` | `ppo/lightning.py:415` | Critic MSE. The one loss that should fall monotonically. The clearest single convergence signal. |
| `loss/entropy_loss` | `ppo/lightning.py:422` | Negative entropy. Rising *toward zero* = policy sharpening. A fast collapse toward 0 early is premature determinism. |

Lightning suffixes these `_step` and `_epoch`. With `on_epoch=True` and one log call per epoch
the two are equal; prefer `_epoch`.

### Reward components (per epoch, from training rollouts)

`reward/components/<calculator>` and `reward/components/<calculator>/<sub_component>`.

Every value is a **per-step mean over the rollout**, built by four successive divisions:

1. `phase_manager.py:152–204` — per-model calculators summed, divided by `n_alive` (`:170`); sub-breakdowns keyed `<name>/<component>`, also divided by `n_alive` (`:172`); global calculators added flat (`:178`).
2. `wargame.py:577–581` — `step()` accumulates each key across the episode.
3. `agent_base.py:113–117` — divided by `episode_reward_steps`.
4. `ppo/lightning.py:600–618` — step-weighted across rollout episodes, rescaled if the rollout is truncated to `n_steps`, divided by `total_steps`.

So `reward/components/vp_gain = -0.10` means **the player loses 0.10 normalized net VP on
every step**, not once per episode. Read all these keys as per-step rates.

#### Calculator semantics

| Component | Kind | Formula / meaning | Healthy direction |
|---|---|---|---|
| `vp_gain` | global | `(player_vp_delta - opponent_vp_delta) / cap_per_turn` (`vp_gain.py:33`). Signed: **negative means the opponent is out-scoring you.** | → positive |
| `objective_coverage` | global | Fraction of objectives the player controls, paid every step. Rewards spreading across *distinct* objectives. | ↑ toward 1 |
| `closest_objective` | per-model | Distance-shaping toward the nearest objective. Sub-keys: `progress`, `distance_delta`, `base_penalty`, `best_distance_bonus`. | `progress` ↑ |
| `closest_objective_v2` | per-model | As above plus de-stacking. Extra sub-keys: `target_obj_idx`, `target_switched`, `overstack_penalty`. High `target_switched` (≳0.3) means the policy is dithering between targets. | `target_switched` ↓ |
| `group_cohesion` | per-model | Penalty proportional to distance beyond `group_max_distance` from the nearest same-group model. 0 when in range or alone in group. **Exactly 0 for a whole run means disabled or never violated — not "good cohesion".** | 0 or ↑ |
| `killing` | global | `bonus_killing_opponent` per newly killed opponent. | context-dependent |
| `objective_flip_bonus` | global | Change in an objective-control potential, summed over objectives. At `loss_penalty_scale == 1.0` it is a pure potential (farming-proof). | ↑ |
| `terminal_success_bonus` | terminal | Awarded on one terminating step when criteria hold, scaled by remaining-turns fraction (`phase_manager.py:190–193`). See trap Ⓑ. | ↑ |
| `terminal_vp_bonus` | terminal | Awarded at episode end when player VP clears the phase threshold. Same dilution caveat as above. | ↑ |

#### Success criteria

`success_rate` means something different per phase, depending on the configured criteria:

| Criteria | Succeeds when |
|---|---|
| `all_at_objectives` | Every alive model is within an objective radius |
| `all_models_grouped` | Every model is within `max_distance` of a same-group member (sole members count as grouped) |
| `player_vp_min` | Player VP ≥ a threshold derived from mission, objective count and round count |
| `player_ahead_on_vp` | `player_vp > opponent_vp` |

**Always resolve the phase's criteria from the run config before interpreting `success_rate`.**
A 90% success rate under `all_at_objectives` says nothing about whether the player is winning.

---

## 3. Reading rules

Traps that produce confidently wrong conclusions.

Ⓐ, Ⓑ and Ⓒ are defects rather than inherent properties, and are marked in the source as
`TODO(metrics-trap-A|B|C)` at the site where the fix belongs — grep for `metrics-trap` to find
them. Ⓓ, Ⓔ and Ⓕ are correct by design; this document is their only fix.

**Ⓐ `success_rate` silently changes definition.** `lightning_base.py:112–115`: when
`episode_successes` is empty — which happens if `env.last_step_context` is `None` — it falls
back to `(steps < max_turns).mean()`, i.e. "episode ended early" rather than "criteria met".
Same key, different meaning, no warning. If `mean_episode_steps == max_turns` *and*
`success_rate == 0`, suspect the fallback rather than a failing policy.

**Ⓑ `terminal_success_bonus` is not comparable to the other components.** It is paid once per
episode but divided by every step (stages 3–4 above), so its magnitude scales inversely with
episode length. It moves when episode length moves, with no change in policy quality. Never
compare it across runs with different `mean_episode_steps`, and never read its value as the
configured bonus.

**Ⓒ `reward_phase` is double-logged** — through Lightning at `:127` and again through raw
`wandb.log(step=self.global_step)` at `:136`, on different step counters. Prefer the value on
the epoch-aligned row.

**Ⓓ Percent vs. fraction.** The logged `success_rate` is `sr * 100` (`:119`) but
`success_threshold` in config is a fraction in `[0, 1]`. Divide the metric by 100 before
comparing.

**Ⓔ Eval and training rollouts are different episodes.** `reward/mean_episode_reward` (eval,
greedy) and `reward/components/*` (training, exploratory) will not reconcile. Do not compute
one from the other.

**Ⓕ Components are means over alive models.** Per-model calculators divide by `n_alive`
(`phase_manager.py:170`). As models die, the same per-survivor behaviour yields a different
component value. Cross-check against casualties before attributing a shift to policy change.

---

## 4. Evaluation procedure

A recommended order of operations for an agent asked to assess a run.

### Step 1 — Establish context before reading any number

Fetch the run config and resolve: `reward_phases` (names, calculators, weights,
`success_criteria`, `success_threshold`), `n_episodes`, `max_turns`, `number_of_opponent_models`
and `opponent_policy`. Without the active criteria, `success_rate` is uninterpretable.

### Step 2 — Triage

| Check | Verdict |
|---|---|
| `state == "crashed"` and `_step == 0` | Startup failure. Report and stop — no metrics to read. |
| `state == "crashed"` with steps logged | Partial run. Metrics valid up to the last step; do not treat final values as converged. |
| `phase_advanced_at_epoch` absent and `reward_phase == 0` | Never cleared phase 0. Investigate the phase-0 criteria before anything else. |
| `mean_episode_steps == max_turns` throughout | Episodes never end early. Either `terminate_on_success: false`, or success is never reached. |

### Step 3 — Convergence

- `loss/value_loss_epoch` should fall and flatten. Still falling at run end ⇒ undertrained.
- `loss/entropy_loss_epoch` should rise toward 0 gradually. Reaching near-0 in the first ~10% of epochs ⇒ premature collapse; suspect `ent_coef` too low.
- `loss/policy_loss_epoch` oscillating around 0 is normal.

### Step 4 — Task performance

- `success_rate` trend across epochs, **not** single points (Ⓐ, and ±1/`n_episodes` resolution).
- `reward/mean_episode_reward` trend; check `min_episode_reward` for a failure mode the mean hides.
- Per-component: is the component the current phase actually weights the one improving?

### Step 5 — Plateau detection

Find the epoch after which the primary components change by less than ~5% of their range.
Everything past that is wasted compute. Report it as an epoch budget recommendation.

### Step 6 — Red flags

| Flag | Condition | Meaning |
|---|---|---|
| **Losing the VP race** | `vp_gain < 0` and flat | Opponent out-scores the player every step; nothing in the reward opposes it. Especially likely when criteria are objective-based rather than VP-based. |
| **Dithering** | `closest_objective_v2/target_switched ≳ 0.3` | Policy re-targets ~1 step in 3. |
| **Dead component** | any component exactly `0` for the whole run | Disabled, unreachable, or zero-weighted. Not evidence of good behaviour. |
| **Reward/criteria mismatch** | components improve, `success_rate` flat | Optimizing something the criteria do not measure. |
| **Success without winning** | `success_rate` high, `vp_gain` negative | The criteria do not select for winning. Check whether that is intended. |

### Step 7 — Report

State the verdict, the evidence with metric keys and epochs, and — for every claim — whether
it rests on a full history or a sampled one. Distinguish "the 5 runs I sampled" from "the
project". If a red flag depends on a trap above, say which.

---

## 5. Programmatic access

Via the W&B MCP server (see the `wandb` entry in `~/.claude.json`):

| Need | Tool |
|---|---|
| Project shape, available keys | `probe_project_tool(entity, project)` |
| Run list with final summaries | `query_wandb_tool` (GraphQL, `order: "-createdAt"`) |
| Training curves for one run | `get_run_history_tool(entity, project, run_id, samples=N)` |
| Two runs side by side | `compare_runs_tool` |

`get_run_history_tool` with an explicit `keys` list can return zero rows even when the keys
exist; call it without `keys` and filter client-side.

Because eval and training metrics land on different steps, history rows are sparse — each row
carries one family or the other. Group by family before computing trends rather than assuming
every row has every key.

For per-step ground truth rather than these aggregates, record an event log and use
`analyze_match` instead — see [game-state-io.md](game-state-io.md).
