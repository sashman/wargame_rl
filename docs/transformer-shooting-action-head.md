# Transformer Shooting Action Head (Transformer-Only)

> **Status: implemented.** Lives in `TransformerNetwork` (`wargame_rl/wargame/model/net.py`),
> covered by `tests/test_transformer_shooting_policy.py`. This document describes the
> shipped design, not a plan.
>
> The scope below was written when DQN and `MLPNetwork` still existed; both were
> removed on 2026-08-09. The "explicitly unchanged" carve-outs are kept as a record
> of what this change did and did not touch — they no longer name live code.

## 1. Problem Statement and Scope

The environment supports a shooting action slice in the union action space, with phase-aware and feasibility-aware masking. Before this change the model policy path did not produce shooting-aware logits from model interactions in a way that used opponent-token context explicitly for target selection.

`TransformerNetwork` policy output now produces shooting logits while preserving the external output contract.

In scope:
- `TransformerNetwork` internals (`wargame_rl/wargame/model/net.py`)
- Transformer-focused tests and documentation

Out of scope:
- `MLPNetwork` changes (explicitly unchanged)
- Env action registry/mask semantics (already provided by env)
- PPO/DQN caller interface changes

## 2. Output Contract and Compatibility

The policy output contract remains:
- Shape: `(batch, n_models, n_actions)`
- Action index layout: exact env registry order (`stay`, `movement`, `shooting`)

Compatibility guarantees:
- PPO code paths (`forward`, `get_action`, `evaluate_actions`) continue to consume the same logits shape.
- DQN action selection remains compatible with existing mask application.
- Shared-transformer PPO remains compatible because the change is policy-head internal.

## 3. Core Design

### 3.1 Alive-token Masking Before Transformer

Given:
- Player model tensor: `P` with shape `(B, Np, Fp)`
- Opponent model tensor: `O` with shape `(B, No, Fo)`
- Action mask tensor: `M` with shape `(B, Np, A)` for batched and `(Np, A)` for single observation

Derive alive masks (both from the per-model `alive` feature column, located via
`_alive_feature_index`, which is the first of the trailing
`N_WOUND_FEATURES + N_COMBAT_STATS` columns):
- Player alive mask `Ap`: `P[..., alive] > 0.5`
- Opponent alive mask `Ao`: `O[..., alive] > 0.5`

Token positions are fixed (`[game, objectives, all players, all opponents]`),
so player `p` is always at index `n_prefix + p` and opponent `o` at
`n_prefix + Np + o`. Dead rows are **not** removed; instead they are excluded
from attention via a boolean key-padding mask:
1. Build the alive masks `Ap` `(B, Np)` and `Ao` `(B, No)` from the `alive`
   feature column.
2. Form a key-padding mask `(B, L)` that is `True` for the prefix (game +
   objectives) and for alive player/opponent rows, `False` for dead rows.
3. Run the transformer in a **single batched pass**, passing the mask as
   `attn_mask = key_mask[:, None, None, :]` into each block's attention
   (`scaled_dot_product_attention(attn_mask=...)`).
4. Return `player_alive` on the `EncodedState` so reconstruction can force dead
   player rows to stay-only without relying on the env mask.

Excluding dead rows as *keys* is equivalent to removing them for the encodings
of live tokens (and the game token), but keeps one batched forward for the whole
minibatch. Since the game + objective tokens are always valid keys, no query is
ever fully masked, so there is no softmax NaN.

### 3.2 Bilinear Shooting Head

Add a separate shooting head with higher capacity than raw dot product:
- `shoot_query_proj: Linear(E, H)`
- `shoot_key_proj: Linear(E, H)`
- Optional scale `1/sqrt(H)` on pairwise score

Let (token positions are fixed, so all rows are present — dead ones were already
excluded as attention *keys* in §3.1):
- `Zp` = encoded player token latents `(B, Np, E)`, sliced at `[n_prefix : n_prefix + Np]`
- `Zo` = encoded opponent token latents `(B, No, E)`, sliced at `[n_prefix + Np : …]`
- `Q = shoot_query_proj(Zp)` `(B, Np, H)`
- `K = shoot_key_proj(Zo)` `(B, No, H)`

Shooting logits:
- `S = Q @ K^T / sqrt(H)` with shape `(B, Np, No)`

This head is separate from movement/stay logits so the model can allocate dedicated capacity to targeting.

### 3.3 Movement/Stay + Shooting Merge

`policy_from_encoded` is fully vectorized — no per-sample loop, no gather/scatter
index maps:

1. Base logits from the shared policy head over player latents: `(B, Np, A)`.
2. Overwrite the shooting slice in place: `base_logits[:, :, s0 : s0 + No] = S`.
   Opponent `o` maps directly to column `s0 + o`.
3. Dead player rows are replaced wholesale by a row that is `-inf` everywhere
   except `stay` (index 0) at `0.0`, selected with
   `torch.where(player_alive, base_logits, dead_row)`. Dead rows are therefore
   stay-only even when the env mask is absent.
4. The env-provided action mask (carried on `EncodedState.mask_tensor`) is then
   applied with `masked_fill`, but only when its shape matches `(Np, A)`.
5. Finally, any row that ended up entirely `-inf` gets `stay` reset to `0.0`, so
   `Categorical(logits=...)` never sees a fully-invalid row.

Result: final logits match exact env action indexing while shooting values come from player-opponent bilinear interaction.

## 4. Edge Cases and Failure Modes

1. No shooting slice in env (`shooting_slice is None`):
- Skip shooting head merge; behavior stays movement-only.

2. No alive players:
- Every row becomes the dead row: `-inf` except finite `stay`.

3. No opponent models (`n_opponents == 0`):
- Shooting head is skipped entirely; movement/stay unchanged. Dead *opponents*
  are handled by the env action mask, which zeroes their shooting columns.

4. Batched observations with variable alive counts:
- No special handling needed — token positions are fixed, so the whole batch is
  one masked `torch.where` over the same shape.

5. Numerical stability:
- Keep invalid entries at `-inf` so downstream `Categorical(logits=...)` + action masks remain consistent.

## 5. Public Interfaces and Internal API Changes

No public interface changes.

Internal `TransformerNetwork` adjustments:
- `encode_state` returns an `EncodedState` dataclass carrying the encoded
  sequence plus alive-index / mask metadata; `policy_from_encoded` and
  `value_from_encoded` take that object (state is passed explicitly, not stored
  on the module).
- Shooting projection layers (`shoot_query_proj`, `shoot_key_proj`) are built in
  policy mode only, and only when the env exposes a non-empty shooting slice
  (`env._action_handler.shooting_slice` → `shooting_slice_start` /
  `shooting_slice_end` constructor args). They are `None` otherwise.
- Helper methods: `_alive_feature_index` / `_alive_from_features` (alive mask
  derivation) and `_shooting_scores` (bilinear scores).

`MLPNetwork` remains unchanged.

## 6. Test Coverage

`tests/test_transformer_shooting_policy.py`:

1. `test_transformer_policy_batched_matches_single_obs` — batch and single-observation
   outputs agree, shape `(B, n_models, n_actions)`.
2. `test_transformer_shooting_scores_land_in_correct_opponent_columns` — shooting logits
   land in the env shooting slice columns indexed by opponent.
3. `test_transformer_policy_dead_player_row_is_stay_only` — dead rows are stay-only.
4. `test_transformer_policy_dead_opponent_shooting_column_is_neginf` — dead targets are
   unselectable.
5. `test_transformer_policy_without_shooting_keeps_shoot_head_disabled` — movement-only
   envs leave the head at `None`.
6. `test_transformer_value_path_runs_with_masking` — value head works on the same encoding.
7. `test_self_attention_key_mask_ignores_masked_positions` /
   `test_dead_tokens_do_not_affect_alive_logits` — key-padding mask is honoured.
8. `test_transformer_policy_no_nan_with_dead_units` — no softmax NaN.
