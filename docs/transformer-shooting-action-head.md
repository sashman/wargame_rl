# Transformer Shooting Action Head Design (Transformer-Only)

## 1. Problem Statement and Scope

The environment already supports a shooting action slice in the union action space, with phase-aware and feasibility-aware masking. The current model policy path does not produce shooting-aware logits from model interactions in a way that uses opponent-token context explicitly for target selection.

This design upgrades `TransformerNetwork` policy output behavior to support shooting logits while preserving the external output contract.

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

### 3.1 Alive-token Compaction Before Transformer

Given:
- Player model tensor: `P` with shape `(B, Np, Fp)`
- Opponent model tensor: `O` with shape `(B, No, Fo)`
- Action mask tensor: `M` with shape `(B, Np, A)` for batched and `(Np, A)` for single observation

Derive alive masks (both from the per-model `alive` feature column, located via
`_alive_feature_index`, which is the first of the trailing
`N_WOUND_FEATURES + N_COMBAT_STATS` columns):
- Player alive mask `Ap`: `P[..., alive] > 0.5`
- Opponent alive mask `Ao`: `O[..., alive] > 0.5`

For each batch item:
1. Gather alive player rows from `P` and alive opponent rows from `O`.
2. Keep the compact→full index maps as `player_alive_indices` /
   `opp_alive_indices` (tensors of original row indices), returned on the
   `EncodedState` so reconstruction is explicit (no module-level state).
3. Build token sequence:
   - `[game_token, objective_tokens, alive_player_tokens, alive_opponent_tokens]`
4. Run the transformer on the compacted sequence.

This removes dead tokens from attention while preserving a reversible mapping
back to full action rows/columns.

> Implementation note: encoding currently loops per batch item (each sample is a
> separate transformer forward, then padded to the batch max length). This keeps
> variable alive-token counts simple but trades away batched throughput; batching
> via padding + a key-padding attention mask is a known future optimization.

### 3.2 Bilinear Shooting Head

Add a separate shooting head with higher capacity than raw dot product:
- `shoot_query_proj: Linear(E, H)`
- `shoot_key_proj: Linear(E, H)`
- Optional scale `1/sqrt(H)` on pairwise score

Let:
- `Zp` = encoded alive player token latents `(B, Np_alive, E)`
- `Zo` = encoded alive opponent token latents `(B, No_alive, E)`
- `Q = shoot_query_proj(Zp)` `(B, Np_alive, H)`
- `K = shoot_key_proj(Zo)` `(B, No_alive, H)`

Shooting logits in compact space:
- `S = Q @ K^T / sqrt(H)` with shape `(B, Np_alive, No_alive)`

This head is separate from movement/stay logits so the model can allocate dedicated capacity to targeting.

### 3.3 Movement/Stay + Shooting Merge

1. Keep existing policy head for base logits from player tokens:
   - `base_logits_compact`: `(B, Np_alive, A)`
2. Reconstruct full tensor:
   - Initialize `final_logits` with `-inf` shape `(B, Np_full, A)`
3. Scatter compact player rows back into full player rows using `player_compact_to_full`.
4. For shooting slice `[s0:s1)`:
   - For each alive player row and alive opponent col, scatter `S[..., i, j]` into
     `final_logits[..., player_full_i, s0 + opp_full_j]`.
5. Keep non-shooting columns from `base_logits_compact` scatter result.
6. Dead player rows remain `-inf` except `stay` can be retained as finite (env mask also enforces stay-only).

Result: final logits match exact env action indexing while shooting values come from player-opponent bilinear interaction.

## 4. Edge Cases and Failure Modes

1. No shooting slice in env (`shooting_slice is None`):
- Skip shooting head merge; behavior stays movement-only.

2. No alive players:
- Return all `-inf` except optional finite `stay` entries per row.

3. No alive opponents:
- Shooting slice remains `-inf` for all players; movement/stay unchanged.

4. Batched observations with variable alive counts:
- Use per-sample gather/scatter logic with explicit index maps.

5. Numerical stability:
- Keep invalid entries at `-inf` so downstream `Categorical(logits=...)` + action masks remain consistent.

## 5. Public Interfaces and Internal API Changes

No public interface changes.

Internal `TransformerNetwork` adjustments:
- `encode_state` returns an `EncodedState` dataclass carrying the encoded
  sequence plus alive-index / mask metadata; `policy_from_encoded` and
  `value_from_encoded` take that object (state is passed explicitly, not stored
  on the module).
- Add shooting projection layers (`shoot_query_proj`, `shoot_key_proj`) in policy mode.
- Add helper methods for:
  - alive mask derivation
  - token compaction + index tracking
  - scatter reconstruction to full `(B, Np, A)` tensor
  - shooting-slice merge

`MLPNetwork` remains unchanged.

## 6. Implementation Checklist

1. Add policy-only shooting head layers in `TransformerNetwork.__init__`.
2. Refactor state encoding to support alive compaction and index map outputs.
3. Keep value-network path functional with compacted encoding (shared backbone compatible).
4. Build full-size logits tensor initialized to `-inf`.
5. Scatter base logits and merge shooting logits into env shooting slice indices.
6. Preserve final output shape and action order contract.
7. Do not modify `MLPNetwork`.

## 7. Deferred Test Checklist (for Implementation Phase)

1. Transformer policy shape regression:
- Single observation and batch outputs remain `(B, n_models, n_actions)`.

2. Shooting merge correctness:
- Shooting logits land exactly in env shooting slice columns mapped by opponent full indices.

3. Dead-player behavior:
- Dead rows are effectively invalid except stay-compatible entries.

4. No-shooting env compatibility:
- Movement-only envs still produce valid logits without shooting-head usage.

5. Shared-transformer PPO path:
- `share_transformer=True` forward path stays operational with unchanged output shapes.
