# 02-02 Execution Summary

## TransformerNetwork Changes

### Constructor
- New arg: `terrain_size: int = 0`
- `terrain_embedding: nn.Linear | None` — created when `terrain_size > 0`,
  otherwise `None` (no extra parameters in state_dict)

### `_embed_terrain(terrain_tensor, is_batched)` → `Tensor | None`
Returns None when `terrain_embedding is None` or tensor has 0 rows.

### `encode_state(xs)`
- Reads terrain from `xs[4]`, mask from `xs[5]` (was `xs[4]`)
- Embeds terrain tokens and appends **after opponents** in the token sequence
- Terrain tokens are always attendable (no alive/dead masking)
- Player and opponent positions unchanged: `n_prefix`, `n_wargame_models`,
  `n_opponents` unaffected

### `share_backbone_with`
Now shares `terrain_embedding` alongside other embeddings.

### `from_env`
Reads `terrain_size` from `tensors[4].shape[-1]` (0 when no terrain).

## MLPNetwork Changes

- `forward`: `xs[:5]` instead of `xs[:4]` (includes terrain in flat concat)
- `from_env`: `obs_size = sum(t.numel() for t in tensors[:5])`

## PPOModel / PPO_Transformer

No changes needed — they pass the full tensor list through to the networks.

## Tests Added

### `tests/test_terrain_observation.py` (15 tests)

**TestTerrainObservation** (TERR-08):
- `test_terrain_obs_has_correct_token_count`
- `test_terrain_obs_carries_normalised_geometry`
- `test_terrain_tensor_shape`
- `test_terrain_tensor_batch_shape`
- `test_terrain_is_static_across_steps`

**TestNoTerrainBackwardCompat** (TERR-09):
- `test_no_terrain_tensor_has_zero_rows`
- `test_no_terrain_obs_is_empty`
- `test_no_terrain_observation_shape_unchanged`

**TestNetworkTerrainIntegration**:
- `test_transformer_no_terrain_has_no_embedding`
- `test_transformer_with_terrain_has_embedding`
- `test_transformer_forward_with_terrain`
- `test_transformer_value_with_terrain`
- `test_mlp_forward_with_terrain`
- `test_transformer_no_terrain_forward_unchanged`
- `test_player_token_positions_unchanged_with_terrain`

## Deviations

None.
