# PPO Self-Play + Elo

This project supports an AlphaZero-style PPO v1 workflow where the agent alternates between:

- **update epochs** against the base scripted opponent
- **self-play epochs** against frozen snapshots of itself

It also maintains an **Elo ladder** against scripted opponents and snapshot opponents.

## Defaults

- Self-play is enabled in `PPOTrainingConfig.self_play`.
- Activation waits until the reward curriculum reaches its **final phase**.
- Cadence is fixed by default to **3 update epochs : 1 self-play epoch**.
- Elo defaults:
- initial rating: `1000`
- K-factor: `32`
- match score from terminal VP:
  - `player_vp > opponent_vp` => `1.0`
  - `player_vp < opponent_vp` => `0.0`
  - tie => `0.5`

## Training Flow

1. PPO trains normally with reward phases.
2. Once final reward phase is active, the self-play callback activates.
3. At each epoch end, current PPO policy weights are saved to:
   - `checkpoints/<run_name>/self_play_snapshots/`
4. Epoch mode alternates:
   - update mode: restores base opponent policy from env config
   - self-play mode: sets opponent policy to `type: model` using one snapshot from the rolling pool
5. Elo ladder runs each epoch after activation and is persisted to:
   - `checkpoints/<run_name>/elo_ratings.json`

## Standalone Elo Evaluation

Use:

```bash
uv run evaluate_elo.py --checkpoint-path <path/to/checkpoint_or_snapshot.pt> --env-config-path <path/to/env.yaml>
```

Optional:

- `--episodes-per-opponent` to control match count per opponent
- `--snapshot-dir` to include snapshot opponents in the ladder
- `--output-json-path` to persist ratings output

`evaluate_elo.py` always includes scripted baselines:

- `random`
- `scripted_advance_to_objective`

and optionally includes all `*.pt` snapshots from `--snapshot-dir`.
