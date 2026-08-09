# Goals & Roadmap

## Vision

Train reinforcement learning agents that learn to play tabletop wargames — navigating units across a grid, capturing objectives, coordinating in groups, engaging in combat, and ultimately competing against opponent forces. The project aims to produce agents whose emergent behaviour resembles the tactical reasoning a human player develops over many games.

## Project Goals

1. **Faithful environment modelling** — Encode the core mechanics of miniature wargames (movement, shooting, melee, morale, terrain) as a Gymnasium environment so RL agents can interact with them through a standard API.

2. **Scalable training pipeline** — Maintain a PPO training loop that handles multi-unit action spaces, per-model credit assignment, and experiment tracking via Wandb.

3. **Emergent tactical behaviour** — Through reward shaping and environment design, encourage agents to learn recognisable tactics: advancing on objectives, maintaining unit cohesion, using cover, focusing fire, and retreating when outmatched.

4. **Extensible architecture** — Keep the codebase modular so new game phases (shooting, melee, morale) can be added without rewriting the observation/action pipeline.

5. **Human-readable replays** — Provide rendering and recording so that trained agents' games can be watched, analysed, and shared.

## What Exists Today

| Area | Status |
|------|--------|
| Discrete grid environment | Done — configurable board size, deployment zones |
| Polar coordinate movement | Done — 16 angles × 6 speed bins, clamped to grid |
| Multi-unit control | Done — independent actions per model |
| Group cohesion | Done — reward penalty when units stray from group |
| Objective capture | Done — distance-based reward + termination on capture |
| DQN (MLP) | Removed 2026-08-09 — unused for months |
| DQN (Transformer) | Removed 2026-08-09 — the NanoGPT-style network stayed, PPO trains it |
| Training pipeline | Done — Lightning, replay buffer, epsilon decay |
| Experiment tracking | Done — Wandb integration |
| Human rendering | Done — Pygame with tooltips, arrows, panels |
| Episode recording | Done — MP4 capture during training |
| Fixed & random placement | Done |
| YAML-driven env config | Done |
| Reward phases (curriculum) | Done — phased reward configs, success criteria, phase advancement (min_epochs, min_epochs_above_threshold), logged to Wandb |
| VP reward and success | Done — `vp_gain` calculator, `player_vp_min` success criteria, optional terminal VP bonus; observation includes `player_vp_delta` |
| PPO | Done — actor-critic with GAE, clipped surrogate; default algorithm |
| Per-model credit assignment | Done — IPPO-style: per-model reward vector, value and importance ratio, so each model is paid for its own contribution |
| Missions & VP scoring | Done — VP calculators + registry (see [missions-and-vp.md](missions-and-vp.md)) |
| Shooting | Done — target-selection action slice, phase-gated masking, D6 hit/wound/save resolution, wounds and elimination (see [shooting.md](shooting.md)) |
| Terrain & line of sight | Done — footprint ruins block LOS only; encoded in the observation (see [terrain.md](terrain.md)) |
| Opponent policies | Done — registry with `random`, `scripted_advance_to_objective`, `scripted_advance_and_shoot` (see [opponent-policies.md](opponent-policies.md)) |
| Scripted baselines | Done — `envs/baseline/` registry (`random`, `greedy_nearest`, `split_evenly`, `squad_march`, `squad_march_shoot`), scored through the same path as checkpoints and logged as `eval/baseline_*` |
| Batched lockstep evaluation | Done — eval episodes run in waves, one batched forward pass per step |
| Game-state I/O | Done — event log, replay, narration and match analysis (see [game-state-io.md](game-state-io.md)) |

## Roadmap

### Phase 1 — Strengthen the Foundation

Harden what already works before adding new game mechanics.

- [ ] **Per-model movement speed** — Let each model define its own `max_move_speed`; speed bins become fractions of that model's maximum. The action space stays uniform.
- [ ] **Positional encoding for transformer** — Add learned or sinusoidal positional encodings to the transformer network so it can distinguish token order.
- [ ] **Hyperparameter sweep tooling** — Add a `just sweep` target backed by Wandb Sweeps (or Optuna) for systematic hyperparameter search.
- [x] **Curriculum learning** — Reward phases with success criteria and phase advancement (see [reward-phases.md](reward-phases.md)).
- [ ] **Improved metrics & dashboards** — Win rate, reward components breakdown and scripted-baseline scores (`eval/baseline_*`) are now logged, and every key is documented in [metrics.md](metrics.md). Average turns to completion and group violation rate are still missing.

### Phase 2 — Combat: Shooting Phase

Introduce ranged attacks so models can damage each other.

- [x] **Wounds & elimination** — `max_wounds` / `current_wounds` are functional; models at 0 wounds are dead and masked out of actions, observations and reward.
- [x] **Shooting action type** — A shooting slice in the union action space; each model picks a target within range, resolved with D6 hit / wound / save rolls (see [shooting.md](shooting.md)).
- [x] **Line of sight** — Footprint-based LOS blocking in `domain/los.py`, with see-out / see-into exceptions (see [terrain.md](terrain.md)).
- [x] **Action type selection** — The action space is a phase-gated union of `stay`, `movement` and `shooting` slices; the valid slice is masked per battle phase rather than chosen freely.
- [x] **Reward shaping for combat** — `model_kills` (per-model) and `killing` (global) calculators pay for opponents killed; opponent shooting is available via `scripted_advance_and_shoot`. Penalising own losses is still not modelled.

### Phase 3 — Terrain & Board Features

Make the grid more than a flat plane.

- [ ] **Terrain types** — Ruins (footprint rectangles) exist and are encoded in the observation space (see [terrain.md](terrain.md)). Cover and difficult ground are not yet modelled.
- [ ] **Cover mechanics** — Models behind cover gain a defensive bonus during shooting resolution.
- [ ] **Difficult terrain** — Reduce movement speed when traversing difficult cells.
- [ ] **Blocking terrain** — Ruins already block line of sight; making them impassable to movement is outstanding.
- [ ] **Map generation** — Procedural or template-based board layouts for training variety.

### Phase 4 — Opponent AI & Self-Play

Move from single-agent objective capture to adversarial gameplay.

- [x] **Scripted opponent** — Rule-based opponents as a training baseline: `scripted_advance_to_objective` and `scripted_advance_and_shoot` (the only one that deliberately targets and fires — no env config uses it yet, and switching to it invalidates every score measured on that config).
- [ ] **Two-agent environment** — Refactor the env to support two sides, each controlling their own models, with alternating or simultaneous turns.
- [ ] **Self-play training** — Train the agent against copies of itself; periodically freeze opponents from the checkpoint pool.
- [ ] **Elo tracking** — Rate agent versions against each other to measure improvement over training.

### Phase 5 — Advanced Mechanics

Layer in the remaining tabletop systems.

- [ ] **Melee combat** — Close-range attacks when models are adjacent; higher damage, no LOS requirement.
- [ ] **Morale / suppression** — Units that take casualties test their nerve; failures cost them objective control. See [rules/01-core-concepts.md](rules/01-core-concepts.md#suppression).
- [ ] **Command abilities** — Special per-model actions (e.g. buff nearby allies, call in support) to increase tactical depth.
- [x] **Multi-phase turns** — Each `env.step()` advances one battle phase (command → movement → shooting → charge → fight). The opponent's full turn is auto-executed after the player completes theirs. Non-movement phases are skipped by default (`skip_phases` config) until their mechanics are implemented; set `skip_phases: []` for full per-phase stepping. Only movement has real actions currently; other phases allow only "stay".

### Phase 6 — Scale & Polish

- [x] **Larger scenarios** — 25v25 configs train and evaluate; evaluation runs in lockstep waves so a wave costs one batched forward pass per step.
- [x] **PPO** — Implemented and default algorithm. MAPPO, QMIX, etc. remain for future exploration.
- [ ] **Web replay viewer** — Browser-based replay viewer (replacing or complementing Pygame) for easier sharing.
- [ ] **Community scenarios** — Publish a library of env configs representing classic tabletop missions (hold the line, king of the hill, escort, etc.).

## Principles

- **Iterate in small, testable increments.** Each new mechanic should be trainable and observable in isolation before combining with others.
- **Reward shaping is a first-class concern.** Every new game mechanic needs a corresponding reward signal; otherwise the agent has no gradient to learn from.
- **Keep the observation space honest.** Only expose information a real player would have (no perfect information about hidden units, fog of war, etc.).
- **Prefer generality over shortcuts.** Design systems (action encoding, observation encoding, config schema) that extend cleanly rather than special-casing each new feature.
