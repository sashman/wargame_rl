# Goals & Roadmap

## Vision

Train reinforcement learning agents that learn to play tabletop wargames — navigating units across a grid, capturing objectives, coordinating in groups, engaging in combat, and ultimately competing against opponent forces. The project aims to produce agents whose emergent behaviour resembles the tactical reasoning a human player develops over many games.

## Project Goals

1. **Faithful environment modelling** — Encode the core mechanics of miniature wargames (movement, shooting, melee, morale, terrain) as a Gymnasium environment so RL agents can interact with them through a standard API.

2. **Scalable training pipeline** — Maintain a DQN-based training loop (with room to swap in other algorithms) that handles multi-unit action spaces, experience replay, and experiment tracking via Wandb.

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
| DQN (MLP) | Done |
| DQN (Transformer) | Done — NanoGPT-style, default architecture |
| Training pipeline | Done — Lightning, replay buffer, epsilon decay |
| Experiment tracking | Done — Wandb integration |
| Human rendering | Done — Pygame with tooltips, arrows, panels |
| Episode recording | Done — MP4 capture during training |
| Fixed & random placement | Done |
| YAML-driven env config | Done |
| Reward phases (curriculum) | Done — phased reward configs, success criteria, phase advancement (min_epochs, min_epochs_above_threshold), logged to Wandb |
| VP reward and success | Done — `vp_gain` calculator, `player_vp_min` success criteria, optional terminal VP bonus; observation includes `player_vp_delta` |

## Roadmap

### Phase 1 — Strengthen the Foundation

Harden what already works before adding new game mechanics.

- [ ] **Per-model movement speed** — Let each model define its own `max_move_speed`; speed bins become fractions of that model's maximum. The action space stays uniform.
- [ ] **Positional encoding for transformer** — Add learned or sinusoidal positional encodings to the transformer network so it can distinguish token order.
- [ ] **Hyperparameter sweep tooling** — Add a `just sweep` target backed by Wandb Sweeps (or Optuna) for systematic hyperparameter search.
- [x] **Curriculum learning** — Reward phases with success criteria and phase advancement (see [reward-phases.md](reward-phases.md)).
- [ ] **Improved metrics & dashboards** — Track win rate, average turns to completion, reward components breakdown, and group violation rate as first-class Wandb metrics.

### Phase 2 — Combat: Shooting Phase

Introduce ranged attacks so models can damage each other.

- [ ] **Wounds & elimination** — Make `max_wounds` / `current_wounds` functional; models with 0 wounds are removed from play.
- [ ] **Shooting action type** — Add a shoot action alongside movement. Each model selects a target within range; resolve hits with configurable accuracy / damage.
- [ ] **Line of sight** — Implement raycasting or grid-based LOS checks so models cannot shoot through obstacles.
- [ ] **Action type selection** — Expand the action space so each model chooses *what* to do (move, shoot, or stay) and *how* (direction/speed or target).
- [ ] **Reward shaping for combat** — Reward dealing damage, penalise losing models, balance against objective-capture incentives.

### Phase 3 — Terrain & Board Features

Make the grid more than a flat plane.

- [ ] **Terrain types** — Define cells as open, cover, blocking, or difficult ground. Encode terrain in the observation space.
- [ ] **Cover mechanics** — Models behind cover gain a defensive bonus during shooting resolution.
- [ ] **Difficult terrain** — Reduce movement speed when traversing difficult cells.
- [ ] **Blocking terrain** — Impassable cells that also block line of sight.
- [ ] **Map generation** — Procedural or template-based board layouts for training variety.

### Phase 4 — Opponent AI & Self-Play

Move from single-agent objective capture to adversarial gameplay.

- [x] **Scripted opponent** — Rule-based opponent (e.g. advance-to-objective) as a training baseline.
- [ ] **Two-agent environment** — Refactor the env to support two sides, each controlling their own models, with alternating or simultaneous turns.
- [ ] **Self-play training** — Train the agent against copies of itself; periodically freeze opponents from the checkpoint pool.
- [ ] **Elo tracking** — Rate agent versions against each other to measure improvement over training.

### Phase 5 — Advanced Mechanics

Layer in the remaining tabletop systems.

- [ ] **Melee combat** — Close-range attacks when models are adjacent; higher damage, no LOS requirement.
- [ ] **Morale / battleshock** — Models that take casualties test morale; failures cause debuffs or retreat.
- [ ] **Command abilities** — Special per-model actions (e.g. buff nearby allies, call in support) to increase tactical depth.
- [x] **Multi-phase turns** — Each `env.step()` advances one battle phase (command → movement → shooting → charge → fight). The opponent's full turn is auto-executed after the player completes theirs. Non-movement phases are skipped by default (`skip_phases` config) until their mechanics are implemented; set `skip_phases: []` for full per-phase stepping. Only movement has real actions currently; other phases allow only "stay".

### Phase 6 — Scale & Polish

- [ ] **Larger scenarios** — Support 10+ models per side with efficient batched inference.
- [x] **PPO** — Implemented and default algorithm. MAPPO, QMIX, etc. remain for future exploration.
- [ ] **Web replay viewer** — Browser-based replay viewer (replacing or complementing Pygame) for easier sharing.
- [ ] **Community scenarios** — Publish a library of env configs representing classic tabletop missions (hold the line, king of the hill, escort, etc.).

## Principles

- **Iterate in small, testable increments.** Each new mechanic should be trainable and observable in isolation before combining with others.
- **Reward shaping is a first-class concern.** Every new game mechanic needs a corresponding reward signal; otherwise the agent has no gradient to learn from.
- **Keep the observation space honest.** Only expose information a real player would have (no perfect information about hidden units, fog of war, etc.).
- **Prefer generality over shortcuts.** Design systems (action encoding, observation encoding, config schema) that extend cleanly rather than special-casing each new feature.
