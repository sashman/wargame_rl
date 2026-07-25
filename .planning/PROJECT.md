# Wargame RL

## What This Is

A reinforcement learning project that trains agents (DQN, PPO) to play tabletop miniature wargames on a discrete grid. Agents control multiple models (units) using polar-coordinate movement to capture objectives, maintain group cohesion, and eventually engage in combat. The goal is to produce agents whose emergent behaviour resembles the tactical reasoning a human player develops over many games.

## Core Value

Agents learn recognisable tactical behaviour — advancing on objectives, maintaining unit cohesion, using cover, focusing fire, retreating when outmatched — through reward shaping and environment design.

## Requirements

### Validated

- ✓ Discrete grid environment with configurable board size and deployment zones — existing
- ✓ Polar coordinate movement (16 angles × 6 speed bins, clamped to grid) — existing
- ✓ Multi-unit control with independent actions per model — existing
- ✓ Group cohesion reward penalty when units stray from group — existing
- ✓ Objective capture with distance-based reward and termination — existing
- ✓ DQN agent with MLP and Transformer (NanoGPT-style) networks — existing
- ✓ PPO actor-critic agent (default algorithm) — existing
- ✓ Reward phases / curriculum learning with success criteria and phase advancement — existing
- ✓ Victory Points scoring with configurable missions — existing
- ✓ YAML-driven environment configuration — existing
- ✓ Scripted and random opponent policies — existing
- ✓ Multi-phase turns (command, movement, shooting, charge, fight; non-movement phases skipped) — existing
- ✓ PyTorch Lightning training pipeline with replay buffer and epsilon decay — existing
- ✓ Wandb experiment tracking and video recording — existing
- ✓ Pygame human rendering with tooltips, arrows, panels — existing
- ✓ Multi-run parallel training with grouped Wandb logging — existing
- ✓ DDD-structured environment with BattleView protocol for read-only state access — existing
- ✓ Line-of-sight query on discrete grid (Bresenham, injectable blocking, `WargameEnv.has_line_of_sight_between_cells`) — Phase 3
- ✓ Shooting action space (target selection in ActionRegistry, phase-gated masks combining LOS/range/alive, WeaponProfile config) — Phase 4
- ✓ Shooting resolution (hit → wound → save → damage with stochastic D6 rolls, configurable weapon profiles and defensive stats, combat stats + expected damage in observation) — Phase 5
- ✓ Canonical game-state model (`GameStateSnapshot` from `BattleView`, JSON + JSON Schema, `WargameEnv.to_snapshot()`) — v9.0
- ✓ Bidirectional state I/O (`GameClock.set_state()`, `WargameEnv.load_state()`, `validate_snapshot()`, round-trip fidelity) — v9.0
- ✓ LLM-readable text representation (`StepNarrator`, public `describe_action()`, combat narrative with probabilities/expected damage, reward-phase context) — v9.0
- ✓ Append-only event stream with delta encoding and deterministic replay behind a pluggable codec interface — v9.0

### Active

Organised by target milestone. Each milestone builds on the previous — later milestones
assume earlier ones are complete. Milestones are created via `/gsd-new-milestone` when
the current one finishes.

#### v1.0 — Ranged Combat & Model Destruction [WIP · HIGH]

- ✓ Wounds & elimination (models with 0 wounds removed from play) — Phase 1
- ✓ Alive-aware observation (alive flags, wound status in tensor, no shape changes mid-episode) — Phase 2
- ✓ Line of sight service (Bresenham ray tracing, optional `blocking_mask`, domain `los.py`, env + render hooks) — Phase 3
- ✓ Shooting action space (target selection registered in ActionRegistry, phase-gated masks) — Phase 4
- ✓ Shooting resolution (hit → wound → save → damage with configurable weapon profiles) — Phase 5
- [ ] Combat reward & curriculum (damage dealt / models lost calculators, shooting curriculum phase)

#### Future milestones (not active)

| Milestone | Theme | Priority |
|-----------|-------|----------|
| v2.0 | Terrain & battlefield geometry (cover, blocking, elevation) | NEXT |
| v3.0 | Advanced movement & deployment (advance, fall back, reserves) | LOW |
| v4.0 | Weapon systems & attack modifiers (tags, area, mortal damage) | LOW |
| v5.0 | Morale & unit resilience (resolve, shaken, leadership) | LOWEST |
| v6.0 | Tactical resources & reactions (command points, overwatch) | LOWEST |
| v7.0 | Adversarial play & self-play (two-agent, Elo, opponent pool) | HIGH |
| v8.0 | Scale, missions & polish (10+ models, web replay) | LOWEST |

#### v9.0 — Structured game state & event streaming [✅ SHIPPED 2026-06-19]

Complete — see `.planning/milestones/v9-ROADMAP.md`. All 14 SGS-* requirements verified with
test evidence. Delivered: canonical `GameStateSnapshot`, bidirectional I/O, LLM-readable
narration, and an append-only event stream with deterministic replay.

#### Foundation (cross-cutting, slotted into milestones as needed)

- [ ] Positional encoding for transformer network
- [ ] Transformer shooting head: structural alignment between opponent tokens and target selection (avoid relying on learned implicit index correspondence via a single linear head)
- [ ] Attention masking for dead entities: prevent dead model tokens from diluting attention signal and wasting network capacity learning to ignore them
- [ ] Hyperparameter sweep tooling (Wandb Sweeps or Optuna)
- [ ] Improved metrics & dashboards (win rate, avg turns, reward breakdown, group violation rate)

### Out of Scope

Melee combat, transports, aircraft, psychic powers, faction-specific rules, army list building, multiplayer (3+), mobile/web deployment, real-time gameplay.

## Context

Brownfield project: working env, DQN + PPO algorithms, mature training pipeline. DDD structure in `envs/` with `BattleView` protocol. v1.0 (ranged combat) complete. v9.0 (structured state) shipped 2026-06-19. Next up: v2.0 (terrain) or v7.0 (self-play).

## Constraints

- **GPU preferred**: Training should target GPU acceleration; fall back to CPU via `CUDA_VISIBLE_DEVICES=""` when CUDA setup is broken
- **Python 3.13**: UV package manager, strict mypy, ruff formatting
- **Gymnasium 1.x**: Environment must conform to standard Gym API
- **Backward compatibility**: New config fields must default to no-op values so existing YAML configs keep working
- **Observation honesty**: Only expose information a real player would have (no perfect information)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Polar coordinate movement encoding | Uniform action space across all models; extends cleanly to per-model speed | ✓ Good |
| PPO as default algorithm over DQN | Better sample efficiency for multi-unit continuous-like problems | ✓ Good |
| Transformer (NanoGPT-style) as default network | Handles variable-length entity sequences; attention over models and objectives | ✓ Good |
| DDD structure in envs/ with BattleView protocol | Clean separation; reward/render consumers don't mutate state | ✓ Good |
| Registry pattern for reward calculators, criteria, opponents | YAML-extensible without code changes to core | ✓ Good |
| Reward phases for curriculum learning | Breaks sparse reward problem into learnable stages | ✓ Good |
| Skip non-movement phases by default | Keeps training fast until mechanics are implemented | ✓ Good |
| Phase 3 LOS: interior cells only for blocking; optional YAML `blocking_mask`; single `domain/los.py` | Matches tabletop-style trace; v2 terrain maps onto mask; no duplicate Bresenham in render | ✓ Good |
| v9.0: canonical `GameStateSnapshot` projected from `BattleView`, decoupled from RL observation tensors | Same data serves two consumers (RL pipeline + external/LLM) without coupling | ✓ Good |
| v9.0: default JSON encoding behind a pluggable codec interface (event log uses JSONL) | Extensible to binary/compact formats without changing the canonical model | ✓ Good |
| v9.0: combat results carry attacker-target pairing + analytical context (probabilities, expected damage) | Enables an LLM to judge whether the agent chose the optimal target | ✓ Good |
| v9.0: append-only event stream with delta encoding between full-snapshot anchors; deterministic replay | Compact history + reproducible seek/fast-forward from a known initial configuration | ✓ Good |

---
*Last updated: 2026-07-25*
