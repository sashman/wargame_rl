# Wargame RL

## What This Is

A reinforcement learning project that trains agents (DQN, PPO) to play tabletop miniature wargames on a discrete grid. Agents control multiple models (units) using polar-coordinate movement to capture objectives, maintain group cohesion, and eventually engage in combat. The goal is to produce agents whose emergent behaviour resembles the tactical reasoning a human player develops over many games.

## Core Value

Agents learn recognisable tactical behaviour — advancing on objectives, maintaining unit cohesion, using cover, focusing fire, retreating when outmatched — through reward shaping and environment design.

## Current Milestone: v1.1 Self-Play Stabilization & League Training

**Goal:** Make PPO self-play and Elo evaluation reliable enough for day-to-day model iteration and promotion decisions.

**Target features:**
- Checkpoint/snapshot compatibility metadata and strict loader validation
- Robust model-opponent loading across checkpoint formats and network settings
- League-style opponent sampling (recent, strong, and diverse historical snapshots)
- Elo confidence reporting and reproducible evaluation runs
- CI smoke coverage for self-play and Elo CLI workflows
- Training dashboards for self-play mode mix, opponent mix, and Elo trend

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

### Active

- [ ] Persist architecture metadata in PPO snapshots/checkpoints and validate at load time
- [ ] Harden `model` opponent policy to handle checkpoint variants with explicit mismatch errors
- [ ] Add league opponent sampling strategy (recent + high-Elo + random historical)
- [ ] Add Elo reliability metrics (win rate, draw rate, uncertainty/confidence)
- [ ] Add CI smoke tests for self-play training loop and `evaluate_elo.py`
- [ ] Add dashboards/metrics for opponent mix, self-play ratio, and Elo trend

### Out of Scope

- Full army list building / points system — focus is on tactical play, not list construction
- Multiplayer (3+ players) — two-player only, per tabletop standard
- Mobile or web deployment of the training pipeline — local/CI training only
- Real-time gameplay — turn-based discrete steps only

## Context

This is a brownfield project with a working environment, two RL algorithms, and a mature training pipeline. The codebase follows DDD principles in the environment layer (`domain/` for rules, `BattleView` protocol for consumers). Extension points are well-documented in `docs/ddd-envs.md`.

Branch `feat/self-play` introduced a first pass of PPO self-play, snapshot opponents, and Elo evaluation tooling. This milestone focuses on production-hardening that pipeline: compatibility safety, opponent league quality, reproducible evaluation, and operational observability.

Combat/terrain roadmap items remain important but are deferred while self-play infrastructure is stabilized, because training quality and model comparison now depend on this pipeline.

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

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-04 after starting milestone v1.1*
