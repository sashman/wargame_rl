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

### Active

Organised by target milestone. Each milestone builds on the previous — later milestones
assume earlier ones are complete. Milestones are created via `/gsd-new-milestone` when
the current one finishes.

#### v1.0 — Ranged Combat & Model Destruction (current milestone)

- ✓ Wounds & elimination (models with 0 wounds removed from play) — Phase 1
- ✓ Alive-aware observation (alive flags, wound status in tensor, no shape changes mid-episode) — Phase 2
- ✓ Line of sight service (Bresenham ray tracing, optional `blocking_mask`, domain `los.py`, env + render hooks) — Phase 3
- [ ] Shooting action space (target selection registered in ActionRegistry, phase-gated masks)
- [ ] Shooting resolution (hit → wound → save → damage with configurable weapon profiles)
- [ ] Combat reward & curriculum (damage dealt / models lost calculators, shooting curriculum phase)

#### v2.0 — Terrain & Battlefield Geometry

- [ ] Terrain cell types (open, light cover, dense cover, blocking, difficult ground)
- [ ] Cover bonus (+1 armour save against ranged attacks when in/behind cover)
- [ ] Dense terrain visibility (models wholly inside are never fully visible from outside)
- [ ] Blocking terrain (impassable, blocks line of sight)
- [ ] Difficult terrain (movement speed penalty when traversing)
- [ ] Elevation and height advantage (improved AP from elevated positions)
- [ ] Board templates and procedural terrain placement for training variety
- [ ] Terrain encoded in observation space
- [ ] Variable base sizes: models have different base radii affecting movement, coherency, engagement, objective control, and LOS (research spike — investigate impact on grid representation and DL training)

#### v3.0 — Advanced Movement & Deployment

- [ ] Advance move (move + random bonus, forfeits shooting that turn)
- [ ] Fall back move (disengage from threat zone, forfeits shooting that turn)
- [ ] Remain stationary bonus (heavy-class weapons get +1 accuracy when unit doesn't move)
- [ ] Threat zone / engagement range (1" zone around enemies — can't end normal move inside)
- [ ] Per-model movement speed (speed bins as fractions of each model's max)
- [ ] Reserve deployment (units arrive from board edges on turn 2+)
- [ ] Drop insertion (units arrive anywhere on board, >9" from enemies)
- [ ] Forward deployment (set up outside deployment zone before battle, >9" from enemies)
- [ ] Reconnaissance advance (free pre-battle move up to X")

#### v4.0 — Weapon Systems & Attack Modifiers

- [ ] Weapon tags: mobile-fire (shoot after advancing), braced (+1 accuracy when stationary), burst-fire (extra attacks at half range)
- [ ] Area weapons (bonus attacks proportional to target unit size)
- [ ] Close-range overcharge (extra damage at half range)
- [ ] Auto-hit weapons (spray/torrent — skip accuracy roll)
- [ ] Armour penetration tiers (weapons reduce saves by different amounts)
- [ ] Alternative saves / energy shields (unaffected by AP)
- [ ] Damage resistance (per-wound chance to ignore damage)
- [ ] Mortal damage (bypasses all saves, excess carries to next model)
- [ ] Multiple weapon loadouts per model with profile selection
- [ ] Weapon keywords: precision-allocation (target leaders in attached units), volatile (risk of self-damage), critical-bypass (critical wounds become mortal damage)

#### v5.0 — Morale & Unit Resilience

- [ ] Resolve test when below half strength (roll vs Leadership characteristic)
- [ ] Shaken status effects (objective control value = 0, restricted tactical options)
- [ ] Desperate escape (falling back while shaken risks losing additional models)
- [ ] Leadership characteristic per unit/model
- [ ] Recovery mechanic (shaken clears at start of next command phase)
- [ ] Stricter unit coherency enforcement (models must stay within 2" of each other; 7+ model units need 2 neighbours)

#### v6.0 — Tactical Resources & Reactions

- [ ] Command resources generated each command phase (1 per turn, capped)
- [ ] Tactical actions: spend resources for one-time effects during specific phases
- [ ] Re-roll action (spend 1 resource to re-roll any single dice)
- [ ] Opportunity fire (reactive shooting during opponent's movement, heavily reduced accuracy — hits on 6s only, once per turn)
- [ ] Emergency cover action (infantry gains temporary invulnerable save + cover for a phase)
- [ ] Smoke cover action (temporary concealment for a unit)
- [ ] Action economy as a learned strategic decision

#### v7.0 — Adversarial Play & Self-Play

- [ ] Two-agent environment with alternating full turns
- [ ] Self-play training against frozen checkpoint opponents
- [ ] Elo rating system for agent version comparison
- [ ] Opponent pool diversity (mix of scripted, frozen, and live opponents)
- [ ] Competitive reward design (win/loss, VP differential, margin of victory)

#### v8.0 — Scale, Missions & Polish

- [ ] Larger scenarios (10+ models per side with batched inference)
- [ ] Mission variety (different objective layouts, primary/secondary objectives)
- [ ] Progressive scoring (earn VP each turn for controlled objectives)
- [ ] Battle size configurations (small skirmish to full battle)
- [ ] Web replay viewer (browser-based, replacing/complementing Pygame)
- [ ] Community scenario library (env configs for classic missions)

#### Foundation (cross-cutting, slotted into milestones as needed)

- [ ] Positional encoding for transformer network
- [ ] Hyperparameter sweep tooling (Wandb Sweeps or Optuna)
- [ ] Improved metrics & dashboards (win rate, avg turns, reward breakdown, group violation rate)

### Out of Scope

- Full army list building / points system — focus is on tactical play, not list construction
- Multiplayer (3+ players) — two-player only, per tabletop standard
- Mobile or web deployment of the training pipeline — local/CI training only
- Real-time gameplay — turn-based discrete steps only
- Melee combat (charge & fight phases, pile in, consolidate) — extremely complex interaction model, deferred indefinitely
- Transports (embark/disembark mechanics) — niche mechanic, high implementation cost
- Aircraft (minimum move, forced reserves) — specialised sub-system, low RL training value
- Psychic powers / abilities — faction-specific, not core mechanics
- Faction-specific rules / detachments — keeps the domain generic

## Context

This is a brownfield project with a working environment, two RL algorithms, and a mature training pipeline. The codebase follows DDD principles in the environment layer (`domain/` for rules, `BattleView` protocol for consumers). Extension points are well-documented in `docs/ddd-envs.md`.

The project models a tabletop miniatures wargame with detailed rules (see `docs/tabletop-rules-reference.md`). The environment currently implements movement and objective control. The long-term vision spans 8 milestones (v1.0–v8.0) progressing from ranged combat through terrain, advanced movement, weapon diversity, morale, tactical resources, adversarial self-play, and scale. Melee combat is explicitly out of scope.

v1.0 (Ranged Combat & Model Destruction) is the active milestone — Phases 1–3 delivered; next is Phase 4 (shooting action space). Future milestones are captured in the Active requirements above and will be created via `/gsd-new-milestone` as each completes.

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
*Last updated: 2026-04-04 — Phase 3 complete; LOS validated; v1.0 progress through Phase 3*
