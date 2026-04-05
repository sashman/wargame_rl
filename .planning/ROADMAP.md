# Roadmap: Shooting & Model Destruction

## Overview

This milestone adds combat to the wargame environment. Models gain durable wound state, a line-of-sight service enables target validity, the action space grows to include shooting, and combat rewards drive curriculum learning. The build order follows dependency: wounds first (shooting needs something to damage), LOS next (shooting needs validity checks), then action space, resolution, and finally rewards. Phases 2 and 3 are independent and can execute in parallel.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Wounds & Elimination** - Domain foundation: wound tracking, elimination logic, and termination on full wipe
- [x] **Phase 2: Alive-Aware Observation** - Observation pipeline handles eliminated models with alive flags and wound status
- [x] **Phase 3: Line of Sight Service** - Single domain service for Bresenham LOS reused by rules, masks, and rendering (completed 2026-04-04)
- [ ] **Phase 4: Shooting Action Space** - Extend ActionRegistry with shooting targets, phase-gated masks combining LOS/range/alive
- [ ] **Phase 5: Shooting Resolution** - Tabletop attack sequence (hit→wound→save→damage) with configurable weapon profiles
- [ ] **Phase 6: Combat Reward & Curriculum** - Reward calculators for damage/losses and curriculum phases for learning to shoot

## Phase Details

### Phase 1: Wounds & Elimination
**Goal**: Models have durable wound state that changes during an episode; eliminated models are removed from play
**Depends on**: Nothing (first phase)
**Requirements**: WOUND-01, WOUND-02, WOUND-03, WOUND-05
**Success Criteria** (what must be TRUE):
  1. A model configured with max_wounds=2 starts each episode with 2 wounds and the value can be reduced during play
  2. A model reduced to 0 wounds is eliminated — excluded from action selection, movement, and objective control
  3. The episode terminates when all models on one side are eliminated
  4. Existing YAML configs without wound settings still work with backward-compatible defaults
**Plans:** 2 plans
Plans:
- [x] 01-01-PLAN.md — Domain foundation: take_damage, is_alive, config default, termination extension, unit tests
- [x] 01-02-PLAN.md — Alive-filtering across env loop, env step wiring, integration tests

### Phase 2: Alive-Aware Observation
**Goal**: The RL agent can distinguish alive from eliminated models and see wound status in its observations
**Depends on**: Phase 1
**Requirements**: WOUND-04, OBS-01, OBS-03
**Success Criteria** (what must be TRUE):
  1. The observation tensor includes current_wounds/max_wounds for all models without shape changes mid-episode
  2. Eliminated models are clearly flagged (alive=0) so the policy distinguishes alive from dead
  3. A training run completes without observation shape mismatches when models are eliminated mid-episode
**Plans**:
- [x] `02-01-PLAN.md` — Types, `to_space`, observation builder + mask, tensor +3 features, tests, suite sweep

### Phase 3: Line of Sight Service
**Goal**: A single authoritative LOS query exists in the domain layer, reusable by rules, masks, and renderers
**Depends on**: Nothing (independent of Phase 2; depends on domain layer existing)
**Requirements**: LOS-01, LOS-02, LOS-04
**Success Criteria** (what must be TRUE):
  1. LOS queries correctly report visibility between any two grid positions using Bresenham ray tracing
  2. The LOS service is a single domain module callable from rules, action masks, and renderers
  3. LOS results are deterministic and tested against known board configurations with blocking cells
**Plans:** 1 plan
Plans:
- [x] `03-01-PLAN.md` — `domain/los.py`, tests, optional `blocking_mask`, env helpers, human LOS debug (L)

### Phase 4: Shooting Action Space
**Goal**: Models can select shoot-target actions during the shooting phase with correct validity masking
**Depends on**: Phase 1, Phase 2, Phase 3
**Requirements**: ACT-01, ACT-02, ACT-03, ACT-04, LOS-03, SHOT-03
**Success Criteria** (what must be TRUE):
  1. The action space includes shooting target indices registered as a new ActionRegistry slice
  2. Shooting actions are masked out in non-shooting phases; movement actions are masked out in shooting phase
  3. Action masks correctly filter shoot targets by LOS, weapon range, and target alive status
  4. Each model selects an action type per phase — move (movement), shoot (shooting), or stay (any)
**Plans:** 2 plans
Plans:
- [ ] 04-01-PLAN.md — Config types (WeaponProfile), ActionHandler shooting slice + phase-aware apply, unit tests
- [ ] 04-02-PLAN.md — Shooting mask function, observation builder overlay, env wiring, integration tests

### Phase 5: Shooting Resolution
**Goal**: Shooting actions resolve damage through the tabletop attack sequence with configurable weapons
**Depends on**: Phase 4
**Requirements**: SHOT-01, SHOT-02, SHOT-04, SHOT-05, SHOT-06, OBS-02
**Success Criteria** (what must be TRUE):
  1. A shoot action resolves via hit roll → wound roll → save → damage, applying wounds to the target
  2. Weapon profiles (range, attacks, BS, strength, AP, damage) are configurable per model in YAML
  3. Models that advanced cannot shoot; models in engagement range cannot shoot
  4. Weapon-relevant stats appear in the agent's observation for informed targeting decisions
**Plans**: TBD

### Phase 6: Combat Reward & Curriculum
**Goal**: The agent learns to use shooting effectively through reward shaping and curriculum progression
**Depends on**: Phase 5
**Requirements**: CRWD-01, CRWD-02, CRWD-03, CRWD-04
**Success Criteria** (what must be TRUE):
  1. A `damage_dealt` reward calculator is registered and configurable in YAML reward phases
  2. A `models_lost` penalty calculator is registered and configurable in YAML reward phases
  3. A curriculum phase exists where the agent learns to shoot before combining shooting with movement and objectives
  4. An agent trained with combat reward phases demonstrates shooting at valid targets during simulation
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6
(Phases 2 and 3 are independent and can execute in parallel)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Wounds & Elimination | 2/2 | Complete | (see phase detail) |
| 2. Alive-Aware Observation | 1/1 | Complete | (see phase detail) |
| 3. Line of Sight Service | 1/1 | Complete | 2026-04-04 |
| 4. Shooting Action Space | 0/0 | Not started | - |
| 5. Shooting Resolution | 0/0 | Not started | - |
| 6. Combat Reward & Curriculum | 0/0 | Not started | - |
