# Requirements: Wargame RL — Shooting & Destruction

**Defined:** 2026-04-02
**Core Value:** Agents learn recognisable tactical behaviour through reward shaping and environment design

## v1 Requirements

Requirements for the shooting & model destruction milestone. Each maps to roadmap phases.

### Wounds & Elimination

- [x] **WOUND-01**: Each model has configurable max wounds and tracks current wounds during an episode
- [x] **WOUND-02**: Models reduced to 0 wounds are eliminated and removed from active play
- [ ] **WOUND-03**: Eliminated models are excluded from action selection, movement, and objective control
- [ ] **WOUND-04**: Observation space handles eliminated models gracefully (alive flags or padding, no shape changes mid-episode)
- [x] **WOUND-05**: Episode terminates when all models on one side are eliminated

### Shooting

- [x] **SHOT-01**: Models can select a shoot action targeting an enemy model within weapon range
- [x] **SHOT-02**: Shooting resolves via the tabletop attack sequence: hit roll → wound roll → save → damage
- [x] **SHOT-03**: Shooting is only valid during the shooting phase (phase-gated via action masks)
- [x] **SHOT-04**: Models that advanced or fell back cannot shoot (consistent with tabletop rules)
- [x] **SHOT-05**: Models in engagement range cannot shoot (locked in combat restriction)
- [x] **SHOT-06**: Weapon profiles are configurable per model (range, attacks, BS, strength, AP, damage)

### Line of Sight

- [x] **LOS-01**: A model can only shoot targets it has line of sight to *(query implemented Phase 3; enforcement Phase 4)*
- [x] **LOS-02**: LOS is computed via grid-based ray tracing (Bresenham) checking for blocking cells
- [x] **LOS-03**: LOS results are used in action masking so the agent cannot select invalid shoot targets
- [x] **LOS-04**: The LOS query is a single domain service reused by rules, masks, and rendering

### Action Space

- [x] **ACT-01**: Each model selects an action type per phase: move (movement phase), shoot (shooting phase), or stay (any phase)
- [x] **ACT-02**: Shooting actions are registered in ActionRegistry as a new slice with shooting-phase validity
- [x] **ACT-03**: Action masks combine phase validity, LOS, range, and model alive status
- [x] **ACT-04**: The total action space grows to accommodate shooting target indices alongside existing movement actions

### Combat Reward

- [ ] **CRWD-01**: A reward calculator rewards dealing damage to opponent models (registered in reward registry)
- [ ] **CRWD-02**: A reward calculator penalises losing own models to opponent fire
- [ ] **CRWD-03**: Combat reward is balanced against existing objective-capture incentives via reward phase weights
- [ ] **CRWD-04**: A curriculum phase exists where the agent learns to shoot before combining shooting with movement and objectives

### Observation

- [ ] **OBS-01**: Agent observation includes model wound status (current wounds / max wounds) for all visible models
- [x] **OBS-02**: Agent observation includes weapon profiles or combat-relevant stats for decision making
- [ ] **OBS-03**: Eliminated models are flagged in the observation so the policy can distinguish alive from dead

## v2 Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Terrain & Cover

- **TERR-01**: Terrain types encoded in board cells (open, cover, blocking, difficult)
- **TERR-02**: Cover mechanics grant defensive bonus during shooting resolution
- **TERR-03**: Difficult terrain reduces movement speed
- **TERR-04**: Blocking terrain is impassable and blocks LOS
- **TERR-05**: Procedural or template-based map generation

### Self-Play & Multi-Agent

- **SELF-01**: Two-agent environment with alternating turns
- **SELF-02**: Self-play via frozen checkpoint opponent pool
- **SELF-03**: Elo tracking for agent version comparison

### Advanced Mechanics

- **ADV-01**: Melee combat when models are adjacent
- **ADV-02**: Morale / battleshock from casualties
- **ADV-03**: Command abilities (per-model special actions)

### Foundation

- **FOUND-01**: Per-model movement speed
- **FOUND-02**: Positional encoding for transformer
- **FOUND-03**: Hyperparameter sweep tooling
- **FOUND-04**: Improved metrics & dashboards

### Scale & Polish

- **SCALE-01**: Larger scenarios (10+ models, batched inference)
- **SCALE-02**: Web replay viewer
- **SCALE-03**: Community scenario library

### Structured state & events (v9.0)

Deferred to v9.0. Roadmap phases TBD when the milestone is activated.

- **SGS-01**: A canonical programmatic game-state model exists (board, entities, phase, scoring, etc.), sourced from domain / read-only views, not tied to RL observation tensors
- **SGS-02**: Serialised state is suitable for external APIs and for LLM-facing validation (stable identifiers, documented semantics, explicit schema version)
- **SGS-03**: A layered change protocol expresses updates as full snapshots and/or granular deltas at defined abstraction levels to minimise redundancy
- **SGS-04**: Default encoding is JSON; a codec or encoder interface allows additional formats without changing the canonical model
- **SGS-05**: An append-only, ordered event stream can represent a complete match history for storage or streaming
- **SGS-06**: Replay is deterministic: events (optionally with periodic snapshots) applied from a known initial configuration reconstruct any requested historical state (fast-forward / seek)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full weapon keyword matrix (Melta, Blast, Lethal Hits, etc.) | Complexity; start with basic hit/wound/save/damage pipeline |
| Overwatch / reaction fire | Requires out-of-turn action framework; defer to advanced mechanics |
| Indirect fire (shooting without LOS) | Special rule; basic LOS-required shooting first |
| Fog of war / partial observability | Design decision deferred; current milestone assumes perfect information |
| Terrain interaction with shooting (cover saves) | Terrain is v2; LOS uses blocking cells only if terrain exists |
| Morale tests from shooting casualties | Morale system is v2 |
| Pistol weapons in engagement range | Edge case; basic "no shooting in engagement" rule first |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| WOUND-01 | Phase 1 | Complete |
| WOUND-02 | Phase 1 | Complete |
| WOUND-03 | Phase 1 | Pending |
| WOUND-04 | Phase 2 | Pending |
| WOUND-05 | Phase 1 | Complete |
| SHOT-01 | Phase 5 | Complete |
| SHOT-02 | Phase 5 | Complete |
| SHOT-03 | Phase 4 | Complete |
| SHOT-04 | Phase 5 | Complete |
| SHOT-05 | Phase 5 | Complete |
| SHOT-06 | Phase 5 | Complete |
| LOS-01 | Phase 3 | Complete (query; mask in Phase 4) |
| LOS-02 | Phase 3 | Complete |
| LOS-03 | Phase 4 | Complete |
| LOS-04 | Phase 3 | Complete |
| ACT-01 | Phase 4 | Complete |
| ACT-02 | Phase 4 | Complete |
| ACT-03 | Phase 4 | Complete |
| ACT-04 | Phase 4 | Complete |
| CRWD-01 | Phase 6 | Pending |
| CRWD-02 | Phase 6 | Pending |
| CRWD-03 | Phase 6 | Pending |
| CRWD-04 | Phase 6 | Pending |
| OBS-01 | Phase 2 | Pending |
| OBS-02 | Phase 5 | Complete |
| OBS-03 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 26 ✓
- Unmapped: 0
- v9.0 (structured state & events): 6 requirements (**SGS-01**–**SGS-06**); phases not yet defined

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-05 — added v9.0 SGS-* requirements (roadmap TBD)*
