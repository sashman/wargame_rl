# Requirements: Wargame RL - Self-Play Stabilization & League Training

**Defined:** 2026-04-04
**Core Value:** Agents learn recognisable tactical behaviour through reward shaping and environment design

## v1 Requirements

Requirements for the self-play stabilization milestone. Each maps to roadmap phases.

### Checkpoint Compatibility

- [ ] **SELF-04**: PPO snapshots persist policy architecture metadata needed to reconstruct the policy network safely
- [ ] **SELF-05**: `model` opponent policy validates snapshot/checkpoint compatibility before loading weights and emits actionable mismatch errors
- [ ] **SELF-06**: Self-play callback skips or quarantines incompatible snapshots instead of failing the epoch

### League Opponent Sampling

- [ ] **LEAG-01**: Self-play opponent selection uses weighted sampling across recent snapshots, high-Elo snapshots, and random historical snapshots
- [ ] **LEAG-02**: Snapshot retention policy keeps a bounded pool while preserving opponent diversity
- [ ] **LEAG-03**: Each training epoch logs selected opponent source and snapshot identifier for traceability

### Elo Evaluation Reliability

- [ ] **ELO-01**: Elo evaluation reports win rate, draw rate, and confidence information alongside ratings
- [ ] **ELO-02**: Elo evaluation supports deterministic seeding for reproducible comparisons
- [ ] **ELO-03**: `evaluate_elo.py` writes a structured JSON report schema with ratings and summary statistics
- [ ] **ELO-04**: Training-time Elo history is persisted per epoch for trend analysis

### CI & Observability

- [ ] **OPS-01**: CI smoke test runs a minimal PPO training loop that exercises self-play opponent swapping
- [ ] **OPS-02**: CI smoke test runs `evaluate_elo.py` against a generated checkpoint artifact
- [ ] **OPS-03**: Training logging includes self-play mode ratio, opponent mix, Elo trend, and snapshot pool size metrics
- [ ] **OPS-04**: Docs include a self-play failure and recovery runbook

## v2 Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Combat Mechanics

- **SHOT-01**: Models can select and resolve shooting attacks with tabletop hit/wound/save/damage flow
- **LOS-01**: Line-of-sight service gates valid shooting targets
- **CRWD-01**: Combat reward shaping balances damage/loss incentives against objectives

### Terrain & Advanced Rules

- **TERR-01**: Terrain types (open, cover, blocking, difficult) affect movement and combat
- **ADV-01**: Melee, morale, and command abilities are modeled in turn flow

## Out of Scope

| Feature | Reason |
|---------|--------|
| New combat mechanics in this milestone | Focus is stability and evaluation quality of self-play pipeline |
| Terrain system changes | Not required to validate self-play infrastructure |
| Distributed multi-process league training | Added operational complexity; defer until single-run league is stable |
| Large UI work beyond dashboards and docs | Milestone is training-system hardening, not product UI |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SELF-04 | Phase 7 | Pending |
| SELF-05 | Phase 7 | Pending |
| SELF-06 | Phase 7 | Pending |
| LEAG-01 | Phase 8 | Pending |
| LEAG-02 | Phase 8 | Pending |
| LEAG-03 | Phase 8 | Pending |
| ELO-01 | Phase 9 | Pending |
| ELO-02 | Phase 9 | Pending |
| ELO-03 | Phase 9 | Pending |
| ELO-04 | Phase 9 | Pending |
| OPS-01 | Phase 10 | Pending |
| OPS-02 | Phase 10 | Pending |
| OPS-03 | Phase 10 | Pending |
| OPS-04 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-04-04*
*Last updated: 2026-04-04 after milestone v1.1 scoping*
