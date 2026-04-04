# Roadmap: Self-Play Stabilization & League Training

## Overview

This milestone hardens the newly introduced PPO self-play and Elo tooling so model progression decisions are reliable. The order follows dependency: checkpoint compatibility first, then better league opponent sampling, then Elo/statistical reliability, then CI/observability guardrails. Phase numbering continues from the prior milestone and starts at Phase 7.

## Phases

**Phase Numbering:**
- Integer phases (7, 8, 9, 10): Planned milestone work
- Decimal phases (7.1, 7.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 7: Snapshot Compatibility Hardening** - Make snapshot/checkpoint loading safe and architecture-aware
- [ ] **Phase 8: League Opponent Sampling** - Improve self-play opponent quality with weighted league selection and pool retention
- [ ] **Phase 9: Elo Reliability & Reporting** - Add reproducible evaluation and richer statistics around Elo ratings
- [ ] **Phase 10: CI & Operational Observability** - Add smoke tests, metrics, and runbooks for reliable operations

## Phase Details

### Phase 7: Snapshot Compatibility Hardening
**Goal**: Ensure self-play and model-opponent loading never fail silently or crash due to checkpoint format and architecture mismatches
**Depends on**: Nothing (first phase)
**Requirements**: SELF-04, SELF-05, SELF-06
**Success Criteria** (what must be TRUE):
  1. Saved snapshots include required policy metadata (architecture and shape-relevant fields)
  2. `model` opponent policy reconstructs compatible policies correctly and reports precise mismatch reasons when incompatible
  3. Self-play callback handles incompatible snapshots gracefully (skip/quarantine/log) without aborting training
**Plans**: TBD

### Phase 8: League Opponent Sampling
**Goal**: Raise training opponent quality and diversity with league-aware sampling and bounded pool management
**Depends on**: Phase 7
**Requirements**: LEAG-01, LEAG-02, LEAG-03
**Success Criteria** (what must be TRUE):
  1. Opponent selection supports weighted categories (recent, high-Elo, random historical)
  2. Snapshot retention policy enforces pool limits while preserving diversity
  3. Training logs expose selected opponent source and snapshot id per epoch
**Plans**: TBD

### Phase 9: Elo Reliability & Reporting
**Goal**: Make Elo comparisons reproducible, interpretable, and suitable for model promotion decisions
**Depends on**: Phase 8
**Requirements**: ELO-01, ELO-02, ELO-03, ELO-04
**Success Criteria** (what must be TRUE):
  1. Elo outputs include rating plus win/draw rates and confidence information
  2. Evaluations are reproducible when a fixed seed is supplied
  3. `evaluate_elo.py` emits structured JSON reports with summary stats
  4. Training stores epoch-indexed Elo history for trend plotting and analysis
**Plans**: TBD

### Phase 10: CI & Operational Observability
**Goal**: Protect self-play pipeline quality with automated smoke checks and explicit operational diagnostics
**Depends on**: Phase 7, Phase 8, Phase 9
**Requirements**: OPS-01, OPS-02, OPS-03, OPS-04
**Success Criteria** (what must be TRUE):
  1. CI includes a smoke path exercising self-play opponent swaps in PPO training
  2. CI includes a smoke path for `evaluate_elo.py` on generated checkpoints
  3. Logged metrics and dashboards show self-play ratio, opponent mix, Elo trend, and pool size
  4. Documentation provides a concrete self-play troubleshooting and recovery runbook
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 7 -> 8 -> 9 -> 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 7. Snapshot Compatibility Hardening | 0/0 | Not started | - |
| 8. League Opponent Sampling | 0/0 | Not started | - |
| 9. Elo Reliability & Reporting | 0/0 | Not started | - |
| 10. CI & Operational Observability | 0/0 | Not started | - |
