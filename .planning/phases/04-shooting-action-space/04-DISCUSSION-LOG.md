# Phase 4: Shooting Action Space - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 04-shooting-action-space
**Areas discussed:** Step semantics, Shooting target encoding, Weapon range for masking, Shooting application & recording

---

## Step Semantics with Shooting Phase

| Option | Description | Selected |
|--------|-------------|----------|
| One step per phase | Remove shooting from skip_phases. Agent gets separate step for movement and shooting. Two env.step() calls per turn. Reuses existing clock/registry/mask infrastructure. | ✓ |
| Bundled compound action | One env.step() takes movement + target selection. Single call per turn, product action space. Would require reworking action structure. | |

**User's choice:** One step per phase
**Notes:** Follows existing GameClock / ActionRegistry / phase-gated mask design exactly.

### Follow-up: Shooting phase activation

| Option | Description | Selected |
|--------|-------------|----------|
| Automatic | Env removes shooting from skip_phases when shooting actions are registered. | |
| Explicit YAML opt-in | User must configure skip_phases to exclude shooting. More control. | ✓ |

**User's choice:** Explicit YAML opt-in
**Notes:** Keeps backward compatibility clean — existing configs unchanged.

---

## Shooting Target Encoding

| Option | Description | Selected |
|--------|-------------|----------|
| One index per opponent model | Shooting slice has number_of_opponent_models indices. Fixed at init, masked at runtime. | ✓ |
| Fixed max targets with padding | Always N slots regardless of opponent count. Decouples from config but adds waste. | |

**User's choice:** One index per opponent model
**Notes:** Extended discussion on implications for transformer learning. Key concerns addressed:

- **Dynamic index list?** No — fixed at init, dead models masked out. Same pattern as movement.
- **Maps onto opponent observation?** Yes — positional alignment between observation slot K and action index K. Must be maintained as invariant.
- **Hinders transformer learning?** Learnable with flat encoding + positional alignment. Pointer-network cross-attention flagged as future enhancement if learning plateaus.

User requested explicit note about pointer-network attention as deferred enhancement.

---

## Weapon Range for Masking

| Option | Description | Selected |
|--------|-------------|----------|
| Single weapon_range field | One integer on ModelConfig. Simple, Phase 5 replaces with full profiles. | |
| Stub weapon profile list | weapons: list[WeaponProfile] on ModelConfig with range-only. Phase 5 fills remaining fields. Avoids config migration. | ✓ |

**User's choice:** Stub weapon profile list from day one
**Notes:** Aligns with non-uniform weapon counts across models (mostly 1-2, up to 7). Phase 5 adds fields to WeaponProfile rather than replacing a flat field.

### Follow-up: Required vs optional

| Option | Description | Selected |
|--------|-------------|----------|
| Optional with default | weapons defaults to empty list. Models without weapons can't shoot (all targets masked). Backward compatible. | ✓ |
| Required when shooting enabled | Validation error if shooting active but no weapons defined. | |

**User's choice:** Optional with empty list default

---

## Shooting Application & Recording

| Option | Description | Selected |
|--------|-------------|----------|
| Record in domain state | Store shooting_target on model entity. Phase 5 reads it. Self-describing state. | |
| Stateless — re-derive | Don't record. Phase 5 decodes target from the action int. No domain state change. | ✓ |

**User's choice:** Stateless
**Notes:** Extended discussion about future explainability architecture. User wants:

- Explicit record of targeting decisions with observation context (what transformer saw)
- Resolution (dice rolls) as separate observable events
- Precomputed attacker × defender probability matrices as both observation features and explainability tools
- These are perfect information (real players can calculate exhaustively)

All deferred — Phase 4 keeps dispatch simple. Event/decision log infrastructure recorded as future plan, feeds into v9.0 event streaming.

---

## Claude's Discretion

- Phase-aware dispatch mechanism (extend ActionHandler vs separate ShootingHandler)
- Distance metric for range check (match existing DistanceCache)
- WeaponProfile file location
- Shooting mask internal implementation

## Deferred Ideas

- Multi-weapon targeting via sub-steps within shooting phase (per weapon)
- Pointer-network cross-attention for target selection learning
- Decision/event log infrastructure for explainability
- Precomputed probability matrices (observation + explainability)
- Configurable fire/hold per weapon (default: all eligible weapons fire)
