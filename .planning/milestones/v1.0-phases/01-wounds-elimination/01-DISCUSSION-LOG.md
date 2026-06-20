# Phase 1: Wounds & Elimination - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 01-wounds-elimination
**Areas discussed:** Wound reduction trigger, Elimination representation, Default max_wounds value

---

## Wound Reduction Trigger

| Option | Description | Selected |
|--------|-------------|----------|
| Domain method + unit tests only | Add `take_damage(amount)` on WargameModel, test via unit tests. No in-env damage source until shooting (Phase 5). | ✓ |
| Scripted opponent damage | Extend opponent policy to deal damage each round for testing. Adds temporary scaffolding. | |
| Test-only env wrapper | Thin test harness calling `take_damage` at specific steps during integration tests. | |

**User's choice:** Domain method + unit tests only
**Notes:** Simplest approach. Keeps Phase 1 focused on data structure and domain logic. Shooting (Phase 5) is the first real consumer.

---

## Elimination Representation

| Option | Description | Selected |
|--------|-------------|----------|
| Flag-based (stay in list) | Model remains in lists with `is_alive` property (`current_wounds > 0`). Fixed array shapes. Downstream filters on alive. | ✓ |
| Removal from list | Model physically removed when eliminated. Observation shape changes mid-episode (breaks NN input). | |
| Separate alive/dead lists | Model moves to `player_eliminated` list. Dual bookkeeping. | |

**User's choice:** Flag-based (stay in list)
**Notes:** Strongly favored by WOUND-04 requirement ("no shape changes mid-episode") and transformer network's need for fixed-size entity sequences.

---

## Default max_wounds Value

| Option | Description | Selected |
|--------|-------------|----------|
| Keep 100 as default | Effectively invulnerable sentinel. Safe backward compat. | |
| Change default to 1 | Matches tabletop infantry. Existing configs get 1-wound models but safe since no damage source until Phase 5. | ✓ |
| Change default to 2 | Middle ground, elite infantry. Same compat concern as option 2. | |

**User's choice:** Change default to 1
**Notes:** Acceptable because no damage source exists until Phase 5. Combat configs will be purpose-built with explicit values. User confirmed existing configs being affected is fine.

---

## Claude's Discretion

- Clamping vs error behavior for `take_damage`
- Internal alive-filtering helpers (property on Battle vs utility function)
- Test structure and parameterization

## Deferred Ideas

None — discussion stayed within phase scope
