# Phase 2: Alive-Aware Observation - Discussion Log

> **Audit trail only.** Decisions live in `02-CONTEXT.md`.

**Date:** 2026-04-04
**Phase:** 2 — Alive-Aware Observation
**Areas discussed:** structured fields, observation action mask, tensor layout, checkpoint policy, testing

---

## 1. Structured observation fields

| Option | Description | Selected |
|--------|-------------|----------|
| A | `alive` float + `current_wounds` / `max_wounds` ints; keep real locations | ✓ |
| B | Wounds only, infer alive | |
| C | Sentinel location for dead | |

**User's choice:** all areas discussed — **A** (explicit alive + wound ints, no sentinel coords; distances from stepped cache with alive_mask).

**Notes:** Symmetric for opponents.

---

## 2. Observation action mask

| Option | Description | Selected |
|--------|-------------|----------|
| A | `alive_mask_for` → `get_model_action_masks` in `build_observation` | ✓ |
| B | Document-only; defer wiring | |

**User's choice:** **A** — must match `ActionHandler` / STAY-only for dead.

---

## 3. Tensor layout and normalization

| Option | Description | Selected |
|--------|-------------|----------|
| A | +3 floats: `alive`, `current/max_wounds`, `max_wounds/100` | ✓ |
| B | Raw integer channels in tensor | |

**User's choice:** **A** — append after existing per-model features; recompute `feature_dim`.

---

## 4. Checkpoint compatibility

| Option | Description | Selected |
|--------|-------------|----------|
| A | Breaking change; retrain | ✓ |
| B | Adapter / dual-head | |

**User's choice:** **A** — no shim in Phase 2.

---

## 5. Testing

| Option | Description | Selected |
|--------|-------------|----------|
| A | Env + tensor unit/integration tests required; rely on existing CI smoke | ✓ |
| B | Mandate new long training test in CI | |

**User's choice:** **A**.

---

## Claude's Discretion

Normalization constants and minor dtype edge cases left to implementation.

## Deferred ideas

OBS-02 weapons, checkpoint adapters, dropping redundant `alive` channel.
