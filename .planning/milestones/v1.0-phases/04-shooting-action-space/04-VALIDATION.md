---
phase: 4
slug: shooting-action-space
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `uv run pytest tests/ -x -q --tb=short` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q --tb=short`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | ACT-02 | unit | `uv run pytest tests/ -k "shooting" -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | ACT-03, LOS-03 | unit | `uv run pytest tests/ -k "shooting_mask" -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | ACT-04 | unit | `uv run pytest tests/ -k "action_space" -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | ACT-01, SHOT-03 | integration | `uv run pytest tests/ -k "phase_gate" -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_shooting_actions.py` — stubs for ACT-01 through ACT-04, LOS-03, SHOT-03
- [ ] Existing `tests/conftest.py` — extend fixtures with weapon-configured models

*Existing infrastructure covers test framework requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Shooting LOS overlay renders correctly | LOS-03 (visual) | Pygame rendering requires human inspection | Run `just test-env` with shooting config, press L to toggle LOS overlay, verify lines drawn to valid targets |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
