---
phase: 1
slug: wounds-elimination
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | WOUND-01 | unit | `uv run pytest tests/test_wounds.py -k "test_wound_config" -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | WOUND-02 | unit | `uv run pytest tests/test_wounds.py -k "test_elimination" -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | WOUND-03 | unit | `uv run pytest tests/test_wounds.py -k "test_excluded" -x` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | WOUND-05 | integration | `uv run pytest tests/test_wounds.py -k "test_termination" -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_wounds.py` — stubs for WOUND-01, WOUND-02, WOUND-03, WOUND-05
- [ ] Shared fixtures in `tests/conftest.py` — wound-configured env fixture

*Existing test infrastructure covers framework and runner.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Renderer greys out eliminated models | N/A (visual) | Visual check only | Run `just simulate-latest` and verify dead models appear differently |

*Core wound/elimination logic is fully automatable.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
