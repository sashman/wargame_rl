---
phase: 03
slug: line-of-sight-service
status: draft
nyquist_compliant: false
wave_0_complete: true
created: 2026-04-04
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` / `pytest.ini` (project defaults) |
| **Quick run command** | `uv run pytest tests/test_los.py -q --tb=short` |
| **Full suite command** | `just validate` |
| **Estimated runtime** | ~60–120 seconds full suite |

---

## Sampling Rate

- **After every task commit:** `uv run pytest tests/test_los.py -q --tb=short` (when LOS files touched)
- **After every plan wave:** `just validate`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-T1 | 01 | 1 | LOS-02 | unit | `uv run pytest tests/test_los.py -q` | ⬜ | ⬜ pending |
| 03-01-T2 | 01 | 1 | LOS-02 | unit | `uv run pytest tests/test_los.py -q` | ⬜ | ⬜ pending |
| 03-01-T3 | 01 | 1 | LOS-04 | unit + lint | `uv run pytest tests/test_los.py -q` + `uv run ruff check wargame_rl/wargame/envs/wargame.py` | ⬜ | ⬜ pending |
| 03-01-T4 | 01 | 1 | LOS-04 | manual / optional | Visual: LOS overlay matches `iter_los_cells` | ⬜ | ⬜ pending |

*LOS-01 (shooting only with LOS) is enforced in Phase 4; Phase 3 supplies the query used by that phase.*

---

## Wave 0 Requirements

- [x] Existing pytest + `tests/` — no new framework
- [x] `tests/conftest.py` — reuse fixtures

*Existing infrastructure covers phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|---------------------|
| Pygame overlay alignment | LOS-04 | Pixel / visual | Enable debug LOS if implemented; eyeball grid vs `iter_los_cells` |

*If overlay task is deferred, mark N/A in execution SUMMARY.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or documented manual step
- [ ] Sampling continuity: LOS tests run after domain changes
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set in frontmatter after execution

**Approval:** pending
