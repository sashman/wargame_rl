---
phase: 5
slug: shooting-resolution
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-06
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `uv run pytest tests/test_shooting_resolution.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -x` |
| **Estimated runtime** | ~120 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_shooting_resolution.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | SHOT-01 | unit+integration | `uv run pytest tests/test_shooting_resolution.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | SHOT-02 | unit | `uv run pytest tests/test_shooting_resolution.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | SHOT-04 | unit | `uv run pytest tests/test_shooting_resolution.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | SHOT-05 | unit | `uv run pytest tests/test_shooting_resolution.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | SHOT-06 | unit | `uv run pytest tests/test_shooting_resolution.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | OBS-02 | unit | `uv run pytest tests/test_shooting_resolution.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_shooting_resolution.py` — stubs for SHOT-01, SHOT-02, SHOT-04, SHOT-05, SHOT-06, OBS-02
- [ ] Fixtures in `tests/conftest.py` — weapon-equipped env config, seeded RNG

*Existing test infrastructure (pytest, conftest.py) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Stochastic D6 outcomes match tabletop probabilities over many rolls | SHOT-02 | Statistical — requires large sample size | Run 10k resolution calls with fixed stats, verify hit/wound/save rates within 2% of expected |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
