---
phase: 1
slug: terrain-los-blocking
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-20
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `01-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (+ `hypothesis` for the LOS-symmetry property test) |
| **Config file** | `pyproject.toml` (pytest config); fixtures in `tests/conftest.py` |
| **Quick run command** | `uv run pytest tests/test_los.py tests/test_terrain.py -x` |
| **Full suite command** | `just test` (or `uv run pytest`) |
| **Estimated runtime** | ~30–60 seconds (quick: a few seconds) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_los.py tests/test_terrain.py -x`
- **After every plan wave:** Run `just test`
- **Before `/gsd-verify-work`:** `just validate` (format + lint + full test) must be green
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-* | 01 | 1 | TERR-01 | unit | `uv run pytest tests/test_los.py -k terrain_config -x` | ❌ W0 | ⬜ pending |
| 01-01-* | 01 | 1 | TERR-02 | unit | `uv run pytest tests/test_los.py -k terrain_validation -x` | ❌ W0 | ⬜ pending |
| 01-01-* | 01 | 1 | TERR-04 | unit | `uv run pytest tests/test_terrain.py -x` | ❌ W0 | ⬜ pending |
| 01-02-* | 02 | 2 | TERR-04 | integration | `uv run pytest tests/test_los.py -k terrain_los -x` | ❌ W0 | ⬜ pending |
| 01-02-* | 02 | 2 | TERR-07 | property | `uv run pytest tests/test_los.py -k symmetry -x` | ❌ W0 | ⬜ pending |
| 01-02-* | 02 | 2 | TERR-03 | regression | `uv run pytest tests/test_los.py -x` | ⚠️ extend | ⬜ pending |
| 01-02-* | 02 | 2 | TERR-06 | integration | `uv run pytest tests/test_shooting_resolution.py -k terrain -x` | ❌ W0 | ⬜ pending |
| 01-02-* | 02 | 2 | TERR-05 | integration | `uv run pytest tests/test_env.py -k terrain_movement -x` | ❌ W0 | ⬜ pending |
| 01-02-* | 02 | 2 | TERR-10 | smoke | `uv run pytest tests/test_render*.py -k terrain -x` (or verdict-helper unit) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_terrain.py` — `Footprint`/`Terrain` domain units: `contains` (corner-inclusive), corner normalisation, `blocking_footprints_for_endpoints` excludes footprints containing either endpoint (TERR-04 building blocks)
- [ ] `tests/test_los.py` — terrain golden boards: blocked / see-into / see-out / per-ruin / off-line + interior-only regression + `blocking_mask`+footprint co-existence (TERR-04, TERR-03)
- [ ] `tests/test_los.py` — LOS-symmetry `hypothesis` property test (TERR-07)
- [ ] `tests/test_los.py` (or `test_fixed_placement.py`) — config validation: off-board corner → `ValueError`, overlapping footprints → `ValueError`, overlap with zone/objective allowed, `terrain=None` default (TERR-01, TERR-02, TERR-03)
- [ ] `tests/test_shooting_resolution.py` — mask↔resolution consistency with terrain (TERR-06)
- [ ] `tests/test_env.py` — movement through/into footprint cells (TERR-05)
- [ ] (light) renderer verdict-colour / overlay-source check (TERR-10)
- [ ] Framework check: `uv run python -c "import hypothesis"`; if absent `uv add --dev hypothesis`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Footprint overlay visual appearance (translucent fill + outline + label) | TERR-10 | Pixel-exact pygame assertions are brittle | Run `just simulate-latest` (or a render of the demo terrain config) and eyeball the footprint overlay and verdict-coloured LOS line |

*Automated coverage asserts the renderer reads `view.terrain` and the verdict helper matches the LOS seam; only the visual styling is manual.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
