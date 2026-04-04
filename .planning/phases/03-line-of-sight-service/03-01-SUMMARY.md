---
phase: 03-line-of-sight-service
plan: 01
subsystem: envs domain + config + render
tags: [los, bresenham, blocking_mask]

requires: [03-CONTEXT.md, 03-RESEARCH.md]
provides:
  - "domain/los.py: iter_los_cells, has_line_of_sight (interior blocking only)"
  - "WargameEnvConfig.blocking_mask optional; shape validated vs board"
  - "WargameEnv.has_line_of_sight_between_cells / iter_los_cells_between_cells"
  - "HumanRender L toggles debug LOS polyline via iter_los_cells"
affects: [04-shooting-action-space]

requirements-completed: [LOS-01, LOS-02, LOS-04]

duration: session
completed: 2026-04-04
---

# Phase 03 Plan 01: Line of sight service — Summary

**Domain Bresenham LOS with injectable blocking; optional YAML mask; env helpers; human debug overlay (L).**

## Accomplishments

- Added `wargame_rl/wargame/envs/domain/los.py` with `_bresenham_line`, `iter_los_cells`, `has_line_of_sight` (no Gym imports).
- Re-exported LOS symbols from `domain/__init__.py`.
- `WargameEnvConfig.blocking_mask`: `list[list[bool]] | None`, YAML 0/1 coerced; `validate_blocking_mask_shape` enforces `(board_height, board_width)` with `mask[y][x]`.
- `WargameEnv`: `_make_is_blocking`, `has_line_of_sight_between_cells`, `iter_los_cells_between_cells`.
- `HumanRender`: `_debug_los` toggled with **L**; `_draw_debug_los_line` uses `iter_los_cells` between first alive player and opponent.
- `tests/test_los.py`: same cell, horizontal, blocked mid, diagonal, OOB, golden trace, endpoint interior-only semantics, env + mask integration, config default None.

## Verification

- `uv run pytest -n 0 -q` — 336 passed.
- `uv run mypy` on touched modules — clean.
- `uv run ruff check` on touched files — clean.

## Self-Check: PASSED
