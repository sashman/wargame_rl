---
phase: quick-260404-plotting-removal
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - wargame_rl/plotting/
  - pyproject.toml
  - uv.lock
autonomous: true
requirements:
  - QUICK-260404-plotting-unused-removal
user_setup: []
must_haves:
  truths:
    - "No wargame_rl/plotting package remains in the repo"
    - "No production or test code imports wargame_rl.plotting"
    - "Direct dependencies used only by that code are removed from pyproject.toml"
  artifacts: []
  key_links:
    - from: "pyproject.toml"
      to: "uv.lock"
      via: "uv remove (regenerates lock)"
      pattern: "plotly|pandas|matplotlib"
---

<objective>
Remove the unused `wargame_rl/plotting/` package and drop direct dependencies that existed only to support it (`plotly`, `pandas`, `matplotlib`).

Purpose: Dead code and unused deps add maintenance noise and install weight; nothing in `tests/` or the package imports this module.

Output: Deleted `plotting/` tree; updated `pyproject.toml` and `uv.lock`.
</objective>

<execution_context>
@$HOME/.cursor/get-shit-done/workflows/execute-plan.md
@$HOME/.cursor/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

**Pre-check (planner):** `rg -i plotting wargame_rl tests train.py simulate.py main.py pyproject.toml` — only hits are `wargame_rl/plotting/training.py` and `pyproject.toml` listing plotly/pandas/matplotlib. No `from wargame_rl.plotting` or `wargame_rl.plotting` imports anywhere. **Do not** edit `.planning/` or `CLAUDE.md` unless a separate docs task is requested.

Follow project workflow: `just` for validate; use `uv remove` for deps (never hand-edit `uv.lock`).
</context>

<tasks>

<task type="auto">
  <name>Task 1: Delete plotting package and confirm no code references</name>
  <files>wargame_rl/plotting/__init__.py, wargame_rl/plotting/training.py</files>
  <action>Delete the entire directory `wargame_rl/plotting/` (including `__init__.py`, `training.py`, and `__pycache__` if present). Re-run search: `rg "wargame_rl\.plotting|from wargame_rl import plotting|plotting/training" wargame_rl tests` — expect no matches. If any match appears outside deleted paths, stop and report (should not happen per pre-check).</action>
  <verify>
    <automated>bash -lc 'cd /home/sash/Workspace/wargame_rl &amp;&amp; test ! -d wargame_rl/plotting &amp;&amp; ! rg -q "wargame_rl\\.plotting" wargame_rl tests'</automated>
  </verify>
  <done>Directory gone; ripgrep finds no `wargame_rl.plotting` references under `wargame_rl/` or `tests/`</done>
</task>

<task type="auto">
  <name>Task 2: Remove unused plotting-related direct dependencies</name>
  <files>pyproject.toml, uv.lock</files>
  <action>From project root, run `uv remove plotly pandas matplotlib` so `pyproject.toml` and `uv.lock` no longer list these as direct dependencies. They are unused elsewhere in Python sources after Task 1. If `uv remove` reports a conflict (unexpected), document and stop.</action>
  <verify>
    <automated>bash -lc 'cd /home/sash/Workspace/wargame_rl &amp;&amp; just validate'</automated>
  </verify>
  <done>`just validate` passes; no direct plotly/pandas/matplotlib entries remain in `[project] dependencies` unless still required by another first-party import (should not be)</done>
</task>

</tasks>

<verification>
- `just validate` (format, lint, tests) passes after both tasks.
- No `wargame_rl/plotting/` on disk.
</verification>

<success_criteria>
Plotting package removed; orphaned direct deps removed; CI-equivalent validation green.
</success_criteria>

<output>
After completion, create `.planning/quick/260404-soy-check-if-plotting-directory-is-used-remo/260404-soy-SUMMARY.md` (brief: what was deleted, deps removed, validate result).
</output>
