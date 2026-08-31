#!/usr/bin/env python3
"""Flag live documentation that the current branch may have invalidated.

Runs as a `PostToolUse` hook after `gh pr create` / `just ship` (registered in
`.claude/settings.json`) and returns its findings as `additionalContext`, so the
doc review happens while the branch is still checked out and cheap to amend.

The motivation is measured, not hypothetical: a full audit on 2026-08-04 found
~60 factual defects across 20 docs — a package in the layout tree that had been
deleted, a documented default of 100 against a real default of 1, a doc still
describing shooting as unimplemented a phase after it shipped. Every one was
trivial at the moment the code changed.

Deliberately stdlib-only and git-only so it never touches the uv environment,
and deliberately silent when nothing is implicated — a hook that speaks on every
PR gets ignored.

Usage: python3 .claude/hooks/docs_check.py [--dry-run] [<base>..<head>]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# Reference documentation, kept current. Anything not listed is out of scope.
LIVE_DOC_PATHS = (
    "README.md",
    "CLAUDE.md",
    "tests/CLAUDE.md",
    "wargame_rl/wargame/envs/CLAUDE.md",
    "wargame_rl/wargame/model/CLAUDE.md",
)
# ⚠ `**` and not `*`. `docs/*.md` does not recurse, so for as long as this hook
# has existed it has been blind to `docs/rules/` -- 20 of the 39 live docs,
# including `implementation-status.md`, the per-rule gap map that is this
# repo's register of what is NOT implemented. That is not an incidental gap: the
# one doc whose whole job is to say which rules are missing was the one doc the
# drift check could not read, which is how it came to rate "targets must be
# visible, in range and unengaged" as implemented when the unengaged clause was
# never written.
LIVE_DOC_GLOBS = ("docs/**/*.md",)

# `reports/` and `.planning/` are dated records of what was believed at the
# time. Editing them to match new code destroys their only value, so they are
# never reported as drift. `ratings/` is the same class: a rating ledger records
# what was measured under one code revision, and is superseded rather than
# edited.
FROZEN_PREFIXES = ("reports/", ".planning/", "configs/", "ratings/")

# Names too generic to implicate a doc on their own -- they appear in prose,
# or are language literals that survive the CamelCase test in
# `documented_symbols` (`True`, `False`, `None`) and would match every doc.
NOISE_SYMBOLS = frozenset(
    {
        "main", "run", "test", "setup", "config", "step", "reset", "name",
        "value", "data", "state", "model", "models", "env", "self", "type",
        "build", "apply", "select", "close", "render", "seed", "info",
        "true", "false", "none", "cls", "args", "kwargs", "list", "dict",
        "note", "text", "path", "file", "line", "lines", "case", "used",
        "justfile", "python", "bash", "json", "yaml",
    }
)  # fmt: skip

MIN_SYMBOL_LENGTH = 4

# Commands that mean "a PR just went up". `just ship` runs `gh pr create`
# inside the recipe, so the command text Claude ran is `just ship ...` and
# matching only on `gh` would miss every PR made the documented way.
#
# This filtering is done here rather than left to the `if` predicate in
# settings.json because that predicate was observed not to gate: every
# registered handler ran on every Bash call, firing the report on unrelated
# commands and firing it once per handler.
# Anchored to command positions -- start of line, or after a shell separator --
# rather than matching anywhere. An unanchored pattern fired on `git commit -m`
# whose message body merely quoted "gh pr create".
TRIGGER_PATTERN = re.compile(
    r"(?:^|&&|\|\||[;|\n])\s*(?:gh\s+pr\s+create|just\s+ship)\b"
)

# Backtick spans, `#` comments and quoted strings inside source are prose, not
# code. Without stripping them a comment explaining `n_episodes='25'` counts as
# "this branch touched n_episodes" and implicates three unrelated docs.
PROSE_IN_SOURCE = re.compile(
    r"`[^`]*`|#.*$|'''.*?'''|\"\"\".*?\"\"\"|'[^']*'|\"[^\"]*\""
)


def strip_prose(line: str) -> str:
    """Remove comment, string and backtick content from a line of source."""
    return PROSE_IN_SOURCE.sub(" ", line)


def git(*args: str) -> str:
    """Run a git command from the repo root, returning stdout (empty on error)."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def repo_root() -> Path | None:
    """Absolute path of the git repo containing the cwd, or None."""
    top = git("rev-parse", "--show-toplevel").strip()
    return Path(top) if top else None


def diff_range(explicit: str | None) -> str | None:
    """The commit range to inspect: the branch's divergence from main.

    Uses `merge-base` rather than a plain two-dot diff so that commits landing
    on main after the branch point are not mistaken for this branch's work.
    """
    if explicit:
        return explicit
    for main_ref in ("origin/main", "main"):
        base = git("merge-base", "HEAD", main_ref).strip()
        if base:
            head = git("rev-parse", "HEAD").strip()
            return f"{base}..{head}" if base != head else None
    return None


def changed_files(commit_range: str) -> list[str]:
    """Repo-relative paths changed in the range, including added and deleted."""
    output = git("diff", "--name-only", commit_range)
    return [line for line in output.splitlines() if line]


def added_removed_lines(commit_range: str, *paths: str) -> list[str]:
    """Added/removed diff lines for the given paths, with the +/- stripped."""
    args = ["diff", "-U0", commit_range]
    if paths:
        args += ["--", *paths]
    lines = []
    for line in git(*args).splitlines():
        if line[:1] in "+-" and line[:3] not in ("+++", "---"):
            lines.append(line[1:])
    return lines


def live_docs(root: Path) -> dict[str, str]:
    """Map every live doc's repo-relative path to its text."""
    paths = [root / p for p in LIVE_DOC_PATHS]
    for pattern in LIVE_DOC_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    docs = {}
    for path in paths:
        try:
            docs[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return docs


def ambiguous_basenames(root: Path) -> frozenset[str]:
    """Filenames that occur more than once in the repo (`registry.py`, ...).

    A doc citing a bare `registry.py` says nothing about *which* registry, so
    matching on those alone implicates every doc that mentions any of them.
    """
    counts: dict[str, int] = {}
    for line in git("ls-files", "*.py").splitlines():
        name = line.rsplit("/", 1)[-1]
        counts[name] = counts.get(name, 0) + 1
    return frozenset(name for name, count in counts.items() if count > 1)


def path_suffixes(path: str, ambiguous: frozenset[str]) -> list[str]:
    """Progressively shorter tail-ends of a path, longest first.

    Docs cite the same file inconsistently -- `ppo/lightning.py` in one place
    and `wargame_rl/wargame/model/net.py` in another -- so a suffix match is
    what actually finds the citations. The bare filename is dropped when it is
    not unique in the repo.
    """
    parts = path.split("/")
    suffixes = ["/".join(parts[i:]) for i in range(len(parts))]
    if parts[-1] in ambiguous:
        suffixes = suffixes[:-1]
    return suffixes


def source_symbols(commit_range: str) -> set[str]:
    """Public class, function and module-constant names touched by the diff.

    Tests are excluded: their local helpers are not API, and a nested `def
    targets(...)` in a test file matched prose in four unrelated docs.
    """
    symbols: set[str] = set()
    patterns = (
        re.compile(r"^\s*class\s+([A-Za-z_]\w*)"),
        re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"),
        re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*[:=]"),
    )
    for line in added_removed_lines(commit_range, "*.py", ":(exclude)tests/*"):
        for pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1)
            if (
                not name.startswith("_")
                and len(name) >= MIN_SYMBOL_LENGTH
                and name.lower() not in NOISE_SYMBOLS
            ):
                symbols.add(name)
    return symbols


def documented_symbols(text: str) -> set[str]:
    """Identifier-shaped code spans a doc has chosen to document.

    Restricted to snake_case and CamelCase so that ordinary prose in backticks
    (`random`, `win`) does not count as a documented symbol.
    """
    spans = re.findall(r"`([A-Za-z_][A-Za-z0-9_]{3,})`", text)
    return {
        s
        for s in spans
        if ("_" in s or not s.islower()) and s.lower() not in NOISE_SYMBOLS
    }


def touched_identifiers(commit_range: str) -> set[str]:
    """Every identifier appearing on a changed line of non-test source.

    Definitions alone are too narrow a signal: a function can keep its
    signature and still invalidate a doc by being called somewhere new. This
    is what catches `compute_shooting_masks` gaining an opponent-side caller.
    """
    identifiers: set[str] = set()
    for line in added_removed_lines(commit_range, "*.py", ":(exclude)tests/*"):
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", strip_prose(line)):
            if name.lower() not in NOISE_SYMBOLS:
                identifiers.add(name)
    return identifiers


def justfile_recipes(commit_range: str) -> set[str]:
    """Recipe names whose definition line was added or removed.

    Recipe bodies are indented, so anchoring at column 0 keeps this to
    definitions. Parameter defaults contain `=` (`n_episodes='25'`), so the
    argument list must not exclude it -- an earlier version did, and silently
    matched nothing.
    """
    pattern = re.compile(r"^([a-z][a-z0-9_-]*)(?:\s+[^:]*?)?:")
    recipes = set()
    for line in added_removed_lines(commit_range, "Justfile"):
        match = pattern.match(line)
        if match:
            recipes.add(match.group(1))
    return recipes


def config_fields(commit_range: str) -> set[str]:
    """Pydantic field names added or removed from the env config schema."""
    pattern = re.compile(r"^\s{4}([a-z]\w{3,}):\s")
    fields = set()
    for line in added_removed_lines(
        commit_range, "wargame_rl/wargame/envs/types/config.py"
    ):
        match = pattern.match(line)
        if match and match.group(1).lower() not in NOISE_SYMBOLS:
            fields.add(match.group(1))
    return fields


def mentions(text: str, needle: str) -> bool:
    """True if `needle` appears in `text` as a whole word."""
    return re.search(rf"(?<![\w/]){re.escape(needle)}(?![\w])", text) is not None


def collect_findings(root: Path, commit_range: str) -> dict[str, list[str]]:
    """Map each implicated live doc to the reasons it is implicated."""
    changed = changed_files(commit_range)
    source_changed = [
        f
        for f in changed
        if not f.startswith(FROZEN_PREFIXES)
        and (f.endswith(".py") or f in ("Justfile", "pyproject.toml"))
    ]
    if not source_changed:
        return {}

    docs = live_docs(root)
    ambiguous = ambiguous_basenames(root)
    findings: dict[str, list[str]] = {}

    def note(doc: str, reason: str) -> None:
        findings.setdefault(doc, [])
        if reason not in findings[doc]:
            findings[doc].append(reason)

    for source in source_changed:
        if not source.endswith(".py"):
            continue
        for suffix in path_suffixes(source, ambiguous):
            hits = [doc for doc, text in docs.items() if suffix in text]
            if hits:
                for doc in hits:
                    note(doc, f"cites `{suffix}`, which this branch changed")
                break

    for symbol in source_symbols(commit_range):
        for doc, text in docs.items():
            if mentions(text, symbol):
                note(doc, f"names `{symbol}`, added or removed here")

    touched = touched_identifiers(commit_range)
    for doc, text in docs.items():
        for symbol in sorted(documented_symbols(text) & touched):
            note(doc, f"documents `{symbol}`, which this branch touched")

    for recipe in justfile_recipes(commit_range):
        for doc, text in docs.items():
            if mentions(text, f"just {recipe}"):
                note(doc, f"documents `just {recipe}`, whose recipe changed")
        note("CLAUDE.md", f"Key Commands table may need `just {recipe}`")

    for field in config_fields(commit_range):
        for doc, text in docs.items():
            if mentions(text, field):
                note(doc, f"documents config field `{field}`, changed here")

    if any(f.startswith("tests/test_") for f in changed):
        added_or_deleted = git(
            "diff", "--name-status", "--diff-filter=AD", commit_range, "--", "tests/"
        )
        if added_or_deleted.strip():
            note("tests/CLAUDE.md", "test files were added or deleted (file map)")

    return findings


MAX_DOCS_REPORTED = 6


def format_report(findings: dict[str, list[str]], already: set[str]) -> str:
    """Render findings as the instruction handed back to Claude.

    Ranked by weight of evidence, because a doc implicated by four separate
    signals is far likelier to be genuinely stale than one that merely cites a
    file the branch happened to touch.
    """
    ranked = sorted(findings, key=lambda d: (-len(findings[d]), d))
    lines = [
        "Documentation drift check (automatic, after PR creation).",
        "",
        "These live docs reference code this branch changed:",
        "",
    ]
    for doc in ranked[:MAX_DOCS_REPORTED]:
        marker = " [already edited on this branch]" if doc in already else ""
        lines.append(f"- **{doc}**{marker}")
        for reason in findings[doc][:4]:
            lines.append(f"  - {reason}")
    dropped = ranked[MAX_DOCS_REPORTED:]
    if dropped:
        lines += [
            "",
            f"{len(dropped)} more matched on weaker evidence and are not listed: "
            + ", ".join(dropped)
            + ". Run `python3 .claude/hooks/docs_check.py --dry-run` for their reasons.",
        ]
    lines += [
        "",
        "Read each one and check whether the change made it wrong.",
        "",
        "**Fix directly** — mechanical drift where the correct text is not a "
        "judgement call: a renamed path or symbol, a changed default value, a "
        "changed CLI flag or recipe argument, or a new entry that belongs in a "
        "table or list the doc already maintains.",
        "",
        "**Only suggest — do not edit** — anything asserting behaviour, intent "
        "or consequence. A confidently wrong rewrite of a behavioural claim is "
        "worse than leaving it stale.",
        "",
        "Report what you fixed and what you are flagging. If a doc turns out to "
        "be fine, say so and move on — this check is a prompt to look, not "
        "evidence that something is wrong.",
        "",
        "`reports/` and `.planning/` are historical records and are exempt.",
    ]
    return "\n".join(lines)


def triggered_by_stdin() -> bool:
    """True when the hook payload describes a PR-creating Bash command.

    Reads the `PostToolUse` payload and matches `tool_input.command`. Anything
    unreadable or unmatched means stay quiet -- a docs reminder that fires on
    unrelated commands is worse than one that occasionally misses.
    """
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command")
    return isinstance(command, str) and TRIGGER_PATTERN.search(command) is not None


def main() -> int:
    """Emit `additionalContext` naming live docs this branch may have staled."""
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]

    if not dry_run and not triggered_by_stdin():
        return 0

    root = repo_root()
    if root is None:
        return 0

    commit_range = diff_range(args[0] if args else None)
    if commit_range is None:
        if dry_run:
            print("no commits on this branch versus main")
        return 0

    findings = collect_findings(root, commit_range)
    if not findings:
        if dry_run:
            print(f"{commit_range}: no live docs implicated")
        return 0

    already = {f for f in changed_files(commit_range) if f in findings}
    report = format_report(findings, already)

    if dry_run:
        print(f"range: {commit_range}\n")
        print(report)
        return 0

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": report,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - a docs reminder must never fail a ship
        sys.exit(0)
