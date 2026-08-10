"""Guard against references to the commercial product this project's rules derive from.

`docs/rules/` is a self-contained specification written for this project. It names no
product, publisher, edition or faction, and neither should anything else in the repo.
This test greps every tracked file for a denylist of such names and fails on any hit, so
a reference cannot creep back in through a docstring, a config comment or a planning
note.

The denylist necessarily *contains* the strings it forbids -- that is unavoidable for a
guard of this shape, and it is the one place they are allowed. It holds names only,
never rules text.

`reports/` is exempt: it records what was believed at the time of each experiment and is
not rewritten.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SELF = Path(__file__).relative_to(REPO_ROOT).as_posix()

EXEMPT_PREFIXES = ("reports/",)

EXEMPT_FILES = frozenset({SELF})

# Binary and lock files that carry hashes rather than prose.
EXEMPT_SUFFIXES = (".lock", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".ico", ".pt")

# Each entry is a regex matched case-insensitively against file contents. Word
# boundaries keep short markers from matching inside hashes and identifiers.
DENYLIST = (
    r"warhammer",
    r"\b40k\b",
    r"\b40,000\b",
    r"games workshop",
    r"wahapedia",
    r"chapter approved",
    r"\b1[01]e\b",  # edition markers
    r"\b1[01]th edition\b",
    r"astartes",
    r"space marine",
    r"primaris",
    r"intercessor",
    r"tyranid",
    r"necron",
    r"aeldari",
    r"drukhari",
    r"adeptus",
    r"\bimperium\b",
    r"\bxenos\b",
    # Rules jargon the reference deliberately replaced -- see docs/rules/00-glossary.md.
    # Only terms with no ordinary-English collision belong here; vocabulary drift in
    # descriptive prose is the glossary's job, not this guard's.
    r"battle-?shock",
    r"\bstratagem",
    r"\bdatasheet",
    r"mortal wound",
)

PATTERN = re.compile("|".join(DENYLIST), re.IGNORECASE)


def _tracked_files() -> list[str]:
    """Every file git would carry, as repo-relative POSIX paths.

    Untracked-but-not-ignored files are included deliberately: a reference added in a
    brand-new file must fail the guard immediately, not only once it is staged.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def _is_scanned(path: str) -> bool:
    """Whether a tracked path takes part in the scan."""
    if path in EXEMPT_FILES:
        return False
    if path.startswith(EXEMPT_PREFIXES):
        return False
    return not path.endswith(EXEMPT_SUFFIXES)


def test_no_product_references_in_tracked_files() -> None:
    """No tracked file names the product, publisher, edition or factions."""
    hits: list[str] = []

    for path in _tracked_files():
        if not _is_scanned(path):
            continue
        try:
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = PATTERN.search(line)
            if match:
                hits.append(f"{path}:{lineno}: {match.group(0)!r} in {line.strip()!r}")

    assert not hits, "Product references found in tracked files:\n" + "\n".join(hits)


def test_the_scan_actually_covers_the_repo() -> None:
    """A broken exemption or a failed git call must not turn the guard into a no-op."""
    scanned = [path for path in _tracked_files() if _is_scanned(path)]

    assert len(scanned) > 100
    assert any(path.startswith("docs/rules/") for path in scanned)
    assert any(path.startswith("wargame_rl/") for path in scanned)
