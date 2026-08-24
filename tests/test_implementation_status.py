"""The gap map must stay true, and must fail the day one of its gaps closes.

`docs/rules/implementation-status.md` is the register of what this environment
does and does not implement. It is only worth having if it is *load-bearing*,
and a register that nobody checks rots silently — this repo has already shipped
a row rating "targets must be unengaged to be shot" as implemented when it was
not, and a `domain/fight.py` docstring asserting "chargers first" while the loop
sorted by group id and read no charge flag at all.

Two guards, and the first is the one that earns its keep:

* every `` `DEFERRED: <namespace>.<symbol>` `` tag names something that must not
  exist in `wargame_rl/`. The day somebody implements it, this test fails and
  names the row that has started lying. That is deliberately the opposite of the
  usual direction — the test protects a **gap**, not a feature.
* every row is well formed: a legal status, and a link that resolves to a real
  file and a real heading.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = REPO_ROOT / "docs" / "rules"
REGISTER = RULES_DIR / "implementation-status.md"
PACKAGE = REPO_ROOT / "wargame_rl"

LEGAL_STATUSES = frozenset({"implemented", "partial", "divergent", "absent"})

DEFERRED = re.compile(r"`DEFERRED: ([a-z_]+)\.([a-z_]+)`")
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)


def _rows() -> list[tuple[str, str, str]]:
    """Every three-column body row of the register, separators excluded."""
    out: list[tuple[str, str, str]] = []
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or set(cells[1]) <= set("-: "):
            continue
        if cells[1].lower() == "status":
            continue
        out.append((cells[0], cells[1], cells[2]))
    return out


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces hyphenated."""
    text = re.sub(r"`|\*|\[|\]|\(|\)", "", heading).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text.strip())


def _definition_patterns(symbol: str) -> list[re.Pattern[str]]:
    """Where an implementation of `symbol` would announce itself.

    A **definition**, never a mention: `fight.passing` would otherwise trip on
    the phrase "passing through an engagement range", which is prose in a
    docstring about a rule that IS implemented. What cannot be faked is
    something actually named after the gap — a function, a class, a config field
    or an attribute.
    """
    pascal = "".join(part.title() for part in symbol.split("_"))
    return [
        re.compile(rf"\bdef\s+_*{symbol}\b"),
        re.compile(rf"\bclass\s+_*{pascal}\b"),
        re.compile(rf"\bself\._*{symbol}\b"),
        re.compile(rf"^\s*_*{symbol}\s*[:=]", re.MULTILINE),
    ]


def test_every_deferred_gap_is_still_a_gap() -> None:
    """A `DEFERRED:` tag whose symbol now exists is a row that has started lying.

    ⚠ **If this fails, the fix is to update the register, not to rename the
    symbol.** Move the row to `implemented` or `partial`, say what remains, and
    delete or re-point the tag.
    """
    # Arrange
    text = REGISTER.read_text(encoding="utf-8")
    tags = DEFERRED.findall(text)
    sources = list(PACKAGE.rglob("*.py"))
    assert tags, "the register carries no DEFERRED tags, so this guard is vacuous"
    assert sources, "no package sources found to search"

    # Act
    landed: list[str] = []
    for namespace, symbol in tags:
        patterns = _definition_patterns(symbol)
        for source in sources:
            body = source.read_text(encoding="utf-8")
            if any(pattern.search(body) for pattern in patterns):
                landed.append(
                    f"{namespace}.{symbol} is defined in "
                    f"{source.relative_to(REPO_ROOT)}"
                )
                break

    # Assert
    assert not landed, (
        "these gaps have closed and the register still calls them absent: "
        + "; ".join(landed)
    )


def test_every_deferred_tag_is_unique() -> None:
    """Two rows sharing a tag means one of them can close unnoticed."""
    # Arrange / Act
    tags = DEFERRED.findall(REGISTER.read_text(encoding="utf-8"))

    # Assert
    assert len(tags) == len(set(tags)), f"duplicate DEFERRED tags: {tags}"


def test_a_deferred_tag_never_sits_on_an_implemented_row() -> None:
    """A tag names something that must not exist, so `implemented` contradicts it.

    `absent`, `partial` and `divergent` may all carry one: a divergent row's tag
    names the thing the environment deliberately does *not* do, which is exactly
    as worth a trip-wire as a gap.
    """
    # Arrange / Act
    offenders = [
        rule
        for rule, status, note in _rows()
        if DEFERRED.search(note) and _status(status) == "implemented"
    ]

    # Assert
    assert not offenders, f"DEFERRED tags on non-gap rows: {offenders}"


def _status(cell: str) -> str:
    """The row's status value, with any qualifier stripped.

    Two rows carry one — "divergent, deliberate" and "implemented
    (degenerately)" — and both say something the bare word does not. The value
    is the leading token either way.
    """
    bare = cell.replace("*", "").strip().lower()
    return re.split(r"[,(]", bare)[0].strip()


def test_the_register_is_well_formed() -> None:
    """Every row states a legal status and links to a heading that exists."""
    # Arrange
    rows = _rows()
    assert len(rows) > 50, "the register looks truncated"
    headings = {
        path.name: {_slug(match) for match in HEADING.findall(path.read_text("utf-8"))}
        for path in RULES_DIR.glob("*.md")
    }

    # Act
    bad_status = [
        rule for rule, status, _ in rows if _status(status) not in LEGAL_STATUSES
    ]
    broken: list[str] = []
    for rule, _status_cell, _note in rows:
        for target in LINK.findall(rule):
            path, _, anchor = target.partition("#")
            if not path or path.startswith("http"):
                continue
            if not (RULES_DIR / path).is_file():
                broken.append(f"{rule}: no such file {path}")
            elif anchor and anchor not in headings.get(path, set()):
                broken.append(f"{rule}: no heading #{anchor} in {path}")

    # Assert
    assert not bad_status, f"rows with an illegal status: {bad_status}"
    assert not broken, "rows linking nowhere: " + "; ".join(broken)
