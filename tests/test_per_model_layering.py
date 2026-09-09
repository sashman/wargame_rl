"""Dependency direction inside the domain and around the per-model facade, AST-walked.

The domain is one bounded context shaped as sub-domains, and the arrows only
point one way: the kernel imports nothing of the domain; each sub-domain
imports the kernel, the aggregate at the root, and the sub-domains listed for
it here; the root modules may name anything. Around it: `envs/per_model/`
never imports the phase facade, and only its named clients import it -- the
facade itself and the set network in `model/per_model/`, the first thing
seated on it. Each rule is checked by reading imports, as
`test_board_layer_is_a_leaf` does, so a violation names its file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ENVS = Path(__file__).resolve().parent.parent / "wargame_rl" / "wargame" / "envs"
DOMAIN = ENVS / "domain"
PACKAGE = ENVS.parent.parent
DOMAIN_PREFIX = "wargame_rl.wargame.envs.domain"

# Which sub-domains each may import, beside the kernel. `battlefield` is
# allowed to shooting for the sight it traces cover with; `movement` to melee
# because a charge, a pile-in and a consolidation are moves.
ALLOWED: dict[str, frozenset[str]] = {
    "kernel": frozenset(),
    "battlefield": frozenset(),
    "sequencing": frozenset(),
    "movement": frozenset(),
    "attacks": frozenset(),
    "shooting": frozenset({"attacks", "battlefield", "movement"}),
    "melee": frozenset({"attacks", "movement", "shooting"}),
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _sub_domain_of(module: str) -> str | None:
    """The sub-domain a dotted domain module belongs to, or None at the root."""
    rest = module[len(DOMAIN_PREFIX) :].lstrip(".")
    head = rest.split(".")[0] if rest else ""
    return head if head in ALLOWED else None


@pytest.mark.parametrize("sub_domain", sorted(ALLOWED))
def test_each_sub_domain_imports_only_the_kernel_and_what_it_is_allowed(
    sub_domain: str,
) -> None:
    """Arrange a sub-domain; act by walking every module's imports; assert each
    project import is the shared kernel `types/`, the domain kernel, or one of
    the sub-domains the table above allows."""
    offenders: dict[str, set[str]] = {}
    for path in _python_files(DOMAIN / sub_domain):
        bad = set()
        for module in _imported_modules(path):
            if not module.startswith("wargame_rl"):
                continue
            if module.startswith("wargame_rl.wargame.envs.types"):
                continue
            if not module.startswith(DOMAIN_PREFIX):
                bad.add(module)
                continue
            target = _sub_domain_of(module)
            if target is None:
                # The aggregate root, its factory and its view are the model
                # every domain service works on; only the kernel may not know
                # them, since they know the kernel.
                if sub_domain == "kernel":
                    bad.add(module)
            elif (
                target != sub_domain
                and target != "kernel"
                and target not in ALLOWED[sub_domain]
            ):
                bad.add(module)
        if bad:
            offenders[path.name] = bad
    assert not offenders, f"{sub_domain} imports upward or sideways: {offenders}"


def test_the_domain_never_imports_the_application() -> None:
    """No module under `domain/` imports anything of `envs/` but `types/`."""
    offenders = {
        path.relative_to(DOMAIN).as_posix(): {
            module
            for module in _imported_modules(path)
            if module.startswith("wargame_rl")
            and not module.startswith(DOMAIN_PREFIX)
            and not module.startswith("wargame_rl.wargame.envs.types")
        }
        for path in _python_files(DOMAIN)
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"domain modules importing the application: {offenders}"


def test_the_per_model_facade_never_imports_the_phase_facade() -> None:
    """The two facades share the domain and nothing else."""
    offenders = {
        path.name
        for path in _python_files(ENVS / "per_model")
        if "wargame_rl.wargame.envs.wargame" in _imported_modules(path)
    }
    assert not offenders, f"per_model modules importing wargame.py: {offenders}"


# The facade's clients, by package prefix relative to `wargame_rl/`. Stage 2's
# set network is the first; a new one is added here on purpose, never by
# naming a directory `per_model`.
FACADE_CLIENTS = ("wargame/envs/per_model", "wargame/model/per_model")
# And the two root drivers, by file name: the play script and the training
# loop. The repo root is walked too, so a new script cannot import the facade
# unlisted.
ROOT_CLIENTS = ("play_per_model.py", "train_per_model.py")
REPO = PACKAGE.parent


def test_only_the_named_clients_import_the_facade() -> None:
    """The facade is imported by its own package, the set network and the two
    root drivers, nothing else."""
    offenders = {
        path.relative_to(PACKAGE).as_posix()
        for path in _python_files(PACKAGE)
        if not path.relative_to(PACKAGE).as_posix().startswith(FACADE_CLIENTS)
        and any(
            module.startswith("wargame_rl.wargame.envs.per_model")
            for module in _imported_modules(path)
        )
    }
    offenders |= {
        path.name
        for path in REPO.glob("*.py")
        if path.name not in ROOT_CLIENTS
        and any(
            module.startswith("wargame_rl.wargame.envs.per_model")
            for module in _imported_modules(path)
        )
    }
    assert not offenders, f"per_model is imported by {offenders}"


def test_the_root_drivers_really_are_clients() -> None:
    """The root allow-list is not vacuous either."""
    for name in ROOT_CLIENTS:
        assert any(
            module.startswith("wargame_rl.wargame.envs.per_model")
            for module in _imported_modules(REPO / name)
        ), name


def test_the_set_network_package_is_a_client_of_the_facade() -> None:
    """The allow-list is not vacuous: the set network really does import it."""
    importers = {
        path.name
        for path in _python_files(PACKAGE / "wargame" / "model" / "per_model")
        if any(
            module.startswith("wargame_rl.wargame.envs.per_model")
            for module in _imported_modules(path)
        )
    }
    assert importers, "model/per_model imports nothing of envs/per_model"
