"""Enforce D-010: the ground-truth oracle must never reach the agent's path.

The oracle reads engine sprite identities that do not exist on the private
evaluation set. Anything that consumes it would work in development and fail
completely at evaluation - the worst possible failure shape, because it looks
like success right up until it counts.

This is checked by static inspection of the source rather than at runtime, so a
violation fails CI whether or not the offending line ever executes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "arc3"

#: Only these may touch the oracle. Everything else in ``src/arc3`` is agent path.
ORACLE_CONSUMERS = {"oracle"}

#: Only the client may import the engine, so perception stays runnable from
#: recorded fixtures alone and cannot accidentally spend an action.
ENGINE_IMPORTERS = {"client", "oracle"}

ENGINE_MODULES = {"arc_agi", "arcengine"}


def _modules() -> list[tuple[Path, ast.Module]]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        out.append((path, ast.parse(path.read_text(encoding="utf-8"), str(path))))
    return out


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                names.add("." * node.level + (node.module or ""))
            elif node.module:
                names.add(node.module.split(".")[0])
                names.add(node.module)
    return names


def _package(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else ""


@pytest.mark.parametrize("path", [p for p, _ in _modules()], ids=lambda p: p.name)
def test_no_agent_module_imports_the_oracle(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    if _package(path) in ORACLE_CONSUMERS:
        return
    for name in _imported_names(tree):
        assert "oracle" not in name, (
            f"{path.relative_to(SRC)} imports {name!r}. The oracle is test-only "
            f"(D-010): it exposes sprite identities absent from the private set."
        )


@pytest.mark.parametrize("path", [p for p, _ in _modules()], ids=lambda p: p.name)
def test_only_the_client_imports_the_engine(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    if _package(path) in ENGINE_IMPORTERS:
        return
    for name in _imported_names(tree):
        assert name not in ENGINE_MODULES, (
            f"{path.relative_to(SRC)} imports {name!r}. Only arc3.client may "
            f"touch the engine, so perception stays runnable from recordings."
        )


def test_perception_imports_cleanly_without_the_engine(monkeypatch):
    """Perception must not need arc_agi even transitively."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ENGINE_MODULES:
            raise AssertionError(f"perception tried to import {name!r}")
        return real_import(name, *args, **kwargs)

    for mod in (
        "arc3.perception.components",
        "arc3.perception.geometry",
        "arc3.perception.diff",
        "arc3.perception.segment",
        "arc3.perception.track",
    ):
        importlib.import_module(mod)  # ensure it is importable at all

    monkeypatch.setattr(builtins, "__import__", guarded)
    for mod in (
        "arc3.perception.geometry",
        "arc3.perception.segment",
        "arc3.perception.track",
    ):
        importlib.reload(importlib.import_module(mod))


def test_oracle_module_documents_its_own_restriction():
    text = (SRC / "oracle" / "sprites.py").read_text(encoding="utf-8")
    assert "D-010" in text
    assert "TEST" in text.upper()
