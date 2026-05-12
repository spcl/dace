# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lint: no hardcoded ``_cpy_in`` / ``_cpy_out`` / ``_mset_out`` literals
outside the libnode bodies that define them.

The intent: external consumers (tests, transformations, codegen) must
reference ``CopyLibraryNode.INPUT_CONNECTOR_NAME`` /
``CopyLibraryNode.OUTPUT_CONNECTOR_NAME`` /
``MemsetLibraryNode.OUTPUT_CONNECTOR_NAME`` — so a future rename is a
single-line change.

Allowed: the libnode files themselves
(``dace/libraries/standard/nodes/{copy,memset}_node.py``) own the literal
strings via module-level constants.
"""
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Literal connector names whose external use is banned.
_BANNED_LITERALS = ("_cpy_in", "_cpy_out", "_mset_out")

# Files whose role is to *define* these names — they are allowed to
# contain the literal strings as module-level constants and as namespaced
# C++ references inside generated tasklet bodies.
_ALLOWED_FILES = {
    REPO_ROOT / "dace/libraries/standard/nodes/copy_node.py",
    REPO_ROOT / "dace/libraries/standard/nodes/memset_node.py",
    # This lint test itself mentions the literals.
    pathlib.Path(__file__).resolve(),
}

_QUOTED_LITERAL = re.compile(r"['\"](?:_cpy_in|_cpy_out|_mset_out)['\"]")


def test_no_libnode_connector_literals_outside_definitions():
    offenders = []
    for path in REPO_ROOT.glob("**/*.py"):
        if path in _ALLOWED_FILES:
            continue
        # Skip caches and external trees.
        rel = path.relative_to(REPO_ROOT)
        if any(part in {".dacecache", "external", ".git"} for part in rel.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _QUOTED_LITERAL.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Hardcoded libnode connector literals found outside their "
        "definition files. Use CopyLibraryNode.INPUT_CONNECTOR_NAME / "
        "OUTPUT_CONNECTOR_NAME / MemsetLibraryNode.OUTPUT_CONNECTOR_NAME "
        "instead:\n  " + "\n  ".join(offenders))
