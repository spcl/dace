# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Lint: external consumers must use ``CopyLibraryNode.INPUT_CONNECTOR_NAME`` etc., not hardcoded
``_cpy_in`` / ``_cpy_out`` / ``_fill_out`` literals; only the libnode definition files may own them."""
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Literal connector names whose external use is banned.
BANNED_LITERALS = ("_cpy_in", "_cpy_out", "_fill_out", "_fill_val")

# Files whose role is to *define* these names -- they are allowed to
# contain the literal strings as module-level constants and as namespaced
# C++ references inside generated tasklet bodies.
ALLOWED_FILES = {
    REPO_ROOT / "dace/libraries/standard/nodes/copy/node.py",
    REPO_ROOT / "dace/libraries/standard/nodes/copy/common.py",
    REPO_ROOT / "dace/libraries/standard/nodes/fill/node.py",
    REPO_ROOT / "dace/libraries/standard/nodes/fill/common.py",
    # This lint test itself mentions the literals.
    pathlib.Path(__file__).resolve(),
}

QUOTED_LITERAL = re.compile("['\"](?:" + "|".join(BANNED_LITERALS) + ")['\"]")

# Only the source trees of this checkout; an installed copy (a virtualenv, ``build/lib``) inside the
# repository would otherwise report the definition files themselves under another path.
SOURCE_TREES = ("dace", "tests", "samples", "tutorials")


def test_no_libnode_connector_literals_outside_definitions():
    """No repo ``.py`` file outside the libnode definition files contains a quoted ``_cpy_in`` /
    ``_cpy_out`` / ``_fill_out`` connector literal."""
    offenders = []
    for path in (p for tree in SOURCE_TREES for p in (REPO_ROOT / tree).glob("**/*.py")):
        if path in ALLOWED_FILES:
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
            if QUOTED_LITERAL.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, ("Hardcoded libnode connector literals found outside their "
                           "definition files. Use CopyLibraryNode.INPUT_CONNECTOR_NAME / "
                           "OUTPUT_CONNECTOR_NAME / FillLibraryNode.OUTPUT_CONNECTOR_NAME "
                           "instead:\n  " + "\n  ".join(offenders))
