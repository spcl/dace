# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Consumers use ``CopyLibraryNode.INPUT_CONNECTOR_NAME`` and the like; only the definition files own the literals."""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

BANNED_LITERALS = ("_cpy_in", "_cpy_out", "_fill_out", "_fill_val")

ALLOWED_FILES = {
    REPO_ROOT / "dace/libraries/standard/nodes/copy/node.py",
    REPO_ROOT / "dace/libraries/standard/nodes/copy/common.py",
    REPO_ROOT / "dace/libraries/standard/nodes/fill/node.py",
    REPO_ROOT / "dace/libraries/standard/nodes/fill/common.py",
    pathlib.Path(__file__).resolve(),
}

QUOTED_LITERAL = re.compile("['\"](?:" + "|".join(BANNED_LITERALS) + ")['\"]")

# An installed copy inside the repository would report the definition files under another path.
SOURCE_TREES = ("dace", "tests", "samples", "tutorials")


def test_no_libnode_connector_literals_outside_definitions():
    offenders = []
    for path in (p for tree in SOURCE_TREES for p in (REPO_ROOT / tree).glob("**/*.py")):
        if path in ALLOWED_FILES:
            continue
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

    assert not offenders, (
        "Hardcoded libnode connector literals found outside their "
        "definition files. Use CopyLibraryNode.INPUT_CONNECTOR_NAME / "
        "OUTPUT_CONNECTOR_NAME / FillLibraryNode.OUTPUT_CONNECTOR_NAME "
        "instead:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    test_no_libnode_connector_literals_outside_definitions()
