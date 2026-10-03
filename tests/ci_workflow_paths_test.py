# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every test path a CI workflow hands to ``pytest`` must exist in the tree.

``pytest`` exits with the usage error 4 when a positional path does not exist, so one stale path turns a
whole workflow into a red step that collects nothing -- and a workflow nobody watches turns silently
vacuous instead. Renaming or moving a test file cannot be caught by running the tests, because the
workflow that names it is the thing that breaks. This guard closes that gap for every workflow at once.
"""
import glob
import os
import re
from typing import List, Tuple

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WORKFLOW_DIR = os.path.join(REPO_ROOT, '.github', 'workflows')

#: A repo-relative test path: ``tests/...`` not preceded by another path character, so ``./pyFV3/tests/x``
#: (a checked-out foreign repository) is correctly left alone.
TEST_PATH = re.compile(r'(?<![\w./-])tests/[\w./*-]+')


def referenced_paths() -> List[Tuple[str, int, str]]:
    """``(workflow, line number, path)`` for every ``tests/`` path on a ``pytest`` command line."""
    found: List[Tuple[str, int, str]] = []
    workflows = glob.glob(os.path.join(WORKFLOW_DIR, '*.yml')) + glob.glob(os.path.join(WORKFLOW_DIR, '*.yaml'))
    for workflow in sorted(workflows):
        with open(workflow, 'r') as handle:
            lines = handle.read().split('\n')
        # A shell line continuation splits one pytest invocation over several YAML lines; join them and
        # report the command's first line.
        command = ''
        start = 1
        for number, line in enumerate(lines, start=1):
            stripped = line.rstrip()
            if not command:
                start = number
            if stripped.endswith('\\'):
                command += stripped[:-1] + ' '
                continue
            command += stripped
            if 'pytest' in command:
                found += [(os.path.basename(workflow), start, path) for path in TEST_PATH.findall(command)]
            command = ''
    return found


@pytest.mark.skipif(not os.path.isdir(WORKFLOW_DIR), reason='no .github/workflows in this checkout')
def test_ci_workflows_reference_existing_test_paths():
    """A workflow naming a moved or deleted test file is a broken workflow."""
    referenced = referenced_paths()
    assert referenced, 'no pytest test paths found in the workflows -- the extractor is broken'

    missing: List[str] = []
    for workflow, number, path in referenced:
        target = os.path.join(REPO_ROOT, path)
        present = bool(glob.glob(target)) if '*' in path else os.path.exists(target)
        if not present:
            missing.append(f'{workflow}:{number} -> {path}')
    assert not missing, 'CI workflows name test paths that do not exist:\n  ' + '\n  '.join(missing)
