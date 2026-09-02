# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pin that ``cache: hash`` stays opt-in only.

``_sdfg_build_folder_getter`` (dace/sdfg/sdfg.py) names the ``hash`` policy's build folder from
``sdfg.hash_sdfg()``. Before that fix the key was ``md5(str(sdfg.to_json()))``, which was observed
unstable across two builds of the SAME, byte-identical-codegen program (folder names
``ae4c565b...`` vs ``b840eb49...``) because ``to_json()`` embeds a fresh ``uuid4`` ``guid`` on
every SDFG/state/node/edge construction. ``hash_sdfg()`` strips ``guid`` and is fixed, but nothing
on this branch actually runs under ``cache: hash`` -- the shipped default is ``name``, and every
CI workflow that sets ``DACE_cache`` explicitly picks ``unique`` or ``single``. These two guards
keep that true, so a policy change is a deliberate edit here rather than a silent drift.
"""
import glob
import os
import re
from typing import List, Tuple

import pytest

from dace.config import Config

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WORKFLOW_DIR = os.path.join(REPO_ROOT, '.github', 'workflows')

#: ``export DACE_cache=<value>`` (shell) or ``DACE_cache: <value>`` (workflow env block).
CACHE_ASSIGNMENT = re.compile(r'DACE_cache[=:]\s*[\'"]?(\w+)[\'"]?')


def cache_assignments() -> List[Tuple[str, int, str]]:
    """``(workflow, line number, value)`` for every ``DACE_cache`` assignment in the workflows."""
    found: List[Tuple[str, int, str]] = []
    workflows = glob.glob(os.path.join(WORKFLOW_DIR, '*.yml')) + glob.glob(os.path.join(WORKFLOW_DIR, '*.yaml'))
    for workflow in sorted(workflows):
        with open(workflow, 'r') as handle:
            for number, line in enumerate(handle, start=1):
                match = CACHE_ASSIGNMENT.search(line)
                if match:
                    found.append((os.path.basename(workflow), number, match.group(1)))
    return found


def test_cache_defaults_to_the_name_policy():
    """``hash`` is the unstable one; the shipped default must stay off it."""
    assert Config.get_default('cache') == 'name'


@pytest.mark.skipif(not os.path.isdir(WORKFLOW_DIR), reason='no .github/workflows in this checkout')
def test_ci_workflows_never_opt_into_the_hash_cache_policy():
    """Every workflow that sets ``DACE_cache`` must pick a policy other than ``hash``."""
    assignments = cache_assignments()
    assert assignments, 'no DACE_cache assignments found in the workflows -- the extractor is broken'

    offending = [f'{workflow}:{number} -> {value}' for workflow, number, value in assignments if value == 'hash']
    assert not offending, 'CI workflows opt into the unstable hash cache policy:\n  ' + '\n  '.join(offending)


if __name__ == '__main__':
    print("Must be called using `pytest`.")
