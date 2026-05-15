"""Put the HLFIR test directory on sys.path so ``from _util import ...`` works
when pytest collects tests from this folder.

Also isolates DaCe's build cache directory per pytest-xdist worker so
parallel runs don't race on the shared ``.dacecache/<sdfg_name>/build``
directory  --  most HLFIR tests reuse the SDFG name ``main`` and would
otherwise clobber each other's CMake state under ``-n N``.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Per-worker DaCe build folder.  ``PYTEST_XDIST_WORKER`` is set by
# pytest-xdist to ``gw0``, ``gw1``, ... on each worker process; absent on
# serial runs (we keep the default ``.dacecache`` so existing tooling
# behaves the same).
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    from dace.config import Config
    Config.set("default_build_folder", value=f".dacecache_{_worker}")

import pytest

# --- f2py-reference teardown-crash guard ---------------------------------
# The e2e tests import f2py-compiled reference extension modules and never
# unload them.  At CPython finalisation numpy's teardown races those
# modules' deallocators, double-freeing a heap block -> SIGABRT (exit
# 134).  It happens strictly after every test ran AND pytest wrote its
# summary, so the verdict is already correct; the crash only corrupts the
# exit path and, under -q, ate buffered output that made results
# unreadable.  We record the real exit status at sessionfinish, then
# os._exit at pytest_unconfigure -- which runs LAST, after the terminal
# summary, exactly where the double-free otherwise fires -- skipping the
# crashing finaliser while preserving pytest's verdict and exit code.
_pytest_exitstatus = [0]


def pytest_sessionfinish(session, exitstatus):
    _pytest_exitstatus[0] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_pytest_exitstatus[0])
