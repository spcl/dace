"""Put the HLFIR test directory on sys.path so ``from _util import ...`` works
when pytest collects tests from this folder.

Also isolates DaCe's build cache directory per pytest-xdist worker so
parallel runs don't race on the shared ``.dacecache/<sdfg_name>/build``
directory — most HLFIR tests reuse the SDFG name ``main`` and would
otherwise clobber each other's CMake state under ``-n N``.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Per-worker DaCe build folder.  ``PYTEST_XDIST_WORKER`` is set by
# pytest-xdist to ``gw0``, ``gw1``, … on each worker process; absent on
# serial runs (we keep the default ``.dacecache`` so existing tooling
# behaves the same).
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    from dace.config import Config
    Config.set("default_build_folder", value=f".dacecache_{_worker}")
