# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pytest fixtures + CLI flags for ``tests/corpus``.

Adds a single per-stage-dump option so the cloudsc parallelize chain can
be re-run manually and post-mortemed step-by-step without rebuilding the
full SDFG every time.

CLI options:

* ``--cloudsc-dump-dir=PATH`` -- if non-empty, save each chain-stage
  SDFG to ``PATH`` as ``<regime>_<stage_index>_<stage_name>.sdfgz``.
  Empty by default (no dump).
"""
import os
import pytest


def pytest_addoption(parser):
    parser.addoption('--cloudsc-dump-dir',
                     action='store',
                     type=str,
                     default='',
                     help=('Directory to dump each cloudsc chain-stage SDFG into '
                           '(filenames ``<regime>_<idx>_<stage>.sdfgz``). Empty (default) '
                           'disables dumping. The directory is created if missing.'))


@pytest.fixture
def cloudsc_dump_dir(request) -> str:
    """Resolved ``--cloudsc-dump-dir`` (created on first use); ``''`` if not set."""
    raw = request.config.getoption('--cloudsc-dump-dir') or ''
    if raw:
        raw = os.path.expanduser(raw)
        os.makedirs(raw, exist_ok=True)
    return raw
