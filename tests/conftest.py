# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
pytest configuration file.
"""
import os

import pytest

# Per-worker DaCe build folder. Tests routinely name their SDFGs generically ("tester",
# "testing", "bypass1", or the default "main"), so two xdist workers building same-named SDFGs
# race each other in the shared .dacecache/<name>/build and one loads a half-linked library
# ("Could not load library libtester.so", "collect2: ld returned 1") -- a name collision that
# reads as a real test failure. PYTEST_XDIST_WORKER is gw0/gw1/... per worker and absent on
# serial runs, which keep the default .dacecache.
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    from dace.config import Config

    # A DACE_* env var OUTRANKS Config.set, so a caller isolating the build tree with
    # DACE_default_build_folder=<dir> would otherwise collapse every worker back onto one folder.
    # Suffix the env var itself so both that isolation and the per-worker split survive.
    _base = os.environ.get("DACE_default_build_folder")
    if _base:
        os.environ["DACE_default_build_folder"] = f"{_base}_{_worker}"
    Config.set("default_build_folder", value=f".dacecache_{_worker}")


@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # If running MPI tests and a failure has been detected, terminate the process to notify MPI to stop the other ranks
    if config.option.markexpr == 'mpi':
        if exitstatus in (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.INTERRUPTED):
            os._exit(1)


def pytest_generate_tests(metafunc):
    """
    This method sets up the parametrizations for the custom fixtures
    """
    if "use_cpp_dispatcher" in metafunc.fixturenames:
        metafunc.parametrize("use_cpp_dispatcher", [
            pytest.param(True, id="use_cpp_dispatcher"),
            pytest.param(False, id="no_use_cpp_dispatcher"),
        ])
