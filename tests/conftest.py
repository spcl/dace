# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
pytest configuration file.
"""

import os

import pytest

# One build folder per xdist worker: tests reuse SDFG names, and two workers building one name race
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    from dace.config import Config

    # A DACE_* variable outranks Config.set, so suffix the variable itself when it is set
    _base = os.environ.get("DACE_default_build_folder")
    _folder = f"{_base}_{_worker}" if _base else f".dacecache_{_worker}"
    if _base:
        os.environ["DACE_default_build_folder"] = _folder
    Config.set("default_build_folder", value=_folder)


@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # If running MPI tests and a failure has been detected, terminate the process to notify MPI to stop the other ranks
    if config.option.markexpr == "mpi":
        if exitstatus in (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.INTERRUPTED):
            os._exit(1)


def pytest_generate_tests(metafunc):
    """
    This method sets up the parametrizations for the custom fixtures
    """
    if "use_cpp_dispatcher" in metafunc.fixturenames:
        metafunc.parametrize(
            "use_cpp_dispatcher",
            [
                pytest.param(True, id="use_cpp_dispatcher"),
                pytest.param(False, id="no_use_cpp_dispatcher"),
            ],
        )
