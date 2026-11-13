# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
pytest configuration file.
"""

import os

import pytest


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


@pytest.fixture
def ctypes_interface(monkeypatch):
    """Pins ``compiler.interface`` to ctypes for tests that assert ctypes-specific behavior.

    A comment at each use site names the per-file divergence.
    ``DACE_compiler_interface`` is dropped as well: the environment is
    re-applied whenever the configuration is LOADED (``temporary_config()``
    restores by loading), so the pin must outlive such a reload - see
    tests/compile_folder_mode_test.py.
    """
    import dace

    monkeypatch.delenv("DACE_compiler_interface", raising=False)
    with dace.config.set_temporary("compiler", "interface", value="ctypes"):
        yield
