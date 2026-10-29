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


@pytest.fixture(autouse=True)
def pinned_cuda_implementation(request):
    """Run a test marked for one CUDA implementation under that implementation, whatever the default is."""
    from dace.config import set_temporary  # Imported late: collection must not depend on importing dace.

    if request.node.get_closest_marker("old_gpu_codegen_only"):
        impl = "legacy"
    elif request.node.get_closest_marker("new_gpu_codegen_only"):
        impl = "experimental"
    else:
        yield
        return
    with set_temporary("compiler", "cuda", "implementation", value=impl):
        yield
