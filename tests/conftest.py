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


def pytest_collection_modifyitems(config, items):
    """Skip tests marked for one CUDA implementation when ``compiler.cuda.implementation`` selects the other."""
    from dace.config import Config  # Imported late: collection must not depend on importing dace.

    impl = Config.get("compiler", "cuda", "implementation")
    for item in items:
        if "old_gpu_codegen_only" in item.keywords and impl != "legacy":
            item.add_marker(pytest.mark.skip(reason="Requires the legacy CUDA codegen"))
        if "new_gpu_codegen_only" in item.keywords and impl != "experimental":
            item.add_marker(pytest.mark.skip(reason="Requires the experimental CUDA codegen"))
