# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
pytest configuration file.
"""
import os

import pytest


@pytest.fixture(scope='session', autouse=True)
def xdist_build_folder():
    """Give each xdist worker its own build directory.

    DaCe keys a build on the SDFG NAME and many tests reuse generic ones ("testing", "tester"), so two
    workers compiling same-named SDFGs race on one .dacecache entry and load a half-written .so. Sets
    the CONFIG rather than ``DACE_default_build_folder``: ``Config.get`` returns an env var before it
    consults the config, so exporting it would defeat every ``set_temporary('default_build_folder')``
    for the whole session. Serial runs are untouched.
    """
    worker = os.environ.get('PYTEST_XDIST_WORKER')
    if not worker:
        return
    from dace.config import Config
    Config.set('default_build_folder', value=os.path.join(Config.get('default_build_folder'), worker))


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


def _active_cuda_impl():
    # Imported lazily so pytest collection works even if the dace package can't be imported.
    from dace.config import Config
    return Config.get('compiler', 'cuda', 'implementation')


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked old_gpu_codegen_only / new_gpu_codegen_only based on the
    current ``compiler.cuda.implementation`` config value."""
    try:
        impl = _active_cuda_impl()
    except Exception:
        return  # If dace config is unavailable, don't interfere with collection.

    skip_old = pytest.mark.skip(reason="Requires legacy CUDA codegen (compiler.cuda.implementation=legacy)")
    skip_new = pytest.mark.skip(reason="Requires experimental CUDA codegen (compiler.cuda.implementation=experimental)")

    for item in items:
        if 'old_gpu_codegen_only' in item.keywords and impl != 'legacy':
            item.add_marker(skip_old)
        if 'new_gpu_codegen_only' in item.keywords and impl != 'experimental':
            item.add_marker(skip_new)
