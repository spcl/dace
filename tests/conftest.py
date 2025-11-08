# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
pytest configuration file.
"""
import os

import pytest


@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # If running MPI tests and a failure has been detected, terminate the process to notify MPI to stop the other ranks
    if config.option.markexpr == 'mpi':
        if exitstatus in (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.INTERRUPTED):
            os._exit(1)


@pytest.hookimpl()
def pytest_collection_modifyitems(config, items):
    """Automatically skip tests marked with @pytest.mark.no_autoopt
    when auto-optimization is enabled."""
    import dace

    # 1. Determine if autooptimize is active
    autoopt_env = os.environ.get("DACE_optimizer_autooptimize")
    autoopt_enabled = False

    if autoopt_env is not None:
        autoopt_enabled = autoopt_env.lower() in ("1", "true", "yes", "on")
    elif dace is not None:
        try:
            autoopt_enabled = bool(dace.Config.get("optimizer", "autooptimize"))
        except Exception:
            autoopt_enabled = False

    # 2. If not enabled, nothing to skip
    if not autoopt_enabled:
        return

    # 3. Apply skip mark to all tests with @pytest.mark.no_autoopt
    skip_marker = pytest.mark.skip(reason="Skipped because autooptimize is enabled")
    skipped = 0
    for item in items:
        if "no_autoopt" in item.keywords:
            item.add_marker(skip_marker)
            skipped += 1

    if skipped:
        config.pluginmanager.get_plugin("terminalreporter").write_line(
            f"[pytest] autooptimize enabled — skipped {skipped} test(s) with @no_autoopt")
