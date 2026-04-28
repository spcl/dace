# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
pytest configuration file.
"""
import hashlib
import os
import re

import pytest

# Sanitizer for pytest test names → C identifiers. ``request.node.name`` looks
# like ``test_copy_sync`` for plain tests and ``test_3x2[cuBLAS]`` for
# parametrized ones; the brackets / dashes / dots / colons need to be folded
# to underscores before the result can land in a cache directory or generated
# C symbol.
_SDFG_NAME_SAFE_CHARS = re.compile(r'[^A-Za-z0-9_]')

# Cap so the resulting ``.dacecache/<sdfg>__<suffix>_<hash>/`` path stays
# under typical filesystem limits (Linux 255-byte filename cap). Tests that
# parametrize over long expression strings (e.g. ``split_tasklets_test``)
# would otherwise blow past the limit.
_SDFG_SUFFIX_MAX_LEN = 64


def _bounded_suffix(name: str) -> str:
    """Sanitized, length-capped test-name suffix. For names that exceed
    ``_SDFG_SUFFIX_MAX_LEN``, keep a readable prefix and append an 8-char
    hash of the full name so distinct parametrize cases stay unique."""
    sanitized = _SDFG_NAME_SAFE_CHARS.sub('_', name).strip('_')
    if len(sanitized) <= _SDFG_SUFFIX_MAX_LEN:
        return sanitized
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    head = sanitized[:_SDFG_SUFFIX_MAX_LEN - len(digest) - 1].rstrip('_')
    return f"{head}_{digest}"


@pytest.fixture(autouse=True)
def _suffix_sdfg_names_with_test_name(request):
    """Append the current test's (sanitized) name to every SDFG compiled
    during the test, so concurrent test workers (xdist) and concurrent
    pytest invocations can never share a ``.dacecache/<sdfg_name>_<hash>/``
    path. The hook fires at ``SDFG.compile`` rather than ``__init__`` so
    construction-time introspection (``sdfg.name == 'foo'`` style) keeps
    seeing the original name; only the on-disk artifacts and emitted
    binaries carry the suffix.

    Idempotent across re-compiles of the same SDFG via the ``endswith``
    guard. Restores the original ``compile`` method when the test ends.
    """
    suffix = _bounded_suffix(request.node.name)
    if not suffix:
        yield
        return

    # Lazy-import: pytest collection should still work if dace imports fail.
    import dace
    orig_compile = dace.SDFG.compile
    marker = f"__{suffix}"

    def _patched_compile(self, *args, **kwargs):
        if not self.name.endswith(marker):
            self.name += marker
        return orig_compile(self, *args, **kwargs)

    dace.SDFG.compile = _patched_compile
    try:
        yield
    finally:
        dace.SDFG.compile = orig_compile


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
