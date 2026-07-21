# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the machine-global build caches (:mod:`dace.codegen.build_cache`).

Both caches are advisory -- a miss only costs speed -- so the interesting property is not that they
exist but that they actually ENGAGE. A precompiled header the compiler silently declines to use, or
a configure seed CMake rejects, is indistinguishable from a correct build except in wall-clock time,
which is exactly the kind of regression that goes unnoticed. These tests assert engagement directly.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

import dace
from dace.codegen import build_cache

N = dace.symbol('N')


@dace.program
def scaled_add(x: dace.float64[N], y: dace.float64[N]):
    y[:] = x * 3.0 + y


def _build(tmp_path, name):
    """Compile ``scaled_add`` into its own build folder and check it computes the right thing."""
    with dace.config.set_temporary('default_build_folder', value=str(tmp_path / name)):
        sdfg = scaled_add.to_sdfg(simplify=True)
        sdfg.name = name
        csdfg = sdfg.compile()
        build_folder = sdfg.build_folder  # resolved against the config, so read it inside the scope
    n = 64
    x = np.random.rand(n)
    y = np.zeros(n)
    csdfg(x=x, y=y, N=n)
    assert np.allclose(y, x * 3.0)
    return build_folder


def _compile_commands(build_folder):
    path = os.path.join(build_folder, 'build', 'compile_commands.json')
    if not os.path.isfile(path):
        path = os.path.join(build_folder, 'compile_commands.json')
    assert os.path.isfile(path), f'compile_commands.json was not exported under {build_folder}'
    with open(path) as f:
        return json.load(f)


def test_compile_commands_are_exported(tmp_path):
    """The build exports a compilation database for the generated sources."""
    entries = _compile_commands(_build(tmp_path, 'ccexport'))
    assert any(e['file'].endswith('.cpp') for e in entries)


@pytest.mark.skipif(os.name != 'posix', reason='precompiled headers are only wired up for GCC/Clang')
def test_precompiled_header_is_actually_used(tmp_path):
    """The generated translation unit must really consume the cached PCH.

    The PCH is only honored when it was built with flags compatible with the translation unit's.
    If ``shared_pch_dir`` ever drifts from what CMake puts on the compile line, GCC silently ignores
    the header and compiles normally -- same object, no speedup, no error. Re-running the exact
    recorded command with ``-Werror=invalid-pch`` turns that silent miss into a failure.
    """
    entries = _compile_commands(_build(tmp_path, 'pchused'))
    generated = [e for e in entries if 'pchused' in e['file']]
    assert generated, 'no compile command recorded for the generated source'
    command = generated[0]['command']
    assert 'dace_prewarm.h' in command, 'the PCH was never put on the compile line'

    # Compile to a throwaway object, promoting "this PCH cannot be used" to a hard error.
    checked = command.replace(' -c ', ' -Winvalid-pch -Werror=invalid-pch -c ')
    result = subprocess.run(checked, shell=True, cwd=generated[0]['directory'], capture_output=True, text=True)
    assert result.returncode == 0, f'the compiler refused the precompiled header:\n{result.stderr}'


def test_configure_cache_seeds_a_working_build(tmp_path):
    """A second program of the same shape reuses the first one's configure and still builds correctly.

    CMake refuses a cache file it finds in a directory other than the one it was created in, so a
    seed that is not retargeted aborts the configure instead of speeding it up. Building a second
    program after the first has published its configure exercises exactly that path.
    """
    _build(tmp_path, 'seedfirst')
    entries = os.listdir(build_cache.cache_root('configure'))
    assert entries, 'the first build published no configure cache entry'
    # The second build seeds from the first; correctness of its result is the assertion in _build.
    _build(tmp_path, 'seedsecond')


def test_caches_disabled_still_builds(tmp_path):
    """With both caches off the build must still work -- they are optimizations, not requirements."""
    with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
        with dace.config.set_temporary('compiler', 'configure_cache', value=False):
            _build(tmp_path, 'nocaches')


def test_seed_declines_when_no_entry_exists(tmp_path):
    """Seeding an unknown key is a no-op rather than an error, so a cold cache just configures."""
    folder = tmp_path / 'empty'
    folder.mkdir()
    assert build_cache.seed_configure_cache(str(folder), 'a-key-that-was-never-published') is False
    assert not os.path.exists(folder / 'CMakeCache.txt')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
