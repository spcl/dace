# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the shared CMake configure cache and precompiled header.

Both are advisory, so the property worth asserting is not that they exist but that they ENGAGE: a
header the compiler silently declines, or a seed CMake rejects, is indistinguishable from a correct
build except in wall-clock time.
"""
import json
import os
import subprocess

import numpy as np
import pytest

import dace

N = dace.symbol('N')


@dace.program
def scaled_add(x: dace.float64[N], y: dace.float64[N]):
    y[:] = x * 3.0 + y


def _build(tmp_path, name):
    """Compile into a private build folder and check the result computes the right thing."""
    with dace.config.set_temporary('default_build_folder', value=str(tmp_path / name)):
        sdfg = scaled_add.to_sdfg(simplify=True)
        sdfg.name = name
        csdfg = sdfg.compile()
        build_folder = sdfg.build_folder  # resolved against the config, so read it inside the scope
    x, y = np.random.rand(64), np.zeros(64)
    csdfg(x=x, y=y, N=64)
    assert np.allclose(y, x * 3.0)
    return build_folder


def test_configure_cache_seeds_a_working_build(tmp_path):
    """A second program reuses the first one's configure and still builds correctly.

    CMake refuses a cache file found in a directory other than the one it was created in, so a seed
    that is not retargeted aborts the configure rather than speeding it up.
    """
    _build(tmp_path, 'seedfirst')
    _build(tmp_path, 'seedsecond')


@pytest.mark.skipif(os.name != 'posix', reason='precompiled headers are only wired up for GCC/Clang')
def test_precompiled_header_is_actually_used(tmp_path):
    """The generated translation unit must really consume the cached header.

    A precompiled header is only honored when its flags are compatible with the translation unit's.
    If they ever drift, the compiler ignores it silently -- same object, no speedup, no error.
    Replaying the recorded command with ``-Werror=invalid-pch`` turns that into a failure.
    """
    build_folder = _build(tmp_path, 'pchused')
    database = os.path.join(build_folder, 'build', 'compile_commands.json')
    assert os.path.isfile(database), 'no compilation database was exported'
    with open(database) as fp:
        generated = [e for e in json.load(fp) if 'pchused' in e['file']]
    assert generated, 'no compile command recorded for the generated source'
    command = generated[0]['command']
    assert 'dace_prewarm.h' in command, 'the precompiled header never reached the compile line'
    checked = command.replace(' -c ', ' -Winvalid-pch -Werror=invalid-pch -c ')
    result = subprocess.run(checked, shell=True, cwd=generated[0]['directory'], capture_output=True, text=True)
    assert result.returncode == 0, f'the compiler refused the precompiled header:\n{result.stderr}'


def test_caches_disabled_still_builds(tmp_path):
    """With both caches off the build must still work -- they are optimizations, not requirements."""
    with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
        with dace.config.set_temporary('compiler', 'configure_cache', value=False):
            _build(tmp_path, 'nocaches')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
