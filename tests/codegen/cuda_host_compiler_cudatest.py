# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""nvcc must build against the same host compiler as the host translation units."""
import json
import os
import pathlib
import shlex
import tempfile

import numpy as np
import pytest

import dace
from dace.codegen import compiler_family
from dace.codegen.target import make_absolute


def compile_gpu_program(build_folder: str) -> str:
    """Build a trivial GPU program and return its ``compile_commands.json`` contents."""

    @dace.program
    def host_compiler_tester(a: dace.float64[64], b: dace.float64[64]):
        for i in dace.map[0:64]:
            b[i] = a[i] + 1.0

    with dace.config.set_temporary('default_build_folder', value=build_folder):
        # A replayed build skips the CMake configure that writes compile_commands.json.
        with dace.config.set_temporary('compiler', 'command_cache', value=False):
            sdfg = host_compiler_tester.to_sdfg()
            sdfg.apply_gpu_transformations()
            csdfg = sdfg.compile()
            # Resolved here: the property follows the config, which the context is about to restore.
            commands = os.path.join(sdfg.build_folder, 'build', 'compile_commands.json')

    a = np.arange(64, dtype=np.float64)
    b = np.zeros(64, dtype=np.float64)
    csdfg(a=a, b=b)
    assert np.allclose(b, a + 1.0)

    if not os.path.exists(commands):
        pytest.skip('the build was replayed, so CMake never wrote compile_commands.json')
    with open(commands) as handle:
        return handle.read()


def ccbin_arguments(commands: str) -> list:
    """Every ``-ccbin`` value nvcc was invoked with."""
    found = []
    for entry in json.loads(commands):
        arguments = shlex.split(entry['command'])
        for argument in arguments:
            if argument.startswith('-ccbin'):
                found.append(argument.split('=', 1)[1] if '=' in argument else '')
    return found


@pytest.mark.gpu
def test_cuda_host_compiler_defaults_to_the_host_toolchain(tmp_path):
    """With no explicit setting, nvcc still gets the compiler the host sources are built with."""
    expected = make_absolute(compiler_family.host_compiler())
    commands = compile_gpu_program(str(tmp_path / 'default'))

    found = ccbin_arguments(commands)
    assert found, 'nvcc was invoked without -ccbin, so it fell back to its own default host compiler'
    assert set(found) == {expected}, found


@pytest.mark.gpu
def test_cuda_host_compiler_follows_the_configured_executable(tmp_path):
    """``compiler.cpu.executable`` steers nvcc, which is what it could not do before."""
    expected = make_absolute(compiler_family.host_compiler())
    with dace.config.set_temporary('compiler', 'cpu', 'executable', value=expected):
        commands = compile_gpu_program(str(tmp_path / 'configured'))

    assert set(ccbin_arguments(commands)) == {expected}


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as scratch:
        test_cuda_host_compiler_defaults_to_the_host_toolchain(pathlib.Path(scratch))
        test_cuda_host_compiler_follows_the_configured_executable(pathlib.Path(scratch))
