# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``compiler.build_type`` is the sole source of the optimization level: args carry no ``-O``, the
build type carries nothing else. The per-family table also feeds the PCH build, so a wrong entry
makes the compiler silently decline the header."""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.codegen.compiler import BUILD_TYPE_FLAGS_BY_FAMILY, CMAKE_BUILD_TYPE_FLAGS
from dace.config import Config, set_temporary

#: Every default flag string that must not carry an optimization level.
ARG_KEYS = (
    ('compiler', 'cpu', 'args'),
    ('compiler', 'cuda', 'args'),
    ('compiler', 'cuda', 'hip_args'),
)

OPT_FLAG = re.compile(r'(?:^|\s)[-/]O')


def cmake_config_flags(language: str = 'CXX', executable: str = '') -> dict:
    """What CMake puts in ``CMAKE_<language>_FLAGS_<CONFIG>`` for this compiler. Asked of CMake, not
    hardcoded: the test is that DaCe's copy tracks the real thing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\n'
                                             f'project(probe {language})\n'
                                             'foreach(C DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)\n'
                                             f'  message(STATUS "PROBE ${{C}}=${{CMAKE_{language}_FLAGS_${{C}}}}")\n'
                                             'endforeach()\n')
        command = ['cmake', '-S', str(root), '-B', str(root / 'b')]
        if executable:
            command.append(f'-DCMAKE_{language}_COMPILER={shutil.which(executable)}')
        out = subprocess.run(command, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
    found = {}
    for line in out.stdout.splitlines():
        match = re.search(r'PROBE (\w+)=(.*)$', line.strip())
        if match:
            found[match.group(1).capitalize()] = match.group(2).split()
    return {
        'Debug': found['Debug'],
        'Release': found['Release'],
        'RelWithDebInfo': found['Relwithdebinfo'],
        'MinSizeRel': found['Minsizerel']
    }


@pytest.mark.parametrize('key', ARG_KEYS, ids=lambda k: '_'.join(k[1:]))
def test_arg_defaults_carry_no_optimization_level(key):
    """An ``-O`` here loses to CMake's per-config flags: no error, no effect -- the worst knob."""
    default = Config.get_default(*key) or ''
    assert not OPT_FLAG.search(default), f'{".".join(key)} default carries an optimization level: {default!r}'


@pytest.mark.parametrize('key', ARG_KEYS, ids=lambda k: '_'.join(k[1:]))
def test_arg_defaults_are_not_fast_math(key):
    """fast-math enables finite-math and reassociation, which change results (NPBench LU fails). The
    granular flags that survive are checked below."""
    default = Config.get_default(*key) or ''
    assert '-ffast-math' not in default and '--use_fast_math' not in default, \
        f'{".".join(key)} default enables fast-math: {default!r}'


def test_cpu_args_keep_the_vectorization_enabling_subset():
    """The four flags that let the vectorizer through libm calls without changing results."""
    default = Config.get_default('compiler', 'cpu', 'args')
    for flag in ('-fno-math-errno', '-fno-trapping-math', '-fno-signed-zeros', '-freciprocal-math'):
        assert flag in default, f'compiler.cpu.args default lost {flag}: {default!r}'


#: One compiler per family we keep a table for, so each table is checked against the compiler it
#: describes. A family with no compiler installed yields no test case rather than a skip.
FAMILY_PROBES = [(family, executable) for family, executable in (('gnu', 'g++'), ('nvhpc', 'nvc++'))
                 if shutil.which(executable)]


@pytest.mark.parametrize('family,executable', FAMILY_PROBES, ids=[f for f, _ in FAMILY_PROBES])
def test_pch_build_type_table_matches_cmake(family, executable):
    """DaCe's per-family table must equal CMake's own values, or the PCH is declined. Parametrized by
    family, not just the host default -- an earlier default-only check passed while every NVHPC entry
    was wrong."""
    assert shutil.which('cmake'), 'cmake is not on PATH, so DaCe cannot build at all'
    ours = BUILD_TYPE_FLAGS_BY_FAMILY[family]
    actual = cmake_config_flags('CXX', executable)
    assert set(ours) == set(actual), f'{family}: DaCe and CMake disagree on the set of build types'
    for build_type, flags in actual.items():
        assert ours[build_type] == flags, \
            (f'{family}/{build_type}: DaCe has {ours[build_type]}, CMake emits {flags}. The PCH would be built '
             'with different flags than the TU and silently ignored.')


@pytest.mark.parametrize('build_type', ['Debug', 'Release', 'RelWithDebInfo'])
def test_compile_line_has_exactly_the_build_types_optimization_level(build_type):
    """End to end: build a real SDFG and read ``compile_commands.json`` -- the only witness that does
    not re-derive the flags the way the code under test does."""

    @dace.program
    def addone(inp: dace.float64[8], out: dace.float64[8]):
        out[:] = inp + 1.0

    sdfg = addone.to_sdfg()
    sdfg.name = f'build_type_flags_{build_type.lower()}'
    with set_temporary('compiler', 'build_type', value=build_type):
        sdfg.compile()
        build_dir = Path(sdfg.build_folder) / 'build'
        commands = json.loads((build_dir / 'compile_commands.json').read_text())

    expected = [f for f in CMAKE_BUILD_TYPE_FLAGS[build_type] if f.startswith('-O')]
    host = [entry for entry in commands if entry['file'].endswith(('.cpp', '.cxx', '.cc'))]
    assert host, 'no host translation unit in compile_commands.json'
    for entry in host:
        levels = [tok for tok in entry['command'].split() if OPT_FLAG.match(' ' + tok)]
        assert levels == expected, \
            f'{build_type}: expected optimization flags {expected}, compile line has {levels}: {entry["command"]}'

    inp = np.arange(8, dtype=np.float64)
    out = np.zeros(8, dtype=np.float64)
    sdfg(inp=inp, out=out)
    assert np.allclose(out, inp + 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
