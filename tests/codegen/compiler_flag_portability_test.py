# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The shipped ``compiler.*.args`` defaults must be accepted by every compiler we claim to support.

They are written for GCC and Clang, and nvc++ rejects four of those switches outright -- an NVHPC
user hits a hard compile error before any DaCe code runs. :mod:`dace.codegen.compiler_family` picks a
per-family default; these tests check that each family's default actually compiles under that family.

Compilers absent from the box yield no test case rather than a skipped one, so this file adds
coverage where a toolchain exists and stays silent where it does not. CI carries no NVHPC, so the
nvc++ case runs on developer machines that have it -- worth knowing when reading a green CI run.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dace.codegen import compiler_family
from dace.config import Config, set_temporary

#: Compilers to try, by the name they are normally installed under. Version-suffixed names are
#: included because a distribution's unsuffixed ``g++`` may be older than the one under test.
CANDIDATES = ('g++', 'g++-12', 'g++-13', 'g++-14', 'g++-15', 'clang++', 'clang++-18', 'clang++-19', 'clang++-20',
              'clang++-21', 'nvc++', 'icpx')

AVAILABLE = [name for name in CANDIDATES if shutil.which(name)]

#: A translation unit small enough that any failure is the flags' fault and not the code's.
PROBE_SOURCE = 'extern "C" int probe(int x) { return x + 1; }\n'


def flags_for(executable: str) -> list:
    """The host flag list DaCe would hand ``executable``, family default included."""
    with set_temporary('compiler', 'cpu', 'executable', value=executable):
        args = compiler_family.cpu_args()
    return args.split() + [f'-std=c++{Config.get("compiler", "cpp_standard")}']


def compile_probe(executable: str, flags: list, tmp_path: Path) -> subprocess.CompletedProcess:
    source = tmp_path / 'probe.cpp'
    source.write_text(PROBE_SOURCE)
    return subprocess.run([executable, *flags, '-c',
                           str(source), '-o', str(tmp_path / 'probe.o')],
                          capture_output=True,
                          text=True,
                          timeout=300)


@pytest.mark.parametrize('executable', AVAILABLE)
def test_default_flags_are_accepted(executable, tmp_path):
    """Every flag DaCe would pass this compiler is one it understands."""
    result = compile_probe(executable, flags_for(executable), tmp_path)
    assert result.returncode == 0, (f'{executable} ({compiler_family.detect(executable)}) rejected DaCe\'s default '
                                    f'flags:\n{result.stderr}')


@pytest.mark.parametrize('executable', AVAILABLE)
def test_family_detection_agrees_with_the_compiler(executable, tmp_path):
    """Detection reads predefined macros, so it must not be fooled by a name.

    Clang and nvc++ both define ``__GNUC__``; a detector that tested it first would call all three
    GNU and hand nvc++ flags it cannot parse.
    """
    family = compiler_family.detect(executable)
    assert family in {name for _, name in compiler_family.FAMILY_MACROS}, f'unknown family {family!r}'
    expected = {'g++': 'gnu', 'clang++': 'clang', 'nvc++': 'nvhpc', 'icpx': 'intelllvm'}
    stem = executable.split('-')[0]
    if stem in expected:
        assert family == expected[stem], f'{executable} detected as {family}, expected {expected[stem]}'


def test_configured_compiler_accepts_its_flags(tmp_path):
    """Always runs, whatever this box has: the compiler DaCe is actually configured to use must
    accept the flags DaCe is actually going to pass it."""
    executable = Config.get('compiler', 'cpu', 'executable') or 'c++'
    assert shutil.which(executable), f'configured compiler {executable!r} is not on PATH'
    result = compile_probe(executable, flags_for(executable), tmp_path)
    assert result.returncode == 0, f'configured compiler {executable} rejected DaCe\'s flags:\n{result.stderr}'


def cmake_compiler_id(executable: str, tmp_path: Path) -> str:
    """What CMake calls ``executable``. The second opinion, from the tool that picks the flags."""
    (tmp_path / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\n'
                                             'project(probe CXX)\n'
                                             'message(STATUS "ID=${CMAKE_CXX_COMPILER_ID}")\n')
    out = subprocess.run(
        ['cmake', '-S',
         str(tmp_path), '-B',
         str(tmp_path / 'b'), f'-DCMAKE_CXX_COMPILER={shutil.which(executable)}'],
        capture_output=True,
        text=True,
        timeout=300)
    assert out.returncode == 0, out.stderr
    match = re.search(r'ID=(\w+)', out.stdout)
    assert match, f'cmake did not report a compiler id for {executable}:\n{out.stdout}'
    return match.group(1)


@pytest.mark.parametrize('executable', AVAILABLE)
def test_family_matches_cmakes_compiler_id(executable, tmp_path):
    """DaCe and CMake must name the same compiler the same way.

    Two independent detectors decide two halves of one build: DaCe's picks the flags, CMake's picks
    the compiler those flags are handed to. Agreeing today is not enough -- if they ever diverge, the
    flags chosen for one compiler are passed to another, which is how ``CXX=nvc++`` used to die on
    ``nvc++-Error-Unknown switch: -fno-math-errno``. Family names are CMake's ids lowercased so this
    comparison is exact rather than a mapping table that can rot.
    """
    assert compiler_family.detect(executable) == cmake_compiler_id(executable, tmp_path).lower()


def test_nvhpc_default_avoids_the_switches_nvcpp_rejects():
    """Pins the four that fail, so a well-meaning sync of the GNU default into the NVHPC one is
    caught even on a box with no nvc++ to compile with."""
    nvhpc = Config.get_metadata('compiler', 'cpu', 'args')['default_nvhpc']
    for flag in ('-fno-math-errno', '-fno-trapping-math', '-freciprocal-math', '-Wno-unused-label'):
        assert flag not in nvhpc, f'nvhpc default carries {flag}, which nvc++ rejects: {nvhpc!r}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
