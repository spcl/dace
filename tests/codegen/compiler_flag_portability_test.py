# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Each family's ``compiler.*.args`` default must compile under that family. The GCC/Clang defaults
have four switches nvc++ rejects, so :mod:`dace.codegen.compiler_family` picks a per-family default.
Absent compilers yield no test case (not a skip); CI has no NVHPC, so nvc++ runs on dev machines."""
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
    """Detection reads predefined macros, not the name. Clang and nvc++ both define ``__GNUC__``, so
    testing it first would call all three GNU and hand nvc++ flags it cannot parse."""
    family = compiler_family.detect(executable)
    assert family in {name for _, name in compiler_family.FAMILY_MACROS}, f'unknown family {family!r}'
    expected = {'g++': 'gnu', 'clang++': 'clang', 'nvc++': 'nvhpc', 'icpx': 'intelllvm'}
    stem = executable.split('-')[0]
    if stem in expected:
        assert family == expected[stem], f'{executable} detected as {family}, expected {expected[stem]}'


def test_configured_compiler_accepts_its_flags(tmp_path):
    """The configured compiler must accept the flags DaCe will pass it. Always runs."""
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
    """DaCe (picks the flags) and CMake (picks the compiler) must name the compiler the same way, or
    flags for one go to another -- how ``CXX=nvc++`` used to die on ``-fno-math-errno``. Family names
    are CMake's ids lowercased, so the comparison is exact."""
    assert compiler_family.detect(executable) == cmake_compiler_id(executable, tmp_path).lower()


def test_appending_flags_keeps_the_family_default(monkeypatch):
    """Appending must extend the family default, not revert to GCC's. ``Config.append`` (``current +=
    value``) is how DaCe adds flags, e.g. LIKWID's ``-DLIKWID_PERFMON``; treating "differs from
    default" as "user chose this" would hand nvc++ its four rejected switches. Family forced."""
    monkeypatch.setattr(compiler_family, 'detect', lambda _executable: 'nvhpc')
    nvhpc = Config.get_metadata('compiler', 'cpu', 'args')['default_nvhpc']
    with set_temporary('compiler', 'cpu', 'args', value=Config.get_default('compiler', 'cpu', 'args')):
        Config.append('compiler', 'cpu', 'args', value=' -DLIKWID_PERFMON -fopenmp ')
        got = compiler_family.cpu_args()
    assert got.startswith(nvhpc), f'append discarded the nvhpc base: {got!r}'
    assert '-DLIKWID_PERFMON' in got and '-fopenmp' in got, f'append lost the appended flags: {got!r}'
    for flag in ('-fno-math-errno', '-fno-trapping-math', '-freciprocal-math', '-Wno-unused-label'):
        assert flag not in got, f'append reintroduced {flag}, which nvc++ rejects: {got!r}'


def test_hand_written_args_are_left_alone(monkeypatch):
    """A value that is not the shipped default plus a suffix belongs to whoever wrote it."""
    monkeypatch.setattr(compiler_family, 'detect', lambda _executable: 'nvhpc')
    with set_temporary('compiler', 'cpu', 'args', value='-fPIC -my-own-flag'):
        assert compiler_family.cpu_args() == '-fPIC -my-own-flag'


def test_nvhpc_default_avoids_the_switches_nvcpp_rejects():
    """Pins the four failing switches, catching a GNU->NVHPC sync even without nvc++ installed."""
    nvhpc = Config.get_metadata('compiler', 'cpu', 'args')['default_nvhpc']
    for flag in ('-fno-math-errno', '-fno-trapping-math', '-freciprocal-math', '-Wno-unused-label'):
        assert flag not in nvhpc, f'nvhpc default carries {flag}, which nvc++ rejects: {nvhpc!r}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
