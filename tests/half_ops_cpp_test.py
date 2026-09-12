# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Host ``dace::float16`` conformance, driven through the real C++ compiler.

``tests/cpp/half_ops_test.cpp`` checks the parts of ``dace::half`` that only exist at
the C++ level -- storage layout, ``constexpr`` folding, overload resolution across the
whole operator surface, compound-assignment semantics and OpenMP reductions. It is
compiled and run once per configuration below, so that both implementations behind
``dace::float16`` (the native binary16 conversion path and the software emulation) are
exercised on any host, including x86 boxes that can never run the AArch64 variant of
the same switch.

Both builds are checked against the same absolute IEEE-754 expectations, which is a
stronger statement than diffing them against each other.
"""

import os
import subprocess

import pytest

from dace import Config

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(TEST_DIR, 'cpp', 'half_ops_test.cpp')
RUNTIME_INCLUDE = os.path.abspath(os.path.join(TEST_DIR, '..', 'dace', 'runtime', 'include'))

#: (test id, extra compiler flags). ``-march=native`` mirrors DaCe's default
#: ``compiler.cpu.args``, and is what enables the hardware conversion path on a host
#: with F16C / AVX512-FP16. The bare configuration pins the x86-64 baseline ISA, where
#: the header must stay on the emulation because ``_Float16`` conversions would lower
#: to libgcc calls that are slower than the inline emulation.
HALF_CONFIGS = [
    ('auto', ['-march=native']),
    ('forced-emulated', ['-march=native', '-DDACE_HALF_NO_NATIVE']),
    ('baseline-isa', []),
]

NATIVE_BANNER = 'dace::half conversion backend: NATIVE'
EMULATED_BANNER = 'dace::half conversion backend: EMULATED'


def cxx():
    """The C++ compiler DaCe itself would invoke for a CPU target."""
    return Config.get('compiler', 'cpu', 'executable') or 'c++'


def compiler_selects_native(extra_flags):
    """Ask the compiler whether <dace/types.h> will pick a native binary16 type.

    Preprocesses the header with the given flags and reports which branch of the
    backend selection it takes, so the test can predict the expected banner on any
    host instead of hard-coding this machine's answer.
    """
    probe = '#include <dace/types.h>\n#if defined(DACE_HALF_NATIVE_T)\nDACE_NATIVE\n#else\nDACE_EMULATED\n#endif\n'
    cmd = [cxx(), '-std=c++20', '-E', '-x', 'c++', '-'] + list(extra_flags) + ['-I', RUNTIME_INCLUDE]
    out = subprocess.run(cmd, input=probe, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return 'DACE_NATIVE' in out.stdout


def build_and_run(binary, extra_flags, strict=True):
    """Compile the conformance test with ``extra_flags`` and run it on 4 threads."""
    cmd = [cxx(), '-std=c++20', '-O2', '-fopenmp', '-Wall', '-Wextra']
    if strict:
        cmd.append('-Werror')
    cmd += list(extra_flags) + ['-I', RUNTIME_INCLUDE, '-o', binary, SOURCE]

    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f'compilation failed:\n{" ".join(cmd)}\n{build.stderr}'
    assert not build.stderr.strip(), f'compiler diagnostics:\n{build.stderr}'

    run = subprocess.run([binary], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS='4'))
    assert run.returncode == 0, f'{run.stdout}\n{run.stderr}'
    return run.stdout


@pytest.mark.parametrize('name, extra_flags', HALF_CONFIGS, ids=[c[0] for c in HALF_CONFIGS])
def test_half_ops_cpp(tmp_path, name, extra_flags):
    """Storage layout, constexpr folding, operator surface and OpenMP reductions."""
    stdout = build_and_run(str(tmp_path / f'half_ops_{name}'), extra_flags)
    assert 'OK (0 failures)' in stdout, stdout

    expected = NATIVE_BANNER if compiler_selects_native(extra_flags) else EMULATED_BANNER
    assert expected in stdout, stdout


def test_baseline_isa_stays_on_the_emulation():
    """Type availability must not be mistaken for hardware support.

    GCC and Clang expose ``_Float16`` on plain x86-64, but without F16C every
    conversion becomes an ``__extendhfsf2`` / ``__truncsfhf2`` libgcc call, which is
    slower than the inline emulation. The header therefore gates on the ISA feature
    macro, and this pins that decision.
    """
    assert not compiler_selects_native([]), ('the x86-64 baseline ISA has no hardware fp16 conversion, so '
                                             '<dace/types.h> must stay on the software emulation there')
    assert not compiler_selects_native(['-march=native', '-DDACE_HALF_NO_NATIVE'])


def test_native_conversion_path_is_selected_for_f16c(tmp_path):
    """Asking for the ISA feature explicitly must flip the header to the native type.

    Compile-only: an ``-mf16c`` binary faults on a CPU without F16C, and the point
    here is the header's branch, not the instruction. Whether the native path is
    additionally *executed* on this host is covered by the ``auto`` configuration
    above, which uses ``-march=native`` and so is always runnable.
    """
    if not compiler_selects_native(['-mf16c']):
        # Not an x86 host with a ``_Float16``-capable compiler. The AArch64 arm of the
        # switch cannot be built or run here either way, so record why rather than
        # passing silently.
        assert not compiler_selects_native([]), 'inconsistent: baseline selects native but -mf16c does not'
        return

    binary = str(tmp_path / 'half_ops_f16c')
    cmd = [cxx(), '-std=c++20', '-O2', '-fopenmp', '-Wall', '-Wextra', '-Werror', '-mf16c']
    cmd += ['-I', RUNTIME_INCLUDE, '-o', binary, SOURCE]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f'{" ".join(cmd)}\n{build.stderr}'
    assert not build.stderr.strip(), build.stderr


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
