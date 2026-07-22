# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Host ``dace::bfloat16`` conformance, driven through the real C++ compiler.

``tests/cpp/bfloat16_ops_test.cpp`` checks the parts of ``dace::bfloat16`` that only
exist at the C++ level -- storage layout, ``constexpr`` folding, overload resolution
across the whole operator surface, compound-assignment semantics, OpenMP reductions,
and exhaustive bit-exactness of both conversion directions.

It is compiled and run once per ISA configuration below. Unlike ``dace::float16``
there is no native/emulated switch to cover (see ``<dace/types.h>``: a native
``__bf16`` would be slower, because GCC lowers its conversions to libgcc calls),
so what these configurations vary is the compiler's freedom to vectorize and
reassociate around the conversions -- which is what would expose an emulation that
is only accidentally correct at one optimization setting.
"""

import os
import subprocess

import pytest

from dace import Config

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(TEST_DIR, 'cpp', 'bfloat16_ops_test.cpp')
RUNTIME_INCLUDE = os.path.abspath(os.path.join(TEST_DIR, '..', 'dace', 'runtime', 'include'))

#: (test id, extra compiler flags). ``-march=native`` mirrors DaCe's default
#: ``compiler.cpu.args``; the bare configuration pins the x86-64 baseline ISA.
BFLOAT16_CONFIGS = [
    ('auto', ['-march=native']),
    ('baseline-isa', []),
    ('unoptimized', ['-O0']),
]


def cxx():
    """The C++ compiler DaCe itself would invoke for a CPU target."""
    return Config.get('compiler', 'cpu', 'executable') or 'c++'


def build_and_run(binary, extra_flags):
    """Compile the conformance test with ``extra_flags`` and run it on 4 threads."""
    cmd = [cxx(), '-std=c++20', '-O2', '-fopenmp', '-Wall', '-Wextra', '-Werror']
    cmd += list(extra_flags) + ['-I', RUNTIME_INCLUDE, '-o', binary, SOURCE]

    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f'compilation failed:\n{" ".join(cmd)}\n{build.stderr}'
    assert not build.stderr.strip(), f'compiler diagnostics:\n{build.stderr}'

    run = subprocess.run([binary], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS='4'))
    assert run.returncode == 0, f'{run.stdout}\n{run.stderr}'
    return run.stdout


@pytest.mark.parametrize('name, extra_flags', BFLOAT16_CONFIGS, ids=[c[0] for c in BFLOAT16_CONFIGS])
def test_bfloat16_ops_cpp(tmp_path, name, extra_flags):
    """Layout, constexpr, operator surface, compound assignment, OpenMP, exhaustive bits.

    Includes an exhaustive sweep of all 2^32 float bit patterns through
    float -> bfloat16, compared against an independently written round-to-nearest-even
    reference, plus all 2^16 bfloat16 patterns through bfloat16 -> float -> bfloat16.
    """
    stdout = build_and_run(str(tmp_path / f'bfloat16_ops_{name}'), extra_flags)
    assert 'OK (0 failures)' in stdout, stdout


def header_defines(macro, extra_flags):
    """Whether <dace/types.h> defines ``macro`` under ``extra_flags``."""
    probe = f'#include <dace/types.h>\n#if defined({macro})\nDACE_YES\n#else\nDACE_NO\n#endif\n'
    cmd = [cxx(), '-std=c++20', '-E', '-x', 'c++', '-'] + list(extra_flags) + ['-I', RUNTIME_INCLUDE]
    out = subprocess.run(cmd, input=probe, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return 'DACE_YES' in out.stdout


def test_bfloat16_conversions_stay_on_the_emulation_by_default():
    """Detecting a native ``__bf16`` must NOT switch the conversions onto it.

    This pins a deliberate decision, so that a future "the compiler supports it, why
    aren't we using it?" edit has to argue with a failing test. Both target compilers
    were measured and both are worse than the emulation:

    * GCC never inlines ``__truncsfbf2``, making narrowing ~32x slower (4.0 ms vs
      129 ms for 2**24 conversions at -O2 -march=native).
    * Clang lowers narrowing to AVX512-BF16 ``VCVTNEPS2BF16``, which flushes
      subnormals to zero and does not consult MXCSR -- every bf16 subnormal silently
      becomes zero.

    Nothing here asserts the host *has* a native ``__bf16``; the point is only that
    wherever it is detected, it stays unused unless explicitly opted into.
    """
    assert not header_defines('DACE_BFLOAT16_NATIVE_T', ['-march=native'])
    assert not header_defines('DACE_BFLOAT16_NATIVE_T', [])

    if header_defines('DACE_NATIVE_BF16', ['-march=native']):
        # The opt-in must actually reach the native type, or the escape hatch is dead.
        assert header_defines('DACE_BFLOAT16_NATIVE_T', ['-march=native', '-DDACE_BFLOAT16_USE_NATIVE'])


def test_bfloat16_builds_under_cxx17(tmp_path):
    """The pre-C++20 arm of the bit-cast helper must stay correct.

    Without ``std::bit_cast`` the header falls back to a union pun and
    ``DACE_LOWP_CE`` degrades from ``constexpr`` to ``inline``. That arm is invisible
    at the default standard, so it is compiled and run explicitly here rather than
    left to rot.
    """
    binary = str(tmp_path / 'bfloat16_ops_cxx17')
    cmd = [cxx(), '-std=c++17', '-O2', '-fopenmp', '-Wall', '-Wextra', '-Werror']
    cmd += ['-I', RUNTIME_INCLUDE, '-o', binary, SOURCE]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f'{" ".join(cmd)}\n{build.stderr}'

    run = subprocess.run([binary], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS='4'))
    assert run.returncode == 0, f'{run.stdout}\n{run.stderr}'
    assert 'OK (0 failures)' in run.stdout, run.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
