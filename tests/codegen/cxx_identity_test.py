# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The host C++ compiler version probe, and the emission-time warning it feeds.

GCC 13's auto-vectorizer mislowers the select the readable code generator emits, returning zeros for
the upper half of every vector. The wrong answer is silent, so codegen names the compiler instead of
leaving it to be rediscovered from a numerical diff. These pin the probe and the warning's conditions
with stub compilers, so no particular GCC has to be installed to run them.
"""
import os
import stat
import warnings

import pytest

from dace.codegen import common, compiler_family

#: What each family predefines. The probe reads these macros rather than the ``--version`` banner:
#: invoked as ``c++`` a GCC banner never names its vendor. clang defines ``__GNUC__`` too, which is
#: why the stub below carries both and the probe still has to answer "clang".
GCC_13 = '#define __GNUC__ 13\n#define __GNUC_MINOR__ 3\n#define __GNUC_PATCHLEVEL__ 0\n'
GCC_15 = '#define __GNUC__ 15\n#define __GNUC_MINOR__ 2\n#define __GNUC_PATCHLEVEL__ 0\n'
CLANG_18 = ('#define __GNUC__ 4\n#define __clang__ 1\n#define __clang_major__ 18\n'
            '#define __clang_minor__ 1\n#define __clang_patchlevel__ 3\n')


def _stub_compiler(tmp_path, name: str, macros: str) -> str:
    """An executable that answers the ``-dM -E`` probe with ``macros`` and nothing else."""
    path = tmp_path / name
    path.write_text(f'#!/bin/sh\ncat <<\'EOF\'\n{macros}EOF\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.mark.skipif(os.name != 'posix', reason='stub compilers are shell scripts')
def test_detect_version_reads_the_predefined_macros(tmp_path):
    gcc13 = _stub_compiler(tmp_path, 'gcc13', GCC_13)
    clang18 = _stub_compiler(tmp_path, 'clang18', CLANG_18)
    assert compiler_family.detect(gcc13) == 'gnu'
    assert compiler_family.detect_version(gcc13) == (13, 3, 0)
    assert compiler_family.detect_version(_stub_compiler(tmp_path, 'gcc15', GCC_15)) == (15, 2, 0)
    # clang defines __GNUC__ 4 as well; the version reported must be clang's own.
    assert compiler_family.detect(clang18) == 'clang'
    assert compiler_family.detect_version(clang18) == (18, 1, 3)


@pytest.mark.skipif(os.name != 'posix', reason='stub compilers are shell scripts')
def test_detect_version_is_none_when_unidentifiable(tmp_path):
    assert compiler_family.detect_version(_stub_compiler(tmp_path, 'mystery', '#define __SOMETHING__ 1\n')) is None
    assert compiler_family.detect_version(str(tmp_path / 'does-not-exist')) is None


@pytest.mark.skipif(os.name != 'posix', reason='stub compilers are shell scripts')
@pytest.mark.parametrize(('name', 'macros', 'expected'), [('wgcc13', GCC_13, True), ('wgcc15', GCC_15, False),
                                                          ('wclang18', CLANG_18, False)])
def test_warning_fires_only_for_the_miscompiling_gcc(tmp_path, monkeypatch, name, macros, expected):
    # Pinned through the environment rather than set_temporary: a DACE_* variable outranks the
    # configuration, and the CI images export DACE_compiler_cpu_executable to keep GCC 13 out of the
    # build -- so set_temporary left the stub unprobed and this asked the runner's own compiler.
    monkeypatch.setenv('DACE_compiler_cpu_implementation', 'experimental_readable')
    monkeypatch.setenv('DACE_compiler_cpu_executable', _stub_compiler(tmp_path, name, macros))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        common.warn_if_cxx_miscompiles_inline_selects()
    assert bool([w for w in caught if 'vectorizer' in str(w.message)]) == expected


@pytest.mark.skipif(os.name != 'posix', reason='stub compilers are shell scripts')
def test_no_warning_for_the_classic_generator(tmp_path, monkeypatch):
    """The classic generator keeps the operands in connector locals, so the inlined select -- the
    only shape the bug is known to hit -- is never emitted and the compiler is not the user's
    problem."""
    monkeypatch.setenv('DACE_compiler_cpu_implementation', 'legacy')
    monkeypatch.setenv('DACE_compiler_cpu_executable', _stub_compiler(tmp_path, 'cgcc13', GCC_13))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        common.warn_if_cxx_miscompiles_inline_selects()
    assert not [w for w in caught if 'vectorizer' in str(w.message)]


if __name__ == '__main__':
    pytest.main([__file__])
