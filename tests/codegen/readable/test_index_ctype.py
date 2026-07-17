# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests for ``compiler.cpu.codegen_params.index_ctype``: the integer type the readable CPU code
    generator's ``<array>_idx`` / ``<array>_size`` helpers compute a flat index in. """
import re

import numpy
import pytest

import dace
from dace.config import set_temporary

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def transpose_add(A: dace.float64[N, M], B: dace.float64[M, N]):
    for i, j in dace.map[0:N, 0:M]:
        B[j, i] = A[i, j] + 1.0


def generate(implementation: str, index_ctype: str = 'int64') -> str:
    sdfg = transpose_add.to_sdfg(simplify=True)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'index_ctype', value=index_ctype):
        return '\n'.join(obj.code for obj in sdfg.generate_code() if obj.language == 'cpp')


def index_signatures(code: str):
    """ The ``<qualifiers> <ctype> <array>_idx(...)`` / ``_size(...)`` helper definition lines. """
    return re.findall(r'^\s*static DACE_HDFI (?:constexpr|consteval) (.+?) \w+_(?:idx|size)\(([^)]*)\)',
                      code,
                      flags=re.MULTILINE)


@pytest.mark.parametrize('index_ctype, expected', [('int64', 'int64_t'), ('int32', 'int32_t')])
def test_index_ctype_applied(index_ctype, expected):
    """ Every generated helper -- return type AND every parameter -- uses the configured type. """
    signatures = index_signatures(generate('experimental_readable', index_ctype))
    assert signatures, 'no index/size helpers were emitted'
    for return_type, params in signatures:
        assert return_type == expected
        for param in (p for p in params.split(',') if p.strip()):
            assert param.strip().rsplit(' ', 1)[0] == expected


def test_int64_is_the_default():
    assert index_signatures(generate('experimental_readable')) == index_signatures(
        generate('experimental_readable', 'int64'))


def test_legacy_ignores_the_flag():
    """ Legacy emits no index helpers and its output must stay byte-identical across the flag. """
    baseline = generate('legacy', 'int64')
    assert baseline == generate('legacy', 'int32')
    assert not index_signatures(baseline)


@pytest.mark.parametrize('index_ctype', ['int64', 'int32'])
def test_both_types_compile_and_run(index_ctype):
    """ Non-square + transposed so a wrong index type or a swapped extent shows up as bad data. """
    with set_temporary('compiler', 'cpu', 'implementation', value='experimental_readable'), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'index_ctype', value=index_ctype):
        A = numpy.random.rand(17, 23)
        B = numpy.zeros((23, 17))
        transpose_add(A=A, B=B, N=17, M=23)
        assert numpy.allclose(B, A.T + 1.0)


if __name__ == '__main__':
    test_index_ctype_applied('int64', 'int64_t')
    test_index_ctype_applied('int32', 'int32_t')
    test_int64_is_the_default()
    test_legacy_ignores_the_flag()
    test_both_types_compile_and_run('int64')
    test_both_types_compile_and_run('int32')
