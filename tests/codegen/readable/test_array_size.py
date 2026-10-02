# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
A heap array of constant or compound symbolic size is allocated over a generated ``<array>_size(...)`` helper; a bare
single symbol keeps its plain name. The kernels also run under both generators and must agree bit-exactly.
"""
import re

import numpy as np

import dace
from dace.config import Config
from dace.dtypes import StorageType
from tests.codegen.readable.conftest import assert_bit_exact, experimental_code, heap_pipeline_1d


def nested_same_name_sdfg(name: str) -> dace.SDFG:
    """A transient ``T`` of size ``N*M`` whose nested SDFG has a transient ``T`` of size ``N*N``."""
    inner_n = dace.symbol('N')
    inner = dace.SDFG('inner')
    inner.add_array('a', [inner_n], dace.float64)
    inner.add_array('b', [inner_n], dace.float64)
    inner.add_transient('T', [inner_n * inner_n], dace.float64, storage=StorageType.CPU_Heap)
    iw = inner.add_state('write')
    ira, iwt = iw.add_read('a'), iw.add_write('T')
    itk = iw.add_tasklet('w', {'ai'}, {'to'}, 'to = ai')
    iw.add_edge(ira, None, itk, 'ai', dace.Memlet('a[0]'))
    iw.add_edge(itk, 'to', iwt, None, dace.Memlet('T[0]'))
    ir = inner.add_state_after(iw, 'read')
    irt, iwb = ir.add_read('T'), ir.add_write('b')
    itk2 = ir.add_tasklet('r', {'ti'}, {'bo'}, 'bo = ti')
    ir.add_edge(irt, None, itk2, 'ti', dace.Memlet('T[0]'))
    ir.add_edge(itk2, 'bo', iwb, None, dace.Memlet('b[0]'))

    n, m = dace.symbol('N'), dace.symbol('M')
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('B', [n], dace.float64)
    sdfg.add_transient('T', [n * m], dace.float64, storage=StorageType.CPU_Heap)
    main = sdfg.add_state('main')
    nsdfg = main.add_nested_sdfg(inner, {'a'}, {'b'}, {'N': 'N'})
    main.add_edge(main.add_read('A'), None, nsdfg, 'a', dace.Memlet('A[0:N]'))
    main.add_edge(nsdfg, 'b', main.add_write('B'), None, dace.Memlet('B[0:N]'))

    later = sdfg.add_state_after(main, 'outer_t')
    outer_t = later.add_access('T')
    seed = later.add_tasklet('seed', {}, {'o'}, 'o = 0.0')
    later.add_edge(seed, 'o', outer_t, None, dace.Memlet('T[0]'))

    sdfg.validate()
    return sdfg


def size_helper_definition(code: str, array: str) -> str:
    """The definition line of ``<array>_size``, matched at a word boundary so ``T`` is not ``inner_T``."""
    pattern = re.compile(r'(?<!\w)%s_size\(' % re.escape(array))
    lines = [line.strip() for line in code.splitlines() if pattern.search(line) and 'return' in line]
    assert lines, 'experimental codegen emitted no %s_size helper' % array
    return lines[0]


def allocation_line(code: str, array: str) -> str:
    """The aligned ``new[]`` statement allocating ``array``."""
    pattern = re.compile(r'(?<!\w)%s = ' % re.escape(array))
    lines = [line.strip() for line in code.splitlines() if 'align_val_t' in line and pattern.search(line)]
    assert lines, 'experimental codegen emitted no aligned allocation for %s' % array
    return lines[0]


def allocation_extent(code: str, array: str) -> str:
    """The element count in the allocation of ``array``."""
    return re.search(r'\[(.*)\];', allocation_line(code, array)).group(1)


def test_symbolic_size_helper():
    """Symbolic ``T[N*M]`` heap transient -> ``constexpr T_size(long long M, long long N)``."""
    n, m = dace.symbol('N'), dace.symbol('M')
    build = lambda name: heap_pipeline_1d(name, n * m, '0:N*M')
    base = dict(A=np.random.rand(48), B=np.zeros(48), N=6, M=8)

    _, experimental = assert_bit_exact(build, 'symsize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'symsize_inspect')
    definition = size_helper_definition(code, 'T')
    assert 'constexpr' in definition, definition
    assert 'long long M' in definition and 'long long N' in definition, definition
    assert '(M * N)' in definition, definition
    assert allocation_extent(code, 'T') == 'T_size(M, N)'


def test_ipow_size_helper():
    """``total_size = ipow(N, 2)`` (``T[N*N]``) -> single-symbol ``constexpr T_size(long long N)``."""
    n = dace.symbol('N')
    build = lambda name: heap_pipeline_1d(name, n * n, '0:N*N')
    base = dict(A=np.random.rand(49), B=np.zeros(49), N=7)

    _, experimental = assert_bit_exact(build, 'ipowsize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'ipowsize_inspect')
    definition = size_helper_definition(code, 'T')
    assert 'constexpr' in definition and 'long long N' in definition, definition
    assert '(N * N)' in definition, definition
    assert allocation_extent(code, 'T') == 'T_size(N)'


def test_constant_size_helper():
    """A constant size gives a nullary helper, ``consteval`` from C++20 on."""
    build = lambda name: heap_pipeline_1d(name, 200, '0:200')
    base = dict(A=np.random.rand(200), B=np.zeros(200))

    _, experimental = assert_bit_exact(build, 'constsize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'constsize_inspect')
    definition = size_helper_definition(code, 'T')
    expected_qual = 'consteval' if int(str(Config.get('compiler', 'cpp_standard')).strip()) >= 20 else 'constexpr'
    assert expected_qual in definition, definition
    assert 'T_size()' in definition and 'return 200;' in definition, definition
    assert allocation_extent(code, 'T') == 'T_size()'


def test_bare_single_symbol_not_wrapped():
    """A bare single-symbol size ``T[N]`` is NOT wrapped (wrapping ``N`` is no win)."""
    n = dace.symbol('N')
    build = lambda name: heap_pipeline_1d(name, n, '0:N')
    base = dict(A=np.random.rand(64), B=np.zeros(64), N=64)

    _, experimental = assert_bit_exact(build, 'baresize', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'baresize_inspect')
    assert 'T_size' not in code, 'a bare single-symbol size must not be wrapped in a helper'
    assert allocation_extent(code, 'T') == 'N'


def test_distinct_size_helpers_across_nested_sdfgs():
    """Same transient name ``T`` at two sizes across a nested SDFG -> distinct helpers."""
    build = nested_same_name_sdfg
    base = dict(A=np.random.rand(5), B=np.zeros(5), N=5, M=3)

    _, experimental = assert_bit_exact(build, 'nestedsize', base)
    assert np.array_equal(experimental['B'][0], base['A'][0])

    code = experimental_code(build, 'nestedsize_inspect')
    inner_def = size_helper_definition(code, 'inner_T')
    outer_def = size_helper_definition(code, 'T')
    assert '(N * N)' in inner_def, inner_def
    assert '(M * N)' in outer_def, outer_def
    assert 'inner_T_size(N)' in allocation_line(code, 'inner_T')
    assert 'T_size(M, N)' in allocation_line(code, 'T')


if __name__ == "__main__":
    test_symbolic_size_helper()
    test_ipow_size_helper()
    test_constant_size_helper()
    test_bare_single_symbol_not_wrapped()
    test_distinct_size_helpers_across_nested_sdfgs()
