# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient that only stores literals is emitted as a ``constexpr`` initializer, with no runtime allocation, zeroing
or initialization, and the results match legacy."""
import re

import numpy as np

import dace
from tests.codegen.readable.conftest import assert_bit_exact, experimental_code, run_variant, EXPERIMENTAL, LEGACY

N = dace.symbol('N')


def literal_table_sdfg(name, length, values):
    """``arr[i] = values[i]`` for the given indices, then ``B = A + arr``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [length], dace.float64)
    sdfg.add_array('B', [length], dace.float64)
    sdfg.add_transient('arr', [length], dace.float64)
    init = sdfg.add_state('init')
    arr = init.add_access('arr')
    for index, value in values.items():
        tasklet = init.add_tasklet(f'set_{index}', {}, {'o'}, f'o = {value}')
        init.add_edge(tasklet, 'o', arr, None, dace.Memlet(f'arr[{index}]'))
    compute = sdfg.add_state_after(init, 'compute')
    me, mx = compute.add_map('m', {'i': f'0:{length}'})
    add = compute.add_tasklet('add', {'a', 'r'}, {'o'}, 'o = a + r')
    compute.add_memlet_path(compute.add_access('A'), me, add, dst_conn='a', memlet=dace.Memlet('A[i]'))
    compute.add_memlet_path(compute.add_access('arr'), me, add, dst_conn='r', memlet=dace.Memlet('arr[i]'))
    compute.add_memlet_path(add, mx, compute.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def scalar_literal_sdfg(name):
    """``s = 3.0`` then ``B[i] = A[i] * s``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('s', [1], dace.float64)
    init = sdfg.add_state('init')
    setter = init.add_tasklet('setc', {}, {'o'}, 'o = 3.0')
    init.add_edge(setter, 'o', init.add_access('s'), None, dace.Memlet('s[0]'))
    compute = sdfg.add_state_after(init, 'compute')
    me, mx = compute.add_map('m', {'i': '0:N'})
    mul = compute.add_tasklet('mul', {'a', 'sc'}, {'o'}, 'o = a * sc')
    compute.add_memlet_path(compute.add_access('A'), me, mul, dst_conn='a', memlet=dace.Memlet('A[i]'))
    compute.add_memlet_path(compute.add_access('s'), me, mul, dst_conn='sc', memlet=dace.Memlet('s[0]'))
    compute.add_memlet_path(mul, mx, compute.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def scalar_constant_subscript_sdfg(name):
    """A tasklet reads ``C[0]``, a 0-dimensional SDFG constant, which has no strides to index with."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('out', [4], dace.float64)
    sdfg.add_constant('C', np.float64(3.0), dace.data.Scalar(dace.float64))
    state = sdfg.add_state('main')
    me, mx = state.add_map('m', {'i': '0:4'})
    tasklet = state.add_tasklet('r', {}, {'o'}, 'o = C[0]', language=dace.Language.Python)
    state.add_edge(me, None, tasklet, None, dace.Memlet())
    state.add_memlet_path(tasklet, mx, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg


def assert_constexpr_without_runtime_init(code, array):
    lines = code.splitlines()
    assert any('constexpr' in line and f'{array}[' in line and '= {' in line for line in lines), code
    for line in lines:
        if re.search(rf'\b{array}\[', line) and 'constexpr' not in line:
            assert 'new' not in line and 'memset' not in line, line


def test_a_literal_scalar_is_constexpr_and_matches_legacy():
    assert_constexpr_without_runtime_init(experimental_code(scalar_literal_sdfg, 'sc_code'), 's')
    base = dict(A=np.random.rand(8), B=np.zeros(8), N=8)
    _, readable = assert_bit_exact(scalar_literal_sdfg, 'sc_run', base)
    assert np.allclose(readable['B'], base['A'] * 3.0)


def test_a_literal_table_is_constexpr_and_matches_legacy():
    build = lambda name: literal_table_sdfg(name, 4, {0: '0.0', 1: '1.0', 2: '2.0', 3: '3.0'})
    assert_constexpr_without_runtime_init(experimental_code(build, 'full_code'), 'arr')
    base = dict(A=np.random.rand(4), B=np.zeros(4))
    _, readable = assert_bit_exact(build, 'full_run', base)
    assert np.allclose(readable['B'], base['A'] + np.array([0., 1., 2., 3.]))


def test_the_unwritten_elements_of_a_literal_table_are_zero():
    build = lambda name: literal_table_sdfg(name, 4, {1: '5.0', 2: '6.0'})
    assert_constexpr_without_runtime_init(experimental_code(build, 'partial_code'), 'arr')
    base = dict(A=np.zeros(4), B=np.zeros(4))
    readable = run_variant(build, 'partial_run_readable', EXPERIMENTAL, base)
    legacy = run_variant(build, 'partial_run_legacy', LEGACY, base)
    assert np.array_equal(readable['B'], [0., 5., 6., 0.])
    assert np.array_equal(legacy['B'][[1, 2]], readable['B'][[1, 2]])


def test_a_subscript_of_a_scalar_constant_reads_the_bare_name():
    code = experimental_code(scalar_constant_subscript_sdfg, 'sc_subscript')
    assert 'C[' not in code and '= C;' in code
