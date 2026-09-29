# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Layout stress tests for the experimental (readable) CPU code generator's per-array
index functions.

Under ``compiler.cpu.implementation = experimental`` every array access is emitted
as ``A[A_idx(i, j, ...)]``, where the generated ``A_idx`` is a ``static ...
constexpr`` function computing ``sum(index_d * strides_d) + sum(offset_d *
strides_d)`` from the descriptor. The legacy generator inlines the same math via
``cpp_offset_expr``. These tests prove the two agree for NON-TRIVIAL layouts
(Fortran/column-major, padded, offset, strided stencils, alignment).

Each case builds ``input A (plain C) -> map writes transient T (exotic layout) ->
map reads T into output B (plain C)``. Only ``T`` carries the exotic
``strides``/``offset``/``total_size``/``alignment``; ``A`` and ``B`` stay plain, so
both generators consume ``T``'s descriptor identically. The SDFG is compiled + run
under BOTH generators on identical inputs (generated once, ``copy.deepcopy``'d per
variant) and the outputs must be BIT-EXACT -- which proves ``T_idx`` reproduces the
legacy flat-offset math for that layout. For a few cases the generated experimental
C++ is additionally inspected (the ``T_idx`` body / the ``aligned_alloc`` call).

CPU kernels are run in a forked child (repo rule: an experimental kernel that
segfaults must not take down pytest) via :func:`conftest.run_isolated`.

Note on DaCe's ``offset`` validation: DaCe validates ``max_index + offset <
shape`` per dimension (``dace/sdfg/validation.py``), so an array with a non-zero
descriptor ``offset`` can only be accessed over an index range small enough that the
offset-shifted access still lands inside ``shape``. The offset cases therefore
restrict their map ranges accordingly; the ``offset`` is still fully exercised in
every ``T_idx`` call (it contributes the constant ``sum(offset_d * strides_d)``).
"""
import copy

import numpy as np
import pytest

import dace
from dace.dtypes import StorageType
from tests.codegen.readable.conftest import EXPERIMENTAL, LEGACY, run_isolated, use_implementation


# --------------------------------------------------------------------------- #
# SDFG builders: A (plain C) -> T (exotic layout) -> B (plain C)
# --------------------------------------------------------------------------- #
def elementwise_2d_sdfg(name, strides, offset, total_size, irange, jrange, alignment=0):
    """``T[i,j] = A[i,j] + 1`` then ``B[i,j] = T[i,j] * 2`` over ``irange x jrange``.

    ``A`` and ``B`` are plain C ``(6, 8)`` arrays; ``T`` carries the exotic layout.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [6, 8], dace.float64)
    sdfg.add_array('B', [6, 8], dace.float64)
    sdfg.add_transient('T', [6, 8],
                       dace.float64,
                       strides=strides,
                       offset=offset,
                       total_size=total_size,
                       alignment=alignment)

    write = sdfg.add_state('write')
    read_a, write_t = write.add_read('A'), write.add_write('T')
    entry, exit_node = write.add_map('write_map', dict(i=irange, j=jrange))
    add = write.add_tasklet('add_one', {'a'}, {'o'}, 'o = a + 1.0')
    write.add_memlet_path(read_a, entry, add, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    write.add_memlet_path(add, exit_node, write_t, src_conn='o', memlet=dace.Memlet('T[i, j]'))

    read = sdfg.add_state_after(write, 'read')
    read_t, write_b = read.add_read('T'), read.add_write('B')
    entry2, exit2 = read.add_map('read_map', dict(i=irange, j=jrange))
    mul = read.add_tasklet('mul_two', {'t'}, {'o'}, 'o = t * 2.0')
    read.add_memlet_path(read_t, entry2, mul, dst_conn='t', memlet=dace.Memlet('T[i, j]'))
    read.add_memlet_path(mul, exit2, write_b, src_conn='o', memlet=dace.Memlet('B[i, j]'))

    sdfg.validate()
    return sdfg


def stencil_1d_sdfg(name, n, strides, offset, total_size, write_range, stencil_range):
    """``T[i] = A[i]`` then ``B[i] = T[i-1] + T[i+1]`` -- exercises ``i-1``/``i+1``
    (and, when ``offset`` is set, ``i - offset``) through the strided ``T_idx``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('B', [n], dace.float64)
    sdfg.add_transient('T', [n], dace.float64, strides=strides, offset=offset, total_size=total_size)

    write = sdfg.add_state('write')
    read_a, write_t = write.add_read('A'), write.add_write('T')
    entry, exit_node = write.add_map('write_map', dict(i=write_range))
    copy_tasklet = write.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    write.add_memlet_path(read_a, entry, copy_tasklet, dst_conn='a', memlet=dace.Memlet('A[i]'))
    write.add_memlet_path(copy_tasklet, exit_node, write_t, src_conn='o', memlet=dace.Memlet('T[i]'))

    read = sdfg.add_state_after(write, 'stencil')
    read_t, write_b = read.add_read('T'), read.add_write('B')
    entry2, exit2 = read.add_map('stencil_map', dict(i=stencil_range))
    stencil = read.add_tasklet('stencil', {'left', 'right'}, {'o'}, 'o = left + right')
    read.add_memlet_path(read_t, entry2, stencil, dst_conn='left', memlet=dace.Memlet('T[i - 1]'))
    read.add_memlet_path(read_t, entry2, stencil, dst_conn='right', memlet=dace.Memlet('T[i + 1]'))
    read.add_memlet_path(stencil, exit2, write_b, src_conn='o', memlet=dace.Memlet('B[i]'))

    sdfg.validate()
    return sdfg


def aligned_1d_sdfg(name, n, alignment):
    """``T[i] = A[i] + 1`` then ``B[i] = T[i] * 2`` with ``T`` a heap array whose
    descriptor requests ``alignment``-byte alignment (drives ``std::aligned_alloc``).
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [n], dace.float64)
    sdfg.add_array('B', [n], dace.float64)
    # CPU_Heap forces heap allocation (a 200-elem array is below max_stack_array_size
    # and would otherwise land on the stack, bypassing aligned_alloc).
    sdfg.add_transient('T', [n], dace.float64, storage=StorageType.CPU_Heap, alignment=alignment)

    write = sdfg.add_state('write')
    read_a, write_t = write.add_read('A'), write.add_write('T')
    entry, exit_node = write.add_map('write_map', dict(i='0:%d' % n))
    add = write.add_tasklet('add_one', {'a'}, {'o'}, 'o = a + 1.0')
    write.add_memlet_path(read_a, entry, add, dst_conn='a', memlet=dace.Memlet('A[i]'))
    write.add_memlet_path(add, exit_node, write_t, src_conn='o', memlet=dace.Memlet('T[i]'))

    read = sdfg.add_state_after(write, 'read')
    read_t, write_b = read.add_read('T'), read.add_write('B')
    entry2, exit2 = read.add_map('read_map', dict(i='0:%d' % n))
    mul = read.add_tasklet('mul_two', {'t'}, {'o'}, 'o = t * 2.0')
    read.add_memlet_path(read_t, entry2, mul, dst_conn='t', memlet=dace.Memlet('T[i]'))
    read.add_memlet_path(mul, exit2, write_b, src_conn='o', memlet=dace.Memlet('B[i]'))

    sdfg.validate()
    return sdfg


# --------------------------------------------------------------------------- #
# Equivalence + codegen-inspection helpers
# --------------------------------------------------------------------------- #
def run_variant(build, name, implementation, base):
    """Build + compile + run one variant on a deep copy of ``base``; return outputs.

    Runs in a forked child (repo rule) via ``run_isolated``. ``base`` is generated
    once by the caller and inherited by both forks, so each variant deep-copies the
    same inputs -- a prerequisite for a meaningful bit-exact comparison, since ``T``
    is only partially overwritten in the offset/stencil cases.
    """

    def work():
        sdfg = build(name)
        arrays = copy.deepcopy(base)
        sdfg.compile()(**arrays)
        return {key: value for key, value in arrays.items() if isinstance(value, np.ndarray)}

    with use_implementation(implementation):
        return run_isolated(work)


def assert_bit_exact(build, base_name, base):
    """Run ``build`` under legacy and experimental; assert every output is bit-exact.

    Bit-exactness proves ``T_idx`` reproduces the legacy flat-offset math for the
    layout under test. Returns the (legacy, experimental) output dicts.
    """
    legacy = run_variant(build, base_name + '_legacy', LEGACY, base)
    experimental = run_variant(build, base_name + '_experimental', EXPERIMENTAL, base)
    assert set(legacy) == set(experimental)
    for key in legacy:
        assert np.array_equal(
            legacy[key], experimental[key]), ('%s: experimental CPU codegen is not bit-exact vs legacy for output %s' %
                                              (base_name, key))
    return legacy, experimental


def experimental_code(build, name):
    """Generated experimental C++ (codegen only, no compile) for inspection."""
    with use_implementation(EXPERIMENTAL):
        sdfg = build(name)
        return sdfg.generate_code()[0].clean_code


def index_function_body(code):
    """The single-line body of the emitted ``T_idx`` index function."""
    lines = [line.strip() for line in code.splitlines() if 'T_idx(' in line and 'return' in line]
    assert lines, 'experimental codegen emitted no T_idx index function'
    return lines[0]


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def test_fortran_column_major(require_experimental):
    """Case 1: column-major / Fortran strides ``[1, 6]`` on a ``(6, 8)`` transient."""
    base = dict(A=np.random.rand(6, 8), B=np.zeros((6, 8)))
    build = lambda name: elementwise_2d_sdfg(
        name, strides=[1, 6], offset=None, total_size=48, irange='0:6', jrange='0:8')

    _, experimental = assert_bit_exact(build, 'fortran', base)
    # Full elementwise write -> analytically B == (A + 1) * 2 everywhere.
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    # The column stride 6 must scale the SECOND index in the index function.
    body = index_function_body(experimental_code(build, 'fortran_inspect'))
    assert '6 * __d1' in body, body


def test_padded_row_strides(require_experimental):
    """Case 2: rows padded to 16 elements -- strides ``[16, 1]`` on ``(6, 8)``."""
    base = dict(A=np.random.rand(6, 8), B=np.zeros((6, 8)))
    build = lambda name: elementwise_2d_sdfg(
        name, strides=[16, 1], offset=None, total_size=6 * 16, irange='0:6', jrange='0:8')

    _, experimental = assert_bit_exact(build, 'padded', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    # The padded row stride 16 must scale the FIRST index in the index function.
    body = index_function_body(experimental_code(build, 'padded_inspect'))
    assert '16 * __d0' in body, body


def test_nonzero_offset(require_experimental):
    """Case 3: non-zero descriptor ``offset [1, 2]`` (C strides ``[8, 1]``).

    Access is restricted to ``i<5, j<6`` so ``max_index + offset < shape`` (DaCe
    validates ``offset`` against ``shape``). The offset contributes a constant
    ``1*8 + 2*1 = 10`` to every ``T_idx`` call.
    """
    base = dict(A=np.random.rand(6, 8), B=np.zeros((6, 8)))
    build = lambda name: elementwise_2d_sdfg(
        name, strides=[8, 1], offset=[1, 2], total_size=58, irange='0:5', jrange='0:6')

    _, experimental = assert_bit_exact(build, 'offset', base)
    expected = base['B'].copy()
    expected[0:5, 0:6] = (base['A'][0:5, 0:6] + 1.0) * 2.0
    assert np.array_equal(experimental['B'], expected)

    # offset*stride == 1*8 + 2*1 == 10 appears as the constant term.
    body = index_function_body(experimental_code(build, 'offset_inspect'))
    assert '+ 10' in body, body


def test_strided_stencil(require_experimental):
    """Case 4: ``i-1``/``i+1`` stencil on a stride-2 ``T[32]`` transient."""
    n = 32
    base = dict(A=np.random.rand(n), B=np.zeros(n))
    build = lambda name: stencil_1d_sdfg(
        name, n=n, strides=[2], offset=None, total_size=64, write_range='0:32', stencil_range='1:31')

    _, experimental = assert_bit_exact(build, 'stencil', base)
    expected = base['B'].copy()
    expected[1:n - 1] = base['A'][0:n - 2] + base['A'][2:n]
    assert np.array_equal(experimental['B'], expected)

    body = index_function_body(experimental_code(build, 'stencil_inspect'))
    assert '2 * __d0' in body, body


def test_offset_strided_stencil(require_experimental):
    """Case 5: ``i-1``/``i+1`` stencil on a strided ``T[32]`` with ``offset [1]``.

    Confirms the ``i - 1`` subtraction lowers correctly through ``T_idx`` when the
    descriptor also carries an offset (stride 3, offset 1 -> constant ``+3``).
    Ranges are restricted so ``max_index + offset < 32``.
    """
    n = 32
    base = dict(A=np.random.rand(n), B=np.zeros(n))
    build = lambda name: stencil_1d_sdfg(
        name, n=n, strides=[3], offset=[1], total_size=99, write_range='0:31', stencil_range='1:30')

    _, experimental = assert_bit_exact(build, 'offset_stencil', base)
    expected = base['B'].copy()
    expected[1:30] = base['A'][0:29] + base['A'][2:31]
    assert np.array_equal(experimental['B'], expected)

    # stride 3 with offset 1 -> body is (3 * __d0) + 3.
    body = index_function_body(experimental_code(build, 'offset_stencil_inspect'))
    assert '3 * __d0' in body and '+ 3' in body, body


def test_alignment(require_experimental):
    """Case 6: heap transient allocated with the same aligned ``new[]`` as the legacy generator."""
    n = 200
    base = dict(A=np.random.rand(n), B=np.zeros(n))
    build = lambda name: aligned_1d_sdfg(name, n=n, alignment=128)

    _, experimental = assert_bit_exact(build, 'aligned', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    # The experimental generator allocates T with the base aligned new[] (T = new T DACE_ALIGN(64)[...]).
    code = experimental_code(build, 'aligned_inspect')
    assert any('T = new ' in line and 'DACE_ALIGN' in line for line in code.splitlines()), \
        'experimental codegen did not use an aligned new[] for T'


if __name__ == '__main__':
    import sys

    sys.exit(pytest.main([__file__, '-q']))
