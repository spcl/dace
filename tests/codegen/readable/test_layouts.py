# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Array accesses are emitted as ``A[A_idx(i, j, ...)]``. For column-major, padded, offset and strided layouts the readable
generator must reproduce the legacy offset arithmetic bit-exactly. Each kernel reads a plain array into a transient
with the layout under test and writes it back to a plain array.
"""
import numpy as np

import dace
from tests.codegen.readable.conftest import assert_bit_exact, experimental_code, heap_pipeline_1d


def elementwise_2d_sdfg(name, strides, offset, total_size, irange, jrange, alignment=0):
    """``T[i,j] = A[i,j] + 1`` then ``B[i,j] = T[i,j] * 2``, with ``T`` laid out as given."""
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
    """``T[i] = A[i]`` then ``B[i] = T[i-1] + T[i+1]``, with ``T`` laid out as given."""
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


def index_function_body(code):
    """The single-line body of the emitted ``T_idx`` index function."""
    lines = [line.strip() for line in code.splitlines() if 'T_idx(' in line and 'return' in line]
    assert lines, 'experimental codegen emitted no T_idx index function'
    return lines[0]


def test_fortran_column_major():
    """Column-major strides ``[1, 6]``."""
    base = dict(A=np.random.rand(6, 8), B=np.zeros((6, 8)))
    build = lambda name: elementwise_2d_sdfg(
        name, strides=[1, 6], offset=None, total_size=48, irange='0:6', jrange='0:8')

    _, experimental = assert_bit_exact(build, 'fortran', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    body = index_function_body(experimental_code(build, 'fortran_inspect'))
    assert '6 * __d1' in body, body


def test_padded_row_strides():
    """Rows padded to 16 elements."""
    base = dict(A=np.random.rand(6, 8), B=np.zeros((6, 8)))
    build = lambda name: elementwise_2d_sdfg(
        name, strides=[16, 1], offset=None, total_size=6 * 16, irange='0:6', jrange='0:8')

    _, experimental = assert_bit_exact(build, 'padded', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    body = index_function_body(experimental_code(build, 'padded_inspect'))
    assert '16 * __d0' in body, body


def test_nonzero_offset():
    """An offset ``[1, 2]`` adds a constant ``1*8 + 2*1`` to every index; the access range keeps the offset in bounds."""
    base = dict(A=np.random.rand(6, 8), B=np.zeros((6, 8)))
    build = lambda name: elementwise_2d_sdfg(
        name, strides=[8, 1], offset=[1, 2], total_size=58, irange='0:5', jrange='0:6')

    _, experimental = assert_bit_exact(build, 'offset', base)
    expected = base['B'].copy()
    expected[0:5, 0:6] = (base['A'][0:5, 0:6] + 1.0) * 2.0
    assert np.array_equal(experimental['B'], expected)

    body = index_function_body(experimental_code(build, 'offset_inspect'))
    assert '+ 10' in body, body


def test_strided_stencil():
    """A stencil on a stride-2 array."""
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


def test_offset_strided_stencil():
    """A stencil on a strided array with an offset, whose constant term is the stride."""
    n = 32
    base = dict(A=np.random.rand(n), B=np.zeros(n))
    build = lambda name: stencil_1d_sdfg(
        name, n=n, strides=[3], offset=[1], total_size=99, write_range='0:31', stencil_range='1:30')

    _, experimental = assert_bit_exact(build, 'offset_stencil', base)
    expected = base['B'].copy()
    expected[1:30] = base['A'][0:29] + base['A'][2:31]
    assert np.array_equal(experimental['B'], expected)

    body = index_function_body(experimental_code(build, 'offset_stencil_inspect'))
    assert '3 * __d0' in body and '+ 3' in body, body


def test_alignment():
    """A heap array keeps its requested alignment."""
    n = 200
    base = dict(A=np.random.rand(n), B=np.zeros(n))
    build = lambda name: heap_pipeline_1d(name, n, f'0:{n}', alignment=128)

    _, experimental = assert_bit_exact(build, 'aligned', base)
    assert np.array_equal(experimental['B'], (base['A'] + 1.0) * 2.0)

    code = experimental_code(build, 'aligned_inspect')
    assert any('T = new ' in line and 'std::align_val_t(128)' in line for line in code.splitlines()), \
        'experimental codegen did not use an aligned new[] for T'
