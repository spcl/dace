# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains test cases for the symbolic memory-volume analysis. """

import pytest
import sympy as sp
import dace
from dace import SDFG
from dace.sdfg.state import LoopRegion
from dace.sdfg.performance_evaluation.total_volume import analyze_sdfg


def make_copy_sdfg(name: str, shape, dtype) -> SDFG:
    """
    Build a minimal SDFG that copies array ``A`` into array ``B``.

    :param name: Name of the SDFG.
    :param shape: Shape of both arrays.
    :param dtype: Element data type of both arrays.
    :return: The constructed SDFG.
    """
    sdfg = SDFG(name)
    sdfg.add_array('A', shape=shape, dtype=dtype)
    sdfg.add_array('B', shape=shape, dtype=dtype)
    state = sdfg.add_state('s0')
    a = state.add_read('A')
    b = state.add_write('B')
    state.add_nedge(a, b, dace.Memlet(f'A[{",".join(f"0:{s}" for s in shape)}]'))
    return sdfg


def test_empty_sdfg():
    sdfg = SDFG('empty')
    sdfg.add_state('s0')
    read, write = analyze_sdfg(sdfg)
    assert sp.simplify(read) == 0
    assert sp.simplify(write) == 0


def test_returns_sympy():
    sdfg = SDFG('empty')
    sdfg.add_state('s0')
    read, write = analyze_sdfg(sdfg)
    assert isinstance(read, sp.Basic)
    assert isinstance(write, sp.Basic)


def test_copy_float64():
    """Copying 8 float64s: expect 64 bytes read and 64 bytes written."""
    sdfg = make_copy_sdfg('copy_f64', [8], dace.float64)
    read, write = analyze_sdfg(sdfg)
    assert sp.simplify(read - 64) == 0
    assert sp.simplify(write - 64) == 0


def test_copy_float32_half_bytes():
    """float32 should produce half the volume of float64 for the same shape."""
    sdfg64 = make_copy_sdfg('copy_f64', [16], dace.float64)
    sdfg32 = make_copy_sdfg('copy_f32', [16], dace.float32)
    r64, w64 = analyze_sdfg(sdfg64)
    r32, w32 = analyze_sdfg(sdfg32)
    assert sp.simplify(r64 - 2 * r32) == 0
    assert sp.simplify(w64 - 2 * w32) == 0


def test_read_write_symmetry_on_copy():
    """A pure copy should read and write the same volume."""
    sdfg = make_copy_sdfg('copy_sym', [32], dace.float64)
    read, write = analyze_sdfg(sdfg)
    assert sp.simplify(read - write) == 0


def test_two_independent_copies():
    """Two sequential copy states should double the volume."""
    sdfg = SDFG('two_copies')
    sdfg.add_array('A', shape=[8], dtype=dace.float64)
    sdfg.add_array('B', shape=[8], dtype=dace.float64)
    sdfg.add_array('C', shape=[8], dtype=dace.float64)

    s0 = sdfg.add_state('s0')
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge())

    for state, src, dst in [(s0, 'A', 'B'), (s1, 'B', 'C')]:
        state.add_nedge(state.add_read(src), state.add_write(dst), dace.Memlet(f'{src}[0:8]'))

    read, write = analyze_sdfg(sdfg)
    assert sp.simplify(read - 128) == 0
    assert sp.simplify(write - 128) == 0


def test_symbolic_shape():
    """An SDFG with a symbolic dimension N should return a symbolic volume."""
    sdfg = SDFG('sym_shape')
    N = dace.symbol('N', dace.int32)
    sdfg.add_array('A', shape=[N], dtype=dace.float64)
    sdfg.add_array('B', shape=[N], dtype=dace.float64)
    state = sdfg.add_state('s0')
    state.add_nedge(state.add_read('A'), state.add_write('B'), dace.Memlet('A[0:N]'))
    read, write = analyze_sdfg(sdfg)
    assert 8 * N - read == 0
    assert 8 * N - write == 0


def test_view_access_node_excluded():
    """Volumes from View arrays should not be double-counted."""
    sdfg = SDFG('view_test')
    sdfg.add_array('A', shape=[16], dtype=dace.float64)
    sdfg.add_view('V', shape=[8], dtype=dace.float64)
    sdfg.add_array('B', shape=[8], dtype=dace.float64)

    state = sdfg.add_state('s0')
    a = state.add_read('A')
    v = state.add_access('V')
    b = state.add_write('B')

    # A -> V (view, should be ignored) -> B
    state.add_nedge(a, v, dace.Memlet('A[0:8]'))
    state.add_nedge(v, b, dace.Memlet('V[0:8]'))

    read, write = analyze_sdfg(sdfg)
    # V is a View, so its edges must not contribute a second time: the read
    # volume should reflect only A, not A + V.
    assert sp.simplify(read - 64) == 0  # 8 elements * 8 bytes from A
    assert sp.simplify(write - 64) == 0  # 8 elements * 8 bytes into B


def test_map_doubles_volume():
    """A map over 2 iterations should double the access volume."""
    sdfg = SDFG('map_test')
    sdfg.add_array('A', shape=[2, 8], dtype=dace.float64)
    sdfg.add_array('B', shape=[2, 8], dtype=dace.float64)

    state = sdfg.add_state('s0')
    a = state.add_read('A')
    b = state.add_write('B')

    me, mx = state.add_map('outer', {'i': '0:2'})
    t = state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')

    state.add_memlet_path(a, me, t, memlet=dace.Memlet('A[i, 0:8]'), dst_conn='inp')
    state.add_memlet_path(t, mx, b, memlet=dace.Memlet('B[i, 0:8]'), src_conn='out')

    read, write = analyze_sdfg(sdfg)
    # 2 iterations * 8 elements * 8 bytes = 128 bytes each
    assert sp.simplify(read - 128) == 0
    assert sp.simplify(write - 128) == 0


def test_loop_multiplies_volume():
    """A loop region iterating N times should scale the volume by N."""
    sdfg = SDFG('loop_test')
    N = dace.symbol('N', dace.int32)
    sdfg.add_array('A', shape=[8], dtype=dace.float64)
    sdfg.add_array('B', shape=[8], dtype=dace.float64)

    loop = LoopRegion('loop',
                      condition_expr='i < N',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      inverted=False,
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.start_block = sdfg.node_id(loop)

    body = loop.add_state('body')
    body.add_nedge(body.add_read('A'), body.add_write('B'), dace.Memlet('A[0:8]'))

    read, write = analyze_sdfg(sdfg)
    # 8 elements * 8 bytes * N iterations = 64*N bytes
    expected = 64 * N
    assert sp.simplify(read - expected) == 0
    assert sp.simplify(write - expected) == 0


# Note: per-kernel volume checks on real kernels (jacobi-1d, jacobi-2d, ...) live in
# ``polybench_analysis_test.py``, which reuses the canonical ``tests/polybench`` kernels
# rather than redefining them here.

if __name__ == '__main__':
    pytest.main([__file__])
