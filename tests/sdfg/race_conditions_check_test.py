# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import pytest


def test_memlet_range_overlap_numeric_ranges():
    sdfg = dace.SDFG('memlet_range_overlap_numeric_ranges')
    sdfg.add_array('A', [5], dace.int32)
    sdfg.add_array('B', [5], dace.int32)

    state = sdfg.add_state()
    t1 = state.add_tasklet('first_tasklet', {'a'}, {'b'}, 'b[0] = a[0] + 10')
    t2 = state.add_tasklet('second_tasklet', {'a'}, {'b'}, 'b[1] = a[1] - 20')

    r = state.add_read('A')
    w = state.add_write('B')

    state.add_edge(r, None, t1, 'a', dace.Memlet('A[0:2]'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[0:2]'))

    state.add_edge(t1, 'b', w, None, dace.Memlet('B[0:2]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[1:4]'))

    A = np.ones([5], dtype=np.int32)
    B = np.zeros([5], dtype=np.int32)

    with pytest.warns(UserWarning):
        sdfg(A=A, B=B)


def test_memlet_range_overlap_symbolic_ranges():
    M = dace.symbol('M')
    N = dace.symbol('N')

    sdfg = dace.SDFG('memlet_range_overlap_symbolic_ranges')
    sdfg.add_array('A', [M], dace.int32)
    sdfg.add_array('B', [N], dace.int32)

    state = sdfg.add_state()
    t1 = state.add_tasklet('first_tasklet', {'a'}, {'b'}, 'b[0] = a[0] + 10')
    t2 = state.add_tasklet('second_tasklet', {'a'}, {'b'}, 'b[1] = a[1] - 20')

    r = state.add_read('A')
    w = state.add_write('B')

    state.add_edge(r, None, t1, 'a', dace.Memlet('A[M-3:M-1]'))
    state.add_edge(r, None, t2, 'a', dace.Memlet('A[M-3:M-1]'))

    state.add_edge(t1, 'b', w, None, dace.Memlet('B[N-4:N-1]'))
    state.add_edge(t2, 'b', w, None, dace.Memlet('B[N-3:N-1]'))

    A = np.ones([5], dtype=np.int32)
    B = np.zeros([5], dtype=np.int32)

    with pytest.warns(UserWarning):
        with pytest.raises(KeyError):
            sdfg(A=A, B=B)


if __name__ == '__main__':
    test_memlet_range_overlap_numeric_ranges()
    test_memlet_range_overlap_symbolic_ranges()
