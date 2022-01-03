# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace
from dace import dtypes
from dace.transformation import coarsening_transformations
from dace.transformation.dataflow import SimpleTaskletFusion
import pytest


@dace.program
def map_with_tasklets(A: dace.float32[20], B: dace.float32[10]):
    C = np.zeros_like(B)
    for i in dace.map[0:10]:
        a = A[i] + B[i]
        b = a * A[2 * i]
        c = a + B[i]
        d = c / b
        C[i] = a * d
    return C


def _make_sdfg(l: str = 'Python'):

    language = dtypes.Language.Python if l == 'Python' else dtypes.Language.CPP
    endl = '\n' if l == 'Python' else ';\n'

    sdfg = dace.SDFG(f'map_with_{l}_tasklets')
    _, arrA = sdfg.add_array('A', (20, ), dace.float32)
    _, arrB = sdfg.add_array('B', (10, ), dace.float32)
    _, arrC = sdfg.add_array('C', (10, ), dace.float32)

    state = sdfg.add_state(is_start_state=True)
    A = state.add_read('A')
    B = state.add_read('B')
    C = state.add_write('C')
    me, mx = state.add_map('Map', {'i': '0:10'})
    inputs = {'__inp1', '__inp2'}
    outputs = {'__out'}
    ta = state.add_tasklet('a', inputs, {'__out1', '__out2', '__out3'},
                           f'__out1 = __inp1 + __inp2{endl}__out2 = __out1{endl}__out3 = __out1{endl}', language)
    tb = state.add_tasklet('b', inputs, outputs, f'__out = __inp1 * __inp2{endl}', language)
    tc = state.add_tasklet('c', inputs, outputs, f'__out = __inp1 + __inp2{endl}', language)
    td = state.add_tasklet('d', inputs, outputs, f'__out = __inp1 / __inp2{endl}', language)
    te = state.add_tasklet('e', inputs, outputs, f'__out = __inp1 * __inp2{endl}', language)
    state.add_memlet_path(A, me, ta, memlet=dace.Memlet('A[i]'), dst_conn='__inp1')
    state.add_memlet_path(B, me, ta, memlet=dace.Memlet('B[i]'), dst_conn='__inp2')
    state.add_memlet_path(A, me, tb, memlet=dace.Memlet('A[2*i]'), dst_conn='__inp2')
    state.add_memlet_path(B, me, tc, memlet=dace.Memlet('B[i]'), dst_conn='__inp2')
    state.add_edge(ta, '__out1', tb, '__inp1', dace.Memlet())
    state.add_edge(ta, '__out2', tc, '__inp1', dace.Memlet())
    state.add_edge(tb, '__out', td, '__inp2', dace.Memlet())
    state.add_edge(tc, '__out', td, '__inp1', dace.Memlet())
    state.add_edge(ta, '__out3', te, '__inp1', dace.Memlet())
    state.add_edge(td, '__out', te, '__inp2', dace.Memlet())
    state.add_memlet_path(te, mx, C, memlet=dace.Memlet('C[i]'), src_conn='__out')

    return sdfg


@pytest.mark.parametrize("l", [pytest.param('Python'), pytest.param('CPP')])
def test_map_with_tasklets(l: str):
    sdfg = _make_sdfg(l)
    coarsening_reduced = [xf for xf in coarsening_transformations() if xf.__name__ != 'SimpleTaskletFusion']
    sdfg.apply_transformations_repeated(coarsening_reduced)
    num = sdfg.apply_transformations_repeated(SimpleTaskletFusion)
    assert (num == 4)
    func = sdfg.compile()
    A = np.arange(1, 21, dtype=np.float32)
    B = np.arange(1, 11, dtype=np.float32)
    C = np.zeros((10, ), dtype=np.float32)
    func(A=A, B=B, C=C)
    ref = map_with_tasklets.f(A, B)
    assert (np.allclose(C, ref))


if __name__ == '__main__':
    test_map_with_tasklets(l='Python')
    test_map_with_tasklets(l='CPP')
