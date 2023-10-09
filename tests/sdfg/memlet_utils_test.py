# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np
import pytest
from dace.sdfg import memlet_utils as mu


def _replace_zero_with_one(memlet: dace.Memlet) -> dace.Memlet:
    for i, s in enumerate(memlet.subset):
        if s == 0:
            memlet.subset[i] = 1
    return memlet


@pytest.mark.parametrize('filter_type', ['none', 'same_array', 'different_array'])
def test_replace_memlet(filter_type):
    # Prepare SDFG
    sdfg = dace.SDFG('replace_memlet')
    sdfg.add_array('A', [2, 2], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state1 = sdfg.add_state()
    state2 = sdfg.add_state()
    state3 = sdfg.add_state()
    end_state = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge('A[0, 0] > 0'))
    sdfg.add_edge(state1, state3, dace.InterstateEdge('A[0, 0] <= 0'))
    sdfg.add_edge(state2, end_state, dace.InterstateEdge())
    sdfg.add_edge(state3, end_state, dace.InterstateEdge())

    t2 = state2.add_tasklet('write_one', {}, {'out'}, 'out = 1')
    t3 = state3.add_tasklet('write_two', {}, {'out'}, 'out = 2')
    w2 = state2.add_write('B')
    w3 = state3.add_write('B')
    state2.add_memlet_path(t2, w2, src_conn='out', memlet=dace.Memlet('B'))
    state3.add_memlet_path(t3, w3, src_conn='out', memlet=dace.Memlet('B'))

    # Filter memlets
    if filter_type == 'none':
        filter = set()
    elif filter_type == 'same_array':
        filter = {'A'}
    elif filter_type == 'different_array':
        filter = {'B'}

    # Replace memlets in conditions
    replacer = mu.MemletReplacer(sdfg.arrays, _replace_zero_with_one, filter)
    for e in sdfg.edges():
        e.data.condition.code[0] = replacer.visit(e.data.condition.code[0])

    # Compile and run
    sdfg.compile()

    A = np.array([[1, 1], [1, -1]], dtype=np.float64)
    B = np.array([0], dtype=np.float64)
    sdfg(A=A, B=B)

    if filter_type in {'none', 'same_array'}:
        assert B[0] == 2
    else:
        assert B[0] == 1


if __name__ == '__main__':
    test_replace_memlet('none')
    test_replace_memlet('same_array')
    test_replace_memlet('different_array')
