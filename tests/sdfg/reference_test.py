# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the use of Reference data descriptors. """
import dace
import numpy as np


def test_reference_branch():
    sdfg = dace.SDFG('refbranch')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [20], dace.float64)
    sdfg.add_symbol('i', dace.int64)
    sdfg.add_reference('ref', [20], dace.float64)
    sdfg.add_array('out', [20], dace.float64)

    # Branch to a or b depending on i
    start = sdfg.add_state()
    a = sdfg.add_state()
    b = sdfg.add_state()
    finish = sdfg.add_state()
    sdfg.add_edge(start, a, dace.InterstateEdge('i < 5'))
    sdfg.add_edge(start, b, dace.InterstateEdge('i >= 5'))
    sdfg.add_edge(a, finish, dace.InterstateEdge())
    sdfg.add_edge(b, finish, dace.InterstateEdge())

    # Copy from reference to output
    a.add_edge(a.add_read('A'), None, a.add_write('ref'), 'set', dace.Memlet('A'))
    b.add_edge(b.add_read('B'), None, b.add_write('ref'), 'set', dace.Memlet('B'))

    r = finish.add_read('ref')
    w = finish.add_write('out')
    finish.add_nedge(r, w, dace.Memlet('ref'))

    A = np.random.rand(20)
    B = np.random.rand(20)
    out = np.random.rand(20)

    sdfg(A=A, B=B, out=out, i=10)
    assert np.allclose(out, B)

    sdfg(A=A, B=B, out=out, i=1)
    assert np.allclose(out, A)


if __name__ == '__main__':
    test_reference_branch()
