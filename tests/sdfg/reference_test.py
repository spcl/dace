# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the use of Reference data descriptors. """
import dace
from dace.sdfg import validation
from dace.transformation.passes.analysis import FindReferenceSources
import numpy as np
import pytest


def test_unset_reference():
    sdfg = dace.SDFG('tester')
    sdfg.add_reference('ref', [20], dace.float64)
    state = sdfg.add_state()
    t = state.add_tasklet('doit', {'a'}, {'b'}, 'b = a + 1')
    state.add_edge(state.add_read('ref'), None, t, 'a', dace.Memlet('ref[0]'))
    state.add_edge(t, 'b', state.add_write('ref'), None, dace.Memlet('ref[1]'))

    with pytest.raises(validation.InvalidSDFGNodeError):
        sdfg.validate()


def _create_branch_sdfg():
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
    return sdfg


def test_reference_branch():
    sdfg = _create_branch_sdfg()

    A = np.random.rand(20)
    B = np.random.rand(20)
    out = np.random.rand(20)

    sdfg(A=A, B=B, out=out, i=10)
    assert np.allclose(out, B)

    sdfg(A=A, B=B, out=out, i=1)
    assert np.allclose(out, A)


def test_reference_sources_pass():
    sdfg = _create_branch_sdfg()
    sources = FindReferenceSources().apply_pass(sdfg, {})
    assert len(sources) == 1  # There is only one SDFG
    sources = sources[0]
    assert len(sources) == 1 and 'ref' in sources  # There is one reference
    sources = sources['ref']
    assert sources == {dace.Memlet('A[0:20]', volume=1), dace.Memlet('B[0:20]', volume=1)}


if __name__ == '__main__':
    test_unset_reference()
    test_reference_branch()
    test_reference_sources_pass()
