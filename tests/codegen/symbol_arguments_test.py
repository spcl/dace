# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np

N = dace.symbol('N')


def test_global_sizes():

    @dace.program
    def tester(A: dace.float64[N]):
        for i in dace.map[0:10]:
            A[i] = 2

    sdfg = tester.to_sdfg()
    # N describes a non-transient shape, so it is part of the interface and stays in the signature
    # even though no code here reads it. Deriving the signature from code use instead made it move
    # under optimization: a pass that eliminated the last use of an extent symbol dropped it, and a
    # later pass that reintroduced one brought it back, so a caller holding the earlier signature
    # could not call the later graph.
    assert 'N' in sdfg.arglist()

    a = np.random.rand(20)
    sdfg(a, N=20)
    assert np.allclose(a[:10], 2)


def test_global_sizes_used():

    @dace.program
    def tester(A: dace.float64[N]):
        for i in dace.map[0:10]:
            with dace.tasklet:
                a >> A[i]
                a = N

    sdfg = tester.to_sdfg()
    # N is used in a tasklet
    assert 'N' in sdfg.arglist()


def test_global_sizes_multidim():

    @dace.program
    def tester(A: dace.float64[N, N]):
        for i, j in dace.map[0:10, 0:10]:
            A[i, j] = 2

    sdfg = tester.to_sdfg()
    # Here N is implicitly used in the index expression, so it should be in the arguments
    assert 'N' in sdfg.arglist()


def test_nested_sdfg_redefinition():
    sdfg = dace.SDFG('tester')
    nsdfg = dace.SDFG('nester')
    state = sdfg.add_state()
    nnode = state.add_nested_sdfg(nsdfg, {}, {}, symbol_mapping=dict(sym=0))

    nstate = nsdfg.add_state()
    nstate.add_tasklet('nothing', {}, {}, 'a = sym')
    nstate2 = nsdfg.add_state()
    nsdfg.add_edge(nstate, nstate2, dace.InterstateEdge(assignments=dict(sym=1)))
    sdfg.compile()


if __name__ == '__main__':
    test_global_sizes()
    test_global_sizes_used()
    test_global_sizes_multidim()
    test_nested_sdfg_redefinition()
