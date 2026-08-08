# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

from dace import graphlib as nx


@dace.program
def sftw(A: dace.float64[20]):
    B = dace.define_local([20], dace.float64)
    C = dace.define_local([20], dace.float64)
    D = dace.define_local([20], dace.float64)
    E = dace.define_local([20], dace.float64)
    dup = dace.define_local([20], dace.float64)

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << A[i]
            b >> B[i]
            b = a

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << B[i]
            b >> dup[i]
            b = a

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << dup[i]
            b >> D[i]
            b = a + 2

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << A[i]
            b >> C[i]
            b = a + 1

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << C[i]
            b >> dup[i]
            b = a + 1

    for i in dace.map[0:20]:
        with dace.tasklet:
            a << dup[i]
            b >> E[i]
            b = a + 3

    for i in dace.map[0:20]:
        with dace.tasklet:
            d << D[i]
            e << E[i]
            a >> A[i]
            a = d + e


def test_sftw():
    A = np.random.rand(20)
    expected = 2 * A + 7
    sdfg = sftw.to_sdfg(simplify=False)
    assert len(sdfg.nodes()) > 2, 'nothing to fuse: the test would be vacuous'
    sdfg.simplify()

    # Ensure almost all states were fused. How far fusion gets depends on how finely ``dup`` is
    # versioned, so this is a bound rather than an exact count -- pinning the count made the test
    # fail on an improvement (fusing to one state) that is provably correct, checked below.
    assert len(sdfg.nodes()) <= 2

    # The hazard this test is named for: ``dup`` is written by two different producers and read by
    # two different consumers, and each read must complete before the next overwrite. Once the
    # states are fused that ordering is carried by edges inside the state, not by state order, so
    # assert it directly -- numerics alone cannot: a fused state that leaves the pair unordered
    # still validates and still happens to compute the right answer whenever the emission order
    # falls the right way (the ``correlation`` miscompile had exactly that shape).
    for state in sdfg.states():
        dups = [n for n in state.data_nodes() if n.data == 'dup']
        readers = [n for n in dups if state.out_degree(n) > 0]
        writers = [n for n in dups if state.in_degree(n) > 0]
        for reader in readers:
            for writer in writers:
                if reader is writer:
                    continue
                ordered = (nx.has_path(state.nx, reader, writer) or nx.has_path(state.nx, writer, reader))
                assert ordered, f'unordered read/overwrite of "dup" in fused state {state.label}'

    sdfg(A=A)

    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_sftw()
