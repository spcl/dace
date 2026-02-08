# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from dace.sdfg.propagation import propagate_memlets_sdfg


def test_conditional():

    @dace.program
    def conditional(in1, out):
        for i in dace.map[0:10]:
            if i >= 1:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    inp = np.random.rand(10)
    outp = np.zeros((10, ))
    conditional(inp, outp)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_conditional_nested():

    @dace.program
    def conditional(in1, out):
        for i in dace.map[0:10]:
            if i >= 1:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    @dace.program
    def nconditional(in1, out):
        conditional(in1, out)

    inp = np.random.rand(10)
    outp = np.zeros((10, ))
    nconditional(inp, outp)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_runtime_conditional():

    @dace.program
    def rconditional(in1, out, mask):
        for i in dace.map[0:10]:
            if mask[i] > 0:
                out[i] = in1[i - 1]
            else:
                out[i] = in1[i]

    inp = np.random.rand(10)
    mask = np.ones((10, ))
    mask[0] = 0
    outp = np.zeros((10, ))
    rconditional(inp, outp, mask)
    expected = inp.copy()
    expected[1:] = inp[0:-1]
    assert np.allclose(outp, expected)


def test_nsdfg_memlet_propagation_with_one_sparse_dimension():
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def sparse(A: dace.float32[M, N], ind: dace.int32[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            A[i, ind[i, j]] += 1

    sdfg = sparse.to_sdfg(simplify=False)
    propagate_memlets_sdfg(sdfg)

    # Verify all memlet subsets and volumes in the main state of the program, i.e. around the NSDFG.
    map_state = sdfg.states()[1]
    i = dace.symbol('i')
    j = dace.symbol('j')

    outer_in = map_state.edges()[0].data
    if outer_in.volume != M * N:
        raise RuntimeError('Expected a volume of M*N on the outer input memlet')
    if outer_in.subset[0] != (0, M - 1, 1) or outer_in.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected subset of outer in memlet to be [0:M, 0:N], found ' + str(outer_in.subset))

    inner_in = map_state.edges()[1].data
    if inner_in.volume != 1:
        raise RuntimeError('Expected a volume of 1 on the inner input memlet')
    if inner_in.subset[0] != (i, i, 1) or inner_in.subset[1] != (j, j, 1):
        raise RuntimeError('Expected subset of inner in memlet to be [i, j], found ' + str(inner_in.subset))

    inner_out = map_state.edges()[2].data
    if inner_out.volume != 1:
        raise RuntimeError('Expected a volume of 1 on the inner output memlet')
    # TODO: (frontend issue) The index `i` is not extracted out of the tasklet for some reason
    if inner_out.subset[0] != (i, i, 1) or inner_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected subset of inner out memlet to be [i, 0:N], found ' + str(inner_out.subset))

    outer_out = map_state.edges()[3].data
    if outer_out.volume != M * N:
        raise RuntimeError('Expected a volume of M*N on the outer output memlet')
    if outer_out.subset[0] != (0, M - 1, 1) or outer_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected subset of outer out memlet to be [0:M, 0:N], found ' + str(outer_out.subset))


def test_nested_conditional_in_loop_in_map():
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def nested_conditional_in_loop_in_map(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(2, N, 1):
                if A[0][0]:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
                A[i, j] = A[i, j] * A[i, j]

    sdfg = nested_conditional_in_loop_in_map.to_sdfg(simplify=True)
    dace.propagate_memlets_sdfg(sdfg)

    # TODO: (frontend issue) A[0][0] does not exist in the SDFG

    # Verify that the memlet propagation works correctly
    i = dace.symbol('i')
    state = sdfg.source_nodes()[0]
    rnode = state.source_nodes()[0]
    # Input memlets for A should be [0:M, 2:N] (immediately outside of nested SDFG should be [0:i+1, 0:N])
    out_edges = state.out_edges(rnode)
    assert len(out_edges) == 1
    assert out_edges[0].data.subset.ranges == [(0, M - 1, 1), (0, N - 1, 1)]
    nsdfg_node = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert state.in_edges(nsdfg_node)[0].data.subset.ranges == [(0, i, 1), (0, N - 1, 1)]
    # Output memlets for A should be [0:M, 2:N] (immediately outside of nested SDFG should be [i, 2:N])
    wnode = state.sink_nodes()[0]
    in_edges = state.in_edges(wnode)
    assert len(in_edges) == 1
    assert in_edges[0].data.subset.ranges == [(0, M - 1, 1), (2, N - 1, 1)]
    assert state.out_edges(nsdfg_node)[0].data.subset.ranges == [(i, i, 1), (2, N - 1, 1)]

    N = 20
    M = 20
    a_test = np.zeros((M, N), dtype=np.float64)
    sdfg(a_test, M=M, N=N)
    a_valid = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(2, N, 1):
            a_valid[i, j] = 4.0

    assert np.allclose(a_test, a_valid)


if __name__ == '__main__':
    test_conditional()
    test_conditional_nested()
    test_runtime_conditional()
    test_nsdfg_memlet_propagation_with_one_sparse_dimension()
    test_nested_conditional_in_loop_in_map()
