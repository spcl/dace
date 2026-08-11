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

    # Verify the memlet subsets and volumes around the map that performs the
    # sparse write. The edges are located by role rather than by position: what
    # sits inside the map (a nested SDFG or a single tasklet) is a frontend
    # choice, but the propagation either way has to reach the whole array
    # outside and a single element inside.
    map_state = next(s for s in sdfg.states() if any(isinstance(n, dace.nodes.MapEntry) for n in s.nodes()))
    map_entry = next(n for n in map_state.nodes() if isinstance(n, dace.nodes.MapEntry))
    map_exit = next(n for n in map_state.nodes() if isinstance(n, dace.nodes.MapExit))
    i = dace.symbol('i')
    j = dace.symbol('j')

    outer_in = next(e.data for e in map_state.in_edges(map_entry) if e.data.data == 'ind')
    if outer_in.volume != M * N:
        raise RuntimeError('Expected a volume of M*N on the outer input memlet')
    if outer_in.subset[0] != (0, M - 1, 1) or outer_in.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected subset of outer in memlet to be [0:M, 0:N], found ' + str(outer_in.subset))

    inner_in = next(e.data for e in map_state.out_edges(map_entry) if e.data.data == 'ind')
    if inner_in.volume != 1:
        raise RuntimeError('Expected a volume of 1 on the inner input memlet')
    if inner_in.subset[0] != (i, i, 1) or inner_in.subset[1] != (j, j, 1):
        raise RuntimeError('Expected subset of inner in memlet to be [i, j], found ' + str(inner_in.subset))

    # One element is written per iteration, but the column is data-dependent, so
    # the subset has to span the row it lands in.
    inner_out = next(e.data for e in map_state.in_edges(map_exit) if e.data.data == 'A')
    if inner_out.volume != 1:
        raise RuntimeError('Expected a volume of 1 on the inner output memlet')
    if inner_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected inner out memlet to span the whole row, found ' + str(inner_out.subset))

    outer_out = next(e.data for e in map_state.out_edges(map_exit) if e.data.data == 'A')
    if outer_out.volume != M * N:
        raise RuntimeError('Expected a volume of M*N on the outer output memlet')
    if outer_out.subset[0] != (0, M - 1, 1) or outer_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected subset of outer out memlet to be [0:M, 0:N], found ' + str(outer_out.subset))


if __name__ == '__main__':
    test_conditional()
    test_conditional_nested()
    test_runtime_conditional()
    test_nsdfg_memlet_propagation_with_one_sparse_dimension()
