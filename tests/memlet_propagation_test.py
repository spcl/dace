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
    if inner_out.subset[0] != (0, i, 1) or inner_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected subset of inner out memlet to be [0:i+1, 0:N], found ' +
            str(inner_out.subset))

    outer_out = map_state.edges()[3].data
    if outer_out.volume != M * N:
        raise RuntimeError('Expected a volume of M*N on the outer output memlet')
    if outer_out.subset[0] != (0, M - 1, 1) or outer_out.subset[1] != (0, N - 1, 1):
        raise RuntimeError('Expected subset of outer out memlet to be [0:M, 0:N], found ' +
            str(outer_out.subset))


def test_nsdfg_memlet_propagation_with_slicing():
    
    dim_X, dim_Y = (dace.symbol(s) for s in ('dim_X', 'dim_Y'))

    def build_nsdfg():
        sdfg = dace.SDFG('mat_to_vec')
        sdfg.add_array('_inp', (dim_X, dim_Y), dace.float64)
        sdfg.add_array('_out', (dim_X,), dace.float64)
        sdfg.add_scalar('_inp_idx', dace.int32)
        state1 = sdfg.add_state()
        state2 = sdfg.add_state()
        sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments={'y':'_inp_idx'}))
        state2.add_edge(
            state2.add_access('_inp'), None,
            state2.add_access('_out'), None,
            dace.Memlet(data='_out', subset='0:dim_X', other_subset='0:dim_X, y')
        )
        return sdfg

    sdfg = dace.SDFG('memlet_propagation_with_slicing')
    sdfg.add_array('mat', (dim_X, dim_Y), dace.float64)
    sdfg.add_array('vec', (dim_X,), dace.float64)
    sdfg.add_symbol('idx_Y', dace.int32)
    sdfg.add_scalar('idx', dace.int32, transient=True)

    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(
        build_nsdfg(),
        sdfg,
        inputs={'_inp', '_inp_idx'},
        outputs={'_out'},
    )
    idx_node = state.add_access('idx')
    state.add_edge(
        state.add_tasklet('get_idx_Y', {}, {'x'}, 'x = idx_Y'),
        'x',
        idx_node,
        None,
        dace.Memlet.simple('idx', '0')
    )
    state.add_edge(
        idx_node,
        None,
        nsdfg_node,
        '_inp_idx',
        dace.Memlet.simple('idx', '0')
    )
    state.add_edge(
        state.add_access('mat'),
        None,
        nsdfg_node,
        '_inp',
        dace.Memlet.from_array('mat', sdfg.arrays['mat'])
    )
    state.add_edge(
        nsdfg_node,
        '_out',
        state.add_access('vec'),
        None,
        dace.Memlet.from_array('vec', sdfg.arrays['vec'])
    )

    propagate_memlets_sdfg(sdfg)

    dim_X.set(10)
    dim_Y.set(20)
    idx_Y = 3
    
    A = np.random.rand(dim_X.get(), dim_Y.get())
    B = np.random.rand(dim_X.get())
    ref = A[:, idx_Y]

    sdfg(mat=A, vec=B, idx_Y=idx_Y, dim_X=dim_X, dim_Y=dim_Y)
    np.allclose(ref, B)


if __name__ == '__main__':
    test_conditional()
    test_conditional_nested()
    test_runtime_conditional()
    test_nsdfg_memlet_propagation_with_one_sparse_dimension()
    test_nsdfg_memlet_propagation_with_slicing()
