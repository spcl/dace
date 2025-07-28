# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import itertools

import numpy as np
import pytest

import dace
from dace import data
from dace.transformation.dataflow.redundant_array import RedundantArray, RemoveSliceView, UnsqueezeViewRemove


def test_redundant_array_removal():

    @dace.program
    def reshape(data: dace.float64[9], reshaped: dace.float64[3, 3]):
        reshaped[:] = np.reshape(data, [3, 3])

    @dace.program
    def test_redundant_array_removal(A: dace.float64[9], B: dace.float64[3]):
        A_reshaped = dace.define_local([3, 3], dace.float64)
        reshape(A, A_reshaped)
        return A_reshaped + B

    data_accesses = {
        n.data
        for n, _ in test_redundant_array_removal.to_sdfg(simplify=True).all_nodes_recursive()
        if isinstance(n, dace.nodes.AccessNode)
    }
    assert "A_reshaped" not in data_accesses

    A = np.arange(9).astype(np.float64)
    B = np.arange(3).astype(np.float64)
    result = test_redundant_array_removal(A.copy(), B.copy())
    assert np.allclose(result, A.reshape(3, 3) + B)


@pytest.mark.gpu
def test_libnode_expansion():

    @dace.program
    def test_broken_matmul(A: dace.float64[8, 2, 4], B: dace.float64[4, 3]):
        return np.einsum("aik,kj->aij", A, B)

    sdfg = test_broken_matmul.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_gpu_transformations()
    sdfg.simplify()

    A = np.random.rand(8, 2, 4).astype(np.float64)
    B = np.random.rand(4, 3).astype(np.float64)
    C = test_broken_matmul(A.copy(), B.copy())

    assert np.allclose(A @ B, C)


@pytest.mark.parametrize(["copy_subset", "nonstrict"], list(itertools.product(["O", "T"], [False, True])))
def test_redundant_array_1_into_2_dims(copy_subset, nonstrict):
    sdfg = dace.SDFG("testing")
    state = sdfg.add_state()

    sdfg.add_array("I", [9], dtype=dace.float32, transient=False)
    sdfg.add_array("T", [9], dtype=dace.float32, transient=True)
    sdfg.add_array("O", [3, 3], dtype=dace.float32, transient=False)

    state.add_mapped_tasklet("add_one",
                             dict(i="0:9"),
                             dict(inp=dace.Memlet("I[i]")),
                             "out = inp + 1",
                             dict(out=dace.Memlet("T[i]")),
                             external_edges=True)
    copy_state = sdfg.add_state_after(state)
    copy_state.add_edge(copy_state.add_read("T"), None, copy_state.add_write("O"), None,
                        sdfg.make_array_memlet(copy_subset))

    sdfg.simplify()
    if nonstrict:
        sdfg.apply_transformations_repeated(RedundantArray, permissive=True)

        # Ensure a view is created
        assert (len([n for n in sdfg.node(0).data_nodes() if type(n.desc(sdfg)) is data.Array]) == 2)

    I = np.ones((9, )).astype(np.float32)
    O = np.zeros((3, 3)).astype(np.float32)
    sdfg(I=I, O=O)
    assert np.allclose(O.flatten(), I + 1)


@pytest.mark.parametrize(["copy_subset", "nonstrict"], list(itertools.product(["O", "T"], [False, True])))
def test_redundant_array_2_into_1_dim(copy_subset, nonstrict):
    sdfg = dace.SDFG("testing")
    state = sdfg.add_state()

    sdfg.add_array("I", [3, 3], dtype=dace.float32, transient=False)
    sdfg.add_array("T", [3, 3], dtype=dace.float32, transient=True)
    sdfg.add_array("O", [9], dtype=dace.float32, transient=False)

    state.add_mapped_tasklet("add_one",
                             dict(i="0:3", j="0:3"),
                             dict(inp=dace.Memlet("I[i, j]")),
                             "out = inp + 1",
                             dict(out=dace.Memlet("T[i, j]")),
                             external_edges=True)
    copy_state = sdfg.add_state_after(state)
    copy_state.add_edge(copy_state.add_read("T"), None, copy_state.add_write("O"), None,
                        sdfg.make_array_memlet(copy_subset))

    sdfg.simplify()
    if nonstrict:
        sdfg.apply_transformations_repeated(RedundantArray, permissive=True)

        # Ensure a view is created
        assert (len([n for n in sdfg.node(0).data_nodes() if type(n.desc(sdfg)) is data.Array]) == 2)

    I = np.ones((3, 3)).astype(np.float32)
    O = np.zeros((9, )).astype(np.float32)
    sdfg(I=I, O=O)
    assert np.allclose(O, (I + 1).flatten())


def test_unsqueeze_view_removal():
    sdfg = dace.SDFG("testing")
    state = sdfg.add_state()

    sdfg.add_view("T", [9], dtype=dace.float32)
    sdfg.add_array("O", [1, 9, 1], dtype=dace.float32, transient=False)

    tnode = state.add_access("T")
    state.add_edge(tnode, None, state.add_write("O"), None, sdfg.make_array_memlet("O"))
    state.add_mapped_tasklet("set_one",
                             dict(i="0:9"), {},
                             "out = 1",
                             dict(out=dace.Memlet("T[i]")),
                             external_edges=True,
                             output_nodes=dict(T=tnode))

    sdfg.apply_transformations_repeated(UnsqueezeViewRemove)

    # Ensure view is removed
    assert (len([n for n in sdfg.node(0).data_nodes() if isinstance(n.desc(sdfg), data.View)]) == 0)

    O = np.zeros((1, 9, 1)).astype(np.float32)
    sdfg(O=O)
    assert np.allclose(O, 1)


def test_view_offset_removal():
    sdfg = dace.SDFG("testing")
    state = sdfg.add_state()

    i = dace.symbol('i')
    sdfg.add_array('inout', [20, 20], dtype=dace.float64, transient=False)
    sdfg.add_transient('tmp', [20 - i], dtype=dace.float64)
    sdfg.add_view('view', [20 - i], dtype=dace.float64)

    r = state.add_read('inout')
    w = state.add_write('inout')
    t = state.add_access('tmp')
    v = state.add_access('view')
    state.add_edge(r, None, v, 'views', dace.Memlet('inout[1, i:20]'))
    state.add_edge(v, None, t, None, dace.Memlet('tmp[0:20-i]'))
    state.add_edge(t, None, w, None, dace.Memlet('inout[2, i:20]'))

    assert sdfg.apply_transformations_repeated(RemoveSliceView) == 1

    inout = np.random.rand(20, 20)

    sdfg(inout=inout, i=1)

    assert np.allclose(inout[1, 1:], inout[2, 1:])


def test_transient_removal_uneven_flow_through_map():
    g = dace.SDFG("testing")
    st0 = g.add_state()
    st1 = g.add_state_after(st0)

    N = dace.symbol('N')

    X, _ = g.add_array('X', [N, N], dtype=dace.float64)
    T0, _ = g.add_transient('T0', [N, N], dtype=dace.float64)
    T1, _ = g.add_transient('T1', [N, N], dtype=dace.float64)
    Y, _ = g.add_array('Y', [N, N], dtype=dace.float64)
    T1a = st0.add_access(T1)
    X, T0, T1b, Y = tuple(st1.add_access(u) for u in (X, T0, T1, Y))

    # Initialize T1 with zeros.
    mE, mX = st0.add_map('T1_set0', dict(i='0:N', j='0:N'))
    mX.add_scope_connectors('out')
    zt = st0.add_tasklet('set_zero', inputs={}, outputs={'out'}, code="out = 0.0")
    st0.add_edge(mE, None, zt, None, dace.Memlet())
    st0.add_edge(zt, 'out', mX, 'IN_out', dace.Memlet(f"{T1a.data}[i, j]"))
    st0.add_edge(mX, 'OUT_out', T1a, None, g.make_array_memlet(T1a.data))
    # Write the full T0 transient.
    c = st1.add_tasklet('copy', inputs={'inp'}, outputs={'out'}, code="out = inp")
    mE, mX = st1.add_map('copy_map', dict(i='0:N', j='0:N'))
    mE.add_scope_connectors('inp')
    mX.add_scope_connectors('out')
    st1.add_edge(X, None, mE, 'IN_inp', g.make_array_memlet(X.data))
    st1.add_edge(mE, 'OUT_inp', c, 'inp', dace.Memlet(f"{X.data}[i, j]"))
    st1.add_edge(c, 'out', mX, 'IN_out', dace.Memlet(f"{T0.data}[i, j]"))
    st1.add_edge(mX, 'OUT_out', T0, None, g.make_array_memlet(T0.data))
    # Forward only the boundary to the T1 transient.
    st1.add_edge(T0, None, T1b, None, dace.Memlet(f"{T0.data}[0:N, N-1] -> [0:N, N-1]"))
    # Forward the full transient.
    st1.add_edge(T1b, None, Y, None, dace.Memlet(f"{T1b.data}[0:N, 0:N] -> [0:N, 0:N]"))

    assert g.apply_transformations_repeated(RedundantArray) == 0

    Xin = np.random.rand(5, 5)
    Yout = np.zeros_like(Xin)
    g(X=Xin, Y=Yout, N=5)

    np.testing.assert_allclose(Yout[:, :4], 0)
    np.testing.assert_allclose(Yout[:, 4], Xin[:, 4])


if __name__ == '__main__':
    test_redundant_array_removal()
    test_redundant_array_1_into_2_dims("O", False)
    test_redundant_array_1_into_2_dims("T", False)
    test_redundant_array_1_into_2_dims("O", True)
    test_redundant_array_1_into_2_dims("T", True)
    test_redundant_array_2_into_1_dim("O", False)
    test_redundant_array_2_into_1_dim("T", False)
    test_redundant_array_2_into_1_dim("O", True)
    test_redundant_array_2_into_1_dim("T", True)
    test_unsqueeze_view_removal()
    test_view_offset_removal()
