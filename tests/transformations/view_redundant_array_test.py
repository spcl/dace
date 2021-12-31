# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import itertools

import dace
from dace import nodes, data
from dace.transformation.dataflow.redundant_array import RedundantArray, UnsqueezeViewRemove


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
        for n, _ in test_redundant_array_removal.to_sdfg(
            strict=True).all_nodes_recursive()
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
    sdfg.coarsen_dataflow()

    A = np.random.rand(8, 2, 4).astype(np.float64)
    B = np.random.rand(4, 3).astype(np.float64)
    C = test_broken_matmul(A.copy(), B.copy())

    assert np.allclose(A @ B, C)


@pytest.mark.parametrize(["copy_subset", "nonstrict"],
                         list(itertools.product(["O", "T"], [False, True])))
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
    copy_state.add_edge(copy_state.add_read("T"), None,
                        copy_state.add_write("O"), None,
                        sdfg.make_array_memlet(copy_subset))

    sdfg.coarsen_dataflow()
    if nonstrict:
        sdfg.apply_transformations_repeated(RedundantArray)

        # Ensure a view is created
        assert (len([
            n for n in sdfg.node(0).data_nodes()
            if type(n.desc(sdfg)) is data.Array
        ]) == 2)

    I = np.ones((9, )).astype(np.float32)
    O = np.zeros((3, 3)).astype(np.float32)
    sdfg(I=I, O=O)
    assert np.allclose(O.flatten(), I + 1)


@pytest.mark.parametrize(["copy_subset", "nonstrict"],
                         list(itertools.product(["O", "T"], [False, True])))
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
    copy_state.add_edge(copy_state.add_read("T"), None,
                        copy_state.add_write("O"), None,
                        sdfg.make_array_memlet(copy_subset))

    sdfg.coarsen_dataflow()
    if nonstrict:
        sdfg.apply_transformations_repeated(RedundantArray)

        # Ensure a view is created
        assert (len([
            n for n in sdfg.node(0).data_nodes()
            if type(n.desc(sdfg)) is data.Array
        ]) == 2)

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
    state.add_edge(tnode, None, state.add_write("O"), None,
                   sdfg.make_array_memlet("O"))
    state.add_mapped_tasklet("set_one",
                             dict(i="0:9"), {},
                             "out = 1",
                             dict(out=dace.Memlet("T[i]")),
                             external_edges=True,
                             output_nodes=dict(T=tnode))

    sdfg.apply_transformations_repeated(UnsqueezeViewRemove)

    # Ensure view is removed
    assert (len([
        n for n in sdfg.node(0).data_nodes()
        if isinstance(n.desc(sdfg), data.View)
    ]) == 0)

    O = np.zeros((1, 9, 1)).astype(np.float32)
    sdfg(O=O)
    assert np.allclose(O, 1)


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
