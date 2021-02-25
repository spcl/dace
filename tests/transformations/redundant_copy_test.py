# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.libraries.blas import Transpose
from dace.transformation.dataflow import RedundantSecondArray

def test_out():
    sdfg = dace.SDFG("test_redundant_copy_out")
    state = sdfg.add_state()
    sdfg.add_array("A", [3, 3], dace.float32)
    sdfg.add_transient("B", [3, 3],
                       dace.float32,
                       storage=dace.StorageType.GPU_Global)
    sdfg.add_transient("C", [3, 3], dace.float32)
    sdfg.add_array("D", [3, 3], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")
    trans = Transpose("transpose", dtype=dace.float32)
    D = state.add_access("D")

    state.add_edge(A, None, B, None, sdfg.make_array_memlet("A"))
    state.add_edge(B, None, C, None, sdfg.make_array_memlet("B"))
    state.add_edge(C, None, trans, "_inp", sdfg.make_array_memlet("C"))
    state.add_edge(trans, "_out", D, None, sdfg.make_array_memlet("D"))

    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 3
    assert B not in state.nodes()
    sdfg.validate()

    A_arr = np.arange(9, dtype=np.float32).reshape(3, 3)
    D_arr = np.zeros_like(A_arr)
    sdfg(A=A_arr, D=D_arr)
    assert (A_arr == D_arr.T).all()


def test_out_success():
    sdfg = dace.SDFG("test_redundant_copy_out_success")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2",
                                             other_subset_str="1, 2, 0:3, 4"))

    sdfg.add_scalar("D", dace.float32, transient=True)
    sdfg.add_array("E", [3, 3, 3], dace.float32)

    me, mx = state.add_map("Map", dict(i='0:3', j='0:3', k='0:3'))
    t = state.add_tasklet("Tasklet", {'__in1', '__in2'}, {'__out'},
                          "__out = __in1 + __in2")
    D = state.add_access("D")
    E = state.add_access("E")

    state.add_memlet_path(B, me, t, memlet=dace.Memlet.simple("B", "i, j"),
                          dst_conn='__in1')
    state.add_memlet_path(B, me, D, memlet=dace.Memlet.simple("B", "j, k"))
    state.add_edge(D, None, t, '__in2', dace.Memlet.simple("D", "0"))
    state.add_memlet_path(t, mx, E, memlet=dace.Memlet.simple("E", "i, j, k"),
                          src_conn='__out')
    

    sdfg.validate()
    sdfg.apply_strict_transformations()
    sdfg.apply_transformations_repeated(RedundantSecondArray)
    assert len(state.nodes()) == 7
    assert B not in state.nodes()
    sdfg.validate()

    A_arr = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    C_arr = np.zeros([5, 5, 5, 5], dtype=np.float32)
    E_arr = np.zeros([3, 3, 3], dtype=np.float32)

    E_ref = np.zeros([3, 3, 3], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                E_ref[i, j, k] = A_arr[0, i, 2 + j] + A_arr[0, j, 2 + k]

    sdfg(A=A_arr, C=C_arr, E=E_arr)
    # This fails, probably due to a bug in the code generator
    # assert np.array_equal(A_arr[0, 0:3, 4], C_arr[1, 2, 0:3, 4])
    assert np.array_equal(E_ref, E_arr)


def test_out_failure_subset_mismatch():
    sdfg = dace.SDFG("test_rco_failure_subset_mismatch")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [8, 8], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2",
                                             other_subset_str="1, 2, 0:3, 4"))

    sdfg.validate()
    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 3
    assert B in state.nodes()


def test_out_failure_no_overlap():
    sdfg = dace.SDFG("test_rco_failure_no_overlap")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [8, 8], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5",
                                             other_subset_str="5:8, 5:8"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2",
                                             other_subset_str="1, 2, 0:3, 4"))

    sdfg.validate()
    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 3
    assert B in state.nodes()


def test_out_failure_partial_overlap():
    sdfg = dace.SDFG("test_rco_failure_partial_overlap")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [8, 8], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5",
                                             other_subset_str="5:8, 5:8"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "4:7, 6",
                                             other_subset_str="1, 2, 0:3, 4"))

    sdfg.validate()
    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 3
    assert B in state.nodes()


def test_in():
    sdfg = dace.SDFG("test_redundant_copy_in")
    state = sdfg.add_state()
    sdfg.add_array("A", [3, 3], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_transient("C", [3, 3],
                       dace.float32,
                       storage=dace.StorageType.GPU_Global)
    sdfg.add_array("D", [3, 3], dace.float32)

    A = state.add_access("A")
    trans = Transpose("transpose", dtype=dace.float32)
    state.add_node(trans)
    B = state.add_access("B")
    C = state.add_access("C")
    D = state.add_access("D")

    state.add_edge(A, None, trans, "_inp", sdfg.make_array_memlet("A"))
    state.add_edge(trans, "_out", B, None, sdfg.make_array_memlet("B"))
    state.add_edge(B, None, C, None, sdfg.make_array_memlet("B"))
    state.add_edge(C, None, D, None, sdfg.make_array_memlet("C"))

    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 3
    assert C not in state.nodes()
    sdfg.validate()

    A_arr = np.arange(9, dtype=np.float32).reshape(3, 3)
    D_arr = np.zeros_like(A_arr)
    sdfg(A=A_arr, D=D_arr)
    assert (A_arr == D_arr.T).all()


if __name__ == '__main__':
    test_in()
    test_out()
    test_out_success()
    test_out_failure_subset_mismatch()
    test_out_failure_no_overlap()
    test_out_failure_partial_overlap()
