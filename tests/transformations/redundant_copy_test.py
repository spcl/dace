# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.libraries.blas import Transpose


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


def test_out_dtd_11_success():
    sdfg = dace.SDFG("redundant_copy_out_dtd_11_success")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)
    # sdfg.add_array("C", [2, 5, 5, 5], dace.float32)
    sdfg.add_array("D", [5, 5, 5, 5], dace.float32)
    sdfg.add_array("E", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")
    D = state.add_access("D")
    E = state.add_access("E")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2"))
    state.add_nedge(B, D, dace.Memlet.simple("B", "0:2, 2",
                                             other_subset_str="0:2, 3, 4, 2"))
    state.add_nedge(B, E, dace.Memlet.simple("E", "1, 2, 3, 4",
                                             other_subset_str="1, 2"))
    
    # sdfg.apply_strict_transformations()
    # assert len(state.nodes()) == 4
    # assert B not in state.nodes()
    # sdfg.validate()

    A_arr = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    C_arr = np.zeros([5, 5, 5, 5], dtype=np.float32)
    # C_arr = np.zeros([2, 5, 5, 5], dtype=np.float32)
    D_arr = np.zeros([5, 5, 5, 5], dtype=np.float32)
    E_arr = np.zeros([5, 5, 5, 5], dtype=np.float32)

    sdfg(A=A_arr, C=C_arr, D=D_arr, E=E_arr)
    assert np.array_equal(A_arr[0, 0:3, 4], C_arr[0:3, 0, 0, 0])
    # assert np.array_equal(A_arr[0, 0:2, 4], C_arr[0:2, 0, 0, 0])
    assert np.array_equal(A_arr[0, 0:2, 4], D_arr[0:2, 3, 4, 2])
    assert np.array_equal(A_arr[0, 1, 4], E_arr[1, 2, 3, 4])


def test_out_dtd_11_fail():
    sdfg = dace.SDFG("redundant_copy_out_dtd_11_fail")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)
    sdfg.add_array("D", [5, 5, 5, 5], dace.float32)
    sdfg.add_array("E", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")
    D = state.add_access("D")
    E = state.add_access("E")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2"))
    state.add_nedge(B, C, dace.Memlet.simple("B", "0:3, 2"))
    state.add_nedge(B, D, dace.Memlet.simple("B", "0:2, 2",
                                             other_subset_str="0:2, 3, 4, 2"))
    state.add_nedge(B, E, dace.Memlet.simple("E", "1, 2, 3, 4",
                                             other_subset_str="1, 2"))
    
    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 5
    assert B in state.nodes()
    sdfg.validate()


def test_out_dtd_13_success():
    sdfg = dace.SDFG("redundant_copy_out_dtd_13_success")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")

    state.add_nedge(A, B, dace.Memlet.simple("A", "0, 0:3, 2:5"))
    state.add_nedge(B, C, dace.Memlet.simple("C", "1, 0:3, 4, 2"))
    
    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 2
    assert B not in state.nodes()
    sdfg.validate()

    A_arr = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    C_arr = np.zeros([5, 5, 5, 5], dtype=np.float32)

    sdfg(A=A_arr, C=C_arr)
    assert np.array_equal(A_arr[0, 0:3, 2], C_arr[1, 0:3, 4, 2])


def test_out_full():
    sdfg = dace.SDFG("redundant_copy_out_full")
    state = sdfg.add_state()

    sdfg.add_array("A", [5, 5, 5], dace.float32)
    sdfg.add_transient("B", [3, 3], dace.float32)
    sdfg.add_array("C", [5, 5, 5, 5], dace.float32)
    sdfg.add_array("D", [2, 5, 4], dace.float32)

    A = state.add_access("A")
    B = state.add_access("B")
    C = state.add_access("C")
    D = state.add_access("D")

    state.add_edge(A, None, B, None, dace.Memlet.simple("A", "0, 0:3, 2:5"))
    state.add_edge(B, None, C, None, dace.Memlet.simple("C", "0:3, 2, 0:3, 4"))
    state.add_edge(B, None, D, None, dace.Memlet.simple(
        "B", "0:2, 2", other_subset_str="0:2, 3, 2"))
    
    sdfg.save("rco_full_before.sdfg")
    sdfg.apply_strict_transformations()
    sdfg.save("rco_full_after.sdfg")
    assert len(state.nodes()) == 3
    assert B not in state.nodes()
    sdfg.validate()

    A_arr = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    C_arr = np.zeros([5, 5, 5, 5], dtype=np.float32)
    D_arr = np.zeros([2, 5, 4], dtype=np.float32)

    sdfg(A=A_arr, C=C_arr, D=D_arr)
    print(A_arr[0, 0:3, 2:5])
    print(C_arr[0:3, 2, 0:3, 4])
    print(D_arr[0:2, 3, 2])
    assert np.array_equal(A_arr[0, 0:3, 2:5], C_arr[0:3, 2, 0:3, 4])
    assert np.array_equal(A_arr[0, 0:2, 4], D_arr[0:2, 3, 2])


def test_out_2():
    sdfg = dace.SDFG("test_redundant_copy_out_2")
    state = sdfg.add_state()
    sdfg.add_array("A", [3, 3], dace.float32)
    sdfg.add_transient("C", [3, 6], dace.float32)
    sdfg.add_array("D", [3, 3], dace.float32)

    A = state.add_access("A")
    C = state.add_access("C")
    trans = Transpose("transpose", dtype=dace.float32)
    D = state.add_access("D")

    state.add_edge(A, None, C, None, dace.Memlet.simple("A", "0:3, 0:3"))
    state.add_edge(C, None, trans, "_inp", dace.Memlet.simple("C", "0:3, 3:6"))
    state.add_edge(trans, "_out", D, None, sdfg.make_array_memlet("D"))

    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 4
    assert C in state.nodes()
    sdfg.validate()


def test_out_3():
    sdfg = dace.SDFG("test_redundant_copy_out_3")
    state = sdfg.add_state()
    sdfg.add_array("A", [5, 5], dace.float32)
    sdfg.add_transient("C", [6, 6], dace.float32)
    sdfg.add_array("D", [3, 3], dace.float32)

    A = state.add_access("A")
    C = state.add_access("C")
    trans = Transpose("transpose", dtype=dace.float32)
    D = state.add_access("D")

    state.add_edge(A, None, C, None, dace.Memlet.simple("A", "1:5, 1:5", other_subset_str="2:6, 2:6"))
    state.add_edge(C, None, trans, "_inp", dace.Memlet.simple("C", "3:6, 2:5"))
    state.add_edge(trans, "_out", D, None, sdfg.make_array_memlet("D"))

    sdfg.apply_strict_transformations()
    assert len(state.nodes()) == 3
    assert C not in state.nodes()
    sdfg.validate()

    A_arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    D_arr = np.zeros((3, 3), dtype=np.float32)
    sdfg(A=A_arr, D=D_arr)
    assert (A_arr[2:5, 1:4] == D_arr.T).all()


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
    # test_in()
    # test_out()
    # test_out_2()
    # test_out_3()
    # test_out_full()
    test_out_dtd_11_success()
    # test_out_dtd_11_fail()
    # test_out_dtd_13_success()
