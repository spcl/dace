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
