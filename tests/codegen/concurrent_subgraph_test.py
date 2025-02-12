# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import Memlet
import numpy as np


def test_duplicate_codegen():

    # Unfortunately I have to generate this graph manually, as doing it with the python
    # frontend wouldn't result in the node ordering that we want

    sdfg = dace.SDFG("dup")
    state = sdfg.add_state()

    c_task = state.add_tasklet("c_task", inputs={"c"}, outputs={"d"}, code='d = c')
    e_task = state.add_tasklet("e_task", inputs={"a", "d"}, outputs={"e"}, code="e = a + d")
    f_task = state.add_tasklet("f_task", inputs={"b", "d"}, outputs={"f"}, code="f = b + d")

    _, A_arr = sdfg.add_array("A", [
        1,
    ], dace.float32)
    _, B_arr = sdfg.add_array("B", [
        1,
    ], dace.float32)
    _, C_arr = sdfg.add_array("C", [
        1,
    ], dace.float32)
    _, D_arr = sdfg.add_array("D", [
        1,
    ], dace.float32)
    _, E_arr = sdfg.add_array("E", [
        1,
    ], dace.float32)
    _, F_arr = sdfg.add_array("F", [
        1,
    ], dace.float32)
    A = state.add_read("A")
    B = state.add_read("B")
    C = state.add_read("C")
    D = state.add_access("D")
    E = state.add_write("E")
    F = state.add_write("F")

    state.add_edge(C, None, c_task, "c", Memlet.from_array("C", C_arr))
    state.add_edge(c_task, "d", D, None, Memlet.from_array("D", D_arr))

    state.add_edge(A, None, e_task, "a", Memlet.from_array("A", A_arr))
    state.add_edge(B, None, f_task, "b", Memlet.from_array("B", B_arr))
    state.add_edge(D, None, f_task, "d", Memlet.from_array("D", D_arr))
    state.add_edge(D, None, e_task, "d", Memlet.from_array("D", D_arr))

    state.add_edge(e_task, "e", E, None, Memlet.from_array("E", E_arr, wcr="lambda x, y: x + y"))
    state.add_edge(f_task, "f", F, None, Memlet.from_array("F", F_arr, wcr="lambda x, y: x + y"))

    A = np.array([1], dtype=np.float32)
    B = np.array([1], dtype=np.float32)
    C = np.array([1], dtype=np.float32)
    D = np.array([1], dtype=np.float32)
    E = np.zeros_like(A)
    F = np.zeros_like(A)

    sdfg(A=A, B=B, C=C, D=D, E=E, F=F)

    assert E[0] == 2
    assert F[0] == 2


if __name__ == "__main__":
    test_duplicate_codegen()
