# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.libraries.blas.nodes import Dot


def test_conjugate_dot_conjugates_the_first_operand():
    sdfg = dace.SDFG("conjugate_dot")
    for name, shape in (("x", [8]), ("y", [8]), ("r", [1])):
        sdfg.add_array(name, shape, dace.complex128)
    state = sdfg.add_state()
    node = Dot("dot", conjugate=True)
    node.implementation = "pure"
    state.add_edge(state.add_read("x"), None, node, "_x", dace.Memlet("x[0:8]"))
    state.add_edge(state.add_read("y"), None, node, "_y", dace.Memlet("y[0:8]"))
    state.add_edge(node, "_result", state.add_write("r"), None, dace.Memlet("r[0]"))
    x = np.arange(8) + 1j * np.arange(8, 0, -1)
    y = np.arange(8, 0, -1) - 2j * np.arange(8)
    r = np.zeros(1, dtype=np.complex128)
    sdfg(x=x, y=y, r=r)
    assert np.allclose(r[0], np.vdot(x, y))


if __name__ == "__main__":
    test_conjugate_dot_conjugates_the_first_operand()
