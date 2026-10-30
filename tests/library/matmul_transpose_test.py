# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest

import dace
from dace.libraries.blas import MatMul


@pytest.mark.parametrize("transA,transB", [(False, False), (True, False), (False, True), (True, True)])
def test_matmul_transposes_its_operands(transA: bool, transB: bool):
    m, k, n = 5, 7, 11
    a_shape = (k, m) if transA else (m, k)
    b_shape = (n, k) if transB else (k, n)
    sdfg = dace.SDFG(f"matmul_transpose_{int(transA)}{int(transB)}")
    sdfg.add_array("A", a_shape, dace.float64)
    sdfg.add_array("B", b_shape, dace.float64)
    sdfg.add_array("C", (m, n), dace.float64)
    state = sdfg.add_state()
    node = MatMul("mm", transA=transA, transB=transB)
    state.add_edge(state.add_read("A"), None, node, "_a", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(state.add_read("B"), None, node, "_b", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(node, "_c", state.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(0)
    a, b = rng.random(a_shape), rng.random(b_shape)
    c = np.zeros((m, n))
    sdfg(A=a, B=b, C=c)
    assert np.allclose(c, (a.T if transA else a) @ (b.T if transB else b))


if __name__ == "__main__":
    for transA, transB in [(False, False), (True, False), (False, True), (True, True)]:
        test_matmul_transposes_its_operands(transA, transB)
