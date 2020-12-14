# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.sdfg import propagation as prop
from dace import nodes
import dace.library
from dace.transformation import transformation as xf


def test_unsqueeze():
    """ Tests for an issue in unsqueeze not allowing reshape. """
    @dace.program
    def callee(A: dace.float64[60, 2]):
        A[:, 1] = 5.0

    @dace.program
    def caller(A: dace.float64[2, 3, 4, 5]):
        callee(A)

    A = np.random.rand(2, 3, 4, 5)
    expected = A[:]
    expected.reshape(60, 2)[:, 1] = 5.0

    sdfg = caller.to_sdfg()
    prop.propagate_memlets_sdfg(sdfg)
    sdfg(A=A)

    assert np.allclose(A, expected)


def test_apply_to_matmul():
    @dace.library.node
    class MyNode(nodes.LibraryNode):
        implementations = {}
        default_implementation = None
        _dace_library_name = "alsdkjalskdj"

    @dace.library.expansion
    class Expansion(xf.ExpandTransformation):
        environments = []

        @classmethod
        def expansion(cls, node, state, sdfg):
            @dace.program
            def matmul(A: dace.float32[8, 12, 10, 10],
                       B: dace.float32[8, 12, 10, 10], C: dace.float32[8, 12,
                                                                       10, 10]):
                C[:] = np.einsum('abik,abkj->abij', A, B)

            return matmul.to_sdfg()

    MyNode.register_implementation("expand", Expansion)

    sdfg = dace.SDFG("test_matmul_apply_to")
    sdfg.add_datadesc("A", dace.float32[8, 12, 10, 10])
    sdfg.add_datadesc("B", dace.float32[8, 12, 10, 10])
    sdfg.add_datadesc("C", dace.float32[8, 12, 10, 10])

    state = sdfg.add_state()
    A = state.add_read("A")
    B = state.add_read("B")
    C = state.add_write("C")

    node = MyNode(name="mynode", inputs={"A", "B"}, outputs={"C"})
    node.implementation = "expand"
    state.add_node(node)
    state.add_edge(A, None, node, "A", sdfg.make_array_memlet("A"))
    state.add_edge(B, None, node, "B", sdfg.make_array_memlet("B"))
    state.add_edge(node, "C", C, None, sdfg.make_array_memlet("C"))

    Expansion._match_node = xf.PatternNode(type(node))
    Expansion.apply_to(sdfg, verify=False, _match_node=node)
    A = np.random.rand(8, 12, 10, 10).astype(np.float32)
    B = np.random.rand(8, 12, 10, 10).astype(np.float32)
    C = np.zeros_like(A)
    sdfg(A=A, B=B, C=C)

    assert np.allclose(C, np.matmul(A, B))


if __name__ == '__main__':
    test_unsqueeze()
    test_apply_to_matmul()
