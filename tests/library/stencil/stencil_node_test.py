# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.libraries.stencil import Stencil
import numpy as np

SIZE = 32
DTYPE = np.float32


def make_sdfg():

    sdfg = dace.SDFG("stencil_node_test")
    _, a_desc = sdfg.add_array("a", (SIZE, SIZE), dtype=DTYPE)
    _, b_desc = sdfg.add_array("b", (SIZE, SIZE), dtype=DTYPE)

    state = sdfg.add_state("stencil_node_test")
    a = state.add_read("a")
    b = state.add_write("b")

    stencil_node = Stencil(
        "stencil_test",
        accesses={"a": ((True, True), ((-1, 0), (1, 0), (0, -1), (0, 1)))},
        code="b[0, 0] = 0.25 * (a[-1, 0] + a[1, 0] + a[0, -1] + a[0, 1])")
    state.add_node(stencil_node)

    state.add_memlet_path(a,
                          stencil_node,
                          dst_conn="a",
                          memlet=dace.Memlet.from_array("a", a_desc))
    state.add_memlet_path(stencil_node,
                          b,
                          src_conn="b",
                          memlet=dace.Memlet.from_array("b", b_desc))

    return sdfg


def test_stencil_node():
    sdfg = make_sdfg()
    a = np.ones((SIZE, SIZE), dtype=DTYPE)
    a[1:-1, 1:-1] = 0
    b = np.empty((SIZE, SIZE), dtype=DTYPE)
    sdfg(a=a, b=b)
    assert np.allclose(
        0.25 * (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]),
        b[1:-1, 1:-1])


if __name__ == "__main__":
    test_stencil_node()
